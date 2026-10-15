"""Classification pipeline: records -> deterministic pre-filter / LLM API -> checkpoint.

One run = one model, one checkpoint, one set of outputs.

Resume
------
The checkpoint is one JSON object per line, keyed by ``hash``. Records whose hash is already
in the checkpoint are skipped, so an interrupted run (crash, Ctrl+C, exhausted daily quota)
continues from where it stopped. Every reader accesses fields with ``.get``, so checkpoints
written by older versions keep working.

What is NOT written to the checkpoint (so it is retried on the next run):
daily quota exhausted, invalid key, API unreachable/overloaded after the retries.
Blocked or unparsable answers are recorded as ``UNDETERMINABLE`` with ``parse_error=true``.
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .checkpoint import CheckpointWriter, load_checkpoint_deduped
from .deterministic import deterministic_classify
from .llm_engine import LLMAbort, LLMConfig, LLMTransientError, query_llm
from .parser import parse_llm_response
from .prompting import PROMPT_VERSION, build_prompt, diff_available, prompt_meta

MAX_CONSECUTIVE_TRANSIENT = 8   # stop after this many back-to-back transient failures


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------
@dataclass
class RunOutcome:
    status: str = "completed"          # completed | stopped | interrupted | nothing_to_do
    reason: str = ""
    n_selected: int = 0                # records selected for this run (before the resume filter)
    n_resumed: int = 0                 # already in the checkpoint
    n_new: int = 0                     # written to the checkpoint during this run
    n_remaining: int = 0               # selected but still not in the checkpoint
    n_no_diff: int = 0                 # skipped: diff truncated in the enriched file (needs --refresh)
    results: List[Dict[str, Any]] = field(default_factory=list)   # whole checkpoint, deduped


# ---------------------------------------------------------------------------
# Record processing
# ---------------------------------------------------------------------------
def process_record(record: Dict[str, Any], cfg: LLMConfig) -> Dict[str, Any]:
    """Full LLM cycle for a single record. Returns a result dict."""
    prompt = build_prompt(record)          # stateless: a pure function of this record
    meta = prompt_meta(record)
    raw_response = query_llm(prompt, cfg)
    parsed = parse_llm_response(raw_response)

    original_target = int(record.get("target", -1))
    predicted_target = parsed["predicted_target"]

    change = record.get("change") or {}
    signals = record.get("signals") or {}
    location = record.get("location") or {}

    return {
        "hash": record.get("hash"),
        "commit_id": record.get("commit_id"),
        "project": record.get("project"),
        "func_name": record.get("func_name"),
        "file": location.get("file"),
        "original_target": original_target,
        "predicted_target": predicted_target,
        "is_noise": original_target != predicted_target and predicted_target != -1,
        "category": parsed["category"],
        "method": "llm",
        "model": cfg.model_name,
        "thinking_level": cfg.thinking_level,
        "prompt_version": meta["prompt_version"],
        "diff_truncated": meta["diff_truncated"],
        "wide_sweep": meta["wide_sweep"],
        # full reasoning
        "step_1_commit_analysis":        parsed["step_1_commit_analysis"],
        "step_2_function_diff_analysis": parsed["step_2_function_diff_analysis"],
        "step_3_conclusion":             parsed["step_3_conclusion"],
        "raw_output":         parsed.get("raw_output"),
        "parse_error":        parsed["parse_error"],
        "parse_error_reason": parsed.get("parse_error_reason", ""),
        # enricher signals (downstream analysis)
        "rank_in_commit":        change.get("rank_in_commit"),
        "share_of_source_churn": change.get("share_of_source_churn"),
        "frac_changed":          change.get("frac_changed"),
        "kind":                  change.get("kind"),
        "siblings_count":        len(record.get("siblings", [])),
        "name_in_message":       signals.get("name_in_message"),
        "cwe_matches_nvd":       signals.get("cwe_matches_nvd"),
        "cwe_missing":           signals.get("cwe_missing_in_dataset", False),
    }


def _error_record(rec: Dict[str, Any], exc: Exception, model: str) -> Dict[str, Any]:
    """Minimal record so that a record-level failure is checkpointed."""
    return {
        "hash": rec.get("hash"),
        "commit_id": rec.get("commit_id"),
        "project": rec.get("project"),
        "func_name": rec.get("func_name"),
        "original_target": int(rec.get("target", -1)),
        "predicted_target": -1,
        "is_noise": False,
        "category": "UNDETERMINABLE",
        "method": "llm",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "parse_error": True,
        "parse_error_reason": str(exc),
        "raw_output": None,
    }


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
def _progress(done: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else float("inf")
    eta_str = f"{eta:.0f}s" if eta < 3600 else f"{eta / 3600:.1f}h"
    sys.stderr.write(f"\r  [{done}/{total}] {elapsed:.0f}s elapsed, ETA {eta_str}    ")
    sys.stderr.flush()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_pipeline(
    records: List[Dict[str, Any]],
    cfg: LLMConfig,
    *,
    workers: int,
    checkpoint: Path,
    resume: bool = True,
    skip_deterministic: bool = False,
    stop: Optional[threading.Event] = None,
) -> RunOutcome:
    """Classify `records` with `cfg`, appending every result to `checkpoint`."""
    out = RunOutcome(n_selected=len(records))
    stop = stop or threading.Event()

    # --- resume: skip what is already in the checkpoint ---
    already_done: set[str] = set()
    if resume and checkpoint.exists():
        previous = load_checkpoint_deduped(checkpoint)
        already_done = {str(r["hash"]) for r in previous}
        _log(f"  → resuming: {len(already_done)} hashes already in {checkpoint}")
        other = {r.get("model") for r in previous
                 if r.get("method") == "llm" and r.get("model") not in (None, cfg.model_name)}
        if other:
            _log(f"  ⚠ checkpoint also contains results from other model(s) {sorted(other)}; "
                 f"they are kept and NOT re-run. Use a different --out-dir (or --no-resume) "
                 f"to evaluate {cfg.model_name} on its own.")
        stale = sum(1 for r in previous
                    if r.get("method") == "llm" and r.get("prompt_version") != PROMPT_VERSION)
        if stale:
            _log(f"  ⚠ {stale} checkpoint results were produced with a different prompt version "
                 f"(current: {PROMPT_VERSION}); they are kept and NOT re-run. Use a new --out-dir "
                 f"(or --no-resume) to re-classify them with the current prompt.")
    todo = [r for r in records if str(r.get("hash", "")) not in already_done]
    out.n_resumed = len(records) - len(todo)
    _log(f"  → {len(todo)} records remaining ({out.n_resumed} already done)")

    # --- deterministic pre-filter (no API call) ---
    to_llm: List[Dict[str, Any]] = []
    determ_results: List[Dict[str, Any]] = []
    if skip_deterministic:
        to_llm = todo
    else:
        for r in todo:
            det = deterministic_classify(r)
            if det is not None:
                determ_results.append(det)
            else:
                to_llm.append(r)
        _log(f"  → deterministic pre-filter: {len(determ_results)} classified, "
             f"{len(to_llm)} sent to the API")

    # records whose function diff was lost (v1 enrichment truncated it): asking the model
    # would only produce a guess -> not sent, not checkpointed, reported
    no_diff = [r for r in to_llm if not diff_available(r)]
    if no_diff:
        skip = {id(r) for r in no_diff}
        to_llm = [r for r in to_llm if id(r) not in skip]
        out.n_no_diff = len(no_diff)
        _log(f"  ⚠ {len(no_diff)} records skipped: their diff was truncated before reaching the "
             f"function (enriched with an old schema). Re-run "
             f"`python -m dv_enricher ... --refresh`, then run again.")

    written = 0
    if todo:
        t0 = time.time()
        consecutive_transient = 0
        n_done = 0
        n_total = len(to_llm)

        def worker(r: Dict[str, Any]):
            if stop.is_set():
                return "skipped", r, None
            try:
                return "ok", r, process_record(r, cfg)
            except LLMAbort as exc:
                stop.set()
                return "abort", r, exc
            except LLMTransientError as exc:
                return "transient", r, exc
            except Exception as exc:  # record-level failure -> checkpointed as UNDETERMINABLE
                _log(f"\n  [error] hash={r.get('hash')}: {exc}")
                return "error", r, _error_record(r, exc, cfg.model_name)

        with CheckpointWriter(checkpoint) as ckpt:
            for det in determ_results:
                ckpt.write(det)
                written += 1

            if to_llm:
                executor = ThreadPoolExecutor(max_workers=max(1, workers))
                try:
                    futs = [executor.submit(worker, r) for r in to_llm]
                    for fu in as_completed(futs):
                        kind, rec, payload = fu.result()
                        if kind == "skipped":
                            continue
                        n_done += 1
                        if kind in ("ok", "error"):
                            consecutive_transient = 0
                            ckpt.write(payload)
                            written += 1
                        elif kind == "transient":
                            consecutive_transient += 1
                            _log(f"\n  [transient] hash={rec.get('hash')} NOT checkpointed "
                                 f"(will be retried on resume): {payload}")
                            if consecutive_transient >= MAX_CONSECUTIVE_TRANSIENT:
                                stop.set()
                                out.status = "stopped"
                                out.reason = (f"{MAX_CONSECUTIVE_TRANSIENT} consecutive transient "
                                              f"failures — API seems down: {payload}")
                        elif kind == "abort":
                            if out.status != "stopped":
                                out.status = "stopped"
                                out.reason = f"{type(payload).__name__}: {payload}"
                        _progress(n_done, n_total, t0)
                except KeyboardInterrupt:
                    stop.set()
                    out.status = "interrupted"
                    out.reason = "KeyboardInterrupt"
                    executor.shutdown(wait=True, cancel_futures=True)
                else:
                    executor.shutdown(wait=True)
                sys.stderr.write("\n")
        if stop.is_set() and out.status == "completed":
            out.status = "interrupted"
            out.reason = "stopped by user"
    else:
        out.status = "nothing_to_do"

    out.n_new = written
    out.results = load_checkpoint_deduped(checkpoint) if checkpoint.exists() else []
    done_now = {str(r.get("hash")) for r in out.results}
    out.n_remaining = sum(1 for r in records if str(r.get("hash", "")) not in done_now)

    if out.status == "stopped":
        _log(f"  ⚠ run stopped: {out.reason}. {out.n_remaining} records still pending — "
             f"re-run to resume from the checkpoint.")
    elif out.status == "interrupted":
        _log(f"  ⚠ run interrupted. {out.n_remaining} records still pending — re-run to resume.")
    else:
        _log("  ✓ complete" if out.n_remaining == 0
             else f"  {out.n_remaining} records still pending")
    return out
