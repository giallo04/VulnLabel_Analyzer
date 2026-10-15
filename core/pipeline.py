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
Blocked or unparsable answers are recorded as ``UNDETERMINABLE`` with ``parse_error=true``
(``retry_errors=True`` re-queries them on resume).

Consistency
-----------
Each commit is characterised ONCE (``core/commit_verdict.py``) and the verdict is injected
unchanged into the prompt of every function of that commit. Optional self-consistency
(``votes`` > 1): majority vote over several sampled answers; a tie is recorded as
``UNDETERMINABLE`` (precision over coverage) with ``vote_tie=true``.
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
from .commit_verdict import CommitVerdicts
from .deterministic import deterministic_classify, undeterminable_no_diff
from .llm_engine import LLMAbort, LLMConfig, LLMTransientError, query_llm_ex
from .parser import parse_llm_response
from .prompting import (GEMINI_RESPONSE_SCHEMA, PROMPT_VERSION, aggregate_votes, build_messages,
                        commit_key, diff_available, prompt_meta)

MAX_CONSECUTIVE_TRANSIENT = 8   # stop after this many back-to-back transient failures
MAX_PARSE_ATTEMPTS = 3          # new samples for the same prompt when the answer is not parsable
VOTE_MIN_TEMPERATURE = 0.7      # with votes > 1 the samples must differ, otherwise voting is moot
ENRICH_SCHEMA_WITH_PAIR_DIFF = 3   # first enricher schema with reliable function-level diffs


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
def _ask_once(msgs: Dict[str, str], cfg: LLMConfig, *, seed: int, temperature: float
              ) -> Dict[str, Any]:
    """One answer for the prompt; up to MAX_PARSE_ATTEMPTS tries if the output is not parsable
    (a cut-off answer, MAX_TOKENS, is retried with a doubled output budget)."""
    budget = cfg.max_output_tokens
    parsed: Dict[str, Any] = {}
    attempts = 0
    for k in range(MAX_PARSE_ATTEMPTS):
        reply = query_llm_ex(msgs["user"], cfg, system=msgs["system"],
                             schema=GEMINI_RESPONSE_SCHEMA, temperature=temperature,
                             seed=seed + k, max_output_tokens=budget)
        attempts += 1
        parsed = parse_llm_response(reply.text)
        if not parsed["parse_error"]:
            break
        if reply.truncated:
            budget = min(budget * 2, 65536)
    parsed["attempts"] = attempts
    return parsed


def _combine(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Majority vote over the samples (parse failures ignored). One sample = that sample."""
    valid = [p for p in samples if not p["parse_error"]]
    if not valid:
        return {**samples[0], "n_votes": len(samples), "agreement": 0.0, "vote_counts": {},
                "vote_tie": False}
    if len(samples) == 1:
        return {**valid[0], "n_votes": 1, "agreement": 1.0,
                "vote_counts": {valid[0]["category"]: 1}, "vote_tie": False}
    agg = aggregate_votes(valid)
    if agg["tie"]:
        rep = dict(valid[0])
        rep.update(category="UNDETERMINABLE", predicted_target=-1)
        return {**rep, "n_votes": len(samples), "agreement": agg["agreement"],
                "vote_counts": agg["counts"], "vote_tie": True}
    rep = next(p for p in valid if p["category"] == agg["category"])
    return {**rep, "n_votes": len(samples), "agreement": agg["agreement"],
            "vote_counts": agg["counts"], "vote_tie": False}


def process_record(record: Dict[str, Any], cfg: LLMConfig, *,
                   verdicts: Optional[CommitVerdicts] = None,
                   ctx: Optional[Dict[str, Any]] = None, votes: int = 1) -> Dict[str, Any]:
    """Full LLM cycle for a single record. Returns a result dict.

    The commit verdict (when `verdicts` is given) is shared by all the functions of the commit.
    """
    verdict = verdicts.get(record, ctx) if verdicts is not None else None
    msgs = build_messages(record, verdict)          # stateless: pure function of record + verdict
    meta = prompt_meta(record, verdict)

    votes = max(1, int(votes))
    temperature = cfg.temperature if votes == 1 else max(cfg.temperature, VOTE_MIN_TEMPERATURE)
    samples = [_ask_once(msgs, cfg, seed=cfg.seed + 101 * i, temperature=temperature)
               for i in range(votes)]
    parsed = _combine(samples)

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
        "diff_method": meta["diff_method"],
        # commit description: identical for every function of the commit
        "commit_nature": meta["commit_nature"],
        "commit_root_cause": meta["commit_root_cause"],
        "commit_fix_pattern": meta["commit_fix_pattern"],
        "commit_fix_scope": meta["commit_fix_scope"],
        "commit_verdict_fallback": bool((verdict or {}).get("fallback", False)),
        # full reasoning
        "step_1_change": parsed.get("step_1_change", ""),
        "step_2_flaw":   parsed.get("step_2_flaw", ""),
        "step_3_role":   parsed.get("step_3_role", ""),
        "raw_output":         parsed.get("raw_output"),
        "parse_error":        parsed["parse_error"],
        "parse_error_reason": parsed.get("parse_error_reason", ""),
        "attempts":           parsed.get("attempts", 1),
        # self-consistency
        "n_votes":     parsed["n_votes"],
        "agreement":   parsed["agreement"],
        "vote_counts": parsed["vote_counts"],
        "vote_tie":    parsed["vote_tie"],
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
    votes: int = 1,
    use_commit_verdict: bool = True,
    retry_errors: bool = False,
) -> RunOutcome:
    """Classify `records` with `cfg`, appending every result to `checkpoint`."""
    out = RunOutcome(n_selected=len(records))
    stop = stop or threading.Event()

    # --- resume: skip what is already in the checkpoint ---
    already_done: set[str] = set()
    if resume and checkpoint.exists():
        previous = load_checkpoint_deduped(checkpoint)
        retry = [r for r in previous if retry_errors and r.get("parse_error")]
        already_done = {str(r["hash"]) for r in previous} - {str(r["hash"]) for r in retry}
        _log(f"  → resuming: {len(already_done)} hashes already in {checkpoint}"
             + (f" ({len(retry)} parse-error records will be retried)" if retry else ""))
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

    # records without a usable function diff: asking the model would only produce a guess
    unusable = [r for r in to_llm if not diff_available(r)]
    if unusable:
        skip = {id(r) for r in unusable}
        to_llm = [r for r in to_llm if id(r) not in skip]
        # schema >= 3: the enricher already did its best (function-pair comparison) -> final
        # UNDETERMINABLE, checkpointed; older schemas: --refresh can still recover the diff
        final = [r for r in unusable
                 if int(r.get("schema_version") or 1) >= ENRICH_SCHEMA_WITH_PAIR_DIFF]
        final_ids = {id(r) for r in final}
        legacy = [r for r in unusable if id(r) not in final_ids]
        for r in final:
            determ_results.append(undeterminable_no_diff(r))
        if final:
            _log(f"  → {len(final)} records recorded as UNDETERMINABLE: no diff could be "
                 f"attributed to the function (function pair not found, ambiguous or "
                 f"inconsistent); no API call")
        if legacy:
            out.n_no_diff = len(legacy)
            _log(f"  ⚠ {len(legacy)} records skipped: their diff was truncated or not reliably "
                 f"attributed to the function (enriched with an old schema). Re-run "
                 f"`python -m dv_enricher ... --refresh`, then run again.")

    # commit context + shared commit verdicts (one API call per commit)
    ctx_by_commit: Dict[str, Dict[str, Any]] = {}
    for r in records:
        c = ctx_by_commit.setdefault(commit_key(r), {"cwes": set(), "n_selected": 0})
        c["cwes"].update(r.get("cwe") or [])
        c["n_selected"] += 1
    for c in ctx_by_commit.values():
        c["cwes"] = sorted(c["cwes"])
    verdicts = (CommitVerdicts(cfg, checkpoint.parent / "commit_verdicts.jsonl")
                if use_commit_verdict else None)
    if verdicts is not None and verdicts.n_loaded:
        _log(f"  → {verdicts.n_loaded} commit verdicts reused from commit_verdicts.jsonl")

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
                return "ok", r, process_record(
                    r, cfg, verdicts=verdicts, ctx=ctx_by_commit.get(commit_key(r)), votes=votes)
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

    if verdicts is not None and verdicts.n_api_calls:
        _log(f"  → commit verdicts: {verdicts.n_api_calls} API calls for "
             f"{len(ctx_by_commit)} commits")
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
