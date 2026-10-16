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
"""
from __future__ import annotations

import concurrent.futures
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .callgraph import CommitGraph, build_graphs, commit_key, result_fields, summarize_graphs
from .checkpoint import CheckpointWriter, load_checkpoint_deduped
from .deterministic import deterministic_classify, undeterminable_no_diff
from .llm_engine import LLMAbort, LLMConfig, LLMTransientError, query_llm_ex
from .parser import parse_llm_response
from .prompting import (GEMINI_RESPONSE_SCHEMA, PROMPT_VERSION, aggregate_votes, analyze_record,
                        build_messages, diff_available, prompt_meta)

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
    """Majority vote over the samples (parse failures ignored)."""
    valid = [p for p in samples if not p["parse_error"]]
    if not valid:
        return {**samples[0], "n_votes": len(samples), "agreement": 0.0, "vote_counts": {},
                "vote_tie": False, "all_parse_errors": True}
    if len(samples) == 1:
        return {**valid[0], "n_votes": 1, "agreement": 1.0,
                "vote_counts": {valid[0]["category"]: 1}, "vote_tie": False,
                "all_parse_errors": False}
    agg = aggregate_votes(valid)
    if agg["tie"]:
        rep = dict(valid[0])
        if not agg["target_tie"]:
            binary_target = agg["predicted_target"]
            rep.update(category="UNDETERMINABLE", predicted_target=binary_target)
            return {**rep, "n_votes": len(samples), "agreement": agg["agreement"],
                    "vote_counts": agg["counts"], "vote_tie": False, "category_tie": True,
                    "all_parse_errors": False}
        rep.update(category="UNDETERMINABLE", predicted_target=-1)
        return {**rep, "n_votes": len(samples), "agreement": agg["agreement"],
                "vote_counts": agg["counts"], "vote_tie": True, "category_tie": True,
                "all_parse_errors": False}
    rep = next(p for p in valid if p["category"] == agg["category"])
    return {**rep, "n_votes": len(samples), "agreement": agg["agreement"],
            "vote_counts": agg["counts"], "vote_tie": False, "category_tie": False,
            "all_parse_errors": False}


def _needs_escalation(sample: Dict[str, Any], original_target: int) -> bool:
    if sample["parse_error"] or sample["category"] == "UNDETERMINABLE":
        return True
    return sample["predicted_target"] != original_target


def process_record(record: Dict[str, Any], cfg: LLMConfig, *,
                   votes: int = 1,
                   vote_policy: str = "always",
                   graph_ctx: Optional[Dict[str, Any]] = None,
                   graphs: Optional[Dict[str, CommitGraph]] = None,
                   analyzed: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Full LLM cycle for a single record. Returns a result dict."""
    if graph_ctx is None and graphs is not None and analyzed is not None:
        g = graphs.get(commit_key(record))
        if g is not None:
            graph_ctx = g.context_for(str(record.get("hash", "")), analyzed)

    diff_ctx = analyze_record(record)
    msgs = build_messages(record, ctx=diff_ctx, graph_ctx=graph_ctx)
    meta = prompt_meta(record, ctx=diff_ctx, graph_ctx=graph_ctx)

    votes = max(1, int(votes))
    temperature = cfg.temperature if votes == 1 else max(cfg.temperature, VOTE_MIN_TEMPERATURE)
    original_target = int(record.get("target", -1))
    
    samples = [_ask_once(msgs, cfg, seed=cfg.seed + 101 * 0, temperature=temperature)]
    if votes > 1 and (vote_policy == "always" or _needs_escalation(samples[0], original_target)):
        samples += [_ask_once(msgs, cfg, seed=cfg.seed + 101 * i, temperature=temperature)
                    for i in range(1, votes)]
    parsed = _combine(samples)

    predicted_target = parsed["predicted_target"]
    change = record.get("change") or {}
    signals = record.get("signals") or {}
    location = record.get("location") or {}

    is_noise = original_target != predicted_target and predicted_target != -1

    return {
        "hash": record.get("hash"),
        "commit_id": record.get("commit_id"),
        "project": record.get("project"),
        "func_name": record.get("func_name"),
        "file": location.get("file"),
        "original_target": original_target,
        "predicted_target": predicted_target,
        "is_noise": is_noise,
        "category": parsed["category"],
        "method": "llm",
        "model": cfg.model_name,
        "thinking_level": getattr(cfg, 'thinking_level', None),
        "prompt_version": meta["prompt_version"],
        "diff_truncated": meta["diff_truncated"],
        "diff_method": meta["diff_method"],
        # call graph context & traceability
        **result_fields(graph_ctx),
        # full reasoning
        "step_1_change": parsed.get("step_1_change", ""),
        "step_2_flaw":   parsed.get("step_2_flaw", ""),
        "step_3_role":   parsed.get("step_3_role", ""),
        "raw_output":         parsed.get("raw_output"),
        "parse_error":        parsed["parse_error"],
        "parse_error_reason": parsed.get("parse_error_reason", ""),
        "attempts":           parsed.get("attempts", 1),
        # self-consistency
        "n_votes":         parsed["n_votes"],
        "agreement":       parsed["agreement"],
        "vote_counts":     parsed["vote_counts"],
        "vote_tie":        parsed["vote_tie"],
        "category_tie":    parsed.get("category_tie", False),
        "all_parse_errors": parsed.get("all_parse_errors", False),
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


def _error_record(rec: Dict[str, Any], exc: Exception, model: str,
                  graph_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        **result_fields(graph_ctx),
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


def estimate_calls(n_records: int, votes: int, vote_policy: str, noise_rate: float = 0.35) -> Dict[str, float]:
    lo = n_records
    hi = n_records * max(1, votes)
    if votes <= 1:
        exp = hi
    elif vote_policy == "always":
        exp = hi
    else:
        exp = n_records * (1 + noise_rate * (votes - 1))
    return {"min": lo, "expected": exp, "max": hi}


def _fmt_hours(h: float) -> str:
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def _log_estimate(n_records: int, votes: int, vote_policy: str, cfg: LLMConfig) -> None:
    if n_records <= 0:
        return
    est = estimate_calls(n_records, votes, vote_policy)
    if getattr(cfg, 'rpm', None) and cfg.rpm > 0:
        floor = {k: v / cfg.rpm / 60 for k, v in est.items()}
        _log(f"  → estimate: ~{est['expected']:.0f} API calls ({est['min']:.0f}-{est['max']:.0f}); "
             f"at rpm={cfg.rpm:g} that is >= {_fmt_hours(floor['expected'])} "
             f"(best case {_fmt_hours(floor['min'])}, worst {_fmt_hours(floor['max'])}) "
             f"before any retry")
        if floor["expected"] > 12:
            _log("  ⚠ the rpm cap, not the workers, bounds the run: check your real quota "
                 "(AI Studio -> Rate limits), use --votes 1 / --vote-policy adaptive, lower "
                 "--thinking for a first pass, or split the run")
    else:
        _log(f"  → estimate: ~{est['expected']:.0f} API calls ({est['min']:.0f}-{est['max']:.0f})")


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
    retry_errors: bool = False,
    vote_policy: str = "adaptive",
    use_callgraph: bool = True,
) -> RunOutcome:
    """Classify `records` with `cfg`, appending every result to `checkpoint`."""
    out = RunOutcome(n_selected=len(records))
    stop = stop or threading.Event()

    # --- resume: skip what is already in the checkpoint ---
    already_done: set[str] = set()
    previous: List[Dict[str, Any]] = []
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

    # --- call graph: one CommitGraph per commit ---
    graphs = build_graphs(records) if use_callgraph else {}
    if use_callgraph and graphs:
        _log(f"  → call graph: {summarize_graphs(graphs)}")

    analyzed: Dict[str, Dict[str, Any]] = {}
    if previous:
        for r in previous:
            h = str(r.get("hash", ""))
            if h and (not retry_errors or not r.get("parse_error")):
                analyzed[h] = r

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

    # Attach call graph context to deterministic results and register in analyzed
    for det in determ_results:
        h = str(det.get("hash", ""))
        ck = commit_key(det)
        g = graphs.get(ck)
        cg = g.context_for(h, analyzed) if (g and use_callgraph) else None
        det.update(result_fields(cg))
        if h:
            analyzed[h] = det

    # Order to_llm so that within each commit, records follow g.order (callees first)
    if use_callgraph:
        to_llm.sort(key=lambda r: (
            commit_key(r),
            graphs[commit_key(r)].pos.get(str(r.get("hash", "")), 0) if commit_key(r) in graphs else 0
        ))

    # Dependency tracking for bottom-up analysis within each commit
    deps: Dict[str, Set[str]] = {}
    to_llm_hashes = {str(r.get("hash", "")) for r in to_llm}
    rec_by_hash = {str(r.get("hash", "")): r for r in to_llm}

    for r in to_llm:
        h = str(r.get("hash", ""))
        ck = commit_key(r)
        g = graphs.get(ck)
        if use_callgraph and g is not None and h in g.pos:
            pos = g.pos[h]
            commit_deps = {pred_h for pred_h in g.order[:pos] if pred_h in to_llm_hashes}
            deps[h] = commit_deps
        else:
            deps[h] = set()

    remaining_deps = {h: set(d) for h, d in deps.items()}
    dependents: Dict[str, List[str]] = defaultdict(list)
    for h, ds in deps.items():
        for d in ds:
            dependents[d].append(h)

    ready = [r for r in to_llm if not remaining_deps[str(r.get("hash", ""))]]

    _log_estimate(len(to_llm), votes, vote_policy, cfg)

    written = 0
    if todo:
        t0 = time.time()
        consecutive_transient = 0
        n_done = 0
        n_total = len(to_llm)
        state_lock = threading.Lock()

        def worker(r: Dict[str, Any]):
            if stop.is_set():
                return "skipped", r, None
            h = str(r.get("hash", ""))
            ck = commit_key(r)
            g = graphs.get(ck) if graphs else None
            graph_ctx = None
            if g is not None and use_callgraph:
                with state_lock:
                    snap = dict(analyzed)
                graph_ctx = g.context_for(h, snap)
            try:
                payload = process_record(
                    r, cfg, votes=votes,
                    vote_policy=vote_policy, graph_ctx=graph_ctx)
                return "ok", r, payload
            except LLMAbort as exc:
                stop.set()
                return "abort", r, exc
            except LLMTransientError as exc:
                return "transient", r, exc
            except Exception as exc:  # record-level failure -> checkpointed as UNDETERMINABLE
                _log(f"\n  [error] hash={r.get('hash')}: {exc}")
                return "error", r, _error_record(r, exc, cfg.model_name, graph_ctx=graph_ctx)

        with CheckpointWriter(checkpoint) as ckpt:
            for det in determ_results:
                ckpt.write(det)
                written += 1

            if to_llm:
                executor = ThreadPoolExecutor(max_workers=max(1, workers))
                active_futures: Dict[Any, Dict[str, Any]] = {}
                try:
                    for r in ready:
                        fu = executor.submit(worker, r)
                        active_futures[fu] = r

                    while active_futures:
                        done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)
                        for fu in done:
                            rec = active_futures.pop(fu)
                            kind, _, payload = fu.result()
                            if kind == "skipped":
                                continue
                            n_done += 1
                            h = str(rec.get("hash", ""))
                            if kind in ("ok", "error"):
                                consecutive_transient = 0
                                ckpt.write(payload)
                                written += 1
                                with state_lock:
                                    analyzed[h] = payload
                                    for dep_h in dependents.get(h, []):
                                        remaining_deps[dep_h].discard(h)
                                        if not remaining_deps[dep_h] and not stop.is_set():
                                            nxt_rec = rec_by_hash[dep_h]
                                            nxt_fu = executor.submit(worker, nxt_rec)
                                            active_futures[nxt_fu] = nxt_rec
                            elif kind == "transient":
                                consecutive_transient += 1
                                _log(f"\n  [transient] hash={rec.get('hash')} NOT checkpointed "
                                     f"(will be retried on resume): {payload}")
                                with state_lock:
                                    for dep_h in dependents.get(h, []):
                                        remaining_deps[dep_h].discard(h)
                                        if not remaining_deps[dep_h] and not stop.is_set():
                                            nxt_rec = rec_by_hash[dep_h]
                                            nxt_fu = executor.submit(worker, nxt_rec)
                                            active_futures[nxt_fu] = nxt_rec
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