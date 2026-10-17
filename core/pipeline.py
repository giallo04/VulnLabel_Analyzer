"""Classification pipeline: records -> rule pre-check / LLM API -> checkpoint.

One run = one model, one checkpoint, one set of outputs.
Exactly ONE LLM request is made per function that reaches the model (no voting, no re-asking
on unparsable output).

Scheduling: per-commit, bottom-up
---------------------------------
The prompt shows, for every callee / caller of the function, the label already issued for it
("label so far", ``pending`` otherwise). That only works if callees are analysed before their
callers, so the unit of parallel work is the COMMIT, not the function:

  * commits run in parallel (one worker thread per commit at a time);
  * inside a commit, functions are processed sequentially in call-graph order (callees first,
    ``CommitGraph.pos``), and every verdict is fed back as a label to the functions that follow;
  * functions of commits without a call graph are independent single-function units.

The labels visible to a function are: verdicts already in the checkpoint (resume), all
rule-based verdicts of the commit (they do not depend on the model), and the verdicts issued
earlier in this run for the same commit. Unparsable answers and transient failures do not
produce a label (the neighbour stays ``pending``).

Same-pattern changes ("twins")
------------------------------
Functions of one commit that received the same edit (``callgraph.find_twins``) are shown to each
other in the "Same-pattern changes" section together with their label so far, so that identical
edits get identical verdicts. Twins live in the same commit, hence in the same work unit and are
processed sequentially: the first one decides, the others see its label. Twins already present
in the checkpoint arrive through the same seed labels as call-graph neighbours.

Resume
------
The checkpoint is one JSON object per line, keyed by ``hash``. Records whose hash is already
in the checkpoint are skipped, so an interrupted run (crash, Ctrl+C, exhausted daily quota)
continues from where it stopped; checkpointed verdicts are re-used as neighbour labels.
Every reader accesses fields with ``.get``, so checkpoints written by older versions keep working.

What is NOT written to the checkpoint (so it is retried on the next run):
daily quota exhausted, invalid key, API unreachable/overloaded after the transport retries.
Blocked or unparsable answers are recorded as ``UNDETERMINABLE`` with ``parse_error=true``
(``retry_errors=True`` re-queries them on resume; this is the only way a function gets a 2nd call).
"""
from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .callgraph import (SummaryCache, build_graphs, commit_key, compact_label, find_twins,
                        result_fields, summarize_graphs)
from .checkpoint import CheckpointWriter, load_checkpoint_deduped
from .deterministic import (DEFAULT_DETERMINISTIC_CONFIG, DeterministicConfig,
                            deterministic_verdict)
from .diffclean import trim_span_artifacts
from .llm_engine import LLMAbort, LLMConfig, LLMTransientError, query_llm_ex
from .prompting import (DEFAULT_CONFIG, PROMPT_VERSION, RESPONSE_JSON_SCHEMA, PromptBuilder,
                        PromptConfig, analyze_record, parse_llm_output, precheck, prompt_meta,
                        resolve_func_name)

MAX_CONSECUTIVE_TRANSIENT = 8      # stop after this many back-to-back transient failures
MAX_CONSECUTIVE_ERRORS = 5         # stop after this many back-to-back record-level exceptions
                                   # (e.g. HTTP 400 on every request = request/config problem)
ENRICH_SCHEMA_WITH_PAIR_DIFF = 3   # first enricher schema with reliable function-level diffs
LLM_METHOD = "llm"
RULE_METHOD = "deterministic"      # deterministic.py + prompting.precheck (same value as before)


def _strip_unsupported(node: Any) -> Any:
    """Gemini's `response_schema` is an OpenAPI subset: it rejects `additionalProperties`."""
    if isinstance(node, dict):
        return {k: _strip_unsupported(v) for k, v in node.items() if k != "additionalProperties"}
    if isinstance(node, list):
        return [_strip_unsupported(v) for v in node]
    return node


# Schema actually sent to the API (the prompt itself does not depend on it).
LLM_RESPONSE_SCHEMA = _strip_unsupported(RESPONSE_JSON_SCHEMA)


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
# Result construction
# ---------------------------------------------------------------------------
def _result(record: Dict[str, Any], verdict: Dict[str, Any], meta: Dict[str, Any], *,
            method: str, model: Optional[str], thinking_level: Optional[str],
            raw_output: Optional[str], output_truncated: bool) -> Dict[str, Any]:
    """Checkpoint row from a verdict (``parse_llm_output`` or ``precheck``) and prompt metadata."""
    data = verdict.get("data") or {}
    original_target = int(record.get("target", -1))
    failed = bool(verdict.get("parse_error"))
    # an unparsable answer must never count as "noise"
    category = "UNDETERMINABLE" if failed else verdict["category"]
    predicted_target = -1 if failed else verdict["predicted_target"]
    is_noise = (not failed) and predicted_target != -1 and predicted_target != original_target

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
        "is_noise": is_noise,
        "category": category,
        "method": method,
        "model": model,
        "thinking_level": thinking_level,
        # prompt version, diff status and call-graph / twin traceability (n_labelled, n_twins, ...)
        **meta,
        # full reasoning
        "step_1_change": data.get("step_1_change", ""),
        "step_2_flaw": data.get("step_2_flaw", ""),
        "step_3_role": data.get("step_3_role", ""),
        "raw_output": raw_output,
        "parse_error": failed,
        "parse_error_reason": verdict.get("parse_error_reason", ""),
        "parse_method": verdict.get("parse_method"),
        "prefilter_rules": verdict.get("prefilter_rules"),
        "prefilter_confidence": verdict.get("prefilter_confidence"),
        "diff_lines_trimmed": change.get("span_trimmed", 0),   # out-of-span lines removed (diffclean)
        "output_truncated": output_truncated,
        # single-answer self-consistency
        "rule": verdict.get("rule"),
        "flaw_visible": verdict.get("flaw_visible"),
        "flags": verdict.get("flags", []),
        "inconsistent": verdict.get("inconsistent", False),
        # constants kept so stats.py / older checkpoints stay compatible
        "n_votes": 1,
        "agreement": 1.0,
        # enricher signals (downstream analysis)
        "rank_in_commit": change.get("rank_in_commit"),
        "share_of_source_churn": change.get("share_of_source_churn"),
        "frac_changed": change.get("frac_changed"),
        "kind": change.get("kind"),
        "siblings_count": len(record.get("siblings", [])),
        "name_in_message": signals.get("name_in_message"),
        "cwe_matches_nvd": signals.get("cwe_matches_nvd"),
        "cwe_missing": signals.get("cwe_missing_in_dataset", False),
    }


def process_record(record: Dict[str, Any], cfg: LLMConfig, *, builder: PromptBuilder,
                   graph_ctx: Optional[Dict[str, Any]] = None,
                   ctx: Optional[Dict[str, Any]] = None,
                   twins: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One LLM request for a single record. Returns a result row."""
    ctx = ctx if ctx is not None else analyze_record(record, builder.cfg)
    prompt = builder.build(record, ctx, graph_ctx, twins)
    meta = prompt_meta(record, ctx, graph_ctx, builder.cfg)
    meta["n_twins"] = len(twins or [])
    meta["n_twins_labelled"] = sum(1 for t in (twins or []) if (t.get("label") or {}).get("category"))

    reply = query_llm_ex(prompt.content, cfg, system=prompt.instructions,
                         schema=LLM_RESPONSE_SCHEMA, temperature=cfg.temperature,
                         seed=cfg.seed, max_output_tokens=cfg.max_output_tokens)
    verdict = parse_llm_output(reply.text)
    return _result(record, verdict, meta, method=LLM_METHOD, model=cfg.model_name,
                   thinking_level=getattr(cfg, "thinking_level", None),
                   raw_output=reply.text, output_truncated=bool(getattr(reply, "truncated", False)))


def _rule_result(record: Dict[str, Any], verdict: Dict[str, Any], ctx: Dict[str, Any],
                 graph_ctx: Optional[Dict[str, Any]], config: PromptConfig) -> Dict[str, Any]:
    """Row for a verdict issued by ``prompting.precheck`` (no API call)."""
    meta = prompt_meta(record, ctx, graph_ctx, config)
    return _result(record, verdict, meta, method=RULE_METHOD, model=None, thinking_level=None,
                   raw_output=None, output_truncated=False)


def _error_record(rec: Dict[str, Any], exc: Exception, model: str,
                  graph_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Minimal row so that a record-level failure is checkpointed."""
    return {
        "hash": rec.get("hash"),
        "commit_id": rec.get("commit_id"),
        "project": rec.get("project"),
        "func_name": rec.get("func_name"),
        "original_target": int(rec.get("target", -1)),
        "predicted_target": -1,
        "is_noise": False,
        "category": "UNDETERMINABLE",
        "method": LLM_METHOD,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "parse_error": True,
        "parse_error_reason": str(exc),
        "raw_output": None,
        **result_fields(graph_ctx),
    }


def _label_of(result: Dict[str, Any]) -> Dict[str, Any]:
    """Neighbour label ("label so far") derived from a result row / checkpoint row.

    Keys match what ``prompting._label_text`` reads: category, source, rule, flaw_visible."""
    return {
        "category": result.get("category"),
        "source": "llm" if result.get("method") == LLM_METHOD else "rule",
        "rule": result.get("rule"),
        "flaw_visible": result.get("flaw_visible"),
        "predicted_target": result.get("predicted_target"),
    }


# ---------------------------------------------------------------------------
# Progress and shared run state
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


def _fmt_hours(h: float) -> str:
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def _log_estimate(n_calls: int, cfg: LLMConfig) -> None:
    if n_calls <= 0:
        return
    rpm = getattr(cfg, "rpm", None)
    if rpm and rpm > 0:
        _log(f"  → {n_calls} API calls (exactly one per function); at rpm={rpm:g} that is "
             f">= {_fmt_hours(n_calls / rpm / 60)} before any transport retry")
    else:
        _log(f"  → {n_calls} API calls (exactly one per function)")


@dataclass
class _RunState:
    """Counters shared by the worker threads; every mutation (and checkpoint write) is locked."""
    total: int
    written: int = 0
    done: int = 0
    consecutive_transient: int = 0
    consecutive_errors: int = 0
    status: str = "completed"
    reason: str = ""
    t0: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, ckpt: CheckpointWriter, result: Dict[str, Any]) -> None:
        with self.lock:
            ckpt.write(result)
            self.written += 1
            self.done += 1
            self.consecutive_transient = 0
            self.consecutive_errors = 0
            _progress(self.done, self.total, self.t0)

    def error(self, ckpt: CheckpointWriter, result: Dict[str, Any], exc: Exception,
              stop: threading.Event) -> None:
        """Record-level exception: checkpointed as UNDETERMINABLE, but a streak stops the run."""
        with self.lock:
            ckpt.write(result)
            self.written += 1
            self.done += 1
            self.consecutive_errors += 1
            if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS and self.status != "stopped":
                stop.set()
                self.status = "stopped"
                self.reason = (f"{MAX_CONSECUTIVE_ERRORS} consecutive record-level errors — "
                               f"likely a request/config problem, not the data: {exc}")
            _progress(self.done, self.total, self.t0)

    def transient(self, rec: Dict[str, Any], exc: Exception, stop: threading.Event) -> None:
        with self.lock:
            self.done += 1
            self.consecutive_transient += 1
            _log(f"\n  [transient] hash={rec.get('hash')} NOT checkpointed "
                 f"(will be retried on resume): {exc}")
            if self.consecutive_transient >= MAX_CONSECUTIVE_TRANSIENT:
                stop.set()
                self.status = "stopped"
                self.reason = (f"{MAX_CONSECUTIVE_TRANSIENT} consecutive transient "
                               f"failures — API seems down: {exc}")
            _progress(self.done, self.total, self.t0)

    def abort(self, exc: Exception, stop: threading.Event) -> None:
        stop.set()
        with self.lock:
            self.done += 1
            if self.status != "stopped":
                self.status = "stopped"
                self.reason = f"{type(exc).__name__}: {exc}"
            _progress(self.done, self.total, self.t0)


@dataclass
class _Unit:
    """Work processed by ONE thread, strictly in order (callees before callers)."""
    records: List[Dict[str, Any]]
    graph: Any = None                                         # CommitGraph or None
    labels: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # seed labels of the commit


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
    retry_errors: bool = False,
    use_callgraph: bool = True,
    prompt_config: PromptConfig = DEFAULT_CONFIG,
    deterministic_config: DeterministicConfig = DEFAULT_DETERMINISTIC_CONFIG,
) -> RunOutcome:
    """Classify `records` with `cfg`, appending every result to `checkpoint`."""
    out = RunOutcome(n_selected=len(records))
    stop = stop or threading.Event()
    builder = PromptBuilder(prompt_config)

    # --- drop diff lines outside the labelled function span (return-type / export-macro lines) ---
    # Done on `records` itself so that prompts, neighbour excerpts and the rule checks all see it.
    trimmed = [trim_span_artifacts(r) for r in records]
    records = [r for r, _ in trimmed]
    n_trimmed = sum(1 for _, k in trimmed if k)
    if n_trimmed:
        _log(f"  → diff cleanup: {n_trimmed} records had diff lines outside the function span "
             f"(return type / export macro on the line before the name) removed "
             f"({sum(k for _, k in trimmed)} lines)")
    del trimmed

    # --- resume: skip what is already in the checkpoint, keep its verdicts as neighbour labels ---
    already_done: set[str] = set()
    known_labels: Dict[str, Dict[str, Any]] = {}
    if resume and checkpoint.exists():
        previous = load_checkpoint_deduped(checkpoint)
        retry = [r for r in previous if retry_errors and r.get("parse_error")]
        already_done = {str(r["hash"]) for r in previous} - {str(r["hash"]) for r in retry}
        known_labels = {str(r["hash"]): _label_of(r) for r in previous
                        if r.get("category") and not r.get("parse_error")}
        _log(f"  → resuming: {len(already_done)} hashes already in {checkpoint}"
             + (f" ({len(retry)} parse-error records will be retried)" if retry else ""))
        other = {r.get("model") for r in previous
                 if r.get("method") == LLM_METHOD and r.get("model") not in (None, cfg.model_name)}
        if other:
            _log(f"  ⚠ checkpoint also contains results from other model(s) {sorted(other)}; "
                 f"they are kept and NOT re-run (and serve as neighbour labels). Use a different "
                 f"--out-dir (or --no-resume) to evaluate {cfg.model_name} on its own.")
        stale = sum(1 for r in previous
                    if r.get("method") == LLM_METHOD and r.get("prompt_version") != PROMPT_VERSION)
        if stale:
            _log(f"  ⚠ {stale} checkpoint results were produced with a different prompt version "
                 f"(current: {PROMPT_VERSION}); they are kept and NOT re-run. Use a new --out-dir "
                 f"(or --no-resume) to re-classify them with the current prompt.")

    todo = [r for r in records if str(r.get("hash", "")) not in already_done]
    out.n_resumed = len(records) - len(todo)
    _log(f"  → {len(todo)} records remaining ({out.n_resumed} already done)")

    # --- call graph: one CommitGraph per commit; neighbour summaries are deterministic ---
    graphs = build_graphs(records) if use_callgraph else {}
    summaries = SummaryCache(records) if graphs else None
    if graphs:
        _log(f"  → call graph: {summarize_graphs(graphs)}")

    # --- same-pattern changes: functions of one commit that received the same edit ---
    twins_of: Dict[str, List[str]] = (
        find_twins(records) if prompt_config.include_twins and use_callgraph else {})
    by_hash = {str(r.get("hash")): r for r in records}
    if twins_of:
        _log(f"  → same-pattern changes: {len(twins_of)} functions have at least one twin in their commit")

    def twin_entries(h: str, labels: Optional[Dict[str, Dict[str, Any]]]
                     ) -> Optional[List[Dict[str, Any]]]:
        """Twins of `h` with their label so far. None when labels are disabled (ablation)."""
        if labels is None:
            return None
        return [{"name": resolve_func_name(by_hash[t]),
                 "file": (by_hash[t].get("location") or {}).get("file"),
                 "label": compact_label(labels.get(t))}
                for t in twins_of.get(h, []) if t in by_hash]

    def plain_ctx(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call-graph context without neighbour labels (for results that never reach the model)."""
        g = graphs.get(commit_key(r))
        return g.context_for(str(r.get("hash", "")), summaries) if g is not None else None

    # --- rule-based classification (no API call) ---
    # 1) deterministic.py: test/doc paths, non-C files, verified cosmetic, linkage-only
    # 2) diff without usable evidence -> UNDETERMINABLE (Rule 0)
    # 3) prompting.precheck: no change / whitespace / comment-only (Rules 0 and 1)
    to_llm: List[Dict[str, Any]] = []
    rule_results: List[Dict[str, Any]] = []
    n_det = n_pre = n_unattributed = n_legacy = 0
    for r in todo:
        ctx = analyze_record(r, prompt_config)
        if not skip_deterministic:
            verdict = deterministic_verdict(r, deterministic_config)
            if verdict is not None:
                rule_results.append(_rule_result(r, verdict, ctx, plain_ctx(r), prompt_config))
                n_det += 1
                continue
        if ctx["diff_kind"] == "unavailable":
            # asking the model would only produce a guess
            if int(r.get("schema_version") or 1) >= ENRICH_SCHEMA_WITH_PAIR_DIFF:
                rule_results.append(_rule_result(r, precheck(r, ctx), ctx, plain_ctx(r), prompt_config))
                n_unattributed += 1
            else:
                n_legacy += 1          # enriched with an old schema: not checkpointed
            continue
        if not skip_deterministic:
            verdict = precheck(r, ctx)
            if verdict is not None:
                rule_results.append(_rule_result(r, verdict, ctx, plain_ctx(r), prompt_config))
                n_pre += 1
                continue
        to_llm.append(r)

    if not skip_deterministic:
        _log(f"  → deterministic pre-filter: {n_det} classified; diff pre-check: {n_pre} classified; "
             f"{len(to_llm)} sent to the API")
    if n_unattributed:
        _log(f"  → {n_unattributed} records recorded as UNDETERMINABLE: no diff could be "
             f"attributed to the function (function pair not found, ambiguous or "
             f"inconsistent); no API call")
    if n_legacy:
        out.n_no_diff = n_legacy
        _log(f"  ⚠ {n_legacy} records skipped: their diff was truncated or not reliably "
             f"attributed to the function (enriched with an old schema). Re-run "
             f"`python -m dv_enricher ... --refresh`, then run again.")

    # rule-based verdicts do not depend on the model: they are visible as labels from the start
    for res in rule_results:
        known_labels[str(res["hash"])] = _label_of(res)

    # --- work units: one per commit (callees first), singletons otherwise ---
    use_labels = prompt_config.include_callgraph and prompt_config.include_neighbour_labels
    by_commit: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    singles: List[_Unit] = []
    for r in to_llm:
        key = commit_key(r)
        if key in graphs:
            by_commit[key].append(r)
        else:
            singles.append(_Unit([r]))

    units: List[_Unit] = []
    for key, recs in by_commit.items():
        g = graphs[key]
        # bottom-up: callees before callers (CommitGraph.pos is the call-graph order)
        recs.sort(key=lambda r: g.pos.get(str(r.get("hash", "")), 0))
        # seed: every already-known label of the commit (call-graph neighbours AND twins)
        seed = {h: known_labels[h] for h in g.pos if h in known_labels} if use_labels else {}
        units.append(_Unit(recs, g, seed))
    units.sort(key=lambda u: len(u.records), reverse=True)   # longest chains first (less tail idle time)
    units += singles
    if units:
        _log(f"  → {len(units)} work units ({len(by_commit)} commits processed callees-first, "
             f"{len(singles)} independent functions)")

    _log_estimate(len(to_llm), cfg)

    def run_unit(unit: _Unit, ckpt: CheckpointWriter, state: _RunState) -> None:
        # labels is touched by this thread only; None -> prompt without neighbour labels
        labels: Optional[Dict[str, Dict[str, Any]]] = dict(unit.labels) if use_labels else None
        for r in unit.records:
            if stop.is_set():
                return
            h = str(r.get("hash", ""))
            graph_ctx: Optional[Dict[str, Any]] = None
            try:
                if unit.graph is not None:
                    graph_ctx = (unit.graph.context_for(h, summaries, labels=labels)
                                 if labels is not None else unit.graph.context_for(h, summaries))
                result = process_record(r, cfg, builder=builder, graph_ctx=graph_ctx,
                                        twins=twin_entries(h, labels))
            except LLMAbort as exc:
                state.abort(exc, stop)
                return
            except LLMTransientError as exc:
                state.transient(r, exc, stop)      # not checkpointed; stays `pending` for its callers
                continue
            except Exception as exc:  # record-level failure -> checkpointed as UNDETERMINABLE
                _log(f"\n  [error] hash={r.get('hash')}: {exc}")
                state.error(ckpt, _error_record(r, exc, cfg.model_name, graph_ctx=graph_ctx), exc, stop)
                continue
            state.record(ckpt, result)
            if labels is not None and not result.get("parse_error"):
                labels[h] = _label_of(result)      # visible to the functions that follow

    if todo:
        state = _RunState(total=len(to_llm))
        interrupted = False
        with CheckpointWriter(checkpoint) as ckpt:
            for res in rule_results:
                ckpt.write(res)
                state.written += 1

            if units:
                executor = ThreadPoolExecutor(max_workers=max(1, workers))
                futures = [executor.submit(run_unit, u, ckpt, state) for u in units]
                try:
                    for fu in as_completed(futures):
                        fu.result()                # surfaces unexpected errors (e.g. checkpoint I/O)
                except KeyboardInterrupt:
                    interrupted = True
                    stop.set()
                    executor.shutdown(wait=True, cancel_futures=True)
                except BaseException:
                    stop.set()
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)
                sys.stderr.write("\n")

        if state.status == "stopped":
            out.status, out.reason = "stopped", state.reason
        elif interrupted:
            out.status, out.reason = "interrupted", "KeyboardInterrupt"
        elif stop.is_set():
            out.status, out.reason = "interrupted", "stopped by user"
        out.n_new = state.written
    else:
        out.status = "nothing_to_do"

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