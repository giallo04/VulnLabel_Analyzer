"""VulnLabelAnalyzer — main entry point (API-based).

Classifies every function of the enriched DiverseVul dataset with an LLM reached through the
Gemini API, and measures how accurate the original labels are.

Pipeline
--------
1. Deterministic pre-filter (none / whitespace / comment / test) — no API call.
2. LLM classification of the remaining records. Each commit is characterised ONCE (one extra
   API call per commit, cached on disk) and that description is shared by all its functions;
   then one API call per function (``--votes N`` for majority voting).
3. Incremental JSONL checkpoint, saved after each record; a re-run skips processed hashes.
4. Outputs + statistics (see ``stats.py``):  results.csv, results_full.jsonl,
   accuracy_summary.json, stats_report.txt.

The API key is read ONLY from the environment: GEMINI_API_KEY (or GOOGLE_API_KEY).
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import stats
from core.ingestion import load_enriched_dataset
from core.llm_engine import LLMConfig
from core.pipeline import run_pipeline

DEFAULT_MODEL = "gemini-3.8-flash"
API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="VulnLabelAnalyzer — LLM-based label noise detection for DiverseVul "
                    "(Gemini API; key from env GEMINI_API_KEY or GOOGLE_API_KEY)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", default="enriched.jsonl",
                    help="Path to the JSONL file produced by dv_enricher")
    ap.add_argument("--out-dir", default="data/output/gemini",
                    help="Directory with checkpoint.jsonl, results.csv, results_full.jsonl "
                         "and the statistics")

    sel = ap.add_argument_group("record selection")
    sel.add_argument("--limit", type=int, default=0, help="Limit total records to process (0 = all)")
    sel.add_argument(
        "--max-per-commit", type=int, default=0,
        help="Cap the number of records processed per commit (0 = no cap). "
             "Reduces influence of giant tangled commits.")
    sel.add_argument("--skip-deterministic", action="store_true",
                     help="Disable the deterministic pre-filter and send all records to the API")
    sel.add_argument("--retry-errors", action="store_true",
                     help="On resume, re-query the records saved with parse_error=true "
                          "(blocked / unparsable answers) instead of keeping them")
    sel.add_argument("--no-resume", action="store_true",
                     help="Do NOT skip hashes already in the checkpoint (re-process everything; "
                          "new lines are appended, the last line per hash wins)")

    api = ap.add_argument_group("API")
    api.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model id")
    api.add_argument("--workers", type=int, default=4,
                     help="Parallel requests (the actual rate is capped by --rpm)")
    api.add_argument("--rpm", type=float, default=10.0,
                     help="Client-side requests-per-minute cap. Set it to your account's limit "
                          "(AI Studio -> Rate limits); 0 = unlimited.")
    api.add_argument("--temperature", type=float, default=1,
                     help="Sampling temperature (low = same input, same label)")
    api.add_argument("--votes", type=int, default=1,
                     help="Answers per function for a majority vote (odd numbers; 3 recommended "
                          "for maximum precision, costs 3x calls). With >1 the temperature is "
                          "raised to at least 0.7 so that the samples differ; a tie is recorded "
                          "as UNDETERMINABLE")
    api.add_argument("--vote-policy", choices=["adaptive", "always"], default="adaptive",
                     help="adaptive: 1 sample per function, the other --votes-1 only when the "
                          "answer contradicts the dataset label or is unusable (about 1.3x calls "
                          "instead of 3x); always: every function gets all --votes samples")
    api.add_argument("--no-commit-verdict", action="store_true",
                     help="Do NOT characterise each commit once and share it across its "
                          "functions (ablation only: the commit is then not described at all)")
    api.add_argument("--no-schema", action="store_true",
                     help="Do not send responseSchema (structured output); for models that do "
                          "not support it")
    api.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    api.add_argument("--timeout", type=int, default=300, help="Per-request timeout in seconds")
    api.add_argument("--retries", type=int, default=6,
                     help="Attempts for transient errors (5xx / network); exponential back-off "
                          "with jitter (10s .. 120s) shared by all workers")
    api.add_argument("--max-output-tokens", type=int, default=32768,
                     help="Max output tokens (thinking tokens included)")
    api.add_argument("--thinking", choices=["minimal", "low", "medium", "high", "default"],
                     default="high",
                     help="Thinking level; 'default' = do not send thinkingConfig (model's own "
                          "default; use it for models without thinking levels, e.g. Gemma)")
    api.add_argument("--base-url", default="https://generativelanguage.googleapis.com",
                     help="API base URL (change only for proxies / tests)")

    cg = ap.add_argument_group("call graph")
    cg.add_argument("--no-callgraph", action="store_true",
                    help="Do NOT build per-commit call graphs or inject call-graph context into prompts")
    cg.add_argument("--hide-callgraph-labels", action="store_true",
                    help="Include call-graph structure in prompts but hide labels of already-analyzed functions (ablation)")
    return ap


def _apply_per_commit_cap(records: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    """Return records with at most `cap` records per commit, preserving order."""
    if cap <= 0:
        return records
    counter: Dict[str, int] = defaultdict(int)
    out = []
    for r in records:
        cid = r.get("commit_id", "")
        if counter[cid] < cap:
            out.append(r)
            counter[cid] += 1
    dropped = len(records) - len(out)
    if dropped:
        print(f"  [cap] --max-per-commit={cap}: dropped {dropped} records "
              f"from oversized commits", file=sys.stderr)
    return out


def _api_key() -> str | None:
    for env in API_KEY_ENVS:
        v = os.environ.get(env, "").strip()
        if v:
            return v
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = _build_arg_parser().parse_args()

    # --- fail fast on missing credentials ---
    key = _api_key()
    if not key:
        print("error: API key not found. Set GEMINI_API_KEY (or GOOGLE_API_KEY); "
              "get one at https://aistudio.google.com/apikey", file=sys.stderr)
        return 2

    cfg = LLMConfig(
        model_name=args.model,
        api_key=key,
        base_url=args.base_url,
        temperature=args.temperature,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        retries=args.retries,
        rpm=args.rpm,
        thinking_level=None if args.thinking == "default" else args.thinking,
        structured_output=not args.no_schema,
    )
    if args.votes < 1 or (args.votes > 1 and args.votes % 2 == 0):
        print("warning: --votes should be an odd number (an even number can end in ties, "
              "recorded as UNDETERMINABLE)", file=sys.stderr)
    out_dir = Path(args.out_dir)

    # --- load dataset ---
    print(f"Loading dataset from {args.dataset} …", file=sys.stderr)
    records = list(load_enriched_dataset(args.dataset))
    print(f"  → {len(records)} records with status=ok", file=sys.stderr)
    if args.max_per_commit > 0:
        records = _apply_per_commit_cap(records, args.max_per_commit)
    if args.limit > 0:
        records = records[: args.limit]
        print(f"  → limited to {len(records)} records (--limit)", file=sys.stderr)
    if not records:
        print("Nothing to process.", file=sys.stderr)
        return 0

    # --- classify ---
    outcome = run_pipeline(
        records, cfg,
        workers=args.workers,
        checkpoint=out_dir / "checkpoint.jsonl",
        resume=not args.no_resume,
        skip_deterministic=args.skip_deterministic,
        votes=max(1, args.votes),
        use_commit_verdict=not args.no_commit_verdict,
        retry_errors=args.retry_errors,
        vote_policy=args.vote_policy,
        use_callgraph=not args.no_callgraph,
        show_graph_labels=not args.hide_callgraph_labels,
    )

    # --- outputs + statistics (rebuilt from the whole checkpoint) ---
    print(f"\n########  {cfg.model_name}  —  {outcome.status}  ########", file=sys.stderr)
    if outcome.results:
        stats.save_results_csv(outcome.results, out_dir / "results.csv")
        stats.save_results_jsonl(outcome.results, out_dir / "results_full.jsonl")
        stats.summarize(outcome.results, out_dir)
    else:
        print("  (no results yet)", file=sys.stderr)

    print(f"\nnew={outcome.n_new}  in-checkpoint={len(outcome.results)}  "
          f"pending={outcome.n_remaining}  out-dir={out_dir}", file=sys.stderr)
    if outcome.reason:
        print(f"reason: {outcome.reason}", file=sys.stderr)

    # exit code 1 only for configuration/auth failures; a daily-quota stop is expected on
    # limited plans (resume later), so it is not an error.
    fatal = outcome.status == "stopped" and "LLMFatalError" in outcome.reason
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
