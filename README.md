# VulnLabel Analyzer

Detect mislabeled functions in the **DiverseVul** vulnerability dataset using an LLM (Gemini API).

## The problem

DiverseVul labels **every** function touched by a CVE-fixing commit as vulnerable (`target=1`). But fix commits often also contain refactoring, API updates, logging or new features, so many of those functions are not vulnerable (*commit tangling*). VulnLabel Analyzer re-examines each function and estimates how accurate the original labels are.

## Pipeline

```
DiverseVul ──▶ 1. Enrich ──▶ 2. Classify ──▶ 3. Stats
               dv_enricher    main.py         stats/
```

1. **Enrich** – locates each function in its Git repository and extracts the function-level before/after diff, plus commit context.
2. **Classify** – a deterministic pre-filter resolves obvious cases (tests, non-C/C++ files, purely cosmetic changes) without an API call. The rest goes to Gemini, with **one call per function**. The prompt includes:
   - a per-commit verdict (nature, root cause, fix pattern), computed once and shared by all functions of the commit;
   - a static call graph of the commit's functions, analyzed callees-first, so each function sees the labels already assigned to its neighbours;
   - "twin" functions that received the same edit, to keep identical edits consistent.
3. **Stats** – label accuracy with 95% Wilson confidence intervals, broken down by method, category and project. Optional manual validation on a stratified sample.

## Requirements

- Python 3.11+ (standard library only)
- Git in `PATH` (step 1 only)
- A [Gemini API key](https://aistudio.google.com/apikey) in the environment:

```bash
export GEMINI_API_KEY="your-key"      # PowerShell: $env:GEMINI_API_KEY="your-key"
```

## Quick start

```bash
# 1. Enrich the dataset (clones repos into --workdir)
python -m dv_enricher --dataset diversevul_20230702.zip --out enriched.jsonl --workdir work --workers 4

# 2. Classify (resumable: re-run the same command after a crash or quota stop)
python main.py --dataset enriched.jsonl --rpm 10 --workers 4

# 3. Statistics from the checkpoint
python -m stats.stats summary --checkpoint data/output/gemini/checkpoint.jsonl
```

Set `--rpm` to your account's rate limit. Use a new `--out-dir` when changing model or prompt, so results are not mixed.

## Categories

| Category | Label | Meaning |
| --- | --- | --- |
| `CORRECT_LABEL` | 1 | The function contains the vulnerability and the diff fixes it. |
| `WRONG_LABEL_SPREAD` | 0 | Safe before the commit; the change only supports a fix located elsewhere. |
| `WRONG_LABEL_CONSISTENCY` | 0 | Mechanical adaptation (rename, signature, call-site update), no new logic. |
| `WRONG_LABEL_FEATURE` | 0 | New functionality, not a bug fix. |
| `WRONG_LABEL_IRRELEVANT` | 0 | Refactoring, logging, tests, formatting, unrelated changes. |
| `UNDETERMINABLE` | -1 | Missing or unusable evidence. Excluded from the statistics. |

A function is counted as **noise** when the predicted label differs from the original one.

## `main.py` options

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset` | `enriched.jsonl` | Input produced by `dv_enricher` |
| `--out-dir` | `data/output/gemini` | Output directory |
| `--model` | `gemini-3.5-flash-lite` | Gemini model id |
| `--workers` | `4` | Parallel workers (one commit each) |
| `--rpm` | `10` | Client-side requests/minute cap (`0` = unlimited) |
| `--temperature` | `0.4` | Sampling temperature |
| `--thinking` | `high` | `minimal`, `low`, `medium`, `high`, `default` |
| `--limit` | `0` (all) | Process only N records (for testing) |
| `--max-per-commit` | `0` (no cap) | Cap records per commit, limiting giant tangled commits |
| `--skip-deterministic` | off | Send every record to the API |
| `--no-callgraph` | off | Disable call-graph context (ablation) |
| `--retry-errors` | off | On resume, re-query unparsable answers |
| `--no-resume` | off | Ignore the existing checkpoint |

Run `python main.py --help` for the full list.

## Output files

In `--out-dir`:

- `checkpoint.jsonl` – one result per function, written incrementally
- `results.csv`, `results_full.jsonl` – final results (the latter with raw model output)
- `accuracy_summary.json`, `stats_report.txt` – accuracy estimates
- `commit_verdicts.jsonl` – per-commit verdicts

## Manual validation

Check the LLM against your own judgment on a stratified sample (N per category, at most one function per commit):

```bash
python stats/sample_for_review.py data/output/gemini/results_full.jsonl -o stats/review_sample.csv
# Fill in the human_vulnerable column: ok = vulnerable, no = not vulnerable
python stats/evaluate_review.py stats/review_sample.csv
```

The report (`review_report.md`, `review_metrics.json`) includes the confusion matrix, precision/recall/F1, MCC, Cohen's kappa and Wilson intervals.

## Project structure

```
main.py          # step 2: classification entry point
core/            # pipeline, prompts, LLM client, call graph, deterministic filter, checkpoint
dv_enricher/     # step 1: dataset enrichment
stats/           # step 3: statistics and manual-validation scripts
tests/           # unit tests (mocked API)
data/output/     # generated results
```

## Tests

```bash
python -m unittest discover -s tests -v
```

No network or API key needed.

## License

GPL-3.0. See [LICENSE](LICENSE).