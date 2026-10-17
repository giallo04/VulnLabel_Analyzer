# VulnLabel Analyzer

Find wrongly labeled functions in the **DiverseVul** vulnerability dataset, using an LLM (Gemini API).

## The problem

DiverseVul marks **every** function touched by a CVE-fixing commit as vulnerable (`target=1`). But a fix commit often also contains refactoring, API updates, logging, or new features. Those functions get labeled "vulnerable" even though they are not. This is called *commit tangling*.

VulnLabel Analyzer checks each function and tells you whether its label is right.

## How it works

```
DiverseVul ──▶ 1. Enrich ──▶ 2. Classify ──▶ 3. Stats
               (dv_enricher)   (main.py)       (stats.py)
```

1. **Enrich** – finds each function in its Git repo and extracts the before/after diff.
2. **Classify** – Gemini assigns each function a category (see below). Obvious cosmetic changes are filtered out without an API call.
3. **Stats** – computes label accuracy, confidence intervals, and noise per project and commit.

## Requirements

- Python 3.11+ (no external packages needed)
- Git in your PATH (for step 1 only)
- A [Gemini API key](https://aistudio.google.com/apikey), set as an environment variable:

```bash
export GEMINI_API_KEY="your-key"
# PowerShell: $env:GEMINI_API_KEY="your-key"
```

## Quick start

**1. Enrich the dataset**

```bash
python -m dv_enricher --dataset diversevul_20230702.zip --out enriched.jsonl --workdir work --workers 4
```

**2. Classify the functions**

```bash
python main.py --dataset enriched.jsonl --model gemini-3.8-flash --workers 4 --rpm 27
```

Set `--rpm` to your account's rate limit. If the run stops (crash, Ctrl+C, daily quota), just run the same command again: it resumes where it left off.

**3. Get the statistics**

```bash
python stats.py summary --checkpoint data/output/gemini/checkpoint.jsonl
```

## Categories

| Category | Label | Meaning |
| --- | --- | --- |
| `CORRECT_LABEL` | 1 | Contains the vulnerability and its fix. |
| `WRONG_LABEL_SPREAD` | 0 | The fix is spread across functions; this one is only a side part. |
| `WRONG_LABEL_CONSISTENCY` | 0 | API or consistency update caused by the fix, no security meaning. |
| `WRONG_LABEL_FEATURE` | 0 | Adds a feature instead of fixing a bug. |
| `WRONG_LABEL_IRRELEVANT` | 0 | Refactoring, logging, tests, or formatting. |
| `UNDETERMINABLE` | -1 | Not enough context. Excluded from the results. |

## Main options for `main.py`

| Flag | Default | What it does |
| --- | --- | --- |
| `--dataset` | `enriched.jsonl` | Input from `dv_enricher` |
| `--out-dir` | `data/output/gemini` | Where results are saved |
| `--model` | `gemini-3.8-flash` | Gemini model to use |
| `--workers` | `4` | Parallel requests |
| `--rpm` | `10` | Max requests per minute (0 = no limit) |
| `--votes` | `1` | Samples per function for majority voting (odd number, 3 = most accurate) |
| `--thinking` | `high` | `minimal`, `low`, `medium`, `high`, or `default` |
| `--limit` | `0` (all) | Only process N records (good for testing) |
| `--no-resume` | off | Redo everything, ignore the checkpoint |
| `--retry-errors` | off | On resume, retry records that failed to parse |

Tip: to try a different model or prompt, use a new `--out-dir` so old results are not mixed in.

## Output files

Saved in `--out-dir`:

- `checkpoint.jsonl` – one result per function, written as it goes
- `results.csv` and `results_full.jsonl` – final results (the second includes raw model output)
- `accuracy_summary.json` and `stats_report.txt` – accuracy numbers
- `commit_verdicts.jsonl` – one summary per commit, so all functions in a commit are judged consistently

## Checking results by hand

You can verify the LLM's work and get more reliable metrics:

```bash
# 1. Export CSVs to review
python stats.py export-review --checkpoint data/output/gemini/checkpoint.jsonl

# 2. Fill in the "verdetto" column: ok / ERROR / idk (leave empty to skip)

# 3. Compute metrics from your answers
python stats.py review data/output/gemini/review_llm.csv data/output/gemini/review_static.csv \
       --checkpoint data/output/gemini/checkpoint.jsonl
```

This gives accuracy, confusion matrix, precision/recall/F1, MCC, Cohen's kappa, and an estimate of the true accuracy of the dataset labels.

## Project structure

```
├── main.py          # Step 2: classification
├── stats.py         # Step 3: statistics and review
├── evaluate.py      # Evaluation script
├── core/            # LLM client, prompts, parsing, checkpointing, pipeline
├── dv_enricher/     # Step 1: enrichment (run with python -m dv_enricher)
├── tests/           # Run: python -m unittest discover -s tests -v
└── data/output/     # Results
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use a mocked API, so no network or API key is needed.

## License

GPL-3.0. See [LICENSE](LICENSE).