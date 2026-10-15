"""Dataset ingestion: load enriched JSONL with UTF-8 encoding and apply
the deterministic pre-filter before returning records to the caller.

Changes over v1:
- Explicit UTF-8 encoding (with replacement for rare binary glitches)
- Reports filtering statistics to stderr
- Filters status != 'ok' records with a reason logged
- Does NOT apply the deterministic filter here (done in main.py so the
  checkpoint is aware of all results, including deterministic ones)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator, Dict, Any, Tuple


def load_enriched_dataset(
    filepath: str | Path,
) -> Iterator[Dict[str, Any]]:
    """Yield all records with status='ok' from an enriched JSONL file.

    Skips malformed lines and logs counts to stderr.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    total = skipped_status = skipped_json = 0
    # `dv_enricher --refresh` appende le versioni aggiornate: per ogni hash vince l'ultima riga
    latest: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                skipped_json += 1
                print(
                    f"  [ingestion] line {lineno}: JSON parse error — {exc}",
                    file=sys.stderr,
                )
                continue

            latest[str(record.get("hash", f"line{lineno}"))] = record

    for record in latest.values():
        if record.get("status", "") != "ok":
            skipped_status += 1
            continue
        yield record

    print(
        f"  [ingestion] {total} lines read — "
        f"{skipped_status} skipped (status≠ok), "
        f"{skipped_json} skipped (malformed JSON), "
        f"{total - skipped_json - len(latest)} superseded by a later line (same hash)",
        file=sys.stderr,
    )


def load_all_records(filepath: str | Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield (status, record) for ALL lines (for coverage reporting)."""
    path = Path(filepath)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                yield record.get("status", "unknown"), record
            except json.JSONDecodeError:
                yield "malformed", {}