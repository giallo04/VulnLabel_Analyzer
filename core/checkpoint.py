"""Checkpoint writer: saves results incrementally to a JSONL file so that
a crash does not lose completed work. On resume, previously processed
records are skipped by their unique hash.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Set


class CheckpointWriter:
    """Thread-safe incremental JSONL checkpoint.

    Usage::

        with CheckpointWriter("results.jsonl") as ckpt:
            done = ckpt.already_done()        # set of hashes processed
            for record in records:
                if record["hash"] in done:
                    continue
                result = process(record)
                ckpt.write(result)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._fh = None

    # ------------------------------------------------------------------
    def __enter__(self) -> "CheckpointWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # append mode: safe to open even if file does not exist yet
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *_) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    # ------------------------------------------------------------------
    def already_done(self) -> Set[str]:
        """Return the set of function hashes already present in the checkpoint."""
        done: Set[str] = set()
        if not self.path.exists():
            return done
        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    h = obj.get("hash")
                    if h:
                        done.add(str(h))
                except json.JSONDecodeError:
                    pass  # corrupt line — skip
        return done

    def write(self, result: Dict[str, Any]) -> None:
        """Append one result dict as a JSONL line (thread-safe)."""
        line = json.dumps(result, ensure_ascii=False) + "\n"
        with self._lock:
            if self._fh is None:
                raise RuntimeError("CheckpointWriter used outside 'with' block")
            self._fh.write(line)
            self._fh.flush()


def iter_checkpoint(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Iterate over all valid result records in a checkpoint file."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def load_checkpoint_deduped(path: str | Path) -> List[Dict[str, Any]]:
    """All valid checkpoint records, one per hash (the last line wins, order of first seen)."""
    by_hash: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for obj in iter_checkpoint(path):
        h = obj.get("hash")
        if h is None:
            continue
        by_hash[str(h)] = obj
    return list(by_hash.values())
