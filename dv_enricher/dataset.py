"""Lettura in streaming di DiverseVul (json-lines, anche direttamente dallo zip).

Il file completo pesa ~740 MB: non lo carichiamo in pandas. Teniamo in memoria solo
le funzioni con target=1 (~19k) e un contatore di quante funzioni ha ogni commit.
"""
from __future__ import annotations

import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Record:
    hash: str          # `hash` è un intero enorme nel JSON: lo trattiamo come stringa
    func: str
    target: int
    cwe: list
    project: str
    commit_id: str
    size: int          # n. di righe della funzione (verificato: coincide con end-start+1 nei file git)
    message: str


def _open_text(path: Path):
    if path.suffix == ".zip":
        zf = zipfile.ZipFile(path)
        name = next(n for n in zf.namelist() if n.endswith(".json"))
        return io.TextIOWrapper(zf.open(name), encoding="utf-8")
    return open(path, encoding="utf-8")


def iter_raw(path: str | Path) -> Iterator[dict]:
    with _open_text(Path(path)) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _to_record(d: dict) -> Record:
    return Record(
        hash=str(d["hash"]), func=d["func"], target=int(d["target"]),
        cwe=list(d.get("cwe") or []), project=d["project"], commit_id=d["commit_id"],
        size=int(d.get("size") or 0), message=d.get("message") or "",
    )


class DatasetIndex:
    """flagged: commit_id -> funzioni target=1; commit_total: commit_id -> n. funzioni totali."""

    def __init__(self) -> None:
        self.flagged: dict[str, list[Record]] = defaultdict(list)
        self.commit_total: Counter = Counter()
        self.n_rows = 0

    @classmethod
    def load(cls, path: str | Path) -> "DatasetIndex":
        idx = cls()
        for d in iter_raw(path):
            idx.n_rows += 1
            idx.commit_total[d["commit_id"]] += 1
            if int(d["target"]) == 1:
                r = _to_record(d)
                idx.flagged[r.commit_id].append(r)
        return idx

    def all_flagged(self) -> list[Record]:
        return [r for rs in self.flagged.values() for r in rs]

    def select(self, projects: Iterable[str] | None = None, hashes: Iterable[str] | None = None,
               sample: int | None = None, seed: int = 0) -> list[Record]:
        recs = self.all_flagged()
        if projects:
            ps = set(projects)
            recs = [r for r in recs if r.project in ps]
        if hashes:
            hs = {str(h) for h in hashes}
            recs = [r for r in recs if r.hash in hs]
        recs.sort(key=lambda r: r.hash)            # ordine deterministico prima del sampling
        if sample and sample < len(recs):
            recs = random.Random(seed).sample(recs, sample)
        return recs
