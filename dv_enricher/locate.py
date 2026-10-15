"""Localizza il testo di una funzione del dataset dentro i file di un commit.

DiverseVul fornisce solo `func` (testo) senza file né righe. Il matching avviene sul testo
con tutti gli spazi rimossi: robusto a differenze di indentazione/EOL, e permette di
risalire a file e intervallo di righe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_TOKEN = re.compile(r"\S+")


class FileView:
    __slots__ = ("path", "text", "_norm")

    def __init__(self, path: str, text: str) -> None:
        self.path, self.text, self._norm = path, text, None

    @property
    def norm(self) -> str:
        if self._norm is None:
            self._norm = "".join(self.text.split())
        return self._norm

    def norm_span_to_lines(self, k0: int, k1: int) -> tuple[int, int]:
        """[k0, k1) su testo normalizzato -> (riga_inizio, riga_fine) 1-based inclusive."""
        acc = 0
        s = e = None
        for m in _TOKEN.finditer(self.text):
            n = m.end() - m.start()
            if s is None and acc + n > k0:
                s = m.start() + (k0 - acc)
            if acc + n >= k1:
                e = m.start() + (k1 - acc)
                break
            acc += n
        assert s is not None and e is not None
        return self.text.count("\n", 0, s) + 1, self.text.count("\n", 0, e - 1) + 1


def normalize(func: str) -> str:
    return "".join(func.split())


@dataclass
class Match:
    path: str
    start: int
    end: int
    method: str        # 'exact' | 'fuzzy'
    n_matches: int     # occorrenze trovate (>1 => ambiguo)
    score: float = 1.0


def _exact_hits(view: FileView, nf: str, cap: int = 5) -> list[int]:
    hits, pos = [], 0
    while len(hits) < cap:
        i = view.norm.find(nf, pos)
        if i < 0:
            break
        hits.append(i)
        pos = i + 1
    return hits


def _fuzzy(view: FileView, nf: str, thr: float) -> tuple[int, int, float] | None:
    """Fallback: stessa testa (40 caratteri) e testo simile (es. commenti in testa alla funzione)."""
    if not 40 <= len(nf) <= 6000:
        return None
    best, pos = None, 0
    for _ in range(8):
        i = view.norm.find(nf[:40], pos)
        if i < 0:
            break
        pos = i + 1
        window = view.norm[i:i + len(nf)]
        sm = SequenceMatcher(None, window, nf, autojunk=False)
        if sm.real_quick_ratio() < thr or sm.quick_ratio() < thr:
            continue
        r = sm.ratio()
        if r >= thr and (best is None or r > best[2]):
            last = [b for b in sm.get_matching_blocks() if b.size][-1]
            best = (i, i + last.a + last.size, r)
    return best


def locate(func: str, views: list[FileView], fuzzy: bool = True, fuzzy_threshold: float = 0.95) -> Match | None:
    nf = normalize(func)
    if not nf:
        return None
    found: list[tuple[FileView, int]] = []
    for v in views:
        for i in _exact_hits(v, nf):
            found.append((v, i))
    if found:
        v, i = found[0]
        s, e = v.norm_span_to_lines(i, i + len(nf))
        return Match(v.path, s, e, "exact", len(found))
    if fuzzy:
        for v in views:
            r = _fuzzy(v, nf, fuzzy_threshold)
            if r:
                s, e = v.norm_span_to_lines(r[0], r[1])
                return Match(v.path, s, e, "fuzzy", 1, r[2])
    return None
