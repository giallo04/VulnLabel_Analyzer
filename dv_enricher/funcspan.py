"""Diff a livello di FUNZIONE: confronta la funzione prima/dopo la modifica.

Perché esiste
-------------
Il diff di file di git (`git diff`) emette un blocco di modifica come TUTTE le righe '-'
seguite da TUTTE le '+'. Quando un commit riscrive o riformatta un file esteso (libreria
importata/aggiornata, funzioni riordinate), attribuire le righe '+' alla funzione giusta
con ancore di riga è impossibile in generale: al modello arrivava il corpo di *un'altra*
funzione («il diff mostra cJSON_Minify al posto di cJSON_CreateTrue», frac_changed > 1).

Qui si procede al contrario: si trova la STESSA funzione nell'altro lato del commit
(per nome, con similarità del testo e posizione attesa come criterio di scelta) e si
confrontano i due testi. Il diff risultante contiene per costruzione solo righe di quella
funzione, con numeri di riga reali di entrambi i file.

Il modulo è solo stdlib. Le funzioni C/C++ sono individuate con una scansione delle
graffe su testo in cui commenti, stringhe e direttive del preprocessore sono mascherati.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import cached_property

from . import features as F

# ---------------------------------------------------------------------------
# Mascheratura: commenti / stringhe / direttive -> spazi (lunghezza e newline invariati)
# ---------------------------------------------------------------------------
_MASK_RE = re.compile(
    r"""
    //[^\n]*(?:\\\n[^\n]*)*                 # commento di riga (anche con continuazione)
  | /\*.*?\*/                               # commento a blocco
  | (?:u8|[uUL])?"(?:\\.|[^"\\\n])*"        # stringa
  | (?:u8|[uUL])?'(?:\\.|[^'\\\n])*'        # char
  | ^[ \t]*\#(?:[^\n\\]|\\.|\\\n)*          # direttiva del preprocessore (con continuazioni)
    """,
    re.VERBOSE | re.DOTALL | re.MULTILINE,
)


def mask_source(text: str) -> str:
    def blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group())
    return _MASK_RE.sub(blank, text)


# ---------------------------------------------------------------------------
# Enumerazione delle funzioni di un file
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FuncSpan:
    name: str            # nome come lo vede features.extract_func_name (macro wrapper inclusi)
    start: int           # riga 1-based della prima riga della firma
    end: int             # riga 1-based della '}' di chiusura

    @property
    def short(self) -> str:
        return self.name.split("::")[-1]

    @property
    def length(self) -> int:
        return self.end - self.start + 1


_TRANSPARENT_RE = re.compile(
    r"(?:\bnamespace\b[\w:\s]*|\bextern\b\s*|\b(?:class|struct|union)\b[^;{}()=]*)$"
)
_FUNC_TAIL_RE = re.compile(
    r"\)\s*(?:(?:const|volatile|noexcept|override|final|throw\s*\([^)]*\)|&&?|->\s*[\w:<>*&\s]+?)\s*)*$"
)
_ACCESS_RE = re.compile(r"^\s*(?:public|private|protected)\s*:\s*")


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for m in re.finditer(r"\n", text):
        starts.append(m.end())
    return starts


def _line_of(starts: list[int], pos: int) -> int:
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def _header_start(masked: str, brace: int) -> int:
    """Indice dell'inizio della firma che precede la '{' alla posizione `brace`.

    La firma comincia dopo l'ultimo ';' '{' '}' che precede la graffa (le direttive e i
    commenti sono già mascherati); vale anche per costruttori con lista di inizializzazione.
    """
    boundary = max(masked.rfind(";", 0, brace), masked.rfind("}", 0, brace),
                   masked.rfind("{", 0, brace))
    start = boundary + 1
    while start < brace and masked[start].isspace():
        start += 1
    m = _ACCESS_RE.match(masked[start:brace])
    if m:
        start += m.end()
        while start < brace and masked[start].isspace():
            start += 1
    return start


def iter_functions(text: str) -> list[FuncSpan]:
    """Tutte le definizioni di funzione di un file C/C++ (anche metodi in class/namespace)."""
    masked = mask_source(text)
    starts = _line_starts(text)
    out: list[FuncSpan] = []
    stack: list[str] = []          # per ogni '{' aperta: 'T' trasparente | 'F' funzione | 'O' altro
    func_open: list[int] = []      # indici '{' delle funzioni aperte
    i, n = 0, len(masked)
    while i < n:
        c = masked[i]
        if c == "{":
            if "F" in stack or "O" in stack:
                stack.append("O")
            else:
                clause = masked[_header_start(masked, i):i]
                if _FUNC_TAIL_RE.search(clause.rstrip()) or _is_ctor_init(clause):
                    stack.append("F")
                    func_open.append(i)
                elif _TRANSPARENT_RE.search(clause.rstrip()):
                    stack.append("T")
                else:
                    stack.append("O")
        elif c == "}":
            if stack:
                kind = stack.pop()
                if kind == "F":
                    brace = func_open.pop()
                    hs = _header_start(masked, brace)
                    header = text[hs:brace + 1]
                    name = F.extract_func_name(header)
                    if name:
                        out.append(FuncSpan(name, _line_of(starts, hs), _line_of(starts, i)))
        i += 1
    return out


def _is_ctor_init(clause: str) -> bool:
    """`Foo::Foo(int a) : x(a), y(b) {` -> la '{' apre il corpo di un costruttore."""
    m = re.search(r"\)\s*:\s*(?!:)[^;{}]*$", clause)
    return bool(m) and bool(re.search(r"[\w>)\]]\s*$", clause.rstrip()))


class SourceFile:
    """Testo di un file + indice lazy delle sue funzioni."""

    def __init__(self, path: str, text: str) -> None:
        self.path, self.text = path, text
        self.lines = text.split("\n")

    @cached_property
    def functions(self) -> list[FuncSpan]:
        try:
            return iter_functions(self.text)
        except Exception:  # noqa: BLE001 - un file illeggibile non deve fermare il batch
            return []

    def span_text(self, s: FuncSpan) -> list[str]:
        return self.lines[s.start - 1:s.end]


# ---------------------------------------------------------------------------
# Similarità e scelta della controparte
# ---------------------------------------------------------------------------
_ATOM_RE = re.compile(r"""(?:u8|[uUL])?"(?:\\.|[^"\\\n])*"|(?:u8|[uUL])?'(?:\\.|[^'\\\n])*'""")
_WORD_GAP = re.compile(r"(?<=\w)\s+(?=\w)")
_ANY_WS = re.compile(r"\s+")
_GAP = "\x00"


def _squash(seg: str) -> str:
    # tra due caratteri di parola lo spazio è significativo (`int x` != `intx`): ne resta uno
    return _ANY_WS.sub("", _WORD_GAP.sub(_GAP, seg)).replace(_GAP, " ")


def norm_line(s: str) -> str:
    """Testo senza whitespace superfluo; le stringhe/char literal restano INTATTE
    (`"a b"` e `"ab"` sono codice diverso)."""
    out, pos = [], 0
    for m in _ATOM_RE.finditer(s):
        out.append(_squash(s[pos:m.start()]))
        out.append(m.group())
        pos = m.end()
    out.append(_squash(s[pos:]))
    return "".join(out)


def similarity(a: list[str], b: list[str]) -> float:
    na = [norm_line(x) for x in a if x.strip()]
    nb = [norm_line(x) for x in b if x.strip()]
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if len(na) * len(nb) > 4_000_000:
        return SequenceMatcher(None, na, nb, autojunk=False).quick_ratio()
    return SequenceMatcher(None, na, nb, autojunk=False).ratio()


@dataclass
class Counterpart:
    path: str
    start: int
    end: int
    name: str
    score: float
    how: str           # 'same_name' | 'renamed' | 'other_file'


def _short(name: str | None) -> str:
    return (name or "").split("::")[-1]


def _best(cands: list[FuncSpan], src: SourceFile, ref_lines: list[str], expected: int | None
          ) -> tuple[FuncSpan, float] | None:
    scored = []
    for c in cands:
        sim = similarity(ref_lines, src.span_text(c))
        dist = abs(c.start - expected) if expected else 0
        scored.append((sim, -dist, c))
    if not scored:
        return None
    top = max(s for s, _, _ in scored)
    near = [t for t in scored if t[0] >= top - 0.02]       # a pari similarità vince la posizione
    sim, _, cand = max(near, key=lambda t: t[1])
    return cand, sim


def find_counterpart(
    name: str | None,
    ref_lines: list[str],
    same_file: SourceFile | None,
    other_files: Callable[[], list[SourceFile]],
    expected_start: int | None,
) -> Counterpart | None:
    """Trova nell'altro lato del commit la funzione che corrisponde a `ref_lines`.

    Ordine: stesso nome nello stesso file -> funzione simile (rinominata) nello stesso file ->
    stesso nome in un altro file modificato dal commit (file spostato/rinominato).
    """
    short = _short(name)
    if same_file is not None:
        cands = [f for f in same_file.functions if short and f.short == short]
        if cands:
            pick = _best(cands, same_file, ref_lines, expected_start)
            if pick and (len(cands) == 1 or pick[1] >= 0.3):
                c, sim = pick
                return Counterpart(same_file.path, c.start, c.end, c.name, sim, "same_name")
        # rinominata: funzione molto simile (>= 0.7) nello stesso file
        if len(ref_lines) >= 4:
            pick = _best(same_file.functions, same_file, ref_lines, expected_start)
            if pick and pick[1] >= 0.7:
                c, sim = pick
                return Counterpart(same_file.path, c.start, c.end, c.name, sim, "renamed")
    for src in other_files():            # lazy: legge gli altri file solo se serve
        if same_file is not None and src.path == same_file.path:
            continue
        cands = [f for f in src.functions if short and f.short == short]
        pick = _best(cands, src, ref_lines, None) if cands else None
        if pick and pick[1] >= 0.6:
            c, sim = pick
            return Counterpart(src.path, c.start, c.end, c.name, sim, "other_file")
    return None


# ---------------------------------------------------------------------------
# Diff tra le due versioni della funzione
# ---------------------------------------------------------------------------
@dataclass
class PairDiff:
    text: str                 # diff unificato (vuoto se identiche o solo whitespace)
    deleted: list[str]
    inserted: list[str]
    whitespace_only: bool     # i testi differiscono, ma solo per spazi/indentazione
    identical: bool


def _hunk_header(o_start: int, o_len: int, n_start: int, n_len: int) -> str:
    # con lunghezza 0 git indica la riga PRECEDENTE (convenzione dell'unified diff)
    return (f"@@ -{o_start - 1 if o_len == 0 else o_start},{o_len} "
            f"+{n_start - 1 if n_len == 0 else n_start},{n_len} @@")


def pair_diff(old: list[str], new: list[str], old_start: int, new_start: int,
              context: int = 3) -> PairDiff:
    """Diff riga per riga dei due testi. Le righe uguali a meno degli spazi sono contesto.

    `old_start`/`new_start`: numero di riga (1-based) della prima riga di ciascun testo nel
    rispettivo file; gli header @@ riportano quindi numeri di riga reali.
    """
    if old == new:
        return PairDiff("", [], [], False, True)
    # stesso testo a meno degli spazi (anche con righe spezzate/unite): solo riformattazione
    if norm_line("\n".join(old)) == norm_line("\n".join(new)):
        return PairDiff("", [], [], True, False)
    a = [norm_line(x) for x in old]
    b = [norm_line(x) for x in new]
    ops = SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if len(a) * len(b) <= 4_000_000 \
        else [("replace", 0, len(a), 0, len(b))]
    deleted: list[str] = []
    inserted: list[str] = []
    # sequenza di visualizzazione: (tag, testo, riga_old, riga_new)
    seq: list[tuple[str, str, int, int]] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i2 - i1):
                seq.append((" ", new[j1 + k], old_start + i1 + k, new_start + j1 + k))
            continue
        for k in range(i1, i2):
            seq.append(("-", old[k], old_start + k, new_start + j1))
            deleted.append(old[k])
        for k in range(j1, j2):
            seq.append(("+", new[k], old_start + i1, new_start + k))
            inserted.append(new[k])
    if not deleted and not inserted:
        return PairDiff("", [], [], True, False)

    idx = [n for n, s in enumerate(seq) if s[0] != " "]
    wins: list[list[int]] = []
    for n in idx:
        lo, hi = max(n - context, 0), min(n + context, len(seq) - 1)
        if wins and lo <= wins[-1][1] + 1:
            wins[-1][1] = max(wins[-1][1], hi)
        else:
            wins.append([lo, hi])
    out: list[str] = []
    for lo, hi in wins:
        part = seq[lo:hi + 1]
        olds = [p for p in part if p[0] in (" ", "-")]
        news = [p for p in part if p[0] in (" ", "+")]
        o_s = olds[0][2] if olds else part[0][2]
        n_s = news[0][3] if news else part[0][3]
        out.append(_hunk_header(o_s, len(olds), n_s, len(news)))
        out.extend(p[0] + p[1] for p in part)
    return PairDiff("\n".join(out), deleted, inserted, False, False)
