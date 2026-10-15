"""Parser di unified diff con numerazione di riga su entrambi i lati."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


@dataclass
class DiffLine:
    tag: str                 # ' ', '-', '+'
    text: str
    old_ln: int | None       # numero di riga nel vecchio file (None per '+')
    new_ln: int | None       # numero di riga nel nuovo file (None per '-')
    old_anchor: int = 0      # per '+': riga vecchia che segue l'inserzione
    new_anchor: int = 0      # per '-': riga nuova che segue la cancellazione


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    header: str
    lines: list[DiffLine] = field(default_factory=list)

    def render(self) -> str:
        head = f"@@ -{self.old_start},{self.old_len} +{self.new_start},{self.new_len} @@{self.header}"
        return "\n".join([head] + [l.tag + l.text for l in self.lines])


def parse_unified_diff(text: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    cur: Hunk | None = None
    old = new = 0
    for raw in text.split("\n"):
        m = _HUNK.match(raw)
        if m:
            os_, ol, ns, nl = int(m[1]), m[2], int(m[3]), m[4]
            ol = 1 if ol is None else int(ol)
            nl = 1 if nl is None else int(nl)
            cur = Hunk(os_, ol, ns, nl, m[5])
            hunks.append(cur)
            # con lunghezza 0 git indica la riga *precedente* l'inserzione/cancellazione
            old = os_ if ol > 0 else os_ + 1
            new = ns if nl > 0 else ns + 1
            continue
        if cur is None or raw.startswith("\\"):     # header del file / "\ No newline at end of file"
            continue
        tag, body = raw[:1], raw[1:]
        if tag == " ":
            cur.lines.append(DiffLine(" ", body, old, new)); old += 1; new += 1
        elif tag == "-":
            cur.lines.append(DiffLine("-", body, old, None, new_anchor=new)); old += 1
        elif tag == "+":
            cur.lines.append(DiffLine("+", body, None, new, old_anchor=old)); new += 1
    for h in hunks:
        _align_blocks(h)
    return hunks


# --------------------------------------------------------------------------------------
# Allineamento dei blocchi di modifica
# --------------------------------------------------------------------------------------
_ALIGN_MAX_CELLS = 2_000_000      # oltre: mappatura proporzionale invece di SequenceMatcher
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub("", text)


def _align_block(minus: list[DiffLine], plus: list[DiffLine]) -> None:
    """Attribuisce ogni riga '+' alla riga '-' che sostituisce (e viceversa).

    Git emette un blocco di modifica come TUTTE le righe '-' seguite da TUTTE le '+', senza
    dire quale '+' rimpiazza quale '-'. Con l'ancora grezza (ogni '+' ancorata alla riga che
    segue il blocco) in un blocco esteso su più funzioni tutte le '+' finirebbero nell'ultima.
    Allineando i due lati (testo normalizzato) ogni riga va alla funzione giusta.
    """
    m_pos = [l.old_ln for l in minus]
    p_pos = [l.new_ln for l in plus]
    end_old, end_new = m_pos[-1] + 1, p_pos[-1] + 1
    a = [_norm(l.text) for l in minus]
    b = [_norm(l.text) for l in plus]
    if a == b:
        ops = [("equal", 0, len(a), 0, len(b))]
    elif len(a) * len(b) <= _ALIGN_MAX_CELLS:
        ops = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    else:
        ops = [("replace", 0, len(a), 0, len(b))]
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("equal", "replace"):
            nm, npl = i2 - i1, j2 - j1
            for k in range(npl):
                plus[j1 + k].old_anchor = m_pos[i1 + (k * nm) // npl]
            for k in range(nm):
                minus[i1 + k].new_anchor = p_pos[j1 + (k * npl) // nm]
        elif tag == "insert":
            anchor = m_pos[i1] if i1 < len(m_pos) else end_old
            for k in range(j1, j2):
                plus[k].old_anchor = anchor
        else:  # delete
            anchor = p_pos[j1] if j1 < len(p_pos) else end_new
            for k in range(i1, i2):
                minus[k].new_anchor = anchor


def _align_blocks(h: Hunk) -> None:
    L, i = h.lines, 0
    while i < len(L):
        if L[i].tag == " ":
            i += 1
            continue
        j = i
        while j < len(L) and L[j].tag != " ":
            j += 1
        block = L[i:j]
        minus = [l for l in block if l.tag == "-"]
        plus = [l for l in block if l.tag == "+"]
        if minus and plus:
            _align_block(minus, plus)
        i = j


def changed_in_range(hunks: list[Hunk], start: int, end: int, side: str = "old"):
    """Righe cancellate/inserite che ricadono in [start, end] (1-based, inclusivo).

    side='old': range espresso nel file vecchio (pre-fix). Una cancellazione conta se la
    sua riga vecchia è nel range; una inserzione conta se è ancorata a una riga del range.
    side='new': simmetrico sul file nuovo.
    """
    dels: list[str] = []
    inss: list[str] = []
    touched: list[Hunk] = []
    for h in hunks:
        hit = False
        for l in h.lines:
            if l.tag == "-":
                pos = l.old_ln if side == "old" else l.new_anchor
                if start <= pos <= end:
                    dels.append(l.text); hit = True
            elif l.tag == "+":
                pos = l.old_anchor if side == "old" else l.new_ln
                if start <= pos <= end:
                    inss.append(l.text); hit = True
        if hit:
            touched.append(h)
    return dels, inss, touched


# --------------------------------------------------------------------------------------
# Diff ristretto a UNA funzione
# --------------------------------------------------------------------------------------
def _side_key(side: str):
    """Posizione di una riga di diff sul lato `side` (coerente con changed_in_range)."""
    if side == "old":
        return lambda l: l.old_ln if l.old_ln is not None else l.old_anchor
    return lambda l: l.new_ln if l.new_ln is not None else l.new_anchor


def render_function_diff(hunks: list[Hunk], start: int, end: int, side: str = "old",
                         context: int = 3) -> str:
    """Unified diff con SOLO le righe che cadono in [start, end] sul lato `side`.

    `git diff -U3` fonde le modifiche vicine in un unico hunk: in una riscrittura estesa
    (es. una libreria riformattata) un solo hunk copre decine di funzioni. Passare quell'hunk
    al modello e troncarlo dall'inizio fa sparire proprio le righe della funzione esaminata.
    Qui si tengono le righe modificate della funzione (le '-' e le '+' che le sostituiscono,
    grazie all'allineamento dei blocchi) più `context` righe di contesto dentro il range, e
    gli header `@@` sono ricalcolati sulle righe tenute. Stringa vuota se non c'è nulla.
    """
    key = _side_key(side)
    inr = lambda l: start <= key(l) <= end
    out: list[str] = []
    for h in hunks:
        # sequenza di visualizzazione: contesto in range + righe modificate in range,
        # per ogni blocco prima le '-' poi le '+'
        disp: list[tuple[DiffLine, bool]] = []
        L, i = h.lines, 0
        while i < len(L):
            if L[i].tag == " ":
                if inr(L[i]):
                    disp.append((L[i], False))
                i += 1
                continue
            j = i
            while j < len(L) and L[j].tag != " ":
                j += 1
            blk = [l for l in L[i:j] if inr(l)]
            disp += [(l, True) for l in blk if l.tag == "-"] + [(l, True) for l in blk if l.tag == "+"]
            i = j
        idx = [n for n, (_, ch) in enumerate(disp) if ch]
        if not idx:
            continue
        wins: list[list[int]] = []
        for n in idx:
            lo, hi = max(n - context, 0), min(n + context, len(disp) - 1)
            if wins and lo <= wins[-1][1] + 1:
                wins[-1][1] = max(wins[-1][1], hi)
            else:
                wins.append([lo, hi])
        for lo, hi in wins:
            kept = [l for l, _ in disp[lo:hi + 1]]
            old = [l for l in kept if l.tag in (" ", "-")]
            new = [l for l in kept if l.tag in (" ", "+")]
            # con lunghezza 0 git indica la riga PRECEDENTE l'inserzione/cancellazione
            o_start = old[0].old_ln if old else kept[0].old_anchor - 1
            n_start = new[0].new_ln if new else kept[0].new_anchor - 1
            out.append(f"@@ -{o_start},{len(old)} +{n_start},{len(new)} @@")
            out.extend(l.tag + l.text for l in kept)
    return "\n".join(out)


def covered_until(hunks: list[Hunk], side: str = "old") -> int:
    """Ultima riga (sul lato `side`) vista dagli hunk; 0 se non ce ne sono."""
    key = _side_key(side)
    return max((key(l) for h in hunks for l in h.lines), default=0)


def shrink_lines(text: str, limit: int, head_frac: float = 0.7) -> tuple[str, bool, int]:
    """Riduce `text` a ~`limit` caratteri tenendo testa E coda (a righe intere).

    Ritorna (testo, troncato, righe_omesse). Il troncamento tiene l'inizio e la fine della
    modifica perché in funzioni molto riscritte la parte decisiva può stare in coda.
    """
    if len(text) <= limit:
        return text, False, 0
    lines = text.split("\n")
    head, tail, size = [], [], 0
    budget_h, budget_t = int(limit * head_frac), limit - int(limit * head_frac)
    i, j = 0, len(lines) - 1
    used_h = used_t = 0
    while i <= j and used_h + len(lines[i]) + 1 <= budget_h:
        head.append(lines[i]); used_h += len(lines[i]) + 1; i += 1
    while j >= i and used_t + len(lines[j]) + 1 <= budget_t:
        tail.append(lines[j]); used_t += len(lines[j]) + 1; j -= 1
    omitted = j - i + 1
    if omitted <= 0:
        return text, False, 0
    tail.reverse()
    return "\n".join(head + [f"[... {omitted} diff lines omitted ...]"] + tail), True, omitted


def map_line(hunks: list[Hunk], line: int, side: str = "old") -> int:
    """Posizione approssimata sull'altro lato di `line` (1-based) di `side`.

    Somma lo scostamento (righe aggiunte - cancellate) degli hunk che stanno interamente
    prima della riga: serve a scegliere la controparte giusta quando un nome compare più volte.
    """
    delta = 0
    for h in hunks:
        if side == "old":
            end = h.old_start + max(h.old_len, 1) - 1
            if end < line:
                delta += h.new_len - h.old_len
        else:
            end = h.new_start + max(h.new_len, 1) - 1
            if end < line:
                delta += h.old_len - h.new_len
    return max(line + delta, 1)


def anchored_is_plausible(deleted: int, inserted: int, length: int) -> bool:
    """Un diff ancorato per righe è credibile solo se le dimensioni sono compatibili con la funzione.

    Non si possono cancellare più righe di quante la funzione ne abbia, né aggiungerne una
    quantità sproporzionata: è il segno tipico di un blocco di modifica esteso su più funzioni
    (es. una libreria riscritta), dove le righe '+' finiscono attribuite alla funzione sbagliata.
    """
    return deleted <= length and inserted <= 2 * length + 10
