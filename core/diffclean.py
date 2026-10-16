"""Pulizia degli artefatti di span nel `function_diff` (diff di coppia).

Problema
--------
Il campo `func` del dataset parte dalla riga del NOME della funzione, anche quando il tipo di
ritorno o una macro di export stanno sulla riga precedente (`unsigned long`,
`NCURSES_EXPORT(int)`, `static`, ...). Il lato opposto del diff di coppia viene invece estratto
includendo quella riga. Il diff contiene allora una riga fantasma, ad es.

    @@ -56,3 +55,4 @@
    +unsigned long
     arch_get_unmapped_area(struct file *filp, ...

che nel file vecchio esiste già (riga 55) ma cade FUORI dallo span etichettato [start, end].
Il modello la legge come "aggiunge il tipo di ritorno"; con una macro come `NCURSES_EXPORT` il
controparte viene anche scambiata per una rinomina (`diff_method=pair_renamed`).

Correzione (a valle, senza toccare l'enricher)
----------------------------------------------
Si eliminano dal `function_diff` i blocchi di sole righe dell'ALTRO lato (inserimenti puri se il
corpo mostrato è `old`, cancellazioni pure se è `new`) la cui posizione cade fuori da
[start_line, end_line]. I blocchi che contengono righe del lato etichettato non si toccano.
Gli header `@@` vengono ricalcolati; un hunk che resta senza modifiche sparisce.

Salvaguardie: si interviene solo se il diff è nelle coordinate del file (le righe del lato
etichettato dentro lo span coincidono con `func`, e ce n'è almeno una); un hunk incompleto
(troncato) passa invariato. Quando l'enricher sarà corretto la funzione diventa un no-op.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class _Item:
    __slots__ = ("tag", "text", "old", "new", "drop")

    def __init__(self, tag: str, text: str, old: int, new: int) -> None:
        self.tag, self.text, self.old, self.new, self.drop = tag, text, old, new, False


class _Hunk:
    def __init__(self, header: str, tail: str, items: List[_Item], raw: List[str], complete: bool):
        self.header, self.tail, self.items, self.raw, self.complete = header, tail, items, raw, complete


def _parse(diff: str) -> Optional[Tuple[List[str], List[_Hunk]]]:
    """(preambolo, hunk). None se il testo non è un diff leggibile."""
    lines = diff.split("\n")
    preamble: List[str] = []
    hunks: List[_Hunk] = []
    cur: Optional[_Hunk] = None
    old_ln = new_ln = old_left = new_left = 0

    for ln in lines:
        m = _HUNK_RE.match(ln)
        if m:
            a, c = int(m.group(1)), int(m.group(3))
            b = int(m.group(2)) if m.group(2) is not None else 1
            d = int(m.group(4)) if m.group(4) is not None else 1
            old_ln = a + 1 if b == 0 else a
            new_ln = c + 1 if d == 0 else c
            old_left, new_left = b, d
            cur = _Hunk(ln, m.group(5), [], [ln], False)
            hunks.append(cur)
            continue
        if cur is None:
            preamble.append(ln)
            continue
        cur.raw.append(ln)
        if old_left == 0 and new_left == 0:
            continue                       # coda dopo un hunk completo (es. riga vuota finale)
        if ln.startswith("\\"):
            cur.items.append(_Item("\\", ln, old_ln, new_ln))
            continue
        tag, text = (ln[:1] or " "), ln[1:]
        if tag == " " and old_left and new_left:
            cur.items.append(_Item(" ", text, old_ln, new_ln))
            old_ln += 1; new_ln += 1; old_left -= 1; new_left -= 1
        elif tag == "-" and old_left:
            cur.items.append(_Item("-", text, old_ln, new_ln))
            old_ln += 1; old_left -= 1
        elif tag == "+" and new_left:
            cur.items.append(_Item("+", text, old_ln, new_ln))
            new_ln += 1; new_left -= 1
        else:
            return None                    # tag sconosciuto o conteggi incoerenti
        cur.complete = old_left == 0 and new_left == 0
    return (preamble, hunks) if hunks else None


def _coordinates_match(hunks: List[_Hunk], func_lines: List[str], s: int, e: int, is_old: bool) -> bool:
    """Il diff è nelle coordinate del file e le righe del lato etichettato coincidono con `func`."""
    seen = 0
    for h in hunks:
        for it in h.items:
            if it.tag == " " or it.tag == ("-" if is_old else "+"):
                n = it.old if is_old else it.new
                if s <= n <= e:
                    if it.text.strip() != func_lines[n - s].strip():
                        return False
                    seen += 1
    return seen > 0


def _mark_out_of_span(h: _Hunk, s: int, e: int, is_old: bool) -> int:
    """Marca da eliminare i blocchi di sole righe dell'altro lato fuori da [s, e]. Ritorna il
    numero di righe marcate."""
    prim, other = ("-", "+") if is_old else ("+", "-")
    n_marked = 0
    i, items = 0, h.items
    while i < len(items):
        if items[i].tag == " ":
            i += 1
            continue
        j = i
        while j < len(items) and items[j].tag != " ":
            j += 1
        block = items[i:j]                  # run massimale di righe non di contesto
        if not any(it.tag == prim for it in block):
            gap = items[i].old if is_old else items[i].new   # posizione sul lato etichettato
            if not (s < gap <= e):
                for it in block:
                    it.drop = True
                    n_marked += it.tag in ("+", "-")
        i = j
    return n_marked


def _render(h: _Hunk) -> List[str]:
    kept = [it for it in h.items if not it.drop]
    if not any(it.tag in "+-" for it in kept):
        return []                           # nessuna modifica rimasta: l'hunk sparisce
    old_items = [it for it in kept if it.tag in " -"]
    new_items = [it for it in kept if it.tag in " +"]
    first = kept[0]
    old_start = old_items[0].old if old_items else first.old - 1
    new_start = new_items[0].new if new_items else first.new - 1
    head = f"@@ -{old_start},{len(old_items)} +{new_start},{len(new_items)} @@{h.tail}"
    return [head] + [it.text if it.tag == "\\" else it.tag + it.text for it in kept]


def trim_diff(diff: str, func: str, s: int, e: int, side: str) -> Tuple[str, List[Tuple[str, str]]]:
    """(diff pulito, righe eliminate come (tag, testo)). Invariato se nulla da fare/non sicuro."""
    parsed = _parse(diff)
    func_lines = func.split("\n")
    if func_lines and func_lines[-1] == "":
        func_lines.pop()
    if parsed is None or side not in ("old", "new") or len(func_lines) != e - s + 1:
        return diff, []
    preamble, hunks = parsed
    is_old = side == "old"
    if not _coordinates_match(hunks, func_lines, s, e, is_old):
        return diff, []

    dropped: List[Tuple[str, str]] = []
    out: List[str] = list(preamble)
    for h in hunks:
        if not h.complete:
            out.extend(h.raw)               # troncato: invariato
            continue
        if _mark_out_of_span(h, s, e, is_old):
            dropped.extend((it.tag, it.text) for it in h.items if it.drop and it.tag in "+-")
            out.extend(_render(h))
        else:
            out.extend(h.raw)
    if not dropped:
        return diff, []
    text = "\n".join(out)
    if text and not text.endswith("\n") and diff.endswith("\n"):
        text += "\n"
    return text, dropped


def trim_span_artifacts(record: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """(record, righe eliminate). Restituisce una copia superficiale solo se qualcosa cambia."""
    change = record.get("change") or {}
    fd = change.get("function_diff")
    loc = record.get("location") or {}
    func = record.get("func")
    s, e = _int(loc.get("start_line")), _int(loc.get("end_line"))
    if (not isinstance(fd, str) or "@@" not in fd or not isinstance(func, str)
            or s < 1 or e < s):
        return record, 0

    new_fd, dropped = trim_diff(fd, func, s, e, str(loc.get("side") or "").strip().lower())
    if not dropped:
        return record, 0

    ins = sum(1 for tag, _ in dropped if tag == "+")
    dele = sum(1 for tag, _ in dropped if tag == "-")
    new_change = dict(change)
    new_change["function_diff"] = new_fd
    new_change["span_trimmed"] = len(dropped)

    old_i, old_d = _int(change.get("inserted"), 0), _int(change.get("deleted"), 0)
    new_i, new_d = max(0, old_i - ins), max(0, old_d - dele)
    if isinstance(change.get("inserted"), int):
        new_change["inserted"] = new_i
    if isinstance(change.get("deleted"), int):
        new_change["deleted"] = new_d
    if isinstance(change.get("changed"), int) and change["changed"] == old_i + old_d:
        new_change["changed"] = new_i + new_d
    if not new_fd.strip():                  # tutto l'effetto era fantasma: funzione invariata
        new_change.update(changed=0, inserted=0, deleted=0)

    # `NCURSES_EXPORT(int)` sulla riga prima del nome → controparte scambiata per rinomina
    if change.get("diff_method") == "pair_renamed":
        cp_name = (change.get("counterpart") or {}).get("name")
        if cp_name and any(cp_name in text for _, text in dropped):
            new_change["diff_method"] = "pair"
            new_change.pop("counterpart", None)

    new_record = dict(record)
    new_record["change"] = new_change
    return new_record, len(dropped)