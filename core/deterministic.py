"""Pre-filtro deterministico — tassonomia DiverseVul.

Classifica senza LLM le funzioni con modifica puramente cosmetica o in path di test/esempi/fuzz.
Produce risultati nella stessa forma di process_record().

Mapping deterministico → categoria DiverseVul:
  file di test/esempi/fuzz CONFERMATO dal path               → WRONG_LABEL_IRRELEVANT
  modifica cosmetica VERIFICATA sul codice:
      - commento/doc cambiato E nome funzione nel messaggio  → WRONG_LABEL_CONSISTENCY (Q1:
        il commento documenta il comportamento cambiato altrove)
      - altrimenti (whitespace, commenti non legati al fix)  → WRONG_LABEL_IRRELEVANT
  Tutto il resto (incluso ciò che non è verificabile)        → None  (necessita LLM)
`predicted_target` è sempre 0: cambia solo la ripartizione per categoria.

Novità rispetto alla versione precedente
----------------------------------------
- La regola `test_path` scatta anche dal solo path (anche con file_is_test=False) e copre
  `libtests/`, `examples/`, `fuzz/`, `*-ctest.c`, `standalone_fuzz_*`, ecc.: sono wrapper/driver
  che passano argv a API controllate, non riparano la libreria.
- Un commento/doc modificato (token stream identico, ma testo diverso oltre il whitespace) con
  `name_in_message=True` va a CONSISTENCY invece che a IRRELEVANT.

Formato del record atteso (schema_version 1)
--------------------------------------------
  func                 codice della funzione, SOLO sul lato `location.side`
  location             {file, start_line, end_line, side: "old"|"new", method, n_matches, score}
                       start/end_line sono numeri di riga nel file del lato `side`
  change               {deleted, inserted, changed, frac_changed, kind, file_is_test,
                        hunks (diff unificato dell'INTERO file, può essere troncato),
                        hunks_truncated, rank_in_commit, share_of_source_churn}
  siblings, signals, commit

Come viene verificata la natura cosmetica
-----------------------------------------
`change.kind` è trattato come un'ipotesi da verificare, non come un fatto: se
l'enricher lo calcola confrontando insiemi di righe, lo spostamento di istruzioni
può risultare "none"/"whitespace".

1. VERIFICA ORDER-SENSITIVE. Nel record c'è una sola versione della funzione (`func`)
   e il diff è a livello di file. Quindi:
     a. si controlla che le righe del diff cadute nell'intervallo [start_line, end_line]
        coincidano con `func` (altrimenti la localizzazione è inaffidabile);
     b. si raggruppano le righe modificate in blocchi (righe '-'/'+' consecutive) e si
        tengono solo quelli interamente DENTRO la funzione;
     c. si ricostruisce il lato opposto sostituendo in `func` i blocchi con il loro
        contenuto sull'altro lato;
     d. le due versioni vengono tokenizzate (C/C++, commenti rimossi, stringhe
        preservate, newline significativi solo nelle direttive del preprocessore,
        `#define X(a)` distinto da `#define X (a)`) e i flussi di token confrontati
        IN ORDINE. Uno spostamento di istruzioni cambia l'ordine dei token → non è
        cosmetico → va all'LLM.
   Un blocco a cavallo del confine della funzione è accettato solo se il blocco stesso
   ha token identici prima/dopo; altrimenti non si può decidere → LLM.
   Se `hunks` è troncato e non copre tutta la funzione → LLM.
2. Se la verifica NON è possibile (nessun hunk, hunk illeggibili/incoerenti, estensione
   non C-like) il record va SEMPRE all'LLM: nessun fallback euristico sui segnali
   dell'enricher.
3. Il path di test va confermato dal path stesso (directory o nome file); il flag
   `file_is_test` da solo non basta: path come `latest.c` / `contest.c` non sono test.
4. Localizzazione non esatta (method != "exact", n_matches != 1, score < 1) → `kind`
   non è affidabile → LLM.
5. Modifiche strutturali (funzione aggiunta/rimossa/rinominata/spostata) alterano il
   token stream ricostruito, quindi non sono mai considerate cosmetiche.
"""
from __future__ import annotations

import math
import posixpath
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

CATEGORY_IRRELEVANT = "WRONG_LABEL_IRRELEVANT"
CATEGORY_CONSISTENCY = "WRONG_LABEL_CONSISTENCY"
CATEGORY = CATEGORY_IRRELEVANT  # alias retrocompatibile
COSMETIC_KINDS = frozenset({"none", "whitespace", "comment_or_blank"})

# Estensioni in cui whitespace/commenti hanno la semantica del C/C++.
_C_FAMILY_EXT = frozenset({
    ".c", ".h", ".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++",
    ".inc", ".inl", ".ipp", ".tcc", ".cu", ".cuh", ".m", ".mm",
})

# Se l'enricher dovesse esporre uno stato strutturale (non presente nello schema v1).
_STRUCTURAL_STATUS = frozenset({
    "added", "new", "deleted", "removed", "renamed", "moved",
    "signature_changed", "file_moved", "file_renamed",
})


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _norm_str(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _num(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


# --------------------------------------------------------------------------- #
# Tokenizzazione C/C++ (per confronti order-sensitive)
# --------------------------------------------------------------------------- #
_SPLICE_RE = re.compile(r"\\\r?\n")

_TOKEN_RE = re.compile(
    r"""
    (?P<lcomment>//[^\n]*)
  | (?P<bcomment>/\*.*?\*/)
  | (?P<string>(?:u8|[uUL])?"(?:\\.|[^"\\\n])*")
  | (?P<char>(?:u8|[uUL])?'(?:\\.|[^'\\\n])*')
  | (?P<newline>\n)
  | (?P<space>[ \t\r\f\v]+)
  | (?P<ident>[A-Za-z_$]\w*)
  | (?P<number>\.?\d(?:[eEpP][+-]|[\w.])*)
  | (?P<punct>>>=|<<=|\.\.\.|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|\#\#|.)
    """,
    re.VERBOSE | re.DOTALL,
)


def _tokenize(src: str) -> List[str]:
    """Token stream senza whitespace/commenti.

    - Le stringhe/char literal restano intatte (cambiarne il contenuto NON è cosmetico).
    - Il newline è un token solo dentro una direttiva del preprocessore, dove
      termina la direttiva (quindi è semanticamente rilevante).
    - Line-splice (backslash-newline) rimossi come da fase 2 del preprocessore.
    - `#define X(a)` (macro funzione) e `#define X (a)` (macro oggetto) producono
      token diversi: lo spazio dopo il nome è semanticamente rilevante.
    """
    src = _SPLICE_RE.sub("", src)
    tokens: List[str] = []
    line_start = True
    in_directive = False
    prev_ws = False  # c'era whitespace/commento subito prima del token corrente?
    for m in _TOKEN_RE.finditer(src):
        kind, text = m.lastgroup, m.group()
        if kind in ("lcomment", "bcomment", "space"):
            prev_ws = True
            continue
        if kind == "newline":
            prev_ws = True
            if in_directive:
                tokens.append("\n")
                in_directive = False
            line_start = True
            continue
        if kind == "punct" and text == "#" and line_start:
            in_directive = True
        line_start = False
        # '#define X(a)' != '#define X (a)'
        if text == "(" and not prev_ws and len(tokens) >= 3 and tokens[-3:-1] == ["#", "define"]:
            text = "(<fn-macro>"
        prev_ws = False
        tokens.append(text)
    return tokens


# --------------------------------------------------------------------------- #
# Parsing del diff unificato (a livello di file)
# --------------------------------------------------------------------------- #
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class _DiffLine:
    op: str     # ' ' contesto, '-' rimossa, '+' aggiunta
    text: str
    # Numeri di riga nel file old / new. Per ' ' e per l'op del proprio lato sono numeri
    # reali; per l'op dell'altro lato sono la POSIZIONE (prossima riga del lato) a cui
    # la riga si colloca.
    old: int
    new: int
    hunk: int


@dataclass(frozen=True)
class _ParsedDiff:
    lines: List[_DiffLine]
    # Tutte le righe del lato con numero <= known_* sono coperte dal diff (le righe fuori
    # dagli hunk sono invariate). math.inf se il diff è completo.
    known_old: float
    known_new: float


def _parse_hunks(diff: str, truncated: bool) -> Optional[_ParsedDiff]:
    """Parsa gli hunk usando i contatori dell'header (robusto a righe '---'/'+++' nel
    contenuto). Restituisce None se il diff è illeggibile/incoerente.

    Se il diff è troncato si scartano la riga finale (può essere tagliata a metà) e
    l'eventuale blocco di modifiche rimasto aperto: restano solo le righe fino
    all'ultimo contesto / fine hunk, di cui `known_*` indica la copertura.
    """
    raw = diff.split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    while raw and raw[-1].startswith("[..."):  # marker aggiunto dall'enricher
        raw.pop()
        truncated = True
    if truncated and raw:
        raw.pop()

    lines: List[_DiffLine] = []
    safe_len = 0
    known_old = known_new = 0
    old_ln = new_ln = old_left = new_left = 0
    hunk = -1

    for ln in raw:
        if ln.startswith("diff --git "):
            break  # hunk di un altro file
        if ln.startswith("@@"):
            m = _HUNK_RE.match(ln)
            if not m or (hunk >= 0 and (old_left or new_left)):
                return None
            a, c = int(m.group(1)), int(m.group(3))
            b = int(m.group(2)) if m.group(2) is not None else 1
            d = int(m.group(4)) if m.group(4) is not None else 1
            # con lunghezza 0 lo start indica la riga PRECEDENTE l'hunk
            old_ln = a + 1 if b == 0 else a
            new_ln = c + 1 if d == 0 else c
            old_left, new_left = b, d
            hunk += 1
            known_old, known_new, safe_len = old_ln - 1, new_ln - 1, len(lines)
            continue
        if hunk < 0 or ln.startswith("\\") or not (old_left or new_left):
            continue

        tag, text = (ln[:1] or " "), ln[1:]  # contesto vuoto con lo spazio rimosso
        if tag == " ":
            if not (old_left and new_left):
                return None
            lines.append(_DiffLine(" ", text, old_ln, new_ln, hunk))
            old_ln += 1
            new_ln += 1
            old_left -= 1
            new_left -= 1
            known_old, known_new, safe_len = old_ln - 1, new_ln - 1, len(lines)
        elif tag == "-":
            if not old_left:
                return None
            lines.append(_DiffLine("-", text, old_ln, new_ln, hunk))
            old_ln += 1
            old_left -= 1
        elif tag == "+":
            if not new_left:
                return None
            lines.append(_DiffLine("+", text, old_ln, new_ln, hunk))
            new_ln += 1
            new_left -= 1
        else:
            return None
        if not (old_left or new_left):
            known_old, known_new, safe_len = old_ln - 1, new_ln - 1, len(lines)

    if hunk < 0:
        return None
    complete = (not truncated) and not (old_left or new_left)
    if complete:
        return _ParsedDiff(lines, math.inf, math.inf)
    return _ParsedDiff(lines[:safe_len], known_old, known_new)


def _group_blocks(lines: List[_DiffLine]) -> List[List[_DiffLine]]:
    """Blocchi = run massimali di righe '-'/'+' separati da contesto (o da altro hunk)."""
    blocks: List[List[_DiffLine]] = []
    cur: List[_DiffLine] = []
    for ln in lines:
        if ln.op == " ":
            if cur:
                blocks.append(cur)
                cur = []
            continue
        if cur and cur[-1].hunk != ln.hunk:
            blocks.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)
    return blocks


# --------------------------------------------------------------------------- #
# Verifica della natura cosmetica
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Verification:
    verdict: Optional[bool]  # True = cosmetico confermato, False = NON cosmetico, None = non verificabile
    reason: str
    # None ma i dati c'erano (troncati / incoerenti / ambigui). Informativo: in ogni
    # caso verdict=None porta all'LLM.
    blocking: bool = False
    # True: nessuna riga del diff tocca la funzione (conferma solo kind='none').
    untouched: bool = False
    # True: token stream identico ma il testo differisce oltre il whitespace → è cambiato
    # il contenuto di un commento/doc (solo significativo con verdict=True).
    comment_edit: bool = False


def _compare_token_streams(before: List[str], after: List[str]) -> Verification:
    if before == after:
        return Verification(True, "token stream identico (ordine incluso)")
    if Counter(before) == Counter(after):
        return Verification(False, "stessi token ma ordine diverso (istruzioni spostate/riordinate)")
    return Verification(False, "token stream diverso")


def _location_problem(location: Dict[str, Any]) -> Optional[str]:
    """Localizzazione non esatta → anche `kind` (calcolato su di essa) è dubbio."""
    method = location.get("method")
    if isinstance(method, str) and method.strip().lower() != "exact":
        return f"method={method!r}"
    n_matches = _num(location, ("n_matches",))
    if n_matches is not None and n_matches != 1:
        return f"n_matches={int(n_matches)}"
    score = _num(location, ("score",))
    if score is not None and score < 1.0:
        return f"score={score}"
    return None


def _squash_ws(text: str) -> str:
    return "".join(text.split())


def _verify_from_hunks(record: Dict[str, Any], diff: str) -> Verification:
    change = record.get("change") or {}
    location = record.get("location") or {}

    s = _to_int(location.get("start_line"))
    e = _to_int(location.get("end_line"))
    side = _norm_str(location.get("side"))
    func = record.get("func")
    if s < 1 or e < s or side not in ("old", "new") or not isinstance(func, str) or not func.strip():
        return Verification(None, "hunk presenti ma func/location incompleti", blocking=True)

    func_lines = func.split("\n")
    if func_lines and func_lines[-1] == "":
        func_lines.pop()
    if len(func_lines) != e - s + 1:
        return Verification(
            None,
            f"func ({len(func_lines)} righe) incoerente con start/end_line ({e - s + 1} righe)",
            blocking=True,
        )

    parsed = _parse_hunks(diff, bool(change.get("hunks_truncated")))
    if parsed is None:
        return Verification(None, "hunk illeggibili o incoerenti", blocking=True)

    is_old = side == "old"
    prim_op = "-" if is_old else "+"
    known = parsed.known_old if is_old else parsed.known_new

    def pnum(ln: _DiffLine) -> int:
        return ln.old if is_old else ln.new

    # (a) le righe del diff nell'intervallo devono coincidere con `func`
    for ln in parsed.lines:
        if ln.op in (" ", prim_op):
            n = pnum(ln)
            if s <= n <= e and ln.text.strip() != func_lines[n - s].strip():
                return Verification(
                    None, f"riga {n} del diff diversa da func: localizzazione inaffidabile",
                    blocking=True,
                )

    # (b) blocchi dentro / a cavallo / fuori dalla funzione
    edits: List[Tuple[int, int, List[str]]] = []  # (indice inizio, indice fine escl., sostituzione)
    straddle_bad: Optional[str] = None
    touched = False
    for blk in _group_blocks(parsed.lines):
        prim = [ln for ln in blk if ln.op == prim_op]
        other = [ln for ln in blk if ln.op != prim_op]
        if prim:
            lo, hi = pnum(prim[0]), pnum(prim[-1])
            if hi < s or lo > e:
                continue
            touched = True
            if s <= lo and hi <= e:
                edits.append((lo - s, hi - s + 1, [ln.text for ln in other]))
                continue
            # a cavallo del confine: non separabile, accettato solo se il blocco è neutro
            old_txt = "\n".join(ln.text for ln in blk if ln.op == "-")
            new_txt = "\n".join(ln.text for ln in blk if ln.op == "+")
            if _tokenize(old_txt) != _tokenize(new_txt):
                straddle_bad = "blocco di modifica a cavallo del confine della funzione"
            continue
        # inserimento puro sul lato "altro": gap tra la riga gap-1 e la riga gap
        gap = pnum(other[0])
        if s < gap <= e:
            touched = True
            edits.append((gap - s, gap - s, [ln.text for ln in other]))

    # (c) ricostruzione del lato opposto (+ rilevamento di modifiche al testo dei commenti)
    comment_edit = any(
        _squash_ws("".join(func_lines[i:j])) != _squash_ws("".join(repl))
        for i, j, repl in edits
    )
    other_lines = list(func_lines)
    for i, j, repl in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        other_lines[i:j] = repl

    # (d) confronto order-sensitive
    cmp = _compare_token_streams(_tokenize("\n".join(func_lines)), _tokenize("\n".join(other_lines)))
    if cmp.verdict is False:
        return cmp
    if straddle_bad:
        return Verification(None, straddle_bad, blocking=True)
    if known < e:
        return Verification(None, "hunks troncati prima della fine della funzione", blocking=True)
    if not touched:
        return Verification(True, "nessuna riga del diff tocca la funzione", untouched=True)
    return Verification(
        True, "ogni blocco modificato nella funzione ha token stream identico (ordine incluso)",
        comment_edit=comment_edit,
    )


def verify_cosmetic(record: Dict[str, Any], path: str) -> Verification:
    """Verifica indipendente da `kind`, se il record contiene func + hunks + location."""
    if path:
        ext = posixpath.splitext(path.lower())[1]
        if ext not in _C_FAMILY_EXT:
            return Verification(None, f"estensione {ext or '(nessuna)'} non C-like")

    diff = (record.get("change") or {}).get("hunks")
    if not isinstance(diff, str) or "@@" not in diff:
        return Verification(None, "nessun hunk nel record")
    return _verify_from_hunks(record, diff)


def _structural_change(change: Dict[str, Any]) -> Optional[str]:
    for key in ("status", "func_status", "function_status", "change_type"):
        v = change.get(key)
        if isinstance(v, str) and v.strip().lower() in _STRUCTURAL_STATUS:
            return f"{key}={v!r}"
    return None


# --------------------------------------------------------------------------- #
# Regole
# --------------------------------------------------------------------------- #
# Directory che contengono codice di test, esempi, harness di fuzzing, benchmark: wrapper e
# driver che non riparano la libreria.
_TEST_DIRS = frozenset({
    # test
    "test", "tests", "testing", "unittest", "unittests", "unit_test", "unit_tests",
    "unit-test", "unit-tests", "__tests__", "testsuite", "testsuites", "testcases",
    "libtest", "libtests", "ctest", "regress", "regression",
    # fuzzing
    "fuzz", "fuzzer", "fuzzers", "fuzzing", "fuzz_targets", "fuzz-targets", "oss-fuzz", "ossfuzz",
    # esempi / demo / benchmark
    "example", "examples", "sample", "samples", "demo", "demos", "bench", "benchmark", "benchmarks",
})
_TEST_DIR_SUFFIX_RE = re.compile(r"[_\-.](?:tests?|unittests?|fuzz(?:ers?|ing)?)$", re.IGNORECASE)

_TEST_FILE_RE = re.compile(
    r"^[Tt]ests?[_\-.]"                                 # test_foo.c, test-foo.c, test.c
    r"|^Test[A-Z]"                                      # TestFoo.cc
    r"|[_\-](?:test|tests|unittest|unittests)\.\w+$"    # foo_test.c, foo-tests.cc
    r"|[a-z0-9]Tests?\.\w+$"                            # FooTest.cc  (NON 'latest.c', 'contest.c')
    r"|[_\-.]ctest\.\w+$"                               # qpdf-ctest.c  (NON 'contest.c')
    r"|^(?:standalone[_\-])?fuzz(?:er|[_\-])"           # fuzz_foo.c, fuzzer.cc, standalone_fuzz_target_runner.cc
    r"|[_\-]fuzz(?:er|_target|_harness)?\.\w+$"         # foo_fuzz.cc, foo-fuzzer.cc
)
_TEST_FUNC_RE = re.compile(r"^(?:test_|Test[A-Z_])|_test$|^TEST[_(]")


def _test_path_strength(path: str) -> str:
    if not path:
        return "unknown"
    parts = path.replace("\\", "/").strip("/").split("/")
    for p in parts[:-1]:
        if p.lower() in _TEST_DIRS or _TEST_DIR_SUFFIX_RE.search(p):
            return "strong"
    if _TEST_FILE_RE.search(parts[-1]):
        return "strong"
    return "none"


def _rule_test_path(
    record: Dict[str, Any], change: Dict[str, Any], path: str
) -> Optional[Tuple[str, str, str, str]]:
    """Test/esempi/fuzz: scatta dal path anche se l'enricher non ha messo file_is_test."""
    flag = bool(change.get("file_is_test", False))
    strength = _test_path_strength(path)
    if strength == "strong":
        how = "file_is_test=True confermato dal path" if flag else "path di test/esempio/fuzz (file_is_test=False)"
        return ("test_path", "high", f"{how} ({path})", CATEGORY_IRRELEVANT)
    if not flag:
        return None
    func_name = str(record.get("func_name") or "")
    if _TEST_FUNC_RE.search(func_name):
        return ("test_path", "medium", f"file_is_test=True e funzione dal nome di test ({func_name})",
                CATEGORY_IRRELEVANT)
    if strength == "unknown":
        return ("test_path", "medium", "file_is_test=True (path non disponibile per la conferma)",
                CATEGORY_IRRELEVANT)
    # path presente ma non riconducibile a test → il flag è sospetto
    return None


def _rule_cosmetic(
    record: Dict[str, Any], change: Dict[str, Any], kind: str, path: str
) -> Optional[Tuple[str, str, str, str]]:
    """Cosmetico solo se CONFERMATO dal codice; altrimenti None (→ LLM).

    Categoria: CONSISTENCY se è cambiato il testo di un commento/doc E il nome della funzione
    compare nel messaggio di commit (il commento documenta il comportamento cambiato dal fix,
    Q1 del prompt); IRRELEVANT altrimenti (whitespace, commenti non legati al fix).
    """
    if kind not in COSMETIC_KINDS:
        return None
    if _structural_change(change):
        return None
    if _location_problem(record.get("location") or {}):
        return None  # kind calcolato su una localizzazione dubbia → LLM

    ver = verify_cosmetic(record, path)
    if ver.verdict is True:
        if ver.untouched and kind != "none":
            # kind dice "whitespace/commenti" ma il diff non tocca la funzione: incoerenza
            return None
        detail = f"kind={kind!r}; verificato: {ver.reason}"
        name_in_msg = bool((record.get("signals") or {}).get("name_in_message"))
        if kind == "comment_or_blank" and ver.comment_edit and name_in_msg:
            return ("cosmetic_doc", "high",
                    f"{detail}; commento/doc modificato e funzione citata nel messaggio "
                    "(documenta il comportamento del fix)", CATEGORY_CONSISTENCY)
        return ("cosmetic", "high", detail, CATEGORY_IRRELEVANT)
    # verdict False (kind smentito dal codice) o None (non verificabile) → LLM
    return None


# --------------------------------------------------------------------------- #
# API pubblica
# --------------------------------------------------------------------------- #
def deterministic_classify(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Restituisce un result dict se il record è classificabile senza LLM, altrimenti None.

    Una modifica cosmetica viene accettata solo se confermata dai hunk del record;
    tutto ciò che non è verificabile viene lasciato all'LLM.
    """
    change = record.get("change") or {}
    location = record.get("location") or {}
    signals = record.get("signals") or {}
    path = location.get("file") or ""
    kind = _norm_str(change.get("kind"))

    rules = [
        r for r in (
            _rule_test_path(record, change, path),
            _rule_cosmetic(record, change, kind, path),
        ) if r
    ]
    if not rules:
        return None

    confidence = "high" if any(r[1] == "high" for r in rules) else "medium"
    reason = "; ".join(f"{name}: {detail}" for name, _, detail, _ in rules)
    # Il path di test domina: un commento in un file di test resta IRRELEVANT.
    category = CATEGORY_IRRELEVANT if any(r[0] == "test_path" for r in rules) else rules[0][3]

    predicted_target = 0
    original_target = _to_int(record.get("target"), -1)

    return {
        "hash":            record.get("hash"),
        "commit_id":       record.get("commit_id"),
        "project":         record.get("project"),
        "func_name":       record.get("func_name"),
        "file":            location.get("file"),
        "original_target": original_target,
        "predicted_target": predicted_target,
        # target ignoto (-1) → non si può parlare di noise (coerente con process_record)
        "is_noise":        original_target not in (-1, predicted_target),
        # Tassonomia DiverseVul
        "category":        category,
        # Ragionamento
        "step_1_change":   "",
        "step_2_flaw":     f"[deterministico] {reason}",
        "step_3_role":     f"[deterministico] {category} — {reason}",
        "raw_output":      None,
        "parse_error":     False,
        "parse_error_reason": "",
        "method":          "deterministic",
        # Metadati del pre-filtro
        "prefilter_rules":      [r[0] for r in rules],
        "prefilter_confidence": confidence,
        # Segnali enricher (per analisi downstream)
        "rank_in_commit":         change.get("rank_in_commit"),
        "share_of_source_churn":  change.get("share_of_source_churn"),
        "frac_changed":           change.get("frac_changed"),
        "kind":                   kind,
        "siblings_count":         len(record.get("siblings") or []),
        "name_in_message":        signals.get("name_in_message"),
        "cwe_matches_nvd":        signals.get("cwe_matches_nvd"),
        "cwe_missing":            signals.get("cwe_missing_in_dataset", False),
    }


def undeterminable_no_diff(record: Dict[str, Any]) -> Dict[str, Any]:
    """Risultato per un record il cui diff non è attribuibile con sicurezza alla funzione.

    Nessuna chiamata API: il modello non avrebbe evidenza e produrrebbe una congettura.
    `predicted_target=-1` (escluso dal denominatore), `parse_error=False`.
    """
    change = record.get("change") or {}
    location = record.get("location") or {}
    signals = record.get("signals") or {}
    reason = (f"diff non attribuibile con sicurezza alla funzione "
              f"(diff_method={change.get('diff_method')!r}, "
              f"diff_reliable={change.get('diff_reliable')!r})")
    return {
        "hash":            record.get("hash"),
        "commit_id":       record.get("commit_id"),
        "project":         record.get("project"),
        "func_name":       record.get("func_name"),
        "file":            location.get("file"),
        "original_target": _to_int(record.get("target"), -1),
        "predicted_target": -1,
        "is_noise":        False,
        "category":        "UNDETERMINABLE",
        "step_1_change":   "",
        "step_2_flaw":     f"[deterministico] {reason}",
        "step_3_role":     f"[deterministico] UNDETERMINABLE — {reason}",
        "raw_output":      None,
        "parse_error":     False,
        "parse_error_reason": "",
        "method":          "deterministic",
        "prefilter_rules":      ["diff_unrecoverable"],
        "prefilter_confidence": "high",
        "diff_method":     change.get("diff_method"),
        "rank_in_commit":         change.get("rank_in_commit"),
        "share_of_source_churn":  change.get("share_of_source_churn"),
        "frac_changed":           change.get("frac_changed"),
        "kind":                   _norm_str(change.get("kind")),
        "siblings_count":         len(record.get("siblings") or []),
        "name_in_message":        signals.get("name_in_message"),
        "cwe_matches_nvd":        signals.get("cwe_matches_nvd"),
        "cwe_missing":            signals.get("cwe_missing_in_dataset", False),
    }
