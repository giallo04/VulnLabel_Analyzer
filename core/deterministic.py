"""Pre-filtro deterministico — tassonomia DiverseVul.

Classifica senza LLM le funzioni (a) in path di test/esempi/fuzz/documentazione, (b) in file
che non sono codice C/C++, (c) con modifica puramente cosmetica VERIFICATA sul codice e
(d) con modifica limitata ai soli specificatori di linkage/inline dell'intestazione.
Tutto ciò che non è verificabile → None (necessita LLM).

Mapping → categoria DiverseVul (`predicted_target` segue CATEGORY_TO_TARGET: 0, o -1 per
undeterminable_no_diff); tra parentesi la Rule del prompt LLM:
  path di test/esempi/fuzz/doc CONFERMATO dal path               → WRONG_LABEL_IRRELEVANT  (Rule 6)
  file non C/C++ (script, doc, build, dati)                      → WRONG_LABEL_IRRELEVANT  (Rule 6)
  cosmetico verificato, commento/doc modificato E funzione
    citata nel messaggio di commit (documenta il fix)            → WRONG_LABEL_CONSISTENCY (Rule 5)
  cosmetico verificato, altrimenti (whitespace, commenti non
    legati al fix, nessuna riga toccata)                         → WRONG_LABEL_IRRELEVANT  (Rule 1/6)
  solo static/inline/extern/attributi di sezione nell'intestazione,
    corpo identico token per token                               → WRONG_LABEL_CONSISTENCY (Rule 5)
La politica cosmetica è unica e vive in prompting.cosmetic_verdict() (condivisa con
prompting.precheck e con le Rule 1/5/6 del prompt LLM).

Cosa NON è automatizzato (resta all'LLM) e perché
-------------------------------------------------
Rinomine di identificatori, cambi di tipo/costante e cambi di callee hanno lo stesso aspetto
sintattico di molte riparazioni (`u8`→`u32`, `kmalloc`→`kzalloc`, `memcpy`→`memmove`,
`MAX`→`MAX_SAFE`): sono Rule 2 o Rule 5 a seconda della semantica. Anche le modifiche a
istruzioni di log possono essere riparazioni (`printk(buf)`→`printk("%s", buf)`).

Interruttori
------------
`DeterministicConfig` abilita/disabilita ogni regola (ablation / audit di precisione). Il
risultato riporta `prefilter_rules` e `prefilter_confidence`.

Record atteso
-------------
  func       codice della funzione, SOLO sul lato `location.side`
  location   {file, start_line, end_line, side: "old"|"new", method, n_matches, score}
             (start/end_line: righe nel file del lato `side`)
  change     {kind, file_is_test, hunks (diff unificato dell'INTERO file, può essere troncato),
              hunks_truncated, function_diff, function_diff_truncated, diff_method,
              diff_reliable, rank_in_commit, share_of_source_churn, frac_changed, ...}
  signals, siblings, commit

Verifica della natura cosmetica
-------------------------------
`change.kind` è un'ipotesi da verificare, non un fatto (se l'enricher confronta insiemi di
righe, lo spostamento di istruzioni può risultare "none"/"whitespace"; viceversa può dichiarare
"code" per un cambiamento che a livello di token non esiste). Il verdetto cosmetico dipende
quindi SOLO dalla verifica sul codice, non da `kind`.
1. VERIFICA ORDER-SENSITIVE sul diff (`hunks`, poi `function_diff` se i primi non bastano):
   a. le righe del diff nell'intervallo [start_line, end_line] devono coincidere con `func`
      (altrimenti la localizzazione è inaffidabile);
   b. le righe '-'/'+' consecutive formano blocchi; si tengono quelli interamente DENTRO la
      funzione;
   c. si ricostruisce il lato opposto sostituendo in `func` i blocchi con il loro contenuto
      sull'altro lato;
   d. le due versioni sono tokenizzate (C/C++, commenti rimossi, stringhe preservate, newline
      significativi solo nelle direttive, `#define X(a)` ≠ `#define X (a)`) e i token confrontati
      IN ORDINE: uno spostamento di istruzioni cambia l'ordine → non è cosmetico → LLM.
   Un blocco a cavallo del confine è accettato solo se ha token identici prima/dopo. Se il diff
   è troncato e non copre tutta la funzione → LLM.
2. Verifica impossibile (nessun hunk, hunk illeggibili/incoerenti, estensione non C-like) →
   SEMPRE LLM: nessun fallback euristico sui segnali dell'enricher.
3. Il path di test va confermato dal path stesso (directory o nome file): `file_is_test` da solo
   non basta (`latest.c`, `contest.c` non sono test).
4. Localizzazione non esatta (method != "exact", n_matches != 1, score < 1): blocca solo la
   conclusione "nessuna riga tocca la funzione" (che dipende interamente dalla localizzazione).
   Le modifiche verificate riga per riga (punto 1a) non dipendono da essa.
5. Modifiche strutturali (funzione aggiunta/rimossa/rinominata/spostata) alterano il token stream
   ricostruito: mai cosmetiche.
"""
from __future__ import annotations

import math
import posixpath
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from .prompting import (CATEGORY_CONSISTENCY, CATEGORY_IRRELEVANT, CATEGORY_TO_TARGET,
                        PROMPT_VERSION, cosmetic_verdict)

CATEGORY = CATEGORY_IRRELEVANT  # alias retrocompatibile
COSMETIC_KINDS = frozenset({"none", "whitespace", "comment_or_blank"})  # solo informativo


@dataclass(frozen=True)
class DeterministicConfig:
    test_paths: bool = True      # test/esempi/fuzz/doc dal path
    non_c_files: bool = True     # script, documentazione, build, dati
    cosmetic: bool = True        # modifica cosmetica verificata sui token
    linkage_only: bool = True    # solo static/inline/... nell'intestazione


DEFAULT_DETERMINISTIC_CONFIG = DeterministicConfig()

# Estensioni in cui whitespace/commenti hanno la semantica del C/C++.
_C_FAMILY_EXT = frozenset({
    ".c", ".h", ".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++",
    ".inc", ".inl", ".ipp", ".tcc", ".cu", ".cuh", ".m", ".mm",
})

# File che non sono codice C/C++ (script, documentazione, build, dati): nessuna di queste
# può contenere il comportamento vulnerabile di una libreria C/C++. Linguaggi con memoria
# gestita ma propri difetti (js, php, java, go, rs) sono volutamente esclusi.
_NON_C_EXT = frozenset({
    ".py", ".pyi", ".pl", ".pm", ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1",
    ".md", ".rst", ".txt", ".html", ".htm", ".css", ".xml", ".json", ".yml", ".yaml", ".toml",
    ".cmake", ".am", ".mk", ".m4", ".po",
})

# Stati strutturali, se l'enricher li espone.
_STRUCTURAL_STATUS = frozenset({
    "added", "new", "deleted", "removed", "renamed", "moved",
    "signature_changed", "file_moved", "file_renamed",
})

# Token che, se sono l'UNICA differenza nell'intestazione, rendono la modifica puramente
# meccanica (Rule 5: "visibility change (static)"). Niente `const`/`volatile`/`unsigned`:
# cambiano il tipo e possono essere riparazioni.
_LINKAGE_TOKENS = frozenset({
    "static", "inline", "__inline", "__inline__", "extern", "__always_inline", "noinline",
    "__noinline", "__maybe_unused", "__unused", "__init", "__exit", "__cold",
})


class _Rule(NamedTuple):
    name: str
    confidence: str
    detail: str
    category: str
    rule_no: int


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

    - Stringhe/char literal intatte (cambiarne il contenuto NON è cosmetico).
    - Il newline è un token solo dentro una direttiva del preprocessore (ne è il terminatore).
    - Line-splice (backslash-newline) rimossi (fase 2 del preprocessore).
    - `#define X(a)` (macro funzione) e `#define X (a)` (macro oggetto) danno token diversi.
    """
    src = _SPLICE_RE.sub("", src)
    tokens: List[str] = []
    line_start = True
    in_directive = False
    prev_ws = False  # whitespace/commento subito prima del token corrente?
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
    contenuto). None se il diff è illeggibile/incoerente.

    Se il diff è troncato si scartano la riga finale (può essere tagliata a metà) e l'eventuale
    blocco di modifiche rimasto aperto: restano le righe fino all'ultimo contesto / fine hunk,
    di cui `known_*` indica la copertura.
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
    if not truncated and not (old_left or new_left):
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
    # True: nessuna riga del diff tocca la funzione (conferma solo kind='none').
    untouched: bool = False
    # True: token stream identico ma il testo differisce oltre il whitespace → è cambiato il
    # contenuto di un commento/doc (significativo solo con verdict=True).
    comment_edit: bool = False
    # Con verdict=False: i due token stream e se la ricostruzione è COMPLETA (diff non troncato
    # prima della fine della funzione, nessun blocco non neutro a cavallo del confine). Solo
    # con `complete` i flussi descrivono TUTTE le differenze della funzione.
    before: Optional[Tuple[str, ...]] = None
    after: Optional[Tuple[str, ...]] = None
    complete: bool = False


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


def _verify_from_hunks(record: Dict[str, Any], diff: str, truncated: bool) -> Verification:
    location = record.get("location") or {}
    s = _to_int(location.get("start_line"))
    e = _to_int(location.get("end_line"))
    side = _norm_str(location.get("side"))
    func = record.get("func")
    if s < 1 or e < s or side not in ("old", "new") or not isinstance(func, str) or not func.strip():
        return Verification(None, "hunk presenti ma func/location incompleti")

    func_lines = func.split("\n")
    if func_lines and func_lines[-1] == "":
        func_lines.pop()
    if len(func_lines) != e - s + 1:
        return Verification(
            None, f"func ({len(func_lines)} righe) incoerente con start/end_line ({e - s + 1} righe)")

    parsed = _parse_hunks(diff, truncated)
    if parsed is None:
        return Verification(None, "hunk illeggibili o incoerenti")

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
                return Verification(None, f"riga {n} del diff diversa da func: localizzazione inaffidabile")

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
    comment_edit = any(_squash_ws("".join(func_lines[i:j])) != _squash_ws("".join(repl))
                       for i, j, repl in edits)
    other_lines = list(func_lines)
    for i, j, repl in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        other_lines[i:j] = repl

    # (d) confronto order-sensitive
    tok_a, tok_b = _tokenize("\n".join(func_lines)), _tokenize("\n".join(other_lines))
    cmp = _compare_token_streams(tok_a, tok_b)
    if cmp.verdict is False:
        return Verification(False, cmp.reason, before=tuple(tok_a), after=tuple(tok_b),
                            complete=known >= e and not straddle_bad)
    if straddle_bad:
        return Verification(None, straddle_bad)
    if known < e:
        return Verification(None, "hunks troncati prima della fine della funzione")
    if not touched:
        return Verification(True, "nessuna riga del diff tocca la funzione", untouched=True)
    return Verification(
        True, "ogni blocco modificato nella funzione ha token stream identico (ordine incluso)",
        comment_edit=comment_edit)


def _diff_candidates(change: Dict[str, Any]) -> List[Tuple[str, bool]]:
    """Diff utilizzabili in ordine di preferenza: `hunks` (file-level), poi `function_diff`,
    ciascuno con il proprio flag di troncamento. Il secondo serve quando il primo è troncato o
    illeggibile; la verifica (a) scarta un diff le cui righe non coincidono con `func`."""
    out: List[Tuple[str, bool]] = []
    for key, flag in (("hunks", "hunks_truncated"), ("function_diff", "function_diff_truncated")):
        diff = change.get(key)
        if isinstance(diff, str) and "@@" in diff:
            out.append((diff, bool(change.get(flag))))
    return out


def verify_cosmetic(record: Dict[str, Any], path: str) -> Verification:
    """Verifica indipendente da `kind`, se il record contiene func + diff + location."""
    if path:
        ext = posixpath.splitext(path.lower())[1]
        if ext not in _C_FAMILY_EXT:
            return Verification(None, f"estensione {ext or '(nessuna)'} non C-like")
    candidates = _diff_candidates(record.get("change") or {})
    if not candidates:
        return Verification(None, "nessun hunk nel record")
    first: Optional[Verification] = None
    for diff, truncated in candidates:
        ver = _verify_from_hunks(record, diff, truncated)
        if ver.verdict is not None:
            return ver
        first = first or ver
    return first  # type: ignore[return-value]


def _structural_change(change: Dict[str, Any]) -> Optional[str]:
    for key in ("status", "func_status", "function_status", "change_type"):
        v = change.get(key)
        if isinstance(v, str) and v.strip().lower() in _STRUCTURAL_STATUS:
            return f"{key}={v!r}"
    return None


# --------------------------------------------------------------------------- #
# Regole
# --------------------------------------------------------------------------- #
# Directory con codice di test, esempi, harness di fuzzing, benchmark, documentazione: wrapper
# e driver che non riparano la libreria.
_TEST_DIRS = frozenset({
    # test
    "test", "tests", "testing", "unittest", "unittests", "unit_test", "unit_tests",
    "unit-test", "unit-tests", "__tests__", "testsuite", "testsuites", "testcases",
    "libtest", "libtests", "ctest", "regress", "regression", "selftests", "kselftest", "kunit",
    "testdata", "test_data", "test-data", "testfiles", "testutil", "testutils", "test_utils",
    "testlib", "fixture", "fixtures", "corpus", "mock", "mocks",
    "googletest", "gtest", "gmock",
    # fuzzing
    "fuzz", "fuzzer", "fuzzers", "fuzzing", "fuzz_targets", "fuzz-targets", "oss-fuzz", "ossfuzz",
    # esempi / demo / benchmark / documentazione
    "example", "examples", "sample", "samples", "demo", "demos", "bench", "benchmark", "benchmarks",
    "tutorial", "tutorials", "playground", "doc", "docs", "documentation", "manual",
})
_TEST_DIR_SUFFIX_RE = re.compile(r"[_\-.](?:tests?|unittests?|fuzz(?:ers?|ing)?)$", re.IGNORECASE)

_TEST_FILE_RE = re.compile(
    r"^[Tt]ests?[_\-.]"                                 # test_foo.c, test-foo.c, test.c
    r"|^Test[A-Z]"                                      # TestFoo.cc
    r"|^test(?:util|utils|lib|harness|helper|helpers|support|data)"  # testutil.c, testlib.c
    r"|[_\-](?:test|tests|unittest|unittests)\.\w+$"    # foo_test.c, foo-tests.cc
    r"|[_\-](?:test|tests|mock|mocks)[_\-]"             # foo_test_util.c, foo_mock_bar.c
    r"|[a-z0-9]Tests?\.\w+$"                            # FooTest.cc  (NON 'latest.c', 'contest.c')
    r"|[_\-.]ctest\.\w+$"                               # qpdf-ctest.c  (NON 'contest.c')
    r"|^(?:standalone[_\-])?fuzz(?:er|[_\-])"           # fuzz_foo.c, fuzzer.cc, standalone_fuzz_target_runner.cc
    r"|[_\-]fuzz(?:er|_target|_harness)?\.\w+$"         # foo_fuzz.cc, foo-fuzzer.cc
)
_TEST_FUNC_RE = re.compile(r"^(?:test_|Test[A-Z_])|_test$|^TEST[_(]")


def _test_path_strength(path: str) -> str:
    """"strong" (path di test), "none" (path presente, non di test) o "unknown" (nessun path)."""
    if not path:
        return "unknown"
    parts = path.replace("\\", "/").strip("/").split("/")
    if any(p.lower() in _TEST_DIRS or _TEST_DIR_SUFFIX_RE.search(p) for p in parts[:-1]):
        return "strong"
    return "strong" if _TEST_FILE_RE.search(parts[-1]) else "none"


GetVerification = Callable[[], Verification]


def _rule_test_path(record: Dict[str, Any], change: Dict[str, Any], path: str,
                    get_ver: GetVerification) -> Optional[_Rule]:
    """Test/esempi/fuzz/doc: scatta dal path anche se l'enricher non ha messo file_is_test."""
    flag = bool(change.get("file_is_test", False))
    strength = _test_path_strength(path)
    if strength == "strong":
        how = ("file_is_test=True confermato dal path" if flag
               else "path di test/esempio/fuzz/doc (file_is_test=False)")
        return _Rule("test_path", "high", f"{how} ({path})", CATEGORY_IRRELEVANT, 6)
    if not flag:
        return None
    func_name = str(record.get("func_name") or "")
    if _TEST_FUNC_RE.search(func_name):
        return _Rule("test_path", "medium",
                     f"file_is_test=True e funzione dal nome di test ({func_name})",
                     CATEGORY_IRRELEVANT, 6)
    # path presente ma non di test → il flag è sospetto; path assente → serve la conferma dal
    # codice: se risulta modificato (non cosmetico) non si può escludere che sia da analizzare
    if strength == "unknown" and get_ver().verdict is not False:
        return _Rule("test_path", "medium",
                     "file_is_test=True (path non disponibile per la conferma)",
                     CATEGORY_IRRELEVANT, 6)
    return None


def _rule_non_c(path: str) -> Optional[_Rule]:
    """Script, documentazione, build e dati: non possono ospitare un difetto C/C++."""
    if not path:
        return None
    ext = posixpath.splitext(path.lower())[1]
    if ext in _NON_C_EXT:
        return _Rule("non_c_file", "medium", f"estensione {ext} non è codice C/C++ ({path})",
                     CATEGORY_IRRELEVANT, 6)
    return None


def _rule_cosmetic(record: Dict[str, Any], change: Dict[str, Any], kind: str,
                   get_ver: GetVerification) -> Optional[_Rule]:
    """Cosmetico solo se CONFERMATO dal codice; altrimenti None (→ LLM). `kind` non è
    determinante: se la verifica sui token lo smentisce (o lo conferma) vince la verifica.
    La categoria viene da prompting.cosmetic_verdict()."""
    if _structural_change(change):
        return None  # modifica strutturale → LLM
    ver = get_ver()
    if ver.verdict is not True:
        return None  # non verificabile o non cosmetico → LLM
    if ver.untouched and (kind != "none" or _location_problem(record.get("location") or {})):
        return None  # "nessuna riga tocca la funzione" regge solo con localizzazione esatta

    detail = f"kind={kind!r}; verificato: {ver.reason}"
    if not ver.untouched and kind not in COSMETIC_KINDS:
        detail += f" (kind={kind!r} non cosmetico, smentito dai token)"
    category, rule_no = cosmetic_verdict(
        comment_edit=ver.comment_edit,
        name_in_message=bool((record.get("signals") or {}).get("name_in_message")))
    if category == CATEGORY_CONSISTENCY:
        return _Rule("cosmetic_doc", "high",
                     f"{detail}; commento/doc modificato e funzione citata nel messaggio "
                     "(documenta il comportamento del fix)", category, rule_no)
    return _Rule("cosmetic", "high", detail, category, rule_no)


def _rule_linkage(change: Dict[str, Any], get_ver: GetVerification) -> Optional[_Rule]:
    """Unica differenza = specificatori di linkage/inline nell'intestazione; il corpo (dalla
    prima '{') è identico token per token. Es. `static int f()` → `int f()` (Rule 5)."""
    if _structural_change(change):
        return None
    ver = get_ver()
    if ver.verdict is not False or not ver.complete or ver.before is None or ver.after is None:
        return None
    b, a = ver.before, ver.after
    if "{" not in b or "{" not in a:
        return None
    ib, ia = b.index("{"), a.index("{")
    if b[ib:] != a[ia:]:
        return None  # il corpo cambia
    head_b, head_a = b[:ib], a[:ia]
    if head_b == head_a:
        return None
    if ([t for t in head_b if t not in _LINKAGE_TOKENS]
            != [t for t in head_a if t not in _LINKAGE_TOKENS]):
        return None  # cambia altro oltre ai specificatori (tipo, parametri, nome, ...)
    removed = sorted((Counter(head_b) - Counter(head_a)).elements())
    added = sorted((Counter(head_a) - Counter(head_b)).elements())
    return _Rule("linkage_only", "high",
                 f"solo specificatori nell'intestazione (lato funzione: -{removed} +{added}); "
                 "corpo identico token per token", CATEGORY_CONSISTENCY, 5)


# --------------------------------------------------------------------------- #
# Decisione
# --------------------------------------------------------------------------- #
def _decide(record: Dict[str, Any],
            config: DeterministicConfig) -> Optional[Tuple[str, int, str, List[str], str]]:
    """(categoria, numero Rule, motivo, nomi regole, confidenza) oppure None.
    Priorità: path di test > file non C > cosmetico > linkage."""
    change = record.get("change") or {}
    path = (record.get("location") or {}).get("file") or ""
    kind = _norm_str(change.get("kind"))

    memo: List[Verification] = []

    def get_ver() -> Verification:
        if not memo:
            memo.append(verify_cosmetic(record, path))
        return memo[0]

    candidates = (
        (config.test_paths, lambda: _rule_test_path(record, change, path, get_ver)),
        (config.non_c_files, lambda: _rule_non_c(path)),
        (config.cosmetic, lambda: _rule_cosmetic(record, change, kind, get_ver)),
        (config.linkage_only, lambda: _rule_linkage(change, get_ver)),
    )
    for enabled, fire in candidates:
        if not enabled:
            continue
        rule = fire()
        if rule is not None:
            return rule.category, rule.rule_no, f"{rule.name}: {rule.detail}", [rule.name], rule.confidence
    return None


# --------------------------------------------------------------------------- #
# Risultati
# --------------------------------------------------------------------------- #
def deterministic_verdict(record: Dict[str, Any],
                          config: DeterministicConfig = DEFAULT_DETERMINISTIC_CONFIG
                          ) -> Optional[Dict[str, Any]]:
    """Verdetto nella forma di `prompting.precheck` / `parse_llm_output` (più `prefilter_*`),
    oppure None se il record va all'LLM."""
    decided = _decide(record, config)
    if decided is None:
        return None
    category, rule_no, reason, rules, confidence = decided
    data = {
        "step_1_change": reason,
        "step_2_flaw": f"Flaw visible: no. [deterministico] {reason}",
        "step_3_role": f"Rule {rule_no}: [deterministico] {reason}",
        "category": category,
    }
    return {
        "data": data, "category": category, "predicted_target": CATEGORY_TO_TARGET[category],
        "parse_error": False, "parse_error_reason": "", "parse_method": "rule",
        "needs_retry": False, "source": "rule", "rule": rule_no, "flaw_visible": "no",
        "flags": [], "inconsistent": False,
        "prefilter_rules": rules, "prefilter_confidence": confidence,
    }


def _result(record: Dict[str, Any], category: str, reason: str, rules: List[str],
            confidence: str, rule_no: Optional[int]) -> Dict[str, Any]:
    change = record.get("change") or {}
    signals = record.get("signals") or {}
    predicted = CATEGORY_TO_TARGET[category]
    original = _to_int(record.get("target"), -1)
    return {
        "hash":            record.get("hash"),
        "commit_id":       record.get("commit_id"),
        "project":         record.get("project"),
        "func_name":       record.get("func_name"),
        "file":            (record.get("location") or {}).get("file"),
        "original_target": original,
        "predicted_target": predicted,
        # target ignoto (-1) o verdetto non binario (-1) → non si può parlare di noise
        "is_noise":        predicted != -1 and original not in (-1, predicted),
        "category":        category,
        "step_1_change":   "",
        "step_2_flaw":     f"[deterministico] {reason}",
        "step_3_role":     f"[deterministico] {category} — {reason}",
        "raw_output":      None,
        "parse_error":     False,
        "parse_error_reason": "",
        "method":          "deterministic",
        "rule":            rule_no,
        "flaw_visible":    "no",
        "prompt_version":  PROMPT_VERSION,   # stessi campi dei risultati LLM
        # Metadati del pre-filtro
        "prefilter_rules":      rules,
        "prefilter_confidence": confidence,
        "diff_method":     change.get("diff_method"),
        # Segnali enricher (per analisi downstream)
        "rank_in_commit":         change.get("rank_in_commit"),
        "share_of_source_churn":  change.get("share_of_source_churn"),
        "frac_changed":           change.get("frac_changed"),
        "kind":                   _norm_str(change.get("kind")),
        "siblings_count":         len(record.get("siblings") or []),
        "name_in_message":        signals.get("name_in_message"),
        "cwe_matches_nvd":        signals.get("cwe_matches_nvd"),
        "cwe_missing":            signals.get("cwe_missing_in_dataset", False),
    }


# --------------------------------------------------------------------------- #
# API pubblica (forma "result dict", retrocompatibile)
# --------------------------------------------------------------------------- #
def deterministic_classify(record: Dict[str, Any],
                           config: DeterministicConfig = DEFAULT_DETERMINISTIC_CONFIG
                           ) -> Optional[Dict[str, Any]]:
    """Result dict se il record è classificabile senza LLM, altrimenti None.

    Una modifica cosmetica è accettata solo se confermata dai hunk del record; tutto ciò che non
    è verificabile resta all'LLM.
    """
    decided = _decide(record, config)
    if decided is None:
        return None
    category, rule_no, reason, rules, confidence = decided
    return _result(record, category, reason, rules, confidence, rule_no)


def undeterminable_no_diff(record: Dict[str, Any]) -> Dict[str, Any]:
    """Risultato per un record il cui diff non è attribuibile con sicurezza alla funzione.

    Nessuna chiamata API: il modello non avrebbe evidenza e produrrebbe una congettura.
    `predicted_target=-1` (escluso dal denominatore), `parse_error=False`.
    """
    change = record.get("change") or {}
    reason = (f"diff non attribuibile con sicurezza alla funzione "
              f"(diff_method={change.get('diff_method')!r}, "
              f"diff_reliable={change.get('diff_reliable')!r})")
    return _result(record, "UNDETERMINABLE", reason, ["diff_unrecoverable"], "high", 0)