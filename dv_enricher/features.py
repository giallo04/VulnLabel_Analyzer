"""Feature deterministiche (senza LLM) usate come contesto o come segnali di validazione."""
from __future__ import annotations

import re

C_EXTS = (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".inc", ".ipp", ".cu", ".m", ".mm")
_TEST_PATH = re.compile(
    r"(^|/)(tests?|testing|testsuite|unittests?|fuzz(ers?)?|examples?|docs?|contrib|samples?)(/|$)"
    r"|(^|/)test_[^/]*$|_tests?\.(c|cc|cpp|cxx)$", re.I)
_CVE = re.compile(r"CVE-\d{4}-\d{4,7}", re.I)
_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "static", "inline", "extern", "const",
             "struct", "enum", "union", "typedef", "__attribute__", "__declspec", "defined", "else", "do"}


def is_source(path: str) -> bool:
    return path.lower().endswith(C_EXTS)


def is_test_path(path: str) -> bool:
    return bool(_TEST_PATH.search(path))


def find_cves(text: str) -> list[str]:
    return sorted({m.upper() for m in _CVE.findall(text)})


def extract_func_name(func: str) -> str | None:
    """Euristica sul nome della funzione. Gestisce i wrapper a macro (PHP_FUNCTION(x), SYSCALL_DEFINE3(x, ...))."""
    head = func.split("{", 1)[0]
    head = re.sub(r"/\*.*?\*/", "", head, flags=re.S)
    head = re.sub(r"//[^\n]*", "", head)
    head = re.sub(r"__attribute__\s*\(\(.*?\)\)", "", head, flags=re.S)
    for m in re.finditer(r"([A-Za-z_~][\w:~]*)\s*\(", head):
        name = m.group(1)
        if name in _KEYWORDS:
            continue
        if re.fullmatch(r"[A-Z][A-Z0-9_]+", name):        # macro: il nome vero è il primo argomento
            arg = head[m.end():].split(",", 1)[0].split(")", 1)[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", arg):
                return arg
        return name
    return None


def name_in_text(name: str | None, text: str) -> bool:
    if not name or not text:
        return False
    short = name.split("::")[-1]
    return bool(re.search(r"(?<![\w])" + re.escape(short) + r"(?![\w])", text))


def classify_change(deleted: list[str], inserted: list[str]) -> str:
    """'none' | 'whitespace' | 'comment_or_blank' | 'code' sulle righe modificate."""
    if not deleted and not inserted:
        return "none"

    def norm(ls):
        return sorted(re.sub(r"\s+", "", l) for l in ls if l.strip())

    if norm(deleted) == norm(inserted):
        return "whitespace"          # solo riformattazione / riordino di righe identiche

    def cosmetic(l):
        s = l.strip()
        return not s or s.startswith(("//", "/*", "*", "*/"))

    if all(cosmetic(l) for l in deleted + inserted):
        return "comment_or_blank"
    return "code"
