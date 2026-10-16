"""Prompt builder - DiverseVul label audit (zero-shot, role prompting + chain-of-thought).

Tuned for a local ~30B open-weights model; the prompts are model-agnostic.
One category per function:
  CORRECT_LABEL (1) | WRONG_LABEL_SPREAD / _CONSISTENCY / _FEATURE / _IRRELEVANT (0) | UNDETERMINABLE (-1)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from dv_enricher.diffparse import (anchored_is_plausible, covered_until, parse_unified_diff,
                                   render_function_diff, shrink_lines)

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover
    repair_json = None

CATEGORY_CONSISTENCY = "WRONG_LABEL_CONSISTENCY"
CATEGORY_IRRELEVANT = "WRONG_LABEL_IRRELEVANT"
CATEGORY_TO_TARGET: Dict[str, int] = {
    "CORRECT_LABEL": 1,
    "WRONG_LABEL_SPREAD": 0,
    CATEGORY_CONSISTENCY: 0,
    "WRONG_LABEL_FEATURE": 0,
    CATEGORY_IRRELEVANT: 0,
    "UNDETERMINABLE": -1,
}
VALID_CATEGORIES = set(CATEGORY_TO_TARGET)
_CATEGORIES_LONGEST_FIRST = sorted(VALID_CATEGORIES, key=len, reverse=True)
_CATEGORY_ALIASES = {
    "CORRECT": "CORRECT_LABEL",
    "SPREAD": "WRONG_LABEL_SPREAD",
    "CONSISTENCY": CATEGORY_CONSISTENCY,
    "FEATURE": "WRONG_LABEL_FEATURE",
    "IRRELEVANT": CATEGORY_IRRELEVANT,
}

PROMPT_VERSION = "2026-10-zero-shot-cot-v12-minimal"
MAX_DIFF_CHARS = 12000
MAX_FUNC_CHARS = 6000
MAX_MSG_CHARS = 3000
MAX_SIBLINGS_LISTED = 12


def prompt_version() -> str:
    return PROMPT_VERSION


def cosmetic_verdict(*, comment_edit: bool, name_in_message: bool) -> Tuple[str, int]:
    if comment_edit and name_in_message:
        return CATEGORY_CONSISTENCY, 5
    return CATEGORY_IRRELEVANT, (6 if comment_edit else 1)


OLLAMA_MODEL = "gemma4:32b"
OLLAMA_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0, "seed": 0, "num_ctx": 16384, "num_predict": 900,
}
OLLAMA_OPTIONS_SAMPLING: Dict[str, Any] = {
    "temperature": 0.6, "top_p": 0.95, "top_k": 64, "num_ctx": 16384, "num_predict": 900,
}

_SYSTEM_PROMPT = """\
You are a C/C++ security researcher classifying ONE function's role in a CVE fix commit. Analyze the provided code diff. Apply these rules in strict order:

Rule 0: Unusable, missing, or improperly truncated diff -> UNDETERMINABLE.
Rule 1: No changes or whitespace/formatting only -> WRONG_LABEL_IRRELEVANT.
Rule 2: Pre-fix code performs an unsafe operation (buffer, free, bounds, lock) and the changed line repairs it -> CORRECT_LABEL.
Rule 3: Flaw lies elsewhere, but this change adds security logic (limit, error path, check) -> WRONG_LABEL_SPREAD.
Rule 4: Adds new capability/feature -> WRONG_LABEL_FEATURE.
Rule 5: Mechanical adaptation (signature, type cast, rename) -> WRONG_LABEL_CONSISTENCY.
Rule 6: Unrelated (tests, refactor, logging) -> WRONG_LABEL_IRRELEVANT.

Reply with ONE valid JSON object, keep strings on a single line, and use backticks for code:
{
  "step_1_change": "<what changes in this function>",
  "step_2_flaw": "<Flaw visible: yes|no|unclear. explanation>",
  "step_3_role": "<Rule N. explanation>",
  "category": "<CORRECT_LABEL | WRONG_LABEL_SPREAD | WRONG_LABEL_CONSISTENCY | WRONG_LABEL_FEATURE | WRONG_LABEL_IRRELEVANT | UNDETERMINABLE>"
}"""


def _schema(fields: Dict[str, Optional[List[str]]], *, gemini: bool = False) -> Dict[str, Any]:
    def prop(enum: Optional[List[str]]) -> Dict[str, Any]:
        if gemini:
            return {"type": "STRING", **({"format": "enum", "enum": enum} if enum else {})}
        return {"type": "string", **({"enum": enum} if enum else {})}

    names = list(fields)
    schema: Dict[str, Any] = {"type": "OBJECT" if gemini else "object",
                              "properties": {k: prop(v) for k, v in fields.items()},
                              "required": names}
    schema.update({"propertyOrdering": names} if gemini else {"additionalProperties": False})
    return schema


REASONING_FIELDS = ("step_1_change", "step_2_flaw", "step_3_role")
_REPLY_FIELDS: Dict[str, Optional[List[str]]] = {
    **dict.fromkeys(REASONING_FIELDS), "category": list(CATEGORY_TO_TARGET)}

RESPONSE_JSON_SCHEMA = _schema(_REPLY_FIELDS)
GEMINI_RESPONSE_SCHEMA = _schema(_REPLY_FIELDS, gemini=True)

_SIDE_LABEL = {"old": "old (pre-fix)", "new": "new (post-fix)"}
_NO_DIFF = "(no changed lines inside this function)"
_WS_ONLY = "(only whitespace or formatting differs inside this function; the code is unchanged)"
_NO_DIFF_LOST = "(NOT AVAILABLE: diff truncated before reaching it)"
_NO_DIFF_UNATTRIBUTABLE = "(NOT AVAILABLE: changed lines could not be attributed to this function)"
_STATUS_APPROX = "derived from file-level hunks (approximate)"


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def estimate_tokens(text: str) -> int:
    return len(text) // 3 + 1


def _head_tail_lines(text: str, max_chars: int, head_frac: float = 0.6) -> Tuple[str, int]:
    if len(text) <= max_chars:
        return text, 0
    lines = text.split("\n")
    head_budget = int(max_chars * head_frac)
    tail_budget = max_chars - head_budget
    head: List[str] = []
    used = 0
    for ln in lines:
        if used + len(ln) + 1 > head_budget:
            break
        head.append(ln)
        used += len(ln) + 1
    tail: List[str] = []
    used = 0
    for ln in reversed(lines[len(head):]):
        if used + len(ln) + 1 > tail_budget:
            break
        tail.append(ln)
        used += len(ln) + 1
    tail.reverse()
    omitted = len(lines) - len(head) - len(tail)
    if omitted <= 0:
        return text[:max_chars], max(1, text[max_chars:].count("\n"))
    return "\n".join(head + [f"[... {omitted} lines omitted ...]"] + tail), omitted


def _clip_message(msg: Any, limit: int) -> str:
    msg = str(msg or "(no message)").strip() or "(no message)"
    if len(msg) <= limit:
        return msg
    cut = msg[:limit]
    nl = cut.rfind("\n")
    if nl > limit // 2:
        cut = cut[:nl]
    return cut + "\n[... commit message truncated ...]"


_OWN_TAGS = ("commit_context", "function", "body", "diff", "commit_siblings", "commit_callgraph")
_TAG_RE = re.compile(r"</?\s*(?:%s)\s*>" % "|".join(_OWN_TAGS), re.IGNORECASE)


def _defang(text: str) -> str:
    return _TAG_RE.sub(lambda m: m.group(0).replace("<", "[").replace(">", "]"), text)


def _message(commit: Dict[str, Any], limit: int) -> str:
    return _defang(_clip_message(commit.get("message"), limit))


_NOT_A_FUNC_NAME = frozenset({
    "if", "while", "for", "switch", "return", "sizeof", "typeof", "alignof", "__attribute__",
    "void", "int", "char", "short", "long", "float", "double", "unsigned", "signed", "const",
    "static", "struct", "enum", "union", "volatile", "bool", "size_t", "inline", "extern"
})
_ANNOTATION_RE = re.compile(r"^__(?:acquire|release|must_hold|attribute|printf|init|exit|weak)\w*$")


def _is_annotation(name: str) -> bool:
    return name in _NOT_A_FUNC_NAME or bool(_ANNOTATION_RE.match(name))


def _resolve_func_name(record: Dict[str, Any]) -> str:
    name = str(record.get("func_name") or "").strip()
    if name and not _is_annotation(name):
        return name
    func = record.get("func") or ""
    brace = func.find("{")
    header = func[: brace if brace > 0 else 800][:800]
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", header):
        if not _is_annotation(m.group(1)):
            return m.group(1)
    return name or "(unknown)"


def resolve_func_name(record: Dict[str, Any]) -> str:
    return _resolve_func_name(record)


def _status_text(change: Dict[str, Any]) -> str:
    cp = change.get("counterpart") or {}
    base = "this function compared with its post-fix version"
    return {
        "pair": base,
        "pair_renamed": f"{base}, renamed `{cp.get('name', '?')}`",
        "pair_moved": f"{base}, moved to {cp.get('path', '?')}",
        "removed": "exists only before the commit (removed)",
        "added": "exists only after the commit (added)",
    }.get(change.get("diff_method"), _STATUS_APPROX)


def _diff_info(record: Dict[str, Any]) -> Dict[str, Any]:
    change = record.get("change") or {}
    loc = record.get("location") or {}
    start, end, side = _int(loc.get("start_line")), _int(loc.get("end_line")), loc.get("side")
    length = (end - start + 1) if start and end and end >= start else 0

    def out(text: str, truncated: bool, available: bool, status: str) -> Dict[str, Any]:
        return {"text": text, "truncated": truncated, "available": available, "status": status}

    if "function_diff" in change:
        reliable = change.get("diff_reliable")
        if reliable is None:
            reliable = (anchored_is_plausible(int(change.get("deleted") or 0),
                                              int(change.get("inserted") or 0), length)
                        if length else True)
        if not reliable:
            return out(_NO_DIFF_UNATTRIBUTABLE, True, False, "not available")
        text, status = change.get("function_diff") or "", _status_text(change)
        if text.strip():
            return out(text, bool(change.get("function_diff_truncated")), True, status)
        if change.get("whitespace_only"):
            return out(_WS_ONLY, False, True, status)
        if int(change.get("changed") or 0) > 0 and record.get("schema_version", 2) >= 3:
            return out(_NO_DIFF_UNATTRIBUTABLE, True, False, "not available")
        return out(_NO_DIFF, False, True, status)

    raw = change.get("hunks") or ""
    if not raw:
        return out(_NO_DIFF, False, True, _STATUS_APPROX)
    if not start or not end or side not in ("old", "new"):
        return out(raw, bool(change.get("hunks_truncated")), True, _STATUS_APPROX)
    hunks = parse_unified_diff(raw)
    if change.get("hunks_truncated") and covered_until(hunks, side) < end:
        return out(_NO_DIFF_LOST, True, False, "not available")
    return out(render_function_diff(hunks, start, end, side), False, True, _STATUS_APPROX)


_DIFF_FILE_HDR_RE = re.compile(r"^(?:\+\+\+|---) (?:a/|b/|/dev/null)")
_COMMENT_CONT_RE = re.compile(r"^\*+(?:\s|/|$)")
_COSMETIC_KIND = {"blank": "whitespace_only", "comment": "comment_only"}


def _is_trivial_change(content: str) -> bool:
    s = content.strip()
    if not s or s.startswith("//"):
        return True
    if s.startswith("/*"):
        return not (s.split("*/", 1)[1].strip() if "*/" in s else "")
    if s.startswith("*/"):
        return not s[2:].strip()
    if _COMMENT_CONT_RE.match(s):
        return s[-1] not in ";{}"
    return False


def _diff_stats(text: str) -> Tuple[int, int, int, str]:
    added = removed = hunks = 0
    cosmetic = "blank"
    for line in text.splitlines():
        if line.startswith("@@"):
            hunks += 1
            continue
        if _DIFF_FILE_HDR_RE.match(line):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
        else:
            continue
        if cosmetic:
            if not _is_trivial_change(line[1:]):
                cosmetic = ""
            elif line[1:].strip():
                cosmetic = "comment"
    return added, removed, hunks, (cosmetic if added + removed else "")


def analyze_record(record: Dict[str, Any]) -> Dict[str, Any]:
    info = _diff_info(record)
    raw, truncated, available = info["text"], info["truncated"], info["available"]
    added = removed = hunks = 0
    if not available:
        kind = "unavailable"
    elif not raw.strip() or raw == _NO_DIFF:
        kind = "no_change"
    elif raw == _WS_ONLY:
        kind = "whitespace_only"
    else:
        added, removed, hunks, cosmetic = _diff_stats(raw)
        kind = "code" if truncated else _COSMETIC_KIND.get(cosmetic, "code")
        
    if kind in ("unavailable", "no_change") or raw == _WS_ONLY:
        diff_text = raw if raw.strip() else _NO_DIFF
    else:
        diff_text, cut, _ = shrink_lines(raw, MAX_DIFF_CHARS)
        truncated = truncated or cut
        
    return {"diff_text": diff_text, "diff_truncated": truncated, "diff_available": available,
            "diff_status": info["status"], "diff_kind": kind,
            "n_added": added, "n_removed": removed, "n_hunks": hunks}


def _failure(reason: str) -> Dict[str, Any]:
    return {"data": None, "category": None, "predicted_target": None, "parse_error": True,
            "parse_error_reason": reason, "parse_method": "failed", "needs_retry": True,
            "source": "llm", "rule": None, "flaw_visible": None, "flags": [],
            "inconsistent": False}


def _verdict(cat: str, data: Dict[str, Any], method: str, **extra: Any) -> Dict[str, Any]:
    return {**_failure(""), "data": data, "category": cat,
            "predicted_target": CATEGORY_TO_TARGET[cat], "parse_error": False,
            "parse_method": method, "needs_retry": False, **extra}


_PRECHECK_WHY = {
    "unavailable": "No usable diff is available for this function, so no role can be assessed.",
    "no_change": "The function has no changed lines inside it.",
    "whitespace_only": "Only whitespace or formatting differs; the code is unchanged.",
    "comment_only": "Every changed line is a comment or blank line; behaviour is unchanged.",
}


def precheck(record: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    ctx = ctx if ctx is not None else analyze_record(record)
    kind = ctx.get("diff_kind")
    why = _PRECHECK_WHY.get(kind)
    if why is None:
        return None
        
    if kind == "unavailable":
        cat, rule = "UNDETERMINABLE", 0
    elif kind == "no_change":
        cat, rule = CATEGORY_IRRELEVANT, 1
    else:
        cat, rule = cosmetic_verdict(
            comment_edit=kind == "comment_only",
            name_in_message=bool((record.get("signals") or {}).get("name_in_message")))
            
    data = {"step_1_change": why, "step_2_flaw": "Flaw visible: no. " + why,
            "step_3_role": f"Rule {rule}: {why}", "category": cat}
    return _verdict(cat, data, "rule", source="rule", rule=rule, flaw_visible="no")


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_THINK_RE = re.compile(r"<(think|thinking|thought)>.*?</\1>", re.IGNORECASE | re.DOTALL)
_CATEGORY_FIELD_RE = re.compile(r'"category"\s*:\s*"([A-Za-z_ \-]+)"')


def _balanced_objects(text: str) -> List[str]:
    spans: List[str] = []
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start:i + 1])
    if depth > 0 and start >= 0:
        spans.append(text[start:])
    return spans


def _extract_json_blobs(raw: str) -> List[str]:
    text = _FENCE_RE.sub("", _THINK_RE.sub("", raw).strip())
    blobs: List[str] = []
    cands = [s for s in _balanced_objects(text) if ":" in s]
    if cands:
        keyed = [s for s in cands if '"category"' in s]
        blobs.append((keyed or cands)[-1])
    first = text.find("{")
    if first >= 0:
        last = text.rfind("}")
        wide = text[first:last + 1] if last > first else text[first:]
        if wide not in blobs:
            blobs.append(wide)
    return blobs


def _loads(raw: Any) -> Tuple[Optional[dict], str]:
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty output"
    blobs = _extract_json_blobs(raw)
    if not blobs:
        return None, "no JSON object in output"
    reason = "parse failed"
    for blob in blobs:
        attempts = [("strict", lambda b=blob: json.loads(b, strict=False))]
        if repair_json is not None:
            attempts.append(("repaired", lambda b=blob: repair_json(b, return_objects=True)))
        for method, fn in attempts:
            try:
                obj = fn()
            except Exception as exc:
                reason = f"{method}: {exc}"
                continue
            if isinstance(obj, list):
                obj = next((o for o in obj if isinstance(o, dict)), None)
            if isinstance(obj, dict) and obj:
                return obj, method
            reason = f"{method}: not a JSON object"
    return None, reason


def _norm_category(value: Any) -> Optional[str]:
    if not isinstance(value, str): return None
    s = re.sub(r"[^A-Za-z]+", "_", value).strip("_").upper()
    if s in VALID_CATEGORIES: return s
    if s in _CATEGORY_ALIASES: return _CATEGORY_ALIASES[s]
    for cat in _CATEGORIES_LONGEST_FIRST:
        if s.startswith(cat): return cat
    return s


_RULE_LEAD_RE = re.compile(r"^\W*Rule\s*([0-6])\b", re.IGNORECASE)
_RULE_ANY_RE = re.compile(r"\bRule\s*([0-6])\b", re.IGNORECASE)
_FLAW_RE = re.compile(r"Flaw\s+visible\s*:\s*(yes|no|unclear)", re.IGNORECASE)
_RULE_TO_CATEGORIES = {
    0: {"UNDETERMINABLE"},
    1: {CATEGORY_IRRELEVANT},
    2: {"CORRECT_LABEL"},
    3: {"WRONG_LABEL_SPREAD"},
    4: {"WRONG_LABEL_FEATURE"},
    5: {CATEGORY_CONSISTENCY},
    6: {CATEGORY_IRRELEVANT},
}


def _check_consistency(obj: Dict[str, Any], cat: str) -> Dict[str, Any]:
    step3, step2 = obj.get("step_3_role"), obj.get("step_2_flaw")
    m = None
    if isinstance(step3, str):
        m = _RULE_LEAD_RE.search(step3) or _RULE_ANY_RE.search(step3)
    rule = int(m.group(1)) if m else None
    f = _FLAW_RE.search(step2) if isinstance(step2, str) else None
    flaw = f.group(1).lower() if f else None
    flags: List[str] = []
    hard = False
    
    if rule is None:
        flags.append("no_rule_marker")
    elif cat not in _RULE_TO_CATEGORIES[rule]:
        flags.append("rule_category_mismatch")
        hard = True
        
    if flaw is None:
        flags.append("no_flaw_marker")
    elif cat == "CORRECT_LABEL" and flaw == "no":
        flags.append("correct_without_flaw")
        hard = True
        
    return {"rule": rule, "flaw_visible": flaw, "flags": flags, "inconsistent": hard}


def parse_llm_output(raw: Any, *, salvage_category: bool = False) -> Dict[str, Any]:
    obj, info = _loads(raw)
    if obj is not None:
        cat = _norm_category(obj.get("category"))
        if cat in VALID_CATEGORIES:
            obj["category"] = cat
            return _verdict(cat, obj, info, **_check_consistency(obj, cat))
        info = f"{info}: missing or invalid category ({cat!r})"
        
    if salvage_category and isinstance(raw, str):
        found = _CATEGORY_FIELD_RE.findall(_THINK_RE.sub("", raw))
        cat = _norm_category(found[-1]) if found else None
        if cat in VALID_CATEGORIES:
            return _verdict(cat, {"category": cat}, "salvaged",
                            parse_error_reason=f"salvaged after: {info}")
    return _failure(info)


def escalate_options(options: Optional[Dict[str, Any]], *, finish_reason: Optional[str] = None,
                     attempt: int = 1) -> Dict[str, Any]:
    opts = dict(options or OLLAMA_OPTIONS)
    if finish_reason == "length":
        opts["num_predict"] = min(int(opts.get("num_predict", 900)) * 2, 3000)
    else:
        opts["temperature"] = max(float(opts.get("temperature", 0.0)), 0.3)
        opts["seed"] = int(opts.get("seed", 0)) + attempt
    return opts


def _siblings_block(record: Dict[str, Any]) -> str:
    sibs = [x for x in (record.get("siblings") or []) if isinstance(x, dict)]
    if not sibs: return ""
    own_file = (record.get("location") or {}).get("file")
    lines = []
    for x in sibs[:MAX_SIBLINGS_LISTED]:
        name = x.get("name") or "(unnamed)"
        if x.get("located") is False:
            lines.append(f"- {name}: not located in repository")
            continue
        where = x.get("file") or "?"
        same = " [same file]" if own_file and where == own_file else ""
        lines.append(f"- {name} ({where}{same}): {x.get('changed', '?')} lines, kind={x.get('kind', '?')}")
    more = len(sibs) - len(lines)
    if more > 0: lines.append(f"- ... and {more} more")
    return f"<commit_siblings>\nOther flagged functions:\n{_defang(chr(10).join(lines))}\n</commit_siblings>\n\n"


def _format_cg_entry(e: Dict[str, Any]) -> str:
    same = " [same file]" if e.get("same_file") else ""
    hops = f", {e['hops']} hops" if e.get("hops", 1) > 1 else ""
    loc = f"{e.get('file', '?')}{same}{hops}"
    label = e.get("label")
    if label:
        reason = f" ({e['reason']})" if e.get("reason") else ""
        return f"- {e.get('name', '?')} ({loc}): {label}{reason}"
    return f"- {e.get('name', '?')} ({loc}): (no verdict yet)"


def _callgraph_block(graph_ctx: Optional[Dict[str, Any]]) -> str:
    if not graph_ctx: return ""
    rec_note = ", calls itself" if graph_ctx.get("recursive") else ""
    lines = [f"Position: {graph_ctx['position']}/{graph_ctx['total']} (depth {graph_ctx['depth']}{rec_note})."]
    
    if graph_ctx.get("callees"):
        lines.append("Callees:")
        for c in graph_ctx["callees"]: lines.append("  " + _format_cg_entry(c))
    if graph_ctx.get("callers"):
        lines.append("Callers:")
        for c in graph_ctx["callers"]: lines.append("  " + _format_cg_entry(c))
    
    return f"<commit_callgraph>\n{_defang(chr(10).join(lines))}\n</commit_callgraph>\n\n"


def _build_record_part(record: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None,
                       graph_ctx: Optional[Dict[str, Any]] = None) -> str:
    ctx = ctx or analyze_record(record)
    graph_ctx = graph_ctx if graph_ctx is not None else record.get("graph_ctx")
    commit, loc = record.get("commit") or {}, record.get("location") or {}
    cves = commit.get("cves") or []
    side = loc.get("side", "old")

    body, omitted = _head_tail_lines(record.get("func", "") or "", MAX_FUNC_CHARS)
    func_warn = "\nNote: shortened." if omitted else ""
    diff_warn = "\nNote: diff truncated." if ctx["diff_truncated"] and ctx["diff_available"] else ""

    return f"""\
<commit_context>
Message: {_message(commit, MAX_MSG_CHARS)}
CVEs: {', '.join(cves) if cves else 'none'}
</commit_context>

{_callgraph_block(graph_ctx)}{_siblings_block(record)}<function>
Name: {_resolve_func_name(record)}
File: {loc.get("file", "?")} ({loc.get("start_line", "?")}-{loc.get("end_line", "?")})
Version: {_SIDE_LABEL.get(side, side)}
</function>

<body>
{_defang(body)}{func_warn}
</body>

<diff>
Status: {ctx["diff_status"]}
{_defang(ctx["diff_text"])}{diff_warn}
</diff>
"""


def build_messages(record: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None,
                   graph_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    return {"system": _SYSTEM_PROMPT,
            "user": _build_record_part(record, ctx=ctx, graph_ctx=graph_ctx)}


def build_prompt(record: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None,
                 graph_ctx: Optional[Dict[str, Any]] = None) -> str:
    m = build_messages(record, ctx, graph_ctx=graph_ctx)
    return m["system"] + "\n\n" + m["user"]


def estimate_prompt_tokens(record: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None,
                           graph_ctx: Optional[Dict[str, Any]] = None) -> int:
    return estimate_tokens(build_prompt(record, ctx, graph_ctx=graph_ctx))


def to_chat(messages: Dict[str, str], *, system_role: bool = True) -> List[Dict[str, str]]:
    if system_role:
        return [{"role": "system", "content": messages["system"]},
                {"role": "user", "content": messages["user"]}]
    return [{"role": "user", "content": messages["system"] + "\n\n" + messages["user"]}]


def ollama_request(record: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None,
                   *, model: str = OLLAMA_MODEL, options: Optional[Dict[str, Any]] = None,
                   system_role: bool = True, graph_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"model": model, "stream": False, "format": RESPONSE_JSON_SCHEMA,
            "messages": to_chat(build_messages(record, ctx, graph_ctx=graph_ctx), system_role=system_role),
            "options": dict(options or OLLAMA_OPTIONS)}


def prompt_meta(record: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None,
                graph_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from .callgraph import result_fields
    ctx = ctx if ctx is not None else analyze_record(record)
    return {"prompt_version": prompt_version(), 
            "diff_truncated": ctx["diff_truncated"], "diff_available": ctx["diff_available"],
            "diff_kind": ctx.get("diff_kind"),
            "diff_method": (record.get("change") or {}).get("diff_method"),
            **result_fields(graph_ctx)}


def aggregate_votes(results: list, *, drop_inconsistent: bool = True) -> Dict[str, Any]:
    valid = [r for r in results if r.get("category")]
    used = [r for r in valid if not r.get("inconsistent")] if drop_inconsistent else valid
    n_inconsistent = sum(1 for r in valid if r.get("inconsistent"))
    used = used or valid
    votes = [r["category"] for r in used]
    if not votes:
        return {"category": None, "predicted_target": None, "agreement": 0.0, "n_valid": 0,
                "n_inconsistent": 0, "counts": {}, "tie": False, "target_tie": False,
                "needs_retry": True}
    top = Counter(votes).most_common()
    t_top = Counter(CATEGORY_TO_TARGET[v] for v in votes).most_common()
    tie = len(top) > 1 and top[0][1] == top[1][1]
    t_tie = len(t_top) > 1 and t_top[0][1] == t_top[1][1]
    return {"category": None if tie else top[0][0],
            "predicted_target": None if t_tie else t_top[0][0],
            "agreement": top[0][1] / len(votes), "n_valid": len(votes),
            "n_inconsistent": n_inconsistent, "counts": dict(top),
            "tie": tie, "target_tie": t_tie, "needs_retry": False}