"""Per-commit call graph: structural context for the classifier.

For every commit the functions of the dataset that belong to it are linked by a *static,
name-based* call graph built from their bodies (no repository access, no parser dependency):

    comments and string/char literals are blanked  ->  every ``name(`` in the body is a call
    candidate  ->  kept only when ``name`` is another function of the same commit.

What the graph is used for
--------------------------
``CommitGraph.context_for`` describes how a function sits in the graph (callees, callers) and,
for each neighbour, a *deterministic* summary of its own change (kind, +/- counts and the first
changed lines). ``prompting`` renders it as the "Call graph" section.

Labels given until that point
-----------------------------
``context_for(h, summaries, labels=verdicts)`` additionally attaches to every neighbour the label
already issued for it (``verdicts``: hash -> verdict dict as returned by ``parse_llm_output`` /
``precheck``, or a bare category string). A neighbour without a verdict is reported as pending.
Passing ``labels=None`` reproduces the label-free context.

This makes the analysis order meaningful: process a commit's functions in ``graph.order``
(callees first) and store each verdict in ``verdicts`` before moving on. Commits stay independent
of each other and can run in parallel; inside one commit the run is serial. Trade-offs to keep in
mind: callers are mostly ``pending`` (they come after their callees), an early wrong label can
propagate to its callers, and results depend on the order. ``cg_n_labelled`` is stored with every
result so that runs with and without labels can be compared (ablation).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from .prompting import analyze_record, resolve_func_name

MAX_NEIGHBOURS = 6        # callees / callers shown per side (the rest is only counted)
EXCERPT_LINES = 3         # changed lines shown per neighbour
EXCERPT_CHARS = 110       # characters per excerpt line


# ---------------------------------------------------------------------------
# Commit Key
# ---------------------------------------------------------------------------
def commit_key(record: Dict[str, Any]) -> str:
    """Cache key for grouping functions by commit."""
    commit = record.get("commit") or {}
    for cand in (record.get("commit_id"), commit.get("id"), commit.get("sha"), commit.get("hash")):
        if cand:
            return str(cand)
    return f"{record.get('project', '?')}::{(commit.get('message') or '')[:80]}"


# ---------------------------------------------------------------------------
# Call extraction
# ---------------------------------------------------------------------------
_NONCODE_RE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S)
_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")
_THIS_RE = re.compile(r"\bthis\s*->$")


def _blank(match: "re.Match[str]") -> str:
    """Replace a comment/literal by spaces, keeping newlines (offsets stay meaningful)."""
    return re.sub(r"[^\n]", " ", match.group(0))


def extract_calls(func: str) -> Set[str]:
    """Names that appear as ``name(`` in the body of `func` (header excluded), non-member only."""
    code = _NONCODE_RE.sub(_blank, func or "")
    brace = code.find("{")
    body = code[brace + 1:] if brace >= 0 else code
    calls: Set[str] = set()
    for m in _CALL_RE.finditer(body):
        before = body[max(0, m.start() - 16): m.start()].rstrip()
        if before.endswith("->") and not _THIS_RE.search(before):
            continue                      # ptr->f(...): field / method of another object
        if before.endswith(".") or before.endswith("~"):
            continue                      # obj.f(...), destructor
        calls.add(m.group(1))
    return calls


def short_name(name: str) -> str:
    """`Class::method` -> `method` (the graph matches on the last component)."""
    return name.rsplit("::", 1)[-1].strip()


# ---------------------------------------------------------------------------
# Deterministic change summaries (what the prompt shows for a neighbour)
# ---------------------------------------------------------------------------
_HDR_RE = re.compile(r"^(?:\+\+\+|---) (?:a/|b/|/dev/null)")


def _excerpt(diff_text: str) -> List[str]:
    """First few changed lines that are not comments or blanks, whitespace-collapsed."""
    out: List[str] = []
    for line in diff_text.splitlines():
        if not line.startswith(("+", "-")) or _HDR_RE.match(line):
            continue
        content = line[1:].strip()
        if not content or content.startswith(("//", "/*", "*")):
            continue
        out.append((line[0] + " " + re.sub(r"\s+", " ", content))[:EXCERPT_CHARS])
        if len(out) >= EXCERPT_LINES:
            break
    return out


def summarize_change(record: Dict[str, Any]) -> Dict[str, Any]:
    """Kind, +/- counts and a short excerpt of a record's own diff (no LLM involved)."""
    try:
        ctx = analyze_record(record)
    except Exception:                      # a broken record must never break the neighbours
        return {"kind": "unknown", "added": 0, "removed": 0, "excerpt": []}
    kind = ctx["diff_kind"]
    return {"kind": kind, "added": ctx["n_added"], "removed": ctx["n_removed"],
            "excerpt": _excerpt(ctx["diff_text"]) if kind == "code" else []}


class SummaryCache:
    """Lazy hash -> change summary, so only neighbours that are actually shown get analysed."""

    def __init__(self, records: Iterable[Dict[str, Any]]):
        self._records = {str(r.get("hash")): r for r in records}
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, h: str, default: Any = None) -> Any:
        s = self._cache.get(h)
        if s is None:
            rec = self._records.get(h)
            if rec is None:
                return default
            s = self._cache[h] = summarize_change(rec)
        return s


# ---------------------------------------------------------------------------
# Labels given so far
# ---------------------------------------------------------------------------
def compact_label(verdict: Any) -> Optional[Dict[str, Any]]:
    """Reduce a stored verdict to what the prompt shows; None when there is no usable label.

    Accepts the dict produced by ``parse_llm_output`` / ``precheck`` or a bare category string.
    Failed parses (no category) count as "no label yet"."""
    if isinstance(verdict, str):
        return {"category": verdict, "rule": None, "flaw_visible": None, "source": None} if verdict else None
    if not isinstance(verdict, dict) or verdict.get("parse_error") or not verdict.get("category"):
        return None
    return {"category": verdict["category"], "rule": verdict.get("rule"),
            "flaw_visible": verdict.get("flaw_visible"), "source": verdict.get("source")}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
@dataclass
class Node:
    hash: str
    name: str
    file: str
    start: int
    side: str
    calls: Set[str] = field(default_factory=set)

    @property
    def sort_key(self) -> Tuple[str, int, str, str]:
        return (self.file, self.start, self.name, self.hash)


@dataclass
class CommitGraph:
    key: str
    nodes: Dict[str, Node]
    callees: Dict[str, List[str]]          # hash -> callee hashes (graph order)
    callers: Dict[str, List[str]]          # hash -> caller hashes (graph order)
    order: List[str]                       # callees first: THE processing order when labels are used
    pos: Dict[str, int]                    # hash -> index in `order`
    scc_of: Dict[str, int]                 # hash -> component id
    scc_members: Dict[int, List[str]]      # component id -> members (graph order)
    depth: Dict[str, int]                  # hash -> longest call chain below it (0 = leaf)
    recursive: Set[str]                    # functions that call themselves directly
    n_ambiguous: int = 0                   # call names matching functions in >= 2 other files

    def __len__(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return sum(len(v) for v in self.callees.values())

    def has_cycles(self) -> bool:
        return any(len(m) > 1 for m in self.scc_members.values())

    def levels(self) -> List[List[str]]:
        """Bottom-up processing levels: level 0 = leaves (deepest in the call tree), then their
        callers, and so on. Every callee sits in a strictly lower level than its caller, except
        members of the same recursion cycle (same level, run those one after the other in list
        order). Functions inside one level do not depend on each other, so a level can be
        analysed in parallel; levels must be analysed in sequence."""
        by_depth: Dict[int, List[str]] = defaultdict(list)
        for h in self.order:
            by_depth[self.depth[h]].append(h)
        return [by_depth[d] for d in sorted(by_depth)]

    def context_for(self, h: str, summaries: Any,
                    labels: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Plain-data description of function `h`: neighbours + their change summaries.

        `summaries` is anything with ``.get(hash)`` (dict or SummaryCache).
        `labels` maps hash -> verdict for the functions analysed so far; None disables labels,
        an empty dict means "labels enabled, nothing analysed yet" (everything pending)."""
        if h not in self.nodes or len(self.nodes) < 2:
            return None
        me = self.nodes[h]
        labels_given = labels is not None

        def label_of(x: str) -> Optional[Dict[str, Any]]:
            return compact_label(labels.get(x)) if labels_given else None

        def entry(x: str) -> Dict[str, Any]:
            n = self.nodes[x]
            s = summaries.get(x) or {}
            return {"name": n.name, "file": n.file, "same_file": n.file == me.file,
                    "kind": s.get("kind", "unknown"), "added": s.get("added", 0),
                    "removed": s.get("removed", 0), "excerpt": s.get("excerpt", []),
                    "label": label_of(x)}

        def pick(ids: List[str]) -> Tuple[List[Dict[str, Any]], int]:
            ents = [entry(x) for x in ids]
            # most informative first: a neighbour already judged as the real fix, then same
            # file, really changed, bigger change
            ents.sort(key=lambda e: ((e["label"] or {}).get("category") != "CORRECT_LABEL",
                                     not e["same_file"], e["kind"] != "code",
                                     -(e["added"] + e["removed"]), e["name"]))
            return ents[:MAX_NEIGHBOURS], max(0, len(ents) - MAX_NEIGHBOURS)

        callees, more_callees = pick(self.callees[h])
        callers, more_callers = pick(self.callers[h])
        neighbours = set(self.callees[h]) | set(self.callers[h])
        return {
            "position": self.pos[h] + 1, "total": len(self.order), "depth": self.depth[h],
            "recursive": h in self.recursive,
            "callees": callees, "callers": callers,
            "n_callees": len(self.callees[h]), "n_callers": len(self.callers[h]),
            "more_callees": more_callees, "more_callers": more_callers,
            "labels_given": labels_given,
            "n_neighbours": len(neighbours),
            "n_labelled": sum(1 for x in neighbours if label_of(x)) if labels_given else 0,
        }


def result_fields(graph_ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Traceability fields stored next to each result."""
    if not graph_ctx:
        return {"callgraph": False}
    return {"callgraph": True, "cg_position": graph_ctx["position"], "cg_total": graph_ctx["total"],
            "cg_depth": graph_ctx["depth"],
            "cg_n_callees": graph_ctx.get("n_callees", len(graph_ctx["callees"])),
            "cg_n_callers": graph_ctx.get("n_callers", len(graph_ctx["callers"])),
            "cg_labels_given": bool(graph_ctx.get("labels_given")),
            "cg_n_labelled": graph_ctx.get("n_labelled", 0)}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def _tarjan(roots: List[str], succ: Dict[str, List[str]]) -> List[List[str]]:
    """Strongly connected components, iterative. Emitted callees-first (reverse topological)."""
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    sccs: List[List[str]] = []
    counter = 0
    for root in roots:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(succ[root]))]
        while work:
            v, it = work[-1]
            descended = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(succ[w])))
                    descended = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
            if low[v] == index[v]:
                comp: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)
    return sccs


def build_commit_graph(key: str, records: List[Dict[str, Any]]) -> CommitGraph:
    """Call graph of the functions of one commit (one node per distinct hash)."""
    nodes: Dict[str, Node] = {}
    for r in records:
        h = str(r.get("hash"))
        if h in nodes:
            continue
        loc = r.get("location") or {}
        try:
            start = int(loc.get("start_line") or 0)
        except (TypeError, ValueError):
            start = 0
        nodes[h] = Node(hash=h, name=short_name(resolve_func_name(r)),
                        file=str(loc.get("file") or ""), start=start,
                        side=str(loc.get("side") or ""), calls=extract_calls(r.get("func") or ""))

    by_name: Dict[str, List[Node]] = defaultdict(list)
    for n in nodes.values():
        by_name[n.name].append(n)

    pos_key = {h: n.sort_key for h, n in nodes.items()}
    callees: Dict[str, List[str]] = {h: [] for h in nodes}
    recursive: Set[str] = set()
    n_ambiguous = 0
    for h, n in nodes.items():
        targets: Set[str] = set()
        for name in n.calls:
            cands = by_name.get(name)
            if not cands:
                continue
            same = [c for c in cands if c.file == n.file]
            if same:
                chosen = same
            elif len({c.file for c in cands}) == 1:
                chosen = cands
            else:
                n_ambiguous += 1          # same name in several other files: do not guess
                continue
            for c in chosen:
                if c.hash == h:
                    recursive.add(h)
                else:
                    targets.add(c.hash)
        callees[h] = sorted(targets, key=pos_key.__getitem__)

    callers: Dict[str, List[str]] = {h: [] for h in nodes}
    for h, outs in callees.items():
        for c in outs:
            callers[c].append(h)
    for h in callers:
        callers[h].sort(key=pos_key.__getitem__)

    # DFS from the entry points (functions nobody calls) first, so that the order reads like
    # the call tree; whatever is left (cycles without an outside caller) follows.
    all_sorted = sorted(nodes, key=pos_key.__getitem__)
    roots = [h for h in all_sorted if not callers[h]] + [h for h in all_sorted if callers[h]]
    sccs = _tarjan(roots, callees)

    order: List[str] = []
    scc_of: Dict[str, int] = {}
    scc_members: Dict[int, List[str]] = {}
    depth: Dict[str, int] = {}
    for i, comp in enumerate(sccs):
        comp.sort(key=pos_key.__getitem__)
        scc_members[i] = comp
        for h in comp:
            scc_of[h] = i
        order.extend(comp)
        below = [depth[c] for h in comp for c in callees[h] if scc_of[c] != i]
        d = 1 + max(below) if below else 0
        for h in comp:
            depth[h] = d
    return CommitGraph(key=key, nodes=nodes, callees=callees, callers=callers, order=order,
                       pos={h: i for i, h in enumerate(order)}, scc_of=scc_of,
                       scc_members=scc_members, depth=depth, recursive=recursive,
                       n_ambiguous=n_ambiguous)


def build_graphs(records: Iterable[Dict[str, Any]]) -> Dict[str, CommitGraph]:
    """One CommitGraph per commit_key of `records`."""
    by_commit: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_commit[commit_key(r)].append(r)
    return {k: build_commit_graph(k, rs) for k, rs in by_commit.items()}


def summarize_graphs(graphs: Dict[str, CommitGraph]) -> str:
    """One log line about the graphs (for the run log)."""
    multi = [g for g in graphs.values() if len(g) > 1]
    with_edges = [g for g in multi if g.n_edges()]
    edges = sum(g.n_edges() for g in multi)
    cyc = sum(1 for g in multi if g.has_cycles())
    deepest = max((max(g.depth.values()) for g in multi), default=0)
    amb = sum(g.n_ambiguous for g in multi)
    return (f"{len(graphs)} commits, {len(multi)} with >= 2 functions "
            f"({len(with_edges)} with at least one call edge, {edges} edges, {cyc} with "
            f"recursion cycles, max call depth {deepest}, {amb} ambiguous call names ignored)")