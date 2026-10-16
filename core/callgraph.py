"""Per-commit call graph: structural context for the classifier.

For every commit the functions of the dataset that belong to it are linked by a *static,
name-based* call graph built from their bodies (no repository access, no parser dependency):

    comments and string/char literals are blanked  ->  every ``name(`` in the body is a call
    candidate  ->  kept only when ``name`` is another function of the same commit.

The graph is used for two things:

1. **Order.** Functions are analysed bottom-up: callees before callers (post-order of a DFS over
   the strongly connected components; mutually recursive functions form one component and are
   ordered by file/line). ``CommitGraph.order`` is that list.
2. **Context.** When a function is analysed, ``CommitGraph.context_for`` describes how it sits in
   the graph (callees, callers, mutually recursive peers, indirect callees) together with the
   label already assigned to each related function and a one-line reason. ``prompting`` renders
   it as ``<commit_callgraph>``.

Limits (by design): only functions present in the dataset for the commit are nodes, so calls to
code outside that set are invisible; calls through function pointers or macros are not
resolved; member calls (``obj->f()``, ``obj.f()``) are ignored except ``this->f()``. The graph
is evidence about the *structure* of the commit, never a verdict.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .prompting import commit_key, resolve_func_name

MAX_INDIRECT_HOPS = 3        # indirect callees shown up to this distance
REASON_CHARS = 200           # length of the one-line explanation attached to a label


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
# Reason shown next to an assigned label
# ---------------------------------------------------------------------------
_RULE_RE = re.compile(r"^\W*Rule\s*([0-6])\b", re.IGNORECASE)
_SENT_END_RE = re.compile(r"(?<=[.!?])\s")


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def brief_reason(result: Dict[str, Any], limit: int = REASON_CHARS) -> str:
    """Compact explanation of a label, built from fields already in the result (no extra call).

    LLM result: ``Rule N`` of step_3_role + first sentence of step_1_change.
    Deterministic result: the pre-filter rule names.
    """
    if result.get("method") == "deterministic":
        rules = result.get("prefilter_rules") or []
        if rules == ["diff_unrecoverable"]:
            return "no diff could be attributed to this function"
        return "deterministic pre-filter: " + (", ".join(map(str, rules)) or "rule-based")
    parts: List[str] = []
    m = _RULE_RE.match(_one_line(str(result.get("step_3_role") or "")))
    if m:
        parts.append(f"Rule {m.group(1)}")
    change = _one_line(str(result.get("step_1_change") or ""))
    if change:
        first = _SENT_END_RE.split(change, 1)[0]
        parts.append("change: " + first)
    text = "; ".join(parts)
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


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
    order: List[str]                       # analysis order: callees first
    pos: Dict[str, int]                    # hash -> index in `order`
    scc_of: Dict[str, int]                 # hash -> component id
    scc_members: Dict[int, List[str]]      # component id -> members (graph order)
    depth: Dict[str, int]                  # hash -> longest call chain below it (0 = leaf)
    recursive: Set[str]                    # functions that call themselves directly
    n_ambiguous: int = 0                   # call names matching functions in >= 2 other files

    # -- queries ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.nodes)

    def n_edges(self) -> int:
        return sum(len(v) for v in self.callees.values())

    def has_cycles(self) -> bool:
        return any(len(m) > 1 for m in self.scc_members.values())

    def _closure(self, start: str, edges: Dict[str, List[str]]) -> Set[str]:
        seen: Set[str] = set()
        todo = list(edges.get(start, ()))
        while todo:
            x = todo.pop()
            if x in seen or x == start:
                continue
            seen.add(x)
            todo.extend(edges.get(x, ()))
        return seen

    def context_for(self, h: str, analyzed: Dict[str, Dict[str, Any]], *,
                    show_labels: bool = True) -> Optional[Dict[str, Any]]:
        """Plain-data description of function `h` in the graph, or None when the commit has a
        single function (nothing to add).

        `analyzed` maps hash -> result (category, step_*, ...) of the functions of this commit
        that already have a verdict; the others are reported as "no verdict yet".
        """
        if h not in self.nodes or len(self.nodes) < 2:
            return None
        me = self.nodes[h]

        def entry(x: str, hops: int) -> Dict[str, Any]:
            n = self.nodes[x]
            res = analyzed.get(x)
            return {"name": n.name, "file": n.file, "same_file": n.file == me.file,
                    "hops": hops,
                    "label": res.get("category") if res else None,
                    "reason": brief_reason(res) if res else ""}

        def ordered(hs: Iterable[str]) -> List[str]:
            return sorted(set(hs), key=lambda x: self.pos[x])

        callees = [entry(x, 1) for x in self.callees[h]]
        callers = [entry(x, 1) for x in self.callers[h]]
        peers_h = [x for x in self.scc_members[self.scc_of[h]] if x != h]
        peers = [entry(x, 0) for x in peers_h]

        seen = {h, *self.callees[h], *peers_h}
        frontier, indirect_h = list(self.callees[h]), []
        for hops in range(2, MAX_INDIRECT_HOPS + 1):
            nxt: List[str] = []
            for x in frontier:
                for y in self.callees[x]:
                    if y not in seen:
                        seen.add(y)
                        nxt.append(y)
                        indirect_h.append((y, hops))
            frontier = nxt
        indirect = [entry(y, hops) for y, hops in sorted(indirect_h, key=lambda t: self.pos[t[0]])]

        related = ({h} | self._closure(h, self.callees) | self._closure(h, self.callers)
                   | set(peers_h))
        others = Counter(res.get("category") for x, res in analyzed.items()
                         if x in self.nodes and x not in related and res.get("category"))

        return {
            "position": self.pos[h] + 1, "total": len(self.order), "depth": self.depth[h],
            "recursive": h in self.recursive,
            "callees": callees, "callers": callers, "peers": peers, "indirect": indirect,
            "others": {"total": sum(others.values()), "by_label": dict(others.most_common())},
            "show_labels": bool(show_labels),
        }


def result_fields(graph_ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Traceability fields stored next to each result."""
    if not graph_ctx:
        return {"callgraph": False}
    shown = 0
    if graph_ctx.get("show_labels", True):
        for k in ("callees", "callers", "peers", "indirect"):
            shown += sum(1 for e in graph_ctx[k] if e.get("label"))
        shown += graph_ctx["others"]["total"]
    return {"callgraph": True, "cg_position": graph_ctx["position"], "cg_total": graph_ctx["total"],
            "cg_depth": graph_ctx["depth"], "cg_n_callees": len(graph_ctx["callees"]),
            "cg_n_callers": len(graph_ctx["callers"]), "cg_n_peers": len(graph_ctx["peers"]),
            "cg_labels_shown": shown, "cg_labels_hidden": not graph_ctx.get("show_labels", True)}


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
