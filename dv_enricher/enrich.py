"""Orchestrazione: per ogni commit costruisce il contesto di tutte le funzioni flaggate.

Schema del bundle (una riga JSON per funzione):
  status: ok | repo_unresolved | commit_not_found | no_parent | func_not_located
  location: file, righe, lato (old = versione pre-fix), metodo di match
  change:   righe cancellate/inserite nella funzione, tipo di modifica, hunk del diff
  commit:   messaggio, CVE, statistiche del diff, file modificati
  siblings: altre funzioni flaggate dello stesso commit con quante righe le riguardano
  signals:  segnali deterministici (nome nel messaggio/NVD, rank nel commit, ...)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import features as F
from .dataset import Record
from .diffparse import anchored_is_plausible, changed_in_range, map_line, parse_unified_diff, render_function_diff, shrink_lines
from .funcspan import SourceFile, find_counterpart, pair_diff
from .locate import FileView, Match, locate
from .nvd import NvdClient
from .repos import GitRepo, RepoProvider

# v2: change.function_diff = diff ristretto alla funzione (v1 passava hunk interi, troncati da inizio)
# v3: function_diff calcolato confrontando la funzione pre/post-fix (non con ancore di riga):
#     aggiunge change.diff_method / diff_reliable / counterpart / whitespace_only
SCHEMA_VERSION = 3


@dataclass
class EnrichConfig:
    max_hunk_chars: int = 20000       # tetto su change.hunks (hunk interi, usati dal pre-filtro)
    max_function_diff_chars: int = 20000   # tetto su change.function_diff (solo righe della funzione)
    max_siblings: int = 60            # funzioni sorelle da localizzare oltre a quelle richieste
    max_siblings_listed: int = 30
    max_files_listed: int = 40
    max_candidate_files: int = 150    # file sorgente da leggere per cercare le funzioni
    fuzzy: bool = True
    fuzzy_threshold: float = 0.95


@dataclass
class Located:
    rec: Record
    name: str | None
    match: Match | None = None
    side: str = "old"
    deleted: list[str] = field(default_factory=list)
    inserted: list[str] = field(default_factory=list)
    hunks: list = field(default_factory=list)
    pair: object = None              # funcspan.PairDiff (None = diff ancorato)
    pair_method: str = "anchored"    # pair | pair_renamed | pair_moved | removed | added | anchored
    counterpart: dict | None = None
    reliable: bool = True

    @property
    def changed(self) -> int:
        return len(self.deleted) + len(self.inserted)


def _commit_skeleton(commit_id: str, recs: list[Record], n_total: int, n_flagged: int) -> dict:
    msg = recs[0].message
    return {"message": msg, "cves": F.find_cves(msg), "n_flagged_in_dataset": n_flagged,
            "n_functions_in_dataset": n_total, "nvd": []}


def _bundle(rec: Record, status: str, commit: dict, repo_url: str | None, **extra) -> dict:
    name = F.extract_func_name(rec.func)
    b = {"schema_version": SCHEMA_VERSION, "hash": rec.hash, "commit_id": rec.commit_id,
         "project": rec.project, "target": rec.target, "cwe": rec.cwe, "dataset_size": rec.size,
         "status": status, "repo_url": repo_url, "func": rec.func, "func_name": name,
         "location": None, "change": None, "commit": commit, "siblings": [],
         "signals": {"name_in_message": F.name_in_text(name, commit["message"])}}
    b.update(extra)
    return b


def enrich_commit(commit_id: str, requested: list[Record], all_flagged: list[Record], n_total: int,
                  provider: RepoProvider, cfg: EnrichConfig | None = None,
                  nvd: NvdClient | None = None) -> list[dict]:
    cfg = cfg or EnrichConfig()
    project = requested[0].project
    commit = _commit_skeleton(commit_id, requested, n_total, len(all_flagged))
    if nvd:
        commit["nvd"] = [asdict(i) for c in commit["cves"] if (i := nvd.get(c))]

    repo, status = provider.get_repo(project, commit_id)
    if repo is None:
        return [_bundle(r, status, commit, None) for r in requested]
    return _enrich_with_repo(repo, commit_id, requested, all_flagged, commit, cfg)


def _enrich_with_repo(repo: GitRepo, commit_id: str, requested: list[Record], all_flagged: list[Record],
                      commit: dict, cfg: EnrichConfig) -> list[dict]:
    url = repo.url
    parents = repo.parents(commit_id)
    if not parents:
        return [_bundle(r, "no_parent", commit, url) for r in requested]
    base = parents[0]
    commit["parent"] = base
    commit["is_merge"] = len(parents) > 1

    # -- statistiche del diff ------------------------------------------------------------
    stats = sorted(repo.numstat(base, commit_id), key=lambda t: -(t[0] + t[1]))
    files = [{"path": p, "added": a, "deleted": d, "is_source": F.is_source(p), "is_test": F.is_test_path(p)}
             for a, d, p in stats]
    src = [f for f in files if f["is_source"]]
    src_churn = sum(f["added"] + f["deleted"] for f in src)
    commit.update({
        "n_files": len(files), "n_source_files": len(src),
        "n_test_files": sum(f["is_test"] for f in files),
        "total_added": sum(f["added"] for f in files), "total_deleted": sum(f["deleted"] for f in files),
        "source_churn": src_churn,
        "files": files[:cfg.max_files_listed], "files_truncated": len(files) > cfg.max_files_listed,
    })

    # -- localizzazione delle funzioni (richieste + sorelle) -------------------------------
    req_hashes = {r.hash for r in requested}
    siblings = [r for r in all_flagged if r.hash not in req_hashes][:cfg.max_siblings]
    items = [Located(r, F.extract_func_name(r.func)) for r in requested + siblings]

    cand = [f["path"] for f in src if not f["is_test"]] + [f["path"] for f in src if f["is_test"]]
    commit["candidate_files_truncated"] = len(cand) > cfg.max_candidate_files
    cand = cand[:cfg.max_candidate_files]
    old_views = [FileView(p, t) for p, t in repo.read_blobs(base, cand).items() if t is not None]
    new_views: list[FileView] | None = None       # letti solo se serve (funzione già post-fix?)

    for it in items:
        m = locate(it.rec.func, old_views, cfg.fuzzy, cfg.fuzzy_threshold)
        if m is None:
            if new_views is None:
                new_views = [FileView(p, t) for p, t in repo.read_blobs(commit_id, cand).items() if t is not None]
            m = locate(it.rec.func, new_views, cfg.fuzzy, cfg.fuzzy_threshold)
            it.side = "new"
        else:
            it.side = "old"
        it.match = m

    # -- righe modificate per funzione -------------------------------------------------------
    hunk_cache: dict[str, list] = {}
    for it in items:
        if it.match is None:
            continue
        p = it.match.path
        if p not in hunk_cache:
            hunk_cache[p] = parse_unified_diff(repo.diff_file(base, commit_id, p))
        it.deleted, it.inserted, it.hunks = changed_in_range(hunk_cache[p], it.match.start, it.match.end, it.side)

    # -- diff per funzione: confronto della coppia pre/post-fix (fallback: ancore di riga) -----
    sides = _Sides(repo, base, commit_id, cand, old_views)
    for it in items:
        if it.match is None:
            continue
        length = it.match.end - it.match.start + 1
        try:
            pd, method, cp = _function_pair(it, sides, it.hunks)
        except Exception:   # noqa: BLE001 - un'anomalia di parsing non deve fermare il commit
            pd, method, cp = None, "pair_error", None
        if pd is not None:
            it.deleted, it.inserted = list(pd.deleted), list(pd.inserted)
            it.pair, it.pair_method, it.counterpart, it.reliable = pd, method, cp, True
        else:
            # ancore di riga: credibili solo se le dimensioni sono compatibili con la funzione
            it.pair, it.pair_method, it.counterpart = None, "anchored", None
            it.reliable = anchored_is_plausible(len(it.deleted), len(it.inserted), length)

    located = [it for it in items if it.match is not None]
    in_flagged = sum(it.changed for it in located)
    commit["source_churn_in_flagged"] = in_flagged
    commit["source_churn_outside_flagged"] = max(src_churn - in_flagged, 0)
    ranked = sorted(located, key=lambda it: -it.changed)
    rank = {it.rec.hash: i + 1 for i, it in enumerate(ranked)}

    sib_summary = [_sibling(it) for it in ranked][:cfg.max_siblings_listed]
    sib_summary += [{"hash": it.rec.hash, "name": it.name, "located": False}
                    for it in items if it.match is None][:cfg.max_siblings_listed]
    commit["n_flagged_located"] = len(located)

    nvd_text = " ".join(n["description"] for n in commit["nvd"])
    nvd_cwes = sorted({c for n in commit["nvd"] for c in n["cwes"]})

    out = []
    for it in items[:len(requested)]:
        r = it.rec
        if it.match is None:
            b = _bundle(r, "func_not_located", commit, url)
        else:
            b = _bundle(r, "ok", commit, url)
            m = it.match
            b["location"] = {"file": m.path, "start_line": m.start, "end_line": m.end, "side": it.side,
                             "method": m.method, "n_matches": m.n_matches, "score": round(m.score, 4)}
            text, trunc = _render_hunks(it.hunks, cfg.max_hunk_chars)
            if it.pair is not None:
                fdiff, kind = it.pair.text, _kind_of(it.pair)
                ws_only = it.pair.whitespace_only
            elif it.reliable:
                fdiff = render_function_diff(it.hunks, m.start, m.end, it.side)
                kind, ws_only = F.classify_change(it.deleted, it.inserted), False
            else:       # attribuzione non credibile: meglio nessun diff che il diff di un'altra funzione
                fdiff, kind, ws_only = "", F.classify_change(it.deleted, it.inserted), False
            fdiff, fdiff_trunc, _ = shrink_lines(fdiff, cfg.max_function_diff_chars)
            length = m.end - m.start + 1
            b["change"] = {
                "deleted": len(it.deleted), "inserted": len(it.inserted), "changed": it.changed,
                "frac_changed": round(it.changed / length, 4),
                "share_of_source_churn": round(it.changed / src_churn, 4) if src_churn else None,
                "kind": kind,
                "rank_in_commit": rank[r.hash], "hunks": text, "hunks_truncated": trunc,
                "function_diff": fdiff, "function_diff_truncated": fdiff_trunc,
                "diff_method": it.pair_method, "diff_reliable": it.reliable,
                "whitespace_only": ws_only, "counterpart": it.counterpart,
                "file_is_test": F.is_test_path(m.path),
            }
        b["siblings"] = [s for s in sib_summary if s["hash"] != r.hash]
        b["signals"].update({
            "name_in_nvd": F.name_in_text(it.name, nvd_text) if nvd_text else None,
            "cwe_matches_nvd": (bool(set(r.cwe) & set(nvd_cwes)) if r.cwe and nvd_cwes else None),
            "cwe_missing_in_dataset": not r.cwe,
        })
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Diff a livello di funzione (confronto della coppia pre/post-fix)
# ---------------------------------------------------------------------------
def _kind_of(pd) -> str:
    """'none' | 'whitespace' | 'comment_or_blank' | 'code' dal diff della coppia di funzioni."""
    if pd.identical:
        return "none"
    if pd.whitespace_only:
        return "whitespace"

    def cosmetic(l: str) -> bool:
        t = l.strip()
        return not t or t.startswith(("//", "/*", "*", "*/"))

    if all(cosmetic(l) for l in pd.deleted + pd.inserted):
        return "comment_or_blank"
    return "code"


class _Sides:
    """File dei due lati del commit, con cache e lettura pigra del lato post-fix."""

    def __init__(self, repo: GitRepo, base: str, commit_id: str, cand: list[str],
                 old_views: list[FileView]) -> None:
        self.repo, self.commit_id, self.cand = repo, commit_id, cand
        self.old = {v.path: SourceFile(v.path, v.text) for v in old_views}
        self._new: dict[str, SourceFile | None] = {}
        self._all_new = False

    def new(self, path: str) -> SourceFile | None:
        if path not in self._new:
            text = self.repo.read_blobs(self.commit_id, [path]).get(path)
            self._new[path] = SourceFile(path, text) if text is not None else None
        return self._new[path]

    def all_new(self) -> list[SourceFile]:
        if not self._all_new:
            missing = [p for p in self.cand if p not in self._new]
            blobs = self.repo.read_blobs(self.commit_id, missing) if missing else {}
            for p in missing:
                self._new[p] = SourceFile(p, blobs[p]) if blobs.get(p) is not None else None
            self._all_new = True
        return [f for f in self._new.values() if f is not None]

    def all_old(self) -> list[SourceFile]:
        return list(self.old.values())


def _function_pair(it: Located, sides: _Sides, hunks: list):
    """Diff della funzione = confronto con la sua controparte sull'altro lato del commit.

    Ritorna (PairDiff | None, method, counterpart dict | None). None = controparte non
    trovata: il chiamante ripiega sul diff ancorato (se credibile).
    """
    m = it.match
    mine_old = it.side == "old"
    mine = sides.old.get(m.path) if mine_old else sides.new(m.path)
    if mine is None:
        return None, "unlocated", None
    my_lines = mine.lines[m.start - 1:m.end]
    other_same = sides.new(m.path) if mine_old else sides.old.get(m.path)
    expected = map_line(hunks, m.start, "old" if mine_old else "new")
    lazy_other = sides.all_new if mine_old else sides.all_old
    cp = find_counterpart(it.name, my_lines, other_same, lazy_other, expected)
    if cp is None:
        if other_same is None:      # il file non esiste dall'altro lato: funzione tolta/aggiunta
            other_lines, o_start = [], 1
            method = "removed" if mine_old else "added"
            return (pair_diff(my_lines, other_lines, m.start, o_start) if mine_old
                    else pair_diff(other_lines, my_lines, o_start, m.start)), method, None
        return None, "no_counterpart", None
    other_src = (sides.new(cp.path) if mine_old else sides.old.get(cp.path))
    if other_src is None:
        return None, "no_counterpart", None
    other_lines = other_src.lines[cp.start - 1:cp.end]
    if mine_old:
        pd = pair_diff(my_lines, other_lines, m.start, cp.start)
    else:
        pd = pair_diff(other_lines, my_lines, cp.start, m.start)
    method = {"same_name": "pair", "renamed": "pair_renamed", "other_file": "pair_moved"}[cp.how]
    return pd, method, {"path": cp.path, "start_line": cp.start, "end_line": cp.end,
                        "name": cp.name, "score": round(cp.score, 4), "how": cp.how}


def _sibling(it: Located) -> dict:
    return {"hash": it.rec.hash, "name": it.name, "located": True, "file": it.match.path,
            "changed": it.changed, "kind": F.classify_change(it.deleted, it.inserted), "side": it.side}


def _render_hunks(hunks, limit: int) -> tuple[str, bool]:
    text = "\n".join(h.render() for h in hunks)
    if len(text) > limit:
        return text[:limit] + "\n[... diff troncato ...]", True
    return text, False
