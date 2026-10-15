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
from .diffparse import changed_in_range, parse_unified_diff, render_function_diff, shrink_lines
from .locate import FileView, Match, locate
from .nvd import NvdClient
from .repos import GitRepo, RepoProvider

# v2: change.function_diff = diff ristretto alla funzione (v1 passava hunk interi, troncati da inizio)
SCHEMA_VERSION = 2


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
            fdiff = render_function_diff(it.hunks, m.start, m.end, it.side)
            fdiff, fdiff_trunc, _ = shrink_lines(fdiff, cfg.max_function_diff_chars)
            length = m.end - m.start + 1
            b["change"] = {
                "deleted": len(it.deleted), "inserted": len(it.inserted), "changed": it.changed,
                "frac_changed": round(it.changed / length, 4),
                "share_of_source_churn": round(it.changed / src_churn, 4) if src_churn else None,
                "kind": F.classify_change(it.deleted, it.inserted),
                "rank_in_commit": rank[r.hash], "hunks": text, "hunks_truncated": trunc,
                "function_diff": fdiff, "function_diff_truncated": fdiff_trunc,
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


def _sibling(it: Located) -> dict:
    return {"hash": it.rec.hash, "name": it.name, "located": True, "file": it.match.path,
            "changed": it.changed, "kind": F.classify_change(it.deleted, it.inserted), "side": it.side}


def _render_hunks(hunks, limit: int) -> tuple[str, bool]:
    text = "\n".join(h.render() for h in hunks)
    if len(text) > limit:
        return text[:limit] + "\n[... diff troncato ...]", True
    return text, False
