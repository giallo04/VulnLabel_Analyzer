"""Accesso a git: clone parziali (blobless), risoluzione project -> repository, lettura blob."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true", "LC_ALL": "C"}
_DEFAULT_MAP = Path(__file__).with_name("repo_map.json")
GIT_TIMEOUT = float(os.environ.get("DV_GIT_TIMEOUT", "900"))   # secondi per singolo comando git


class GitError(RuntimeError):
    pass


class GitRepo:
    """Wrapper minimale su un repo bare (anche partial clone: i blob arrivano on demand)."""

    def __init__(self, path: str | Path, url: str | None = None) -> None:
        self.path = Path(path)
        self.url = url
        self._fetch_lock = threading.Lock()

    def git(self, *args: str, check: bool = True, input: bytes | None = None,
            timeout: float | None = GIT_TIMEOUT) -> subprocess.CompletedProcess:
        try:
            p = subprocess.run(["git", "--git-dir", str(self.path), *args],
                               capture_output=True, env=_ENV, input=input, timeout=timeout)
        except subprocess.TimeoutExpired as exc:      # clone blobless: blob scaricati uno a uno
            if not check:
                return subprocess.CompletedProcess(exc.cmd, 124, b"", b"timeout")
            raise GitError(f"git {' '.join(args[:2])}: timeout dopo {timeout:.0f}s") from exc
        if check and p.returncode != 0:
            raise GitError(f"git {' '.join(args)}: {p.stderr.decode(errors='replace').strip()}")
        return p

    def has_commit(self, sha: str) -> bool:
        return self.git("cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0

    def fetch_commit(self, sha: str) -> bool:
        """Scarica un commit non raggiungibile dal clone iniziale (branch cancellati, ecc.)."""
        with self._fetch_lock:
            if self.has_commit(sha):
                return True
            p = self.git("fetch", "--quiet", "--filter=blob:none", "origin", sha, check=False)
            return p.returncode == 0 and self.has_commit(sha)

    def parents(self, sha: str) -> list[str]:
        out = self.git("rev-list", "--parents", "-n", "1", sha).stdout.decode().split()
        return out[1:]

    def numstat(self, base: str, sha: str) -> list[tuple[int, int, str]]:
        out = self.git("diff", "--numstat", "-z", "--no-renames", base, sha).stdout.decode(errors="replace")
        rows = []
        for rec in filter(None, out.split("\0")):
            a, d, path = rec.split("\t", 2)
            rows.append((0 if a == "-" else int(a), 0 if d == "-" else int(d), path))  # '-' = binario
        return rows

    def diff_file(self, base: str, sha: str, path: str, context: int = 3) -> str:
        return self.git("diff", "--no-renames", "--no-color", f"-U{context}", base, sha, "--", path
                        ).stdout.decode("utf-8", errors="replace")

    def read_blobs(self, rev: str, paths: list[str]) -> dict[str, str | None]:
        """Legge molti file di una revisione con un solo processo `cat-file --batch`."""
        if not paths:
            return {}
        inp = "".join(f"{rev}:{p}\n" for p in paths).encode()
        out = self.git("cat-file", "--batch", input=inp).stdout
        res: dict[str, str | None] = {}
        i = 0
        for path in paths:
            j = out.index(b"\n", i)
            header = out[i:j].decode(errors="replace")
            i = j + 1
            if header.endswith(" missing") or header.endswith(" ambiguous"):
                res[path] = None
                continue
            _, typ, size = header.rsplit(" ", 2)
            size = int(size)
            data = out[i:i + size]
            i += size + 1
            res[path] = data.decode("utf-8", errors="replace") if typ == "blob" else None
        return res


def _http_json(url: str, headers: dict | None = None, tries: int = 3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < tries - 1:
                time.sleep(20 * (attempt + 1))          # rate limit
                continue
            return None
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    return None


class RepoProvider:
    """Dato (project, commit_id) restituisce un GitRepo che contiene quel commit.

    Ordine dei candidati: ultimo repo che ha funzionato per il progetto, poi la mappa
    statica (repo_map.json + eventuale override utente), poi la ricerca GitHub per hash.
    Il commit viene sempre verificato: i nomi dei progetti nel dataset sono ambigui
    (es. "net", "server", "linux-2.6").
    """

    def __init__(self, workdir: str | Path, repo_map_path: str | Path | None = None,
                 github_token: str | None = None, use_search: bool = True) -> None:
        self.workdir = Path(workdir)
        (self.workdir / "repos").mkdir(parents=True, exist_ok=True)
        self.repo_map: dict[str, list[str]] = json.loads(_DEFAULT_MAP.read_text())
        if repo_map_path:
            self.repo_map.update(json.loads(Path(repo_map_path).read_text()))
        self.token = github_token or os.environ.get("GITHUB_TOKEN")
        self.use_search = use_search
        self._known: dict[str, str] = {}
        self._repos: dict[str, GitRepo | None] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._glock = threading.Lock()
        self._search_file = self.workdir / "search_cache.json"
        self._search_cache: dict[str, list[str]] = (
            json.loads(self._search_file.read_text()) if self._search_file.exists() else {})

    # -- clone -------------------------------------------------------------------------
    def _lock_for(self, url: str) -> threading.Lock:
        with self._glock:
            return self._locks.setdefault(url, threading.Lock())

    def _clone(self, url: str) -> GitRepo | None:
        with self._lock_for(url):
            if url in self._repos:
                return self._repos[url]
            name = url.rstrip("/").removesuffix(".git").split("/")[-2:]
            dest = self.workdir / "repos" / ("__".join(name) + ".git")
            if not dest.exists():
                tmp = dest.with_suffix(".partial")
                p = subprocess.run(["git", "clone", "--bare", "--quiet", "--filter=blob:none", url, str(tmp)],
                                   capture_output=True, env=_ENV)
                if p.returncode != 0:
                    self._repos[url] = None
                    return None
                tmp.rename(dest)
            self._repos[url] = GitRepo(dest, url)
            return self._repos[url]

    # -- ricerca -----------------------------------------------------------------------
    def _search(self, commit: str) -> list[str]:
        if commit in self._search_cache:
            return self._search_cache[commit]
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "dv-enricher"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = _http_json(f"https://api.github.com/search/commits?q=hash:{commit}", headers)
        urls: list[str] = []
        if data:
            items = sorted(data.get("items", []), key=lambda it: it["repository"].get("fork", False))
            urls = [it["repository"]["html_url"] for it in items]
        with self._glock:
            self._search_cache[commit] = urls
            self._search_file.write_text(json.dumps(self._search_cache))
        return urls

    # -- API pubblica ------------------------------------------------------------------
    def get_repo(self, project: str, commit: str) -> tuple[GitRepo | None, str]:
        """Ritorna (repo, status): status in {ok, repo_unresolved, commit_not_found}."""
        cands: list[str] = []
        if project in self._known:
            cands.append(self._known[project])
        cands += [u for u in self.repo_map.get(project, []) if u not in cands]
        tried_any_repo = False
        for phase in ("static", "search"):
            if phase == "search":
                if not self.use_search:
                    break
                cands = [u for u in self._search(commit) if u not in cands][:3]
            for url in cands:
                repo = self._clone(url)
                if repo is None:
                    continue
                tried_any_repo = True
                if repo.has_commit(commit) or repo.fetch_commit(commit):
                    self._known[project] = url
                    return repo, "ok"
        return None, ("commit_not_found" if tried_any_repo else "repo_unresolved")
