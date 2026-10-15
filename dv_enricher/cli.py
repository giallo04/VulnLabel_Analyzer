from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .dataset import DatasetIndex
from .enrich import SCHEMA_VERSION, EnrichConfig, enrich_commit
from .nvd import NvdClient
from .repos import RepoProvider


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dv_enricher", description=__doc__)
    ap.add_argument("--dataset", required=True, help="diversevul_*.json oppure .zip")
    ap.add_argument("--out", "--output", dest="out", required=True, help="output JSONL (riprendibile)")
    ap.add_argument("--workdir", "--repos", dest="workdir", default="./work", help="cache: cloni git, NVD, ricerche")
    ap.add_argument("--project", action="append", help="filtra per progetto (ripetibile)")
    ap.add_argument("--hashes-file", help="file con un hash per riga (output del Sampler)")
    ap.add_argument("--sample", type=int, help="campione casuale di N funzioni")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo-map", help="JSON project -> [url] che estende/sovrascrive quello di default")
    ap.add_argument("--no-search", action="store_true", help="non usare la ricerca GitHub per hash")
    ap.add_argument("--nvd", action="store_true", help="interroga NVD per i CVE citati nel messaggio")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-hunk-chars", type=int, default=20000,
                    help="tetto su change.hunks (hunk interi del file, usati dal pre-filtro)")
    ap.add_argument("--max-function-diff-chars", type=int, default=20000,
                    help="tetto su change.function_diff (solo le righe della funzione, va al modello)")
    ap.add_argument("--refresh", action="store_true",
                    help="rielabora i record scritti da una versione vecchia dello schema "
                         "(schema_version < attuale): necessario per correggere i diff troncati")
    a = ap.parse_args(argv)

    print("Carico il dataset (streaming)...", file=sys.stderr)
    idx = DatasetIndex.load(a.dataset)
    hashes = Path(a.hashes_file).read_text().split() if a.hashes_file else None
    recs = idx.select(a.project, hashes, a.sample, a.seed)

    out = Path(a.out)
    done = set()
    if out.exists():
        with out.open() as f:
            for l in f:
                if not l.strip():
                    continue
                b = json.loads(l)
                if not a.refresh or b.get("schema_version", 1) >= SCHEMA_VERSION:
                    done.add(b["hash"])
    recs = [r for r in recs if r.hash not in done]
    by_commit: dict[str, list] = {}
    for r in recs:
        by_commit.setdefault(r.commit_id, []).append(r)
    print(f"{len(recs)} funzioni in {len(by_commit)} commit da processare ({len(done)} già fatte)", file=sys.stderr)

    workdir = Path(a.workdir)
    if workdir.name == "repos" and not (workdir / "repos").exists():
        workdir = workdir.parent
    provider = RepoProvider(workdir, a.repo_map, use_search=not a.no_search)
    nvd = NvdClient(workdir) if a.nvd else None
    cfg = EnrichConfig(max_hunk_chars=a.max_hunk_chars, max_function_diff_chars=a.max_function_diff_chars)
    lock, counts, t0 = threading.Lock(), Counter(), time.time()

    def work(cid, rs):
        try:
            return enrich_commit(cid, rs, idx.flagged[cid], idx.commit_total[cid], provider, cfg, nvd)
        except Exception as e:                      # un commit rotto non deve fermare il batch
            from .enrich import _bundle, _commit_skeleton
            sk = _commit_skeleton(cid, rs, idx.commit_total[cid], len(idx.flagged[cid]))
            return [_bundle(r, "error", sk, None, error=f"{type(e).__name__}: {e}") for r in rs]

    with out.open("a") as fo, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, c, rs) for c, rs in by_commit.items()]
        for n, fu in enumerate(as_completed(futs), 1):
            bundles = fu.result()
            with lock:
                for b in bundles:
                    fo.write(json.dumps(b, ensure_ascii=False) + "\n")
                    counts[b["status"]] += 1
                fo.flush()
            if n % 20 == 0 or n == len(futs):
                print(f"[{n}/{len(futs)} commit, {time.time()-t0:.0f}s] {dict(counts)}", file=sys.stderr)
    print(json.dumps(dict(counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
