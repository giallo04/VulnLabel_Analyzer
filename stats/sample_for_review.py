#!/usr/bin/env python3
"""
Campionamento stratificato per la validazione manuale dei verdetti del tool.

- N funzioni per categoria (default 10)
- al piu' UNA funzione per commit in tutto il campione (anche tra categorie
  diverse), cosi' le osservazioni sono indipendenti
- output: CSV da compilare + file <output>.meta.json (seed e dimensioni delle
  popolazioni per categoria, usato dallo script di valutazione)

Uso:
    python sample_for_review.py risultati.jsonl -o review_sample.csv
    python sample_for_review.py risultati.jsonl -n 10 --seed 55 --delimiter ";"
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# project (minuscolo) -> "owner/repo" oppure URL base completo.
# Estendibile con --repo-map file.json
REPO_URLS = {
    "linux": "torvalds/linux",
    "tensorflow": "tensorflow/tensorflow",
    "ncurses": "ThomasDickey/ncurses-snapshots",
    "psutil": "giampaolo/psutil",
    "qpdf": "qpdf/qpdf",
    "librsvg": "GNOME/librsvg",
    "jasper": "jasper-software/jasper",
    "unbound": "NLnetLabs/unbound",
    "bdwgc": "ivmai/bdwgc",
    "imagemagick": "ImageMagick/ImageMagick",
    "ffmpeg": "FFmpeg/FFmpeg",
    "qemu": "qemu/qemu",
    "php-src": "php/php-src",
    "openssl": "openssl/openssl",
    "curl": "curl/curl",
    "vim": "vim/vim",
    "redis": "redis/redis",
    "libarchive": "libarchive/libarchive",
    "gpac": "gpac/gpac",
    "radare2": "radareorg/radare2",
    "mruby": "mruby/mruby",
    "wireshark": "wireshark/wireshark",
    "systemd": "systemd/systemd",
    "nginx": "nginx/nginx",
    "httpd": "apache/httpd",
    "openssh-portable": "openssh/openssh-portable",
    "git": "git/git",
    "samba": "samba-team/samba",
    "openjpeg": "uclouvain/openjpeg",
    "libpng": "glennrp/libpng",
    "libxml2": "GNOME/libxml2",
    "v8": "v8/v8",
    "chromium": "chromium/chromium",
}


def load_records(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    records = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[warn] riga {i} non valida, ignorata: {e}", file=sys.stderr)
    return records


def commit_url(project, sha, repo_map):
    base = repo_map.get(str(project or "").lower())
    if not base or not sha:
        return ""
    if base.startswith("http"):
        return f"{base.rstrip('/')}/commit/{sha}"
    return f"https://github.com/{base}/commit/{sha}"


def eligible(rec, excluded):
    if rec.get("parse_error"):
        return False
    if rec.get("category") in excluded or not rec.get("category"):
        return False
    return rec.get("predicted_target") in (0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="file JSONL (o JSON array) con i verdetti")
    ap.add_argument("-o", "--output", default="review_sample.csv")
    ap.add_argument("-n", type=int, default=10, help="funzioni per categoria (default 10)")
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--exclude", nargs="*", default=["UNDETERMINABLE"],
                    help="categorie da non campionare (default: UNDETERMINABLE)")
    ap.add_argument("--max-per-project", type=int, default=0,
                    help="massimo di funzioni dallo stesso progetto nel campione (0 = nessun limite)")
    ap.add_argument("--blind", action="store_true",
                    help="non includere la motivazione dell'LLM nel CSV")
    ap.add_argument("--repo-map", nargs="+",
                    help="uno o piu' JSON {project: 'owner/repo' | URL | [URL]} per estendere/sovrascrivere i link")
    ap.add_argument("--delimiter", default=",", help="delimitatore CSV (usa ';' per Excel in italiano)")
    args = ap.parse_args()

    repo_map = dict(REPO_URLS)
    for mp in args.repo_map or []:
        for k, v in json.loads(Path(mp).read_text(encoding="utf-8")).items():
            if isinstance(v, list):  # formato {"progetto": ["https://..."]}
                v = v[0] if v else ""
            if v:
                repo_map[k.lower()] = v

    records = [r for r in load_records(args.input) if eligible(r, set(args.exclude))]
    if not records:
        sys.exit("Nessun record utilizzabile nel file di input.")

    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    rng = random.Random(args.seed)
    # categorie piu' piccole per prime: sono le piu' vincolate dal requisito
    # "un solo campione per commit"
    order = sorted(by_cat, key=lambda c: len({r["commit_id"] for r in by_cat[c]}))

    used_commits, project_count, selected = set(), Counter(), []
    for cat in order:
        cands = by_cat[cat][:]
        rng.shuffle(cands)
        picked, seen = [], set()
        for r in cands:
            c = r["commit_id"]
            if c in used_commits or c in seen:
                continue
            if args.max_per_project and project_count[r["project"]] >= args.max_per_project:
                continue
            picked.append(r)
            seen.add(c)
            project_count[r["project"]] += 1
            if len(picked) == args.n:
                break
        used_commits |= seen
        if len(picked) < args.n:
            print(f"[warn] categoria {cat}: solo {len(picked)}/{args.n} funzioni "
                  f"disponibili con commit distinti", file=sys.stderr)
        selected.extend(picked)

    rng.shuffle(selected)  # ordine casuale: non raggruppa per categoria

    fields = ["id", "hash", "project", "commit_url", "commit_id", "file", "func_name",
              "llm_category", "llm_verdict"]
    if not args.blind:
        fields.append("llm_rationale")
    fields += ["human_vulnerable", "notes"]

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=args.delimiter)
        w.writeheader()
        for i, r in enumerate(selected, 1):
            row = {
                "id": i,
                "hash": r["hash"],
                "project": r.get("project", ""),
                "commit_url": commit_url(r.get("project"), r["commit_id"], repo_map),
                "commit_id": r["commit_id"],
                "file": r.get("file", ""),
                "func_name": r.get("func_name", ""),
                "llm_category": r["category"],
                "llm_verdict": "VULNERABILE" if r["predicted_target"] == 1 else "NON_VULNERABILE",
                "human_vulnerable": "",   # <- ok = vulnerabile, no = non vulnerabile
                "notes": "",
            }
            if not args.blind:
                row["llm_rationale"] = r.get("step_2_flaw", "")
            w.writerow(row)

    meta = {
        "seed": args.seed,
        "n_per_category": args.n,
        "population": {
            c: {"n": len(rs),
                "n_pred_vuln": sum(1 for r in rs if r["predicted_target"] == 1),
                "n_pred_non_vuln": sum(1 for r in rs if r["predicted_target"] == 0),
                "n_commits": len({r["commit_id"] for r in rs})}
            for c, rs in by_cat.items()
        },
        "sampled": dict(Counter(r["category"] for r in selected)),
    }
    meta_path = Path(args.output).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    missing = sum(1 for r in selected if not commit_url(r.get("project"), r["commit_id"], repo_map))
    print(f"Campionate {len(selected)} funzioni da {len(used_commits)} commit distinti -> {args.output}")
    print(f"Metadati -> {meta_path}")
    for c in sorted(meta["sampled"]):
        print(f"  {c:28s} campionate {meta['sampled'][c]:3d} / popolazione {meta['population'][c]['n']}")
    if missing:
        unknown = sorted({str(r.get('project')) for r in selected
                          if not commit_url(r.get('project'), r['commit_id'], repo_map)})
        print(f"[info] link mancante per {missing} righe (progetti: {', '.join(unknown)}). "
              f"Aggiungili con --repo-map.", file=sys.stderr)


if __name__ == "__main__":
    main()