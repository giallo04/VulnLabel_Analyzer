#!/usr/bin/env python3
"""
build_dataset.py - Ricostruisce DiverseVul con etichette corrette dall'audit.

  - POSITIVI: tutte le funzioni di result.jsonl con predicted_target=1 (dopo i filtri)
  - NEGATIVI: campionati dal dataset originale (target=0) per ottenere la
    frazione di positivi richiesta con --pos-frac

Esempi:
  python build_dataset.py --original diversevul.json --audit result.jsonl  --pos-frac 0.4 --output new_dataset.jsonl

  python build_dataset.py --original diversevul.json --audit result.jsonl  --pos-frac 0.5 --min-agreement 1.0 --exclude-flagged --keep-meta --output bal.jsonl
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter


# ================================================================ animazione
class Progress:
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    CYAN, GREEN, DIM, RESET = "\033[36m", "\033[32m", "\033[2m", "\033[0m"

    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stderr.isatty()
        self.frame = 0
        self.last = 0.0
        self.t0 = time.time()

    def _write(self, s):
        cols = shutil.get_terminal_size((100, 20)).columns
        sys.stderr.write("\r\033[K" + s[: cols - 1])
        sys.stderr.flush()

    def start(self, label):
        self.t0 = time.time()
        self.last = 0.0
        if self.enabled:
            self.update(label, None, None, force=True)
        else:
            print(f"... {label}", file=sys.stderr)

    def update(self, label, done, total, force=False):
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last < 0.07:
            return
        self.last = now
        self.frame = (self.frame + 1) % len(self.FRAMES)
        spin = f"{self.CYAN}{self.FRAMES[self.frame]}{self.RESET}"
        el = now - self.t0
        if total:
            frac = min(done / total, 1.0)
            filled = int(28 * frac)
            bar = "█" * filled + "░" * (28 - filled)
            rate = done / el if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            self._write(f"{spin} {label:<24} {self.CYAN}{bar}{self.RESET} {frac*100:5.1f}% "
                        f"{self.DIM}{el:5.1f}s  ETA {eta:4.0f}s{self.RESET}")
        else:
            self._write(f"{spin} {label:<24} {self.DIM}{el:5.1f}s{self.RESET}")

    def done(self, label, extra=""):
        el = time.time() - self.t0
        msg = f"{self.GREEN}✔{self.RESET} {label:<24} {self.DIM}{el:5.1f}s{self.RESET} {extra}"
        if self.enabled:
            self._write(msg)
            sys.stderr.write("\n")
        else:
            print(f"    ok ({el:.1f}s) {extra}", file=sys.stderr)


# ================================================================ utilità
def norm_hash(h):
    """Nell'originale hash è un intero, nell'audit una stringa."""
    try:
        return str(int(h))
    except (ValueError, TypeError):
        return str(h)


def iter_records(path, prog=None, label=""):
    """Legge JSONL o array JSON; yield (indice, record)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        first = f.read(1)
        f.seek(0)
        if first == b"[":
            data = json.load(f)
            n = len(data)
            for i, rec in enumerate(data):
                if prog and i % 256 == 0:
                    prog.update(label, i, n)
                yield i, rec
            return
        read = 0
        for i, line in enumerate(f):
            read += len(line)
            if prog and i % 256 == 0:
                prog.update(label, read, size)
            line = line.strip()
            if line:
                yield i, json.loads(line)


def load_audit(path, args, prog):
    """hash -> (predicted_target, riga) per le righe che passano i filtri."""
    excluded = set(args.exclude_category or [])
    audit, st = {}, Counter()
    for _, row in iter_records(path, prog, "Lettura audit"):
        st["audit_rows"] += 1
        if row.get("parse_error"):
            st["audit_skip_parse_error"] += 1
            continue
        if row.get("predicted_target") not in (0, 1):
            st["audit_skip_no_prediction"] += 1
            continue
        if args.exclude_flagged and (row.get("flags") or row.get("inconsistent")):
            st["audit_skip_flagged"] += 1
            continue
        if (row.get("agreement") or 0) < args.min_agreement:
            st["audit_skip_low_agreement"] += 1
            continue
        if row.get("category") in excluded:
            st["audit_skip_category"] += 1
            continue
        key = norm_hash(row["hash"])
        prev = audit.get(key)
        if prev is None or (row.get("agreement") or 0) > (prev[1].get("agreement") or 0):
            audit[key] = (int(row["predicted_target"]), row)
    st["audit_usable"] = len(audit)
    return audit, st


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Ricostruisce DiverseVul con etichette corrette.")
    ap.add_argument("--original", required=True, help="DiverseVul originale (JSONL o JSON array)")
    ap.add_argument("--audit", required=True, help="result.jsonl con l'audit dei target=1")
    ap.add_argument("--output", required=True, help="file di output (JSONL)")
    ap.add_argument("--pos-frac", type=float, required=True,
                    help="frazione di positivi nel dataset finale, in (0,1]. "
                         "Es. 0.4 = 40%% vulnerabili, 60%% sane")

    ap.add_argument("--allow-fewer", action="store_true",
                    help="se le sane non bastano, sottocampiona i positivi per mantenere la frazione "
                         "(default: errore)")
    ap.add_argument("--include-flipped", action="store_true",
                    help="aggiungi al pool dei negativi anche i falsi positivi corretti 1->0 "
                         "(default: i negativi vengono solo dalle sane originali)")

    ap.add_argument("--min-agreement", type=float, default=0.0, help="agreement minimo audit (0-1)")
    ap.add_argument("--exclude-flagged", action="store_true", help="scarta righe audit con flags/inconsistent")
    ap.add_argument("--exclude-category", nargs="*", default=None,
                    help="categorie audit da scartare, es. WRONG_LABEL_IRRELEVANT")

    ap.add_argument("--keep-meta", action="store_true",
                    help="aggiungi campi: original_target, audit_category, label_changed, source")
    ap.add_argument("--no-shuffle", action="store_true", help="non mescolare l'output")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="solo statistiche, nessuna scrittura")
    ap.add_argument("--no-animation", action="store_true", help="disattiva l'animazione")
    args = ap.parse_args()

    if not (0 < args.pos_frac <= 1):
        sys.exit("Errore: --pos-frac deve essere nell'intervallo (0, 1].")

    prog = Progress(enabled=not args.no_animation)
    rng = random.Random(args.seed)

    # --- 1) audit
    prog.start("Lettura audit")
    audit, astats = load_audit(args.audit, args, prog)
    prog.done("Lettura audit", f"{len(audit):,} righe utilizzabili")

    # --- 2) scansione originale: pool di positivi e negativi
    prog.start("Scansione originale")
    pos, neg, flipped = [], [], []   # liste di indici di riga
    seen_audit = set()
    st = Counter()
    for idx, rec in iter_records(args.original, prog, "Scansione originale"):
        st["original_rows"] += 1
        orig = int(rec["target"])
        if orig == 0:
            neg.append(idx)
            continue
        key = norm_hash(rec.get("hash"))
        entry = audit.get(key)
        if entry is None:
            st["vuln_not_in_audit_ignored"] += 1
            continue
        seen_audit.add(key)
        if entry[0] == 1:
            pos.append(idx)
        else:
            flipped.append(idx)
    st["audit_hash_not_in_original"] = len(audit) - len(seen_audit)
    st["flipped_1_to_0"] = len(flipped)
    prog.done("Scansione originale",
              f"{len(pos):,} positivi, {len(neg):,} sane, {len(flipped):,} corrette 1→0")

    # --- 3) calcolo quantità
    prog.start("Calcolo rapporto")
    neg_pool = neg + (flipped if args.include_flipped else [])
    flipped_set = set(flipped)
    P, avail_neg = len(pos), len(neg_pool)
    f = args.pos_frac

    if P == 0:
        prog.done("Calcolo rapporto", "FALLITO")
        sys.exit("Errore: nessun positivo disponibile dopo i filtri.")

    needed = round(P * (1 - f) / f)
    if needed > avail_neg:
        if not args.allow_fewer:
            prog.done("Calcolo rapporto", "FALLITO")
            max_p = int(avail_neg * f / (1 - f)) if f < 1 else P
            sys.exit(f"Errore: per avere {f:.0%} di positivi con {P:,} positivi servono {needed:,} "
                     f"negativi, ne sono disponibili {avail_neg:,}. "
                     f"Usa --allow-fewer (ridurrebbe i positivi a ~{max_p:,}) o abbassa --pos-frac.")
        P = int(avail_neg * f / (1 - f))
        needed = avail_neg if f < 1 else 0
        pos = rng.sample(pos, P)
    n_pos, n_neg = P, needed
    prog.done("Calcolo rapporto", f"positivi={n_pos:,}  negativi={n_neg:,}  "
                                  f"(frazione reale {n_pos / (n_pos + n_neg):.3f})")

    sel_neg = rng.sample(neg_pool, n_neg) if n_neg < avail_neg else neg_pool
    sel = {i: 1 for i in pos}
    sel.update({i: 0 for i in sel_neg})

    print("\n=== Statistiche ===", file=sys.stderr)
    for k, v in {**astats, **st}.items():
        print(f"  {k}: {v:,}", file=sys.stderr)
    print(f"  positivi usati: {n_pos:,}", file=sys.stderr)
    print(f"  negativi usati: {n_neg:,}  (di cui corretti 1→0: "
          f"{sum(1 for i in sel_neg if i in flipped_set):,})", file=sys.stderr)
    print(f"  totale: {n_pos + n_neg:,}\n", file=sys.stderr)

    if args.dry_run:
        return

    # --- 4) estrazione record
    prog.start("Estrazione campioni")
    out = []
    for idx, rec in iter_records(args.original, prog, "Estrazione campioni"):
        if idx not in sel:
            continue
        label = sel[idx]
        orig = int(rec["target"])
        if args.keep_meta:
            row = audit.get(norm_hash(rec.get("hash")))
            rec["original_target"] = orig
            rec["audit_category"] = row[1].get("category") if row else None
            rec["label_changed"] = label != orig
            rec["source"] = ("confirmed" if label == 1
                             else "flipped" if orig == 1 else "orig_neg")
        rec["target"] = label
        if label == 0 and orig == 1 and "cwe" in rec:
            rec["cwe"] = []
        out.append(rec)
    prog.done("Estrazione campioni", f"{len(out):,} record")

    if not args.no_shuffle:
        rng.shuffle(out)

    # --- 5) scrittura
    prog.start("Scrittura output")
    with open(args.output, "w", encoding="utf-8") as fh:
        for i, rec in enumerate(out):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 64 == 0:
                prog.update("Scrittura output", i, len(out))
    prog.done("Scrittura output", args.output)

    c = Counter(r["target"] for r in out)
    print(f"\nFatto: {len(out):,} record (target0={c[0]:,}, target1={c[1]:,}, "
          f"frazione positivi {c[1] / len(out):.3f})", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\n\033[33mInterrotto.\033[0m\n")
        sys.exit(130)