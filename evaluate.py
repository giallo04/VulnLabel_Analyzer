#!/usr/bin/env python3
"""
Evaluate a label-analyzer checkpoint (JSONL) against a human-reviewed CSV.

Usage
-----
    python evaluate_checkpoint.py checkpoint.jsonl review_fixed.csv
    python evaluate_checkpoint.py checkpoint.jsonl review_fixed.csv \
        --abstain skip --json report.json --errors-csv errors.csv

Inputs
------
checkpoint.jsonl : one JSON object per line. Keys used:
                   hash (join key), category, predicted_target, parse_error.
review_fixed.csv : columns used: id (join key), category, verdetto, project,
                   func_name, commit_id, commit_url, note.

How ground truth is derived from the CSV
----------------------------------------
The CSV `category` column is the category an earlier run assigned, and
`verdetto` is the human review of that category:

  verdetto == ok      -> the CSV category is correct.
                         true category = CSV category (exact).
                         true binary   = 1 if CORRECT_LABEL else 0 (noise).
  verdetto == ERROR   -> the CSV category is wrong, correct one not recorded.
                         acceptable categories = every category except the CSV one.
                         true binary = 0 (noise) if the CSV category was
                         CORRECT_LABEL; unknown (row skipped for binary metrics)
                         if the CSV category was a WRONG_LABEL_* one.
  verdetto == IDK / ? -> reviewer unsure; excluded from all metrics.

Binary task: positive class = "noise" (function NOT a genuine part of the
vulnerability fix, i.e. any WRONG_LABEL_*). CORRECT_LABEL -> "real".

Metrics
-------
1. Binary (real vs noise): confusion matrix, accuracy (+95% Wilson CI),
   per-class precision / recall / F1, macro-F1, balanced accuracy, Cohen's kappa.
2. Category, "compatible" accuracy: prediction is in the acceptable set
   (all reviewed rows).
3. Category, "exact" accuracy + per-class P/R/F1 + confusion matrix: only rows
   whose true category is fully known (verdetto == ok).
4. Per-project breakdown, and a list of every disagreement.

Abstentions (UNDETERMINABLE, parse errors, unknown labels) are handled with
--abstain: `wrong` (default) counts them as misses, `skip` drops them.
Reviewed rows absent from the checkpoint are always excluded and reported.

Standard library only.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict

CORRECT = "CORRECT_LABEL"
UNDET = "UNDETERMINABLE"
SPREAD = "WRONG_LABEL_SPREAD"
CONSIST = "WRONG_LABEL_CONSISTENCY"
IRREL = "WRONG_LABEL_IRRELEVANT"
FEATURE = "WRONG_LABEL_FEATURE"
ALL_CATS = [CORRECT, SPREAD, CONSIST, IRREL, FEATURE]
ABSTAIN = "ABSTAIN"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_truth(path):
    rows, skipped, all_ids = [], Counter(), set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rid = (r.get("id") or "").strip()
            all_ids.add(rid)
            base = (r.get("category") or "").strip().upper()
            verdict = (r.get("verdetto") or "").strip()
            rec = {
                "id": rid,
                "project": (r.get("project") or "").strip(),
                "func_name": (r.get("func_name") or "").strip(),
                "commit_id": (r.get("commit_id") or "").strip(),
                "commit_url": (r.get("commit_url") or "").strip(),
                "csv_category": base,
                "verdetto": verdict,
                "note": (r.get("note") or "").strip(),
            }
            v = verdict.lower()
            if v == "ok":
                rec["exact"] = base
                rec["gt_set"] = {base}
                rec["gt_bin"] = 1 if base == CORRECT else 0
            elif v == "error":
                rec["exact"] = None
                rec["gt_set"] = set(ALL_CATS) - {base}
                rec["gt_bin"] = 0 if base == CORRECT else None
            else:
                skipped[verdict or "<empty>"] += 1
                continue
            rows.append(rec)
    return rows, skipped, all_ids


def load_checkpoint(path):
    preds, bad_lines, dups = {}, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if not isinstance(rec, dict):
                bad_lines += 1
                continue
            key = rec.get("hash", rec.get("id"))
            if key is None:
                bad_lines += 1
                continue
            key = str(key)
            if key in preds:
                dups += 1  # later lines win (retries append to checkpoints)
            preds[key] = rec
    return preds, bad_lines, dups


def parse_pred(rec):
    """Return (pred_cat, pred_bin, status). status in ok|abstain|invalid."""
    raw = rec.get("category")
    cat = str(raw).strip().upper() if raw not in (None, "") else None
    pt = rec.get("predicted_target")
    if cat in ALL_CATS:
        return cat, (1 if cat == CORRECT else 0), "ok"
    if cat == UNDET or pt == -1:
        return None, None, "abstain"
    if cat is None and pt in (0, 1):  # no category, only a binary prediction
        return None, pt, "invalid"
    return None, None, "invalid"


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def safe_div(a, b):
    return a / b if b else float("nan")


def prf(tp, fp, fn):
    p, r = safe_div(tp, tp + fp), safe_div(tp, tp + fn)
    f = safe_div(2 * p * r, p + r) if not (math.isnan(p) or math.isnan(r)) else float("nan")
    return p, r, f


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x * 100:5.1f}%" if pct else f"{x:5.3f}"


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(g == p for g, p in pairs) / n
    gc, pc = Counter(g for g, _ in pairs), Counter(p for _, p in pairs)
    pe = sum(gc[k] * pc[k] for k in set(gc) | set(pc)) / (n * n)
    return safe_div(po - pe, 1 - pe)


def table(header, rows, widths=None):
    cols = list(zip(*([header] + rows))) if rows else [[h] for h in header]
    widths = widths or [max(len(str(c)) for c in col) for col in cols]
    line = lambda r: "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                               for i, (c, w) in enumerate(zip(r, widths)))
    out = [line(header), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(truth, preds, abstain_policy):
    scored, missing = [], []
    for t in truth:
        rec = preds.get(t["id"])
        if rec is None:
            missing.append(t)
            continue
        cat, pbin, status = parse_pred(rec)
        scored.append({**t, "pred_cat": cat, "pred_bin": pbin, "status": status})

    counts = Counter(s["status"] for s in scored)
    drop_abstain = abstain_policy == "skip"

    # ---- binary --------------------------------------------------------
    bin_pairs = []  # (gt_noise, pred_noise) with noise = 1
    bin_abstained = 0
    for s in scored:
        if s["gt_bin"] is None:
            continue
        gt_noise = 1 - s["gt_bin"]
        if s["pred_bin"] is None:
            bin_abstained += 1
            if drop_abstain:
                continue
            pred_noise = 1 - gt_noise  # forced miss
        else:
            pred_noise = 1 - s["pred_bin"]
        bin_pairs.append((gt_noise, pred_noise))

    tp = sum(g == 1 and p == 1 for g, p in bin_pairs)
    fp = sum(g == 0 and p == 1 for g, p in bin_pairs)
    fn = sum(g == 1 and p == 0 for g, p in bin_pairs)
    tn = sum(g == 0 and p == 0 for g, p in bin_pairs)
    n_bin = len(bin_pairs)
    p_n, r_n, f_n = prf(tp, fp, fn)
    p_r, r_r, f_r = prf(tn, fn, fp)
    binary = {
        "n": n_bin, "abstained": bin_abstained,
        "tp_noise": tp, "fp_noise": fp, "fn_noise": fn, "tn_noise": tn,
        "accuracy": safe_div(tp + tn, n_bin),
        "accuracy_ci95": wilson(tp + tn, n_bin),
        "noise": {"precision": p_n, "recall": r_n, "f1": f_n},
        "real": {"precision": p_r, "recall": r_r, "f1": f_r},
        "macro_f1": safe_div(f_n + f_r, 2),
        "balanced_accuracy": safe_div(r_n + r_r, 2),
        "kappa": kappa(bin_pairs),
    }

    # ---- category: compatible (all reviewed rows) --------------------------
    comp_hits = comp_n = 0
    for s in scored:
        if s["pred_cat"] is None and drop_abstain:
            continue
        comp_n += 1
        comp_hits += s["pred_cat"] in s["gt_set"]
    compatible = {"n": comp_n, "correct": comp_hits,
                  "accuracy": safe_div(comp_hits, comp_n),
                  "accuracy_ci95": wilson(comp_hits, comp_n)}

    # ---- category: exact (rows with fully known truth) --------------------
    labels = ALL_CATS + [ABSTAIN]
    conf = {g: Counter() for g in ALL_CATS}
    ex_n = ex_hits = 0
    for s in scored:
        if s["exact"] is None:
            continue
        pc = s["pred_cat"]
        if pc is None:
            if drop_abstain:
                continue
            pc = ABSTAIN
        conf[s["exact"]][pc] += 1
        ex_n += 1
        ex_hits += pc == s["exact"]
    per_class = {}
    for c in ALL_CATS:
        tp_c = conf[c][c]
        fn_c = sum(conf[c].values()) - tp_c
        fp_c = sum(conf[g][c] for g in ALL_CATS if g != c)
        p, r, f = prf(tp_c, fp_c, fn_c)
        per_class[c] = {"support": sum(conf[c].values()), "precision": p, "recall": r, "f1": f}
    present = [c for c in ALL_CATS if per_class[c]["support"] > 0]
    exact = {"n": ex_n, "correct": ex_hits,
             "accuracy": safe_div(ex_hits, ex_n),
             "accuracy_ci95": wilson(ex_hits, ex_n),
             "macro_f1": safe_div(sum(per_class[c]["f1"] for c in present), len(present)) if present else float("nan"),
             "per_class": per_class,
             "confusion": {g: dict(conf[g]) for g in ALL_CATS}}

    # ---- per project ------------------------------------------------------
    byproj = defaultdict(lambda: {"n": 0, "bin_ok": 0, "bin_n": 0, "comp_ok": 0, "comp_n": 0})
    for s in scored:
        d = byproj[s["project"]]
        if not (s["pred_cat"] is None and drop_abstain):
            d["comp_n"] += 1
            d["comp_ok"] += s["pred_cat"] in s["gt_set"]
        if s["gt_bin"] is not None and not (s["pred_bin"] is None and drop_abstain):
            d["bin_n"] += 1
            d["bin_ok"] += (s["pred_bin"] == s["gt_bin"])
        d["n"] += 1

    # ---- disagreements ----------------------------------------------------
    errors = []
    for s in scored:
        bin_bad = s["gt_bin"] is not None and s["pred_bin"] != s["gt_bin"]
        cat_bad = s["pred_cat"] not in s["gt_set"]
        if bin_bad or cat_bad:
            errors.append({**s, "binary_wrong": bin_bad, "category_wrong": cat_bad})

    return {"scored": scored, "missing": missing, "status_counts": dict(counts),
            "binary": binary, "compatible": compatible, "exact": exact,
            "by_project": byproj, "errors": errors, "labels": labels}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(res, args, truth_n, skipped, ck_n, extra, bad_lines, dups):
    b, c, e = res["binary"], res["compatible"], res["exact"]
    sc = res["status_counts"]
    print("=" * 74)
    print("CHECKPOINT EVALUATION")
    print("=" * 74)
    print(f"checkpoint : {args.checkpoint}  ({ck_n} unique records, {dups} duplicate keys "
          f"overwritten, {bad_lines} unreadable lines)")
    print(f"truth      : {args.truth}  ({truth_n} usable rows; excluded as unsure: "
          f"{dict(skipped) or 0})")
    print(f"abstain    : policy = {args.abstain}")
    print()
    print("COVERAGE")
    print(f"  reviewed rows found in checkpoint : {len(res['scored'])} / {truth_n}")
    print(f"  reviewed rows MISSING from ckpt   : {len(res['missing'])}")
    print(f"  checkpoint rows not in the CSV    : {extra} (ignored; unsure IDK/? rows are in the CSV but excluded)")
    print(f"  valid predictions / abstain / invalid : "
          f"{sc.get('ok', 0)} / {sc.get('abstain', 0)} / {sc.get('invalid', 0)}")
    if res["missing"]:
        print("  WARNING: metrics below cover only the rows present in the checkpoint.")

    print("\n1) BINARY: real (CORRECT_LABEL) vs noise (WRONG_LABEL_*), positive = noise")
    lo, hi = b["accuracy_ci95"]
    print(f"  scored rows: {b['n']}   (abstained on binary: {b['abstained']}; "
          f"ERROR-on-WRONG rows without binary truth are skipped)")
    print(f"  accuracy   : {fmt(b['accuracy'], True)}   95% CI [{fmt(lo, True)}, {fmt(hi, True)}]")
    print(f"  balanced acc {fmt(b['balanced_accuracy'], True)} | macro-F1 {fmt(b['macro_f1'])} "
          f"| Cohen's kappa {fmt(b['kappa'])}")
    print("\n  confusion (rows = human truth, cols = checkpoint)")
    print(table(["", "pred noise", "pred real"],
                [["true noise", b["tp_noise"], b["fn_noise"]],
                 ["true real", b["fp_noise"], b["tn_noise"]]]))
    print()
    print(table(["class", "precision", "recall", "F1"],
                [["noise", fmt(b["noise"]["precision"]), fmt(b["noise"]["recall"]), fmt(b["noise"]["f1"])],
                 ["real", fmt(b["real"]["precision"]), fmt(b["real"]["recall"]), fmt(b["real"]["f1"])]]))

    print("\n2) CATEGORY, compatible match (prediction within the human-acceptable set)")
    lo, hi = c["accuracy_ci95"]
    print(f"  {c['correct']} / {c['n']} = {fmt(c['accuracy'], True)}   95% CI [{fmt(lo, True)}, {fmt(hi, True)}]")

    print("\n3) CATEGORY, exact match (rows with known true category, verdetto = ok)")
    lo, hi = e["accuracy_ci95"]
    print(f"  {e['correct']} / {e['n']} = {fmt(e['accuracy'], True)}   95% CI [{fmt(lo, True)}, {fmt(hi, True)}]"
          f"   macro-F1 {fmt(e['macro_f1'])}")
    print()
    print(table(["class", "support", "precision", "recall", "F1"],
                [[k.replace("WRONG_LABEL_", "WL_"), v["support"], fmt(v["precision"]),
                  fmt(v["recall"]), fmt(v["f1"])]
                 for k, v in e["per_class"].items()]))
    short = lambda s: s.replace("WRONG_LABEL_", "WL_").replace("CORRECT_LABEL", "CORRECT")
    cols = [x for x in res["labels"] if x != ABSTAIN or any(e["confusion"][g].get(ABSTAIN) for g in ALL_CATS)]
    print("\n  confusion (rows = human truth, cols = checkpoint)")
    print(table([""] + [short(x) for x in cols],
                [[short(g)] + [e["confusion"][g].get(x, 0) for x in cols] for g in ALL_CATS]))

    print("\n4) PER PROJECT (projects with >= 3 reviewed rows)")
    rows = []
    for p, d in sorted(res["by_project"].items(), key=lambda kv: -kv[1]["n"]):
        if d["n"] < 3:
            continue
        rows.append([p or "<none>", d["n"], fmt(safe_div(d["bin_ok"], d["bin_n"]), True),
                     fmt(safe_div(d["comp_ok"], d["comp_n"]), True)])
    print(table(["project", "n", "binary acc", "category acc"], rows))

    print(f"\n5) DISAGREEMENTS: {len(res['errors'])} rows "
          f"(binary wrong: {sum(x['binary_wrong'] for x in res['errors'])}, "
          f"category outside acceptable set: {sum(x['category_wrong'] for x in res['errors'])})")
    shown = res["errors"][: args.show_errors]
    if shown:
        print(table(["project", "function", "human", "csv_cat", "checkpoint"],
                    [[x["project"][:14], x["func_name"][:30] or "<none>", x["verdetto"],
                      short(x["csv_category"]), short(x["pred_cat"] or x["status"].upper())]
                     for x in shown]))
        if len(res["errors"]) > len(shown):
            print(f"  ... {len(res['errors']) - len(shown)} more (use --errors-csv for the full list)")


def write_errors_csv(path, errors):
    cols = ["id", "project", "func_name", "commit_id", "verdetto", "csv_category",
            "pred_category", "pred_status", "binary_wrong", "category_wrong", "note", "commit_url"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for x in errors:
            w.writerow([x["id"], x["project"], x["func_name"], x["commit_id"], x["verdetto"],
                        x["csv_category"], x["pred_cat"] or "", x["status"],
                        x["binary_wrong"], x["category_wrong"], x["note"], x["commit_url"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="checkpoint.jsonl produced by the label analyzer")
    ap.add_argument("truth", nargs="?", default="review_fixed.csv", help="human-reviewed CSV")
    ap.add_argument("--abstain", choices=["wrong", "skip"], default="wrong",
                    help="how to treat UNDETERMINABLE / invalid predictions (default: wrong)")
    ap.add_argument("--json", metavar="PATH", help="write the full metrics as JSON")
    ap.add_argument("--errors-csv", metavar="PATH", help="write every disagreement to CSV")
    ap.add_argument("--show-errors", type=int, default=25, help="rows of disagreements to print")
    args = ap.parse_args()

    truth, skipped, all_ids = load_truth(args.truth)
    preds, bad_lines, dups = load_checkpoint(args.checkpoint)
    res = evaluate(truth, preds, args.abstain)
    extra = len(set(preds) - all_ids)
    print_report(res, args, len(truth), skipped, len(preds), extra, bad_lines, dups)

    if args.errors_csv:
        write_errors_csv(args.errors_csv, res["errors"])
        print(f"\nwrote {args.errors_csv}")
    if args.json:
        out = {k: res[k] for k in ("status_counts", "binary", "compatible", "exact")}
        out["by_project"] = dict(res["by_project"])
        out["coverage"] = {"truth_rows": len(truth), "scored": len(res["scored"]),
                           "missing": len(res["missing"]), "extra_in_checkpoint": extra,
                           "excluded_unsure": dict(skipped), "abstain_policy": args.abstain}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=lambda o: None if o != o else str(o))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())