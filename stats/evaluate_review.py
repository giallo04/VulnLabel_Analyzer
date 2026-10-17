#!/usr/bin/env python3
"""
Valuta il tool confrontando i verdetti LLM con l'annotazione manuale.

Convenzioni
- colonna human_vulnerable:  ok = funzione vulnerabile, no = non vulnerabile
- classe positiva = VULNERABILE (target 1)
- rumore nel dataset = funzione etichettata 1 ma in realta' non vulnerabile;
  il tool lo rileva quando predice 0, quindi per il "rilevamento del rumore":
      precision_rumore = NPV,  recall_rumore = specificity

Uso:
    python evaluate_review.py review_sample.csv
    python evaluate_review.py review_sample.csv --report report.md --json metriche.json
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

YES = {"ok", "si", "sì", "yes", "y", "1", "true", "v", "vuln", "vulnerabile"}
NO = {"no", "n", "0", "false", "nv", "non vulnerabile", "non_vulnerabile"}


# ---------------------------------------------------------------- metriche
def div(a, b):
    return a / b if b else None


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)


def compute(tp, fp, fn, tn, ci=True):
    n = tp + fp + fn + tn
    acc = div(tp + tn, n)
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    spec, npv = div(tn, tn + fp), div(tn, tn + fn)

    def f1(p, r):
        return div(2 * p * r, p + r) if p is not None and r is not None else None

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else None
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else None
    kappa = (acc - pe) / (1 - pe) if n and pe is not None and 1 - pe else None
    m = {
        "n": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": acc, "precision": prec, "recall": rec, "specificity": spec,
        "npv": npv, "f1": f1(prec, rec), "f1_neg": f1(npv, spec),
        "balanced_accuracy": (rec + spec) / 2 if rec is not None and spec is not None else None,
        "mcc": mcc, "kappa": kappa,
        "fpr": None if spec is None else 1 - spec,
        "fnr": None if rec is None else 1 - rec,
        "prevalence_vuln_human": div(tp + fn, n),
    }
    if ci:
        for key, (k, d) in {"accuracy": (tp + tn, n), "precision": (tp, tp + fp),
                            "recall": (tp, tp + fn), "specificity": (tn, tn + fp),
                            "npv": (tn, tn + fn)}.items():
            m[key + "_ci95"] = wilson(k, d)
    return m


# ---------------------------------------------------------------- I/O
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        head = f.readline()
        f.seek(0)
        delim = ";" if head.count(";") > head.count(",") else ","
        return list(csv.DictReader(f, delimiter=delim))


def parse_rows(rows):
    data, unannotated, errors = [], 0, []
    for r in rows:
        h = (r.get("human_vulnerable") or "").strip().lower()
        v = (r.get("llm_verdict") or "").strip().upper()
        if v not in ("VULNERABILE", "NON_VULNERABILE"):
            errors.append(f"id {r.get('id')}: llm_verdict non valido ({v!r})")
            continue
        if not h:
            unannotated += 1
            continue
        if h in YES:
            human = 1
        elif h in NO:
            human = 0
        else:
            errors.append(f"id {r.get('id')}: valore human_vulnerable non riconosciuto ({h!r})")
            continue
        data.append({**r, "human": human, "pred": 1 if v == "VULNERABILE" else 0})
    return data, unannotated, errors


def cells(items):
    tp = sum(1 for d in items if d["pred"] == 1 and d["human"] == 1)
    fp = sum(1 for d in items if d["pred"] == 1 and d["human"] == 0)
    fn = sum(1 for d in items if d["pred"] == 0 and d["human"] == 1)
    tn = sum(1 for d in items if d["pred"] == 0 and d["human"] == 0)
    return tp, fp, fn, tn


# ---------------------------------------------------------------- formattazione
def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def fc(x):
    return f"{x:.1f}" if isinstance(x, float) and not float(x).is_integer() else f"{int(x)}"


def fci(ci):
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


METRIC_COLS = [("accuracy", "Acc"), ("precision", "Prec"), ("recall", "Rec"),
               ("specificity", "Spec"), ("npv", "NPV"), ("f1", "F1"), ("f1_neg", "F1-neg"),
               ("balanced_accuracy", "BalAcc"), ("mcc", "MCC"), ("kappa", "Kappa")]


def md_table(named_metrics):
    head = ["Gruppo", "n", "TP", "FP", "FN", "TN"] + [c[1] for c in METRIC_COLS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, m in named_metrics:
        row = [name, fc(m["n"]), fc(m["TP"]), fc(m["FP"]), fc(m["FN"]), fc(m["TN"])]
        row += [f(m[k]) for k, _ in METRIC_COLS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV annotato")
    ap.add_argument("--meta", help="file .meta.json (default: <csv>.meta.json)")
    ap.add_argument("--report", default="review_report.md")
    ap.add_argument("--json", default="review_metrics.json")
    args = ap.parse_args()

    data, unannotated, errors = parse_rows(read_csv(args.csv))
    for e in errors:
        print(f"[errore] {e}", file=sys.stderr)
    if not data:
        sys.exit("Nessuna riga annotata valida.")

    meta_path = Path(args.meta) if args.meta else Path(args.csv).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    if meta is None:
        print(f"[info] {meta_path} non trovato: salto le stime pesate sulla popolazione.", file=sys.stderr)

    by_cat = defaultdict(list)
    for d in data:
        by_cat[d["llm_category"]].append(d)
    cats = sorted(by_cat)

    # --- metriche per categoria e aggregate (pooled, non pesate)
    per_cat = {c: compute(*cells(by_cat[c])) for c in cats}
    pooled = compute(*cells(data))

    # --- stime pesate sulla popolazione (stratificazione per categoria)
    weighted, noise = None, {}
    if meta:
        w_cells, N_tot, pred0 = [0.0] * 4, 0, 0
        h0_weighted = 0.0
        for c in cats:
            pop = meta["population"].get(c)
            if not pop:
                continue
            n_c = len(by_cat[c])
            w = pop["n"] / n_c
            for i, v in enumerate(cells(by_cat[c])):
                w_cells[i] += w * v
            N_tot += pop["n"]
            pred0 += pop["n_pred_non_vuln"]
            h0_weighted += w * sum(1 for d in by_cat[c] if d["human"] == 0)
        if N_tot:
            weighted = compute(*w_cells, ci=False)
            noise = {
                "tool_noise_rate_population": pred0 / N_tot,
                "human_noise_rate_weighted": h0_weighted / N_tot,
            }
    n_h0 = sum(1 for d in data if d["human"] == 0)
    noise["human_noise_rate_pooled"] = n_h0 / len(data)
    noise["human_noise_rate_pooled_ci95"] = wilson(n_h0, len(data))

    # --- disaccordi
    disagreements = [d for d in data if d["pred"] != d["human"]]

    # --- report markdown
    L = []
    L.append("# Report di validazione manuale\n")
    L.append(f"- Righe annotate valide: **{len(data)}** (non compilate: {unannotated}, errori: {len(errors)})")
    L.append(f"- Convenzione: ok = vulnerabile, no = non vulnerabile; classe positiva = VULNERABILE")
    if meta:
        L.append(f"- Seed campionamento: {meta['seed']}, funzioni per categoria: {meta['n_per_category']}")
    L.append("")
    L.append("## Matrice di confusione complessiva (non pesata)\n")
    L.append("| | Umano: vulnerabile | Umano: non vulnerabile |\n|---|---|---|")
    L.append(f"| **LLM: vulnerabile** | TP = {pooled['TP']} | FP = {pooled['FP']} |")
    L.append(f"| **LLM: non vulnerabile** | FN = {pooled['FN']} | TN = {pooled['TN']} |\n")

    L.append("## Metriche per categoria e totali\n")
    rows = [(c, per_cat[c]) for c in cats] + [("TOTALE (non pesato)", pooled)]
    if weighted:
        rows.append(("TOTALE (pesato per popolazione)", weighted))
    L.append(md_table(rows))
    L.append("\n`n/a` = metrica non definita (denominatore nullo): in una categoria che il tool "
             "mappa sempre sulla stessa classe, precision/recall/specificity/NPV sono in parte "
             "indefinite; per queste categorie l'indicatore significativo e' l'accuracy "
             "(= quota di verdetti LLM confermati dall'analisi manuale).\n")

    L.append("## Intervalli di confidenza 95% (Wilson)\n")
    L.append("| Gruppo | Accuracy | Precision | Recall | Specificity | NPV |\n|---|---|---|---|---|---|")
    for name, m in rows:
        if "accuracy_ci95" in m:
            L.append(f"| {name} | " + " | ".join(
                f"{f(m[k])} {fci(m[k + '_ci95'])}" for k in
                ("accuracy", "precision", "recall", "specificity", "npv")) + " |")
    L.append("")

    L.append("## Metriche aggiuntive (totale non pesato)\n")
    for k, label in [("fpr", "False positive rate"), ("fnr", "False negative rate"),
                     ("prevalence_vuln_human", "Prevalenza vulnerabili (umano)")]:
        L.append(f"- {label}: {f(pooled[k])}")
    L.append("")

    L.append("## Rilevamento del rumore nel dataset\n")
    L.append("Rumore = funzione etichettata vulnerabile (1) ma non vulnerabile secondo l'analisi manuale. "
             "Per il rilevamento del rumore: precision = NPV, recall = specificity.\n")
    L.append(f"- Rumore stimato dall'analisi manuale (campione): {f(noise['human_noise_rate_pooled'])} "
             f"{fci(noise['human_noise_rate_pooled_ci95'])}  (campione stratificato: non e' una stima della popolazione)")
    if "human_noise_rate_weighted" in noise:
        L.append(f"- Rumore stimato dall'analisi manuale (pesato per popolazione): {f(noise['human_noise_rate_weighted'])}")
        L.append(f"- Rumore rilevato dal tool (predetto 0) sull'intera popolazione: {f(noise['tool_noise_rate_population'])}")
    L.append(f"- Precision rilevamento rumore (NPV): {f(pooled['npv'])} {fci(pooled.get('npv_ci95'))}")
    L.append(f"- Recall rilevamento rumore (specificity): {f(pooled['specificity'])} {fci(pooled.get('specificity_ci95'))}\n")

    L.append(f"## Disaccordi LLM vs umano ({len(disagreements)})\n")
    if disagreements:
        L.append("| id | project | funzione | categoria LLM | LLM | umano | link | note |\n|---|---|---|---|---|---|---|---|")
        for d in disagreements:
            L.append(f"| {d.get('id')} | {d.get('project')} | `{d.get('func_name')}` | {d['llm_category']} | "
                     f"{d['llm_verdict']} | {'vulnerabile' if d['human'] else 'non vulnerabile'} | "
                     f"{d.get('commit_url') or d.get('commit_id')} | {(d.get('notes') or '').replace('|', '/')} |")
    else:
        L.append("Nessun disaccordo.")
    L.append("")

    if meta:
        L.append("## Note metodologiche\n")
        L.append("- Una sola funzione per commit nell'intero campione (osservazioni indipendenti). "
                 "Le funzioni sono estratte in modo casuale; i commit con piu' funzioni hanno quindi "
                 "probabilita' di selezione maggiore, coerente con i pesi per numero di funzioni.")
        L.append("- Le stime pesate scalano ogni categoria per N_categoria / n_annotate; "
                 "con n = 10 per categoria sono indicative e hanno intervalli ampi (vedi i Wilson sopra).")
    report = "\n".join(L)

    Path(args.report).write_text(report, encoding="utf-8")
    Path(args.json).write_text(json.dumps({
        "n_annotated": len(data), "n_unannotated": unannotated, "errors": errors,
        "per_category": per_cat, "pooled": pooled, "weighted": weighted, "noise": noise,
        "disagreements": [{k: d.get(k) for k in ("id", "hash", "project", "func_name", "llm_category",
                                                 "llm_verdict", "human", "commit_url", "notes")}
                          for d in disagreements],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report)
    print(f"\nReport -> {args.report}\nJSON   -> {args.json}")


if __name__ == "__main__":
    main()