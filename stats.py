#!/usr/bin/env python3
"""stats — l'unico modulo di misura: accuratezza delle label del dataset e qualità del classificatore.

Sostituisce report.py, evaluate.py e core/evaluation.py. Solo stdlib (niente pandas).

Tre comandi
-----------
  summary        accuratezza stimata delle label dal checkpoint (nessuna revisione manuale)
  export-review  CSV compatti per la verifica manuale (review_llm.csv / review_static.csv)
  review         metriche dalla revisione manuale: accuratezza del classificatore, matrice di
                 confusione, precision/recall/F1, MCC, kappa e accuratezza *corretta* del dataset

  python stats.py summary       --checkpoint data/output/gemini/checkpoint.jsonl
  python stats.py export-review --checkpoint data/output/gemini/checkpoint.jsonl
  python stats.py review data/output/gemini/review_llm.csv data/output/gemini/review_static.csv \\
                         --checkpoint data/output/gemini/checkpoint.jsonl

Definizioni
-----------
  record valido      : niente parse_error e predicted_target in {0, 1}
                       (parse error e UNDETERMINABLE sono esclusi dal denominatore e riportati a parte)
  rumore (is_noise)  : il classificatore contraddice l'etichetta originale (predicted != original)
  accuratezza label  : record NON rumorosi / record validi, con intervallo di Wilson al 95%

Verdetti della revisione manuale (colonna `verdetto` dei CSV di revisione):
  ok     -> il classificatore ha ragione  (la sua predizione è corretta)
  ERROR  -> il classificatore ha torto    (la verità è l'opposto della predizione)
  idk    -> non so decidere               (escluso dalle metriche principali)
  vuoto  -> riga non revisionata          (esclusa)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.checkpoint import load_checkpoint_deduped

STATIC_METHOD = "deterministic"
LLM_METHOD = "llm"

SUMMARY_JSON = "accuracy_summary.json"
SUMMARY_TXT = "stats_report.txt"
REVIEW_JSON = "review_metrics.json"
REVIEW_TXT = "review_metrics.txt"

OK, ERR, IDK = "ok", "error", "idk"
VALID_VERDICTS = {OK, ERR, IDK}


# ===========================================================================
# Helpers
# ===========================================================================
def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "si", "sì")
    return bool(v)


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """Intervallo di confidenza di Wilson (95%) per la proporzione k/n."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _pct(x: Optional[float], digits: int = 1) -> str:
    return "n/d" if x is None or x != x else f"{100 * x:.{digits}f}%"


def _ci_txt(ci: Optional[Sequence[float]]) -> str:
    return "" if not ci else f"  IC95 [{_pct(ci[0])} – {_pct(ci[1])}]"


def _clean_nan(obj: Any) -> Any:
    """NaN non è JSON valido: li sostituisce con null."""
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean_nan(obj), indent=2, ensure_ascii=False), encoding="utf-8")


# ===========================================================================
# 1. SUMMARY — accuratezza stimata dal classificatore
# ===========================================================================
def _is_valid(r: Dict[str, Any]) -> bool:
    return not _truthy(r.get("parse_error")) and _as_int(r.get("predicted_target")) in (0, 1)


def _is_noise(r: Dict[str, Any]) -> bool:
    orig, pred = _as_int(r.get("original_target")), _as_int(r.get("predicted_target"))
    if orig in (0, 1) and pred in (0, 1):
        return orig != pred
    return _truthy(r.get("is_noise"))


def _acc_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    noise = sum(1 for r in rows if _is_noise(r))
    clean = n - noise
    return {
        "n": n,
        "noise": noise,
        "clean": clean,
        "accuracy": clean / n if n else None,
        "ci95": wilson(clean, n),
    }


def compute_summary(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Statistiche del dataset dai risultati del classificatore.

    Va chiamata sull'insieme completo dei record (checkpoint deduplicato), mai su un sottoinsieme
    già filtrato.
    """
    by_hash: Dict[str, Dict[str, Any]] = {}
    for r in results:                       # dedup: vince l'ultima riga per hash
        by_hash[str(r.get("hash"))] = r
    records = list(by_hash.values())

    parse_errors = [r for r in records if _truthy(r.get("parse_error"))]
    undeterminable = [r for r in records
                      if not _truthy(r.get("parse_error")) and _as_int(r.get("predicted_target")) == -1]
    valid = [r for r in records if _is_valid(r)]

    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_project: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_commit: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    categories: Counter = Counter()
    for r in valid:
        by_method[STATIC_METHOD if r.get("method") == STATIC_METHOD else LLM_METHOD].append(r)
        by_project[str(r.get("project", ""))].append(r)
        by_commit[str(r.get("commit_id") or "?")].append(r)
        categories[str(r.get("category", "?"))] += 1

    # rumore per commit (cluster-aware: i commit giganti non dominano la media)
    rates = [sum(1 for r in rs if _is_noise(r)) / len(rs) for rs in by_commit.values()]
    mean = sum(rates) / len(rates) if rates else None
    std = (math.sqrt(sum((x - mean) ** 2 for x in rates) / len(rates)) if rates else None)

    n_valid = len(valid)
    return {
        "total_records": len(records),
        "parse_errors_excluded": len(parse_errors),
        "undeterminable_excluded": len(undeterminable),
        "overall": _acc_stats(valid),
        "by_method": {m: _acc_stats(rows) for m, rows in sorted(by_method.items())},
        "by_project": {p: _acc_stats(rows) for p, rows in sorted(by_project.items())},
        "by_category": {c: {"n": k, "share": k / n_valid if n_valid else None}
                        for c, k in sorted(categories.items(), key=lambda kv: -kv[1])},
        "by_commit": {"n_commits": len(rates), "mean_noise_rate": mean, "std_noise_rate": std},
    }


def _fmt_stats(label: str, s: Dict[str, Any]) -> str:
    return (f"  {label:<24} n={s['n']:<6} rumore={s['noise']:<6} "
            f"accuratezza={_pct(s['accuracy'], 2)}{_ci_txt(s['ci95'])}")


def format_summary(summary: Dict[str, Any]) -> str:
    o = summary["overall"]
    lines = [
        "=" * 70,
        "  ACCURATEZZA DELLE LABEL DEL DATASET (stima dal classificatore)",
        "=" * 70,
        f"  Record nel checkpoint        : {summary['total_records']}",
        f"  Esclusi: parse error         : {summary['parse_errors_excluded']}",
        f"  Esclusi: UNDETERMINABLE      : {summary['undeterminable_excluded']}",
        f"  Record validi (denominatore) : {o['n']}",
        "-" * 70,
        _fmt_stats("TOTALE", o),
        "",
        "Per metodo:",
        *[_fmt_stats(m, s) for m, s in summary["by_method"].items()],
        "",
        "Per categoria (sui record validi):",
        *[f"  {c:<28} {v['n']:>6}  ({_pct(v['share'])})" for c, v in summary["by_category"].items()],
        "",
        "Per progetto:",
        *[_fmt_stats(p, s) for p, s in summary["by_project"].items()],
    ]
    c = summary["by_commit"]
    if c["n_commits"]:
        lines += ["", f"Rumore per commit: {_pct(c['mean_noise_rate'])} ± {_pct(c['std_noise_rate'])}"
                      f"  (n={c['n_commits']} commit)"]
    lines.append("=" * 70)
    return "\n".join(lines)


def summarize(results: List[Dict[str, Any]], out_dir: Optional[Path] = None,
              echo: bool = True) -> Dict[str, Any]:
    """Calcola il riepilogo; se `out_dir` è dato salva JSON + testo; se `echo` lo stampa."""
    summary = compute_summary(results)
    text = format_summary(summary)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _dump_json(out_dir / SUMMARY_JSON, summary)
        (out_dir / SUMMARY_TXT).write_text(text + "\n", encoding="utf-8")
    if echo:
        print(text)
    return summary


# ===========================================================================
# 2. RESULTS / EXPORT — file prodotti dal classificatore e CSV di revisione
# ===========================================================================
_RESULT_COLUMNS = [
    "hash", "commit_id", "project", "func_name", "file",
    "original_target", "predicted_target", "is_noise",
    "category", "method", "model",
    "step_1_commit_analysis", "step_2_function_diff_analysis", "step_3_conclusion",
    "rank_in_commit", "share_of_source_churn", "frac_changed",
    "kind", "siblings_count", "name_in_message", "cwe_matches_nvd", "cwe_missing",
    "parse_error", "parse_error_reason",
]


def save_results_csv(results: List[Dict[str, Any]], path: Path) -> None:
    """Tabella completa dei risultati (una riga per funzione)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_RESULT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in _RESULT_COLUMNS})


def save_results_jsonl(results: List[Dict[str, Any]], path: Path) -> None:
    """Come il CSV ma in JSONL e con `raw_output` (risposta grezza del modello)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_LLM_REVIEW_FIELDS = [
    "id", "project", "commit_id", "func_name", "file",
    "original_target", "predicted_target", "category", "is_noise",
    "rank_in_commit", "siblings_count", "commit_url", "verdetto", "note",
]
_STATIC_REVIEW_FIELDS = [
    "id", "project", "commit_id", "func_name", "file",
    "original_target", "predicted_target", "category", "is_noise",
    "kind", "commit_url", "verdetto", "note",
]


def _commit_url(r: Dict[str, Any], repo_map: Dict[str, str]) -> str:
    repo = repo_map.get(str(r.get("project")))
    return f"https://github.com/{repo}/commit/{r.get('commit_id')}" if repo else ""


def _write_review(path: Path, fields: List[str], rows: List[Dict[str, Any]],
                  repo_map: Dict[str, str]) -> int:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["id"] = r.get("hash")
            row["file"] = r.get("file") or ""
            row["commit_url"] = _commit_url(r, repo_map)
            row["verdetto"] = ""      # da compilare: ok / ERROR / idk
            row["note"] = ""
            w.writerow(row)
    return len(rows)


def export_review(results: Iterable[Dict[str, Any]], out_dir: Path, *,
                  repo_map: Optional[Dict[str, str]] = None,
                  only_disagreements: bool = False,
                  include_parse_errors: bool = False) -> Dict[str, int]:
    """Scrive review_llm.csv, review_static.csv e il riepilogo di accuratezza.

    L'accuratezza è calcolata su tutti i record, prima dei filtri di esportazione.
    """
    by_hash: Dict[str, Dict[str, Any]] = {}
    for r in results:
        by_hash[str(r.get("hash"))] = r
    summarize(list(by_hash.values()), out_dir, echo=False)

    llm = [r for r in by_hash.values() if r.get("method") != STATIC_METHOD]
    static = [r for r in by_hash.values() if r.get("method") == STATIC_METHOD]
    if not include_parse_errors:
        llm = [r for r in llm if not _truthy(r.get("parse_error"))]
    if only_disagreements:
        llm = [r for r in llm if _truthy(r.get("is_noise"))]
        static = [r for r in static if _truthy(r.get("is_noise"))]

    # rumore sospetto per primo, poi raggruppato per progetto/commit
    llm.sort(key=lambda r: (0 if _truthy(r.get("is_noise")) else 1, str(r.get("project", "")),
                            str(r.get("commit_id", "")), r.get("rank_in_commit") or 0))
    static.sort(key=lambda r: (str(r.get("kind", "")), str(r.get("project", "")),
                               str(r.get("commit_id", ""))))

    out_dir.mkdir(parents=True, exist_ok=True)
    repo_map = repo_map or {}
    return {
        "llm": _write_review(out_dir / "review_llm.csv", _LLM_REVIEW_FIELDS, llm, repo_map),
        "static": _write_review(out_dir / "review_static.csv", _STATIC_REVIEW_FIELDS, static, repo_map),
    }


# ===========================================================================
# 3. REVIEW — metriche dai verdetti manuali
# ===========================================================================
def _normalize_verdict(v: Any) -> Optional[str]:
    """None = non revisionata; 'invalid' = valore non riconosciuto."""
    s = "" if v is None else str(v).strip().lower()
    if not s:
        return None
    return s if s in VALID_VERDICTS else "invalid"


def _class_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    if tp:
        f1 = _div(2 * precision * recall, precision + recall)
    else:
        f1 = 0.0 if (tp + fp + fn) else float("nan")
    return {"precision": precision, "recall": recall, "specificity": _div(tn, tn + fp),
            "f1": f1, "support": tp + fn}


def load_review_rows(paths: Sequence[str], verdict_col: str = "verdetto") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with Path(p).open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            missing = [c for c in ("predicted_target", verdict_col)
                       if c not in (reader.fieldnames or [])]
            if missing:
                raise SystemExit(f"{p}: colonne mancanti {missing}. "
                                 f"Disponibili: {reader.fieldnames}")
            rows.extend(reader)
    return rows


def review_metrics(rows: List[Dict[str, Any]], *, verdict_col: str = "verdetto",
                   group_by: Sequence[str] = ("category", "project"),
                   summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Metriche del classificatore dai verdetti manuali + accuratezza corretta del dataset.

    Dai verdetti si ricava la "verità implicita":  ok -> verità = pred;  ERROR -> verità = 1 - pred.
    Classe positiva principale: "noise" (pred = 0, il classificatore dice che la label è sbagliata).
    """
    for r in rows:
        r["_v"] = _normalize_verdict(r.get(verdict_col))
        r["_pred"] = _as_int(r.get("predicted_target"))

    total = len(rows)
    n_unrev = sum(1 for r in rows if r["_v"] is None)
    n_invalid = sum(1 for r in rows if r["_v"] == "invalid")
    # righe senza predizione 0/1 (UNDETERMINABLE) non hanno una "verità implicita": escluse
    usable = [r for r in rows if r["_pred"] in (0, 1)]
    n_no_pred = total - len(usable)
    n_ok = sum(1 for r in usable if r["_v"] == OK)
    n_err = sum(1 for r in usable if r["_v"] == ERR)
    n_idk = sum(1 for r in usable if r["_v"] == IDK)
    n_decided, n_reviewed = n_ok + n_err, n_ok + n_err + n_idk

    cov = {"total": total, "unreviewed": n_unrev, "invalid": n_invalid,
           "no_prediction": n_no_pred, "reviewed": n_reviewed,
           "ok": n_ok, "error": n_err, "idk": n_idk}
    if n_decided == 0:
        return {"coverage": cov, "computable": False}

    out: Dict[str, Any] = {"coverage": cov, "computable": True}
    out["accuracy"] = {
        "value": _div(n_ok, n_decided), "ci95": wilson(n_ok, n_decided),
        "idk_as_error": _div(n_ok, n_reviewed), "idk_as_ok": _div(n_ok + n_idk, n_reviewed),
    }

    # --- matrice di confusione sulla verità implicita ---
    d = [r for r in usable if r["_v"] in (OK, ERR)]
    for r in d:
        r["_truth"] = r["_pred"] if r["_v"] == OK else 1 - r["_pred"]
    tp = sum(1 for r in d if r["_truth"] == 0 and r["_pred"] == 0)
    fp = sum(1 for r in d if r["_truth"] == 1 and r["_pred"] == 0)
    fn = sum(1 for r in d if r["_truth"] == 0 and r["_pred"] == 1)
    tn = sum(1 for r in d if r["_truth"] == 1 and r["_pred"] == 1)
    m_noise = _class_metrics(tp, fp, fn, tn)
    m_ok = _class_metrics(tn, fn, fp, tp)
    n = tp + fp + fn + tn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else float("nan")
    out["confusion"] = {"tp_noise": tp, "fp_noise": fp, "fn_noise": fn, "tn_noise": tn}
    out["noise"], out["label_ok"] = m_noise, m_ok
    out["noise_precision_ci95"] = wilson(tp, tp + fp)
    out["macro_f1"] = (m_noise["f1"] + m_ok["f1"]) / 2
    out["balanced_accuracy"] = (m_noise["recall"] + m_ok["recall"]) / 2
    out["mcc"] = _div(tp * tn - fp * fn, den)
    out["kappa"] = _div((tp + tn) / n - pe, 1 - pe) if n else float("nan")

    # --- dettaglio per gruppo ---
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for col in group_by:
        if not any(col in r for r in rows):
            groups[col] = []
            continue
        agg: Dict[str, Counter] = defaultdict(Counter)
        for r in usable:
            if r["_v"] in VALID_VERDICTS:
                agg[str(r.get(col, ""))][r["_v"]] += 1
        tab = [{"group": g, "reviewed": sum(c.values()), "ok": c[OK], "error": c[ERR], "idk": c[IDK],
                "accuracy": _div(c[OK], c[OK] + c[ERR])} for g, c in agg.items()]
        groups[col] = sorted(tab, key=lambda x: -x["reviewed"])
    out["groups"] = groups

    # --- accuratezza CORRETTA del dataset (serve il riepilogo del checkpoint) ---
    # flagged = il classificatore contraddice la label.  Tra i flagged, "ok" = rumore vero;
    # tra i non-flagged, "ERROR" = rumore mancato.
    def _flagged(r: Dict[str, Any]) -> bool:
        orig = _as_int(r.get("original_target"))
        return r["_pred"] != orig if orig in (0, 1) else r["_pred"] == 0

    fl = [r for r in d if _flagged(r)]
    nf = [r for r in d if not _flagged(r)]
    prec = _div(sum(1 for r in fl if r["_v"] == OK), len(fl)) if fl else None
    missed = _div(sum(1 for r in nf if r["_v"] == ERR), len(nf)) if nf else None
    corrected: Dict[str, Any] = {
        "flag_precision": prec,
        "flag_precision_ci95": wilson(sum(1 for r in fl if r["_v"] == OK), len(fl)),
        "missed_noise_rate": missed,
        "missed_noise_rate_ci95": wilson(sum(1 for r in nf if r["_v"] == ERR), len(nf)),
        "corrected_accuracy": None, "estimated_noise": None,
        "assumes_no_missed_noise": prec is not None and missed is None,
    }
    if summary and prec is not None and summary["overall"]["n"]:
        ov = summary["overall"]
        est = ov["noise"] * prec + (ov["clean"] * missed if missed is not None else 0.0)
        corrected["estimated_noise"] = est
        corrected["corrected_accuracy"] = 1 - est / ov["n"]
    out["corrected_dataset"] = corrected

    if n_unrev:
        out["unreviewed_pred_counts"] = dict(Counter(
            str(r["_pred"]) for r in rows if r["_v"] is None))
    return out


def format_review(m: Dict[str, Any], group_by: Sequence[str]) -> str:
    c = m["coverage"]
    L = ["=" * 66, "COPERTURA DELLA REVISIONE", "=" * 66,
         f"Righe totali           : {c['total']}",
         f"Non revisionate        : {c['unreviewed']}  ({_pct(_div(c['unreviewed'], c['total']))})"]
    if c["invalid"]:
        L.append(f"Verdetti non validi    : {c['invalid']}  <-- ESCLUSI, controlla il CSV")
    if c["no_prediction"]:
        L.append(f"Senza predizione 0/1   : {c['no_prediction']}  (UNDETERMINABLE, esclusi)")
    L += [f"Revisionate            : {c['reviewed']}",
          f"  ok                   : {c['ok']}",
          f"  ERROR                : {c['error']}",
          f"  idk                  : {c['idk']}  ({_pct(_div(c['idk'], c['reviewed']))} delle revisionate)"]
    if not m["computable"]:
        L += ["", "Nessuna riga con verdetto ok/ERROR: impossibile calcolare le metriche."]
        return "\n".join(L)

    a = m["accuracy"]
    L += ["", "=" * 66, "ACCURATEZZA DEL CLASSIFICATORE", "=" * 66,
          f"Accuracy (ok / (ok+ERROR))         : {_pct(a['value'])}{_ci_txt(a['ci95'])}",
          f"Error rate                         : {_pct(1 - a['value'])}",
          f"Accuracy, idk = errore (limite inf): {_pct(a['idk_as_error'])}",
          f"Accuracy, idk = ok     (limite sup): {_pct(a['idk_as_ok'])}"]

    cf = m["confusion"]
    L += ["", "=" * 66, "MATRICE DI CONFUSIONE (verità implicita dai verdetti)", "=" * 66,
          f"{'':<24}{'pred: label OK (1)':>20}{'pred: noise (0)':>18}",
          f"{'verità: label OK (1)':<24}{cf['tn_noise']:>20}{cf['fp_noise']:>18}",
          f"{'verità: noise (0)':<24}{cf['fn_noise']:>20}{cf['tp_noise']:>18}",
          "", f"{'classe':<22}{'precision':>11}{'recall':>9}{'F1':>8}{'support':>9}"]
    for name, k in (("noise (pred=0)", m["noise"]), ("label OK (pred=1)", m["label_ok"])):
        L.append(f"{name:<22}{_pct(k['precision']):>11}{_pct(k['recall']):>9}"
                 f"{_pct(k['f1']):>8}{k['support']:>9}")
    L += ["", f"Macro-F1              : {_pct(m['macro_f1'])}",
          f"Balanced accuracy     : {_pct(m['balanced_accuracy'])}",
          f"MCC                   : {m['mcc']:.3f}",
          f"Cohen's kappa         : {m['kappa']:.3f}",
          f"Precision noise{_ci_txt(m['noise_precision_ci95'])}"]

    cd = m["corrected_dataset"]
    L += ["", "=" * 66, "ACCURATEZZA CORRETTA DELLE LABEL DEL DATASET", "=" * 66,
          f"Precisione del flag rumore (flagged)   : {_pct(cd['flag_precision'])}"
          f"{_ci_txt(cd['flag_precision_ci95'])}",
          f"Rumore mancato (non flagged verificati): {_pct(cd['missed_noise_rate'])}"
          f"{_ci_txt(cd['missed_noise_rate_ci95'])}"]
    if cd["corrected_accuracy"] is not None:
        L.append(f"Accuratezza corretta                   : {_pct(cd['corrected_accuracy'], 2)}")
        if cd["assumes_no_missed_noise"]:
            L.append("NB: nessun record non flagged verificato -> si assume 0 rumore mancato; "
                     "la stima è un limite superiore dell'accuratezza.")
    else:
        L.append("(passa --checkpoint per ottenere l'accuratezza corretta del dataset)")

    for col in group_by:
        tab = m["groups"].get(col)
        L += ["", "=" * 66, f"DETTAGLIO PER {col.upper()}", "=" * 66]
        if tab is None or not tab:
            L.append(f"[!] colonna '{col}' assente, dettaglio saltato")
            continue
        L.append(f"{col[:28]:<30}{'rev.':>6}{'ok':>6}{'ERROR':>7}{'idk':>5}{'accuracy':>10}")
        for t in tab:
            L.append(f"{t['group'][:28]:<30}{t['reviewed']:>6}{t['ok']:>6}{t['error']:>7}"
                     f"{t['idk']:>5}{_pct(t['accuracy']):>10}")

    if c["unreviewed"]:
        L += ["", f"[!] {c['unreviewed']} righe non revisionate (predizioni: "
                  f"{m.get('unreviewed_pred_counts', {})}). Se non sono un campione casuale, "
                  "precision/recall della classe interessata possono essere distorte."]
    return "\n".join(L)


# ===========================================================================
# CLI
# ===========================================================================
def _cmd_summary(a: argparse.Namespace) -> int:
    ck = Path(a.checkpoint)
    if not ck.exists():
        raise SystemExit(f"Checkpoint non trovato: {ck}")
    records = load_checkpoint_deduped(ck)
    if not records:
        raise SystemExit(f"Checkpoint vuoto: {ck}")
    out_dir = Path(a.out_dir) if a.out_dir else ck.parent
    summarize(records, out_dir)
    print(f"\n→ {out_dir / SUMMARY_JSON}\n→ {out_dir / SUMMARY_TXT}", file=sys.stderr)
    return 0


def _cmd_export(a: argparse.Namespace) -> int:
    ck = Path(a.checkpoint)
    if not ck.exists():
        raise SystemExit(f"Checkpoint non trovato: {ck}")
    repo_map = json.loads(Path(a.repo_map).read_text(encoding="utf-8")) if a.repo_map else None
    out_dir = Path(a.out_dir) if a.out_dir else ck.parent
    counts = export_review(load_checkpoint_deduped(ck), out_dir, repo_map=repo_map,
                           only_disagreements=a.only_disagreements,
                           include_parse_errors=a.include_parse_errors)
    print(f"review_llm.csv: {counts['llm']} righe | review_static.csv: {counts['static']} righe "
          f"→ {out_dir}", file=sys.stderr)
    print("Compila la colonna `verdetto` (ok / ERROR / idk), poi lancia: python stats.py review ...",
          file=sys.stderr)
    return 0


def _cmd_review(a: argparse.Namespace) -> int:
    rows = load_review_rows(a.csv, a.verdict_col)
    summary = None
    if a.checkpoint:
        ck = Path(a.checkpoint)
        if not ck.exists():
            raise SystemExit(f"Checkpoint non trovato: {ck}")
        summary = compute_summary(load_checkpoint_deduped(ck))
    m = review_metrics(rows, verdict_col=a.verdict_col, group_by=a.group_by, summary=summary)
    text = format_review(m, a.group_by)
    out_dir = Path(a.out_dir) if a.out_dir else Path(a.csv[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REVIEW_TXT).write_text(text + "\n", encoding="utf-8")
    _dump_json(out_dir / REVIEW_JSON, m)
    print(text)
    print(f"\n→ {out_dir / REVIEW_TXT}\n→ {out_dir / REVIEW_JSON}", file=sys.stderr)
    return 0 if m["computable"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Misura l'accuratezza delle label del dataset (DiverseVul) e la qualità "
                    "del classificatore.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_ck = "data/output/gemini/checkpoint.jsonl"

    s = sub.add_parser("summary", help="accuratezza stimata dal checkpoint",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s.add_argument("--checkpoint", default=default_ck)
    s.add_argument("--out-dir", default=None, help="default: cartella del checkpoint")
    s.set_defaults(fn=_cmd_summary)

    e = sub.add_parser("export-review", help="CSV per la verifica manuale",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    e.add_argument("--checkpoint", default=default_ck)
    e.add_argument("--out-dir", default=None, help="default: cartella del checkpoint")
    e.add_argument("--repo-map", default=None,
                   help='JSON {"iperf": "esnet/iperf", ...} per generare commit_url')
    e.add_argument("--only-disagreements", action="store_true",
                   help="solo record con is_noise=True")
    e.add_argument("--include-parse-errors", action="store_true")
    e.set_defaults(fn=_cmd_export)

    r = sub.add_parser("review", help="metriche dai verdetti manuali",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r.add_argument("csv", nargs="+", help="uno o più CSV di revisione compilati")
    r.add_argument("--checkpoint", default=None,
                   help="se dato, calcola anche l'accuratezza corretta del dataset")
    r.add_argument("--verdict-col", default="verdetto")
    r.add_argument("--group-by", nargs="*", default=["category", "project"],
                   help="colonne per le tabelle di dettaglio")
    r.add_argument("--out-dir", default=None, help="default: cartella del primo CSV")
    r.set_defaults(fn=_cmd_review)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
