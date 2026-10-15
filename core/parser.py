"""Parser della risposta LLM — tassonomia DiverseVul.

Schema atteso:
  category  : CORRECT_LABEL | WRONG_LABEL_SPREAD | WRONG_LABEL_CONSISTENCY
               | WRONG_LABEL_IRRELEVANT | UNDETERMINABLE
  predicted_target derivato da category (non dichiarato dal modello).

Robustezza:
- Valida category contro l'insieme ammesso; restituzione di parse_error se invalida
- Gestisce null, liste JSON, confidence_score come stringa ("90%")
- Salva sempre raw_output e tutti e 3 gli step
- parse_error_reason strutturato
"""
from __future__ import annotations

import json
import re

from .prompting import CATEGORY_TO_TARGET, VALID_CATEGORIES


def parse_llm_response(raw_response: str) -> dict:
    """Valida e normalizza la risposta JSON dell'LLM.

    Restituisce un dict con chiavi:
        category, predicted_target,
        step_1_commit_analysis, step_2_function_diff_analysis, step_3_conclusion,
        raw_output, parse_error, parse_error_reason
    """
    clean = _strip_markdown(raw_response)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        return _error(raw_response, f"JSON decode error: {exc}")

    # Gestisce wrapping accidentale in lista
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            data = data[0]
        else:
            return _error(raw_response, "JSON root è una lista senza dict")

    if not isinstance(data, dict):
        return _error(raw_response, f"JSON root è {type(data).__name__}, atteso dict")

    # --- category ---
    category = str(data.get("category", "")).strip().upper()
    if category not in VALID_CATEGORIES:
        return _error(
            raw_response,
            f"category={category!r} non valida; ammesse: {sorted(VALID_CATEGORIES)}",
        )

    predicted_target = CATEGORY_TO_TARGET[category]

    return {
        "category": category,
        "predicted_target": predicted_target,
        "step_1_commit_analysis":      str(data.get("step_1_commit_analysis", "")),
        "step_2_function_diff_analysis": str(data.get("step_2_function_diff_analysis", "")),
        "step_3_conclusion":           str(data.get("step_3_conclusion", "")),
        "raw_output":    raw_response,
        "parse_error":   False,
        "parse_error_reason": "",
        "method": "llm",
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _error(raw: str, reason: str) -> dict:
    return {
        "category":       "UNDETERMINABLE",
        "predicted_target": -1,
        "step_1_commit_analysis": "",
        "step_2_function_diff_analysis": "",
        "step_3_conclusion": "",
        "raw_output":    raw,
        "parse_error":   True,
        "parse_error_reason": reason,
        "method": "llm",
    }