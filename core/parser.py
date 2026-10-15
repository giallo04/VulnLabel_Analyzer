"""Parser della risposta LLM per-funzione — tassonomia DiverseVul.

Un solo parser: quello di ``prompting.parse_llm_output`` (JSON tollerante: fence, caratteri di
controllo, ``json_repair`` se installato). Qui lo si adatta al formato dei risultati della
pipeline.

Schema atteso (vedi ``prompting.GEMINI_RESPONSE_SCHEMA``):
  step_1_change, step_2_flaw, step_3_role : ragionamento, nell'ordine
  category : CORRECT_LABEL | WRONG_LABEL_SPREAD | WRONG_LABEL_CONSISTENCY |
             WRONG_LABEL_FEATURE | WRONG_LABEL_IRRELEVANT | UNDETERMINABLE
  ``predicted_target`` è derivato da ``category`` (non dichiarato dal modello).

Un output non parsabile NON è un verdetto: ``parse_error=True`` e ``predicted_target=-1``
(escluso dal denominatore delle statistiche, ritentabile con ``--retry-errors``).
"""
from __future__ import annotations

from .prompting import CATEGORY_TO_TARGET, REASONING_FIELDS, parse_llm_output


def parse_llm_response(raw_response: str) -> dict:
    """Valida e normalizza la risposta JSON dell'LLM.

    Restituisce un dict con chiavi: category, predicted_target, step_1_change, step_2_flaw,
    step_3_role, raw_output, parse_error, parse_error_reason, parse_method, method.
    """
    parsed = parse_llm_output(raw_response)
    data = parsed.get("data") or {}
    out = {
        "category": parsed["category"] if not parsed["parse_error"] else "UNDETERMINABLE",
        "predicted_target": parsed["predicted_target"] if not parsed["parse_error"] else -1,
        "raw_output": raw_response,
        "parse_error": parsed["parse_error"],
        "parse_error_reason": parsed["parse_error_reason"],
        "parse_method": parsed["parse_method"],
        "method": "llm",
    }
    for f in REASONING_FIELDS:
        out[f] = str(data.get(f, "") or "") if not parsed["parse_error"] else ""
    assert out["category"] in CATEGORY_TO_TARGET
    return out
