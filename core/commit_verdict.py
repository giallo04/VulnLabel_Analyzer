"""Verdetto a livello di commit: UNA chiamata API per commit, riusata da tutte le sue funzioni.

Problema risolto
----------------
Prima ogni funzione del commit ridescriveva il commit da sola (step_1 nel prompt della
funzione): con un campionamento non nullo lo stesso commit riceveva nature e cause radice
diverse da una funzione all'altra, e le etichette ne risentivano.

Ora il commit è caratterizzato una volta sola (natura, causa radice, schema della correzione,
ambito) e il risultato è inserito identico nel prompt di ogni sua funzione.

Garanzie
--------
* una sola chiamata per commit anche con più thread: i thread delle altre funzioni dello
  stesso commit aspettano il risultato (lock per chiave);
* il verdetto è persistito in ``commit_verdicts.jsonl`` accanto al checkpoint: dopo un resume
  (o un cambio di thread) il commit non viene ridescritto e resta coerente con le funzioni già
  classificate;
* temperatura 0 per questa chiamata (il campionamento qui sarebbe solo rumore);
* errori transitori/fatali dell'API si propagano (il record non viene checkpointato e sarà
  ritentato); solo un output non parsabile dopo i retry ripiega su un verdetto ``unclear``
  marcato ``fallback=True`` — tenuto in memoria per il run (stessa descrizione per tutte le
  funzioni) ma NON salvato su disco, così al prossimo run il commit viene ritentato.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .llm_engine import LLMConfig, query_llm_ex
from .prompting import (COMMIT_RESPONSE_SCHEMA, PROMPT_VERSION, build_commit_messages,
                        commit_key, parse_commit_output)

COMMIT_PROMPT_VERSION = f"commit-{PROMPT_VERSION}"
PARSE_ATTEMPTS = 3

_FALLBACK = {"evidence": "commit verdict unavailable (unparsable model output)",
             "root_cause": "unknown", "defect_class": "none",
             "fix_pattern": "unknown", "fix_scope": "unknown", "nature": "unclear"}


class CommitVerdicts:
    """Cache thread-safe dei verdetti di commit, con persistenza JSONL opzionale."""

    def __init__(self, cfg: LLMConfig, path: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.path = Path(path) if path else None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()       # protegge _locks / _cache / file
        self.n_api_calls = 0
        self.n_loaded = 0
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        stale = 0
        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("version") != COMMIT_PROMPT_VERSION or obj.get("model") != self.cfg.model_name:
                    stale += 1
                    continue
                if isinstance(obj.get("verdict"), dict) and obj.get("key"):
                    self._cache[str(obj["key"])] = obj["verdict"]
        self.n_loaded = len(self._cache)
        if stale:
            print(f"  [commit-verdict] {stale} saved verdicts ignored (different model or prompt "
                  f"version): they will be recomputed", file=sys.stderr)

    def _persist(self, key: str, verdict: Dict[str, Any]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"key": key, "version": COMMIT_PROMPT_VERSION, "model": self.cfg.model_name,
               "verdict": verdict}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def get(self, record: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Verdetto del commit di `record` (calcolato al primo accesso, poi dalla cache)."""
        key = commit_key(record)
        with self._guard:
            hit = self._cache.get(key)
            if hit is not None:
                return hit
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:                                   # un solo thread per commit chiama l'API
            with self._guard:
                hit = self._cache.get(key)
            if hit is not None:
                return hit
            verdict = self._compute(record, ctx)
            with self._guard:
                self._cache[key] = verdict
                if not verdict.get("fallback"):
                    self._persist(key, verdict)
            return verdict

    def _compute(self, record: Dict[str, Any], ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        msgs = build_commit_messages(record, ctx)
        budget = self.cfg.max_output_tokens
        last_reason = ""
        for attempt in range(PARSE_ATTEMPTS):
            reply = query_llm_ex(msgs["user"], self.cfg, system=msgs["system"],
                                 schema=COMMIT_RESPONSE_SCHEMA, temperature=0.0,
                                 seed=self.cfg.seed + attempt, max_output_tokens=budget)
            with self._guard:
                self.n_api_calls += 1
            parsed = parse_commit_output(reply.text)
            if not parsed["parse_error"]:
                verdict = dict(parsed["data"])
                verdict["fallback"] = False
                return verdict
            last_reason = parsed["parse_error_reason"]
            if reply.truncated:
                budget = min(budget * 2, 65536)
        print(f"  [commit-verdict] unparsable output for commit {commit_key(record)[:12]} after "
              f"{PARSE_ATTEMPTS} attempts ({last_reason}): using an 'unclear' verdict",
              file=sys.stderr)
        return {**_FALLBACK, "fallback": True}
