"""Lookup opzionale NVD (API 2.0). Funziona solo se il messaggio di commit cita un CVE."""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .repos import _http_json

_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={}"


@dataclass
class NvdInfo:
    cve_id: str
    description: str
    cwes: list[str]
    published: str | None = None


def parse_nvd(payload: dict) -> NvdInfo | None:
    vulns = (payload or {}).get("vulnerabilities") or []
    if not vulns:
        return None
    cve = vulns[0]["cve"]
    desc = next((d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"), "")
    cwes = sorted({w["value"] for wk in cve.get("weaknesses", []) for w in wk.get("description", [])
                   if w.get("value", "").startswith("CWE-")})
    return NvdInfo(cve["id"], desc, cwes, cve.get("published"))


class NvdClient:
    def __init__(self, cache_dir: str | Path, api_key: str | None = None) -> None:
        self.dir = Path(cache_dir) / "nvd"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.key = api_key or os.environ.get("NVD_API_KEY")
        self.delay = 0.7 if self.key else 6.5          # 50 req/30s con chiave, 5 req/30s senza
        self._lock = threading.Lock()

    def get(self, cve_id: str) -> NvdInfo | None:
        f = self.dir / f"{cve_id}.json"
        if f.exists():
            payload = json.loads(f.read_text())
        else:
            with self._lock:
                time.sleep(self.delay)
                headers = {"apiKey": self.key} if self.key else {}
                payload = _http_json(_URL.format(cve_id), headers)
            if payload is None:
                return None
            f.write_text(json.dumps(payload))
        return parse_nvd(payload)
