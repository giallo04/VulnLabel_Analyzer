"""LLM engine — Google Gemini API (REST, stdlib only, no extra dependency).

The framework is fully API-based: this is the only module that talks to a model.

Rate-limit / quota handling (shared by all worker threads):

* client-side requests-per-minute limiter (``cfg.rpm``, 0 = unlimited);
* HTTP 429 handling that tells apart
    - per-minute quota  -> wait the server-suggested ``retryDelay`` and try again,
    - per-day quota     -> ``LLMQuotaExhausted`` (run stops cleanly, resume later),
    - ``limit: 0``      -> ``LLMFatalError`` (model not available for this key/tier);
* 5xx / network errors -> exponential back-off, then ``LLMTransientError``;
* invalid key / unknown model -> ``LLMFatalError``;
* blocked / empty answers -> plain ``RuntimeError`` (record checkpointed as parse error).

Error contract (used by ``core/pipeline.py``)
---------------------------------------------
* ``LLMAbort``          — the whole run must stop (bad API key, daily quota exhausted, ...).
                          The current record is NOT checkpointed, so a later resume retries it.
* ``LLMTransientError`` — the request failed after all retries for a temporary reason
                          (overload, network). The record is NOT checkpointed either.
* any other exception   — record-level failure: the record is checkpointed as UNDETERMINABLE
                          with ``parse_error=True``.

The API key is read by the caller (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``) and passed via
``LLMConfig.api_key``; it is sent in the ``x-goog-api-key`` header, never in the URL.
"""
from __future__ import annotations

import http.client
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LLMAbort(RuntimeError):
    """Unrecoverable for the whole run: stop it, do not checkpoint the current record."""


class LLMFatalError(LLMAbort):
    """Configuration / authentication problem (invalid API key, unknown model, ...)."""


class LLMQuotaExhausted(LLMAbort):
    """Daily quota exhausted: resume the run after the quota reset."""


class LLMTransientError(RuntimeError):
    """Temporary failure after all retries: skip the record without checkpointing it."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    model_name: str = "gemini-3.5-flash"
    api_key: str | None = field(default=None, repr=False)
    base_url: str = "https://generativelanguage.googleapis.com"
    temperature: float = 0.4           # basso: stesso input -> stessa etichetta (riproducibilità)
    seed: int = 55
    max_output_tokens: int = 32768      # thinking tokens included (high thinking can use a lot)
    timeout: float = 300                # seconds per request
    retries: int = 6                    # attempts on transient errors (5xx / network)
    backoff_base: float = 2.0           # exponential factor of the back-off (attempt k waits ~initial*base^(k-1))
    backoff_initial: float = 10.0       # first wait on 5xx/overload (seconds): an overloaded model needs time
    backoff_max: float = 120.0          # cap of a single back-off wait (seconds)
    backoff_jitter: float = 0.25        # +-25% random spread so the workers do not retry in lockstep
    rpm: float = 10.0                   # client-side requests-per-minute cap (0 = unlimited)
    # "minimal" | "low" | "medium" | "high"; None = model default.  Default "high": the labelling
    # task needs real deliberation (the answer fields are the final analysis, not a scratchpad).
    thinking_level: str | None = "high"
    # Output strutturato (responseSchema): JSON sempre valido e `category` sempre dentro l'enum.
    # Disattivalo (--no-schema) per modelli che non lo supportano.
    structured_output: bool = True


_logged_first_call = False


def _log_first_call(cfg: LLMConfig) -> None:
    global _logged_first_call
    if _logged_first_call:
        return
    _logged_first_call = True
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(
        f"  [llm] model={cfg.model_name} base_url={cfg.base_url} seed={cfg.seed} "
        f"temperature={cfg.temperature} thinking={cfg.thinking_level or 'model default'} "
        f"rpm={cfg.rpm} started={ts}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Gemini REST client
# ---------------------------------------------------------------------------
MAX_RATE_LIMIT_WAITS = 30        # per-minute 429 waits tolerated per request
MAX_RATE_LIMIT_SLEEP = 300.0     # never sleep longer than this in one go (seconds)
DEFAULT_RATE_LIMIT_SLEEP = 30.0  # used when the server gives no retryDelay

_BLOCK_FINISH_REASONS = {
    "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY", "RECITATION",
}

# Source code about vulnerabilities routinely trips the "dangerous content" classifier:
# disable the optional filters (this is a security-research labelling task on public code).
_SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


# ---------------------------------------------------------------------------
# Client-side rate limiter (shared by all worker threads)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Thread-safe 'one request every 60/rpm seconds' limiter."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        wait = slot - now
        if wait > 0:
            time.sleep(wait)

    def refund(self) -> None:
        """Give back one slot: a request the server rejected for overload (5xx) did not consume
        the per-minute budget, so the retry must not wait for a second slot."""
        if self.interval <= 0:
            return
        with self._lock:
            self._next = max(time.monotonic(), self._next - self.interval)

    def penalize(self, seconds: float) -> None:
        """Push the next free slot forward (server told us to back off)."""
        if self.interval <= 0:
            return
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


_limiters: Dict[Tuple[str, str, float], RateLimiter] = {}
_limiters_lock = threading.Lock()


def _get_limiter(cfg: LLMConfig) -> RateLimiter:
    key = (cfg.base_url, cfg.model_name, float(cfg.rpm or 0))
    with _limiters_lock:
        lim = _limiters.get(key)
        if lim is None:
            lim = _limiters[key] = RateLimiter(cfg.rpm)
        return lim


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------
def build_request_body(prompt: str, cfg: LLMConfig, *, system: Optional[str] = None,
                       schema: Optional[Dict[str, Any]] = None,
                       temperature: Optional[float] = None, seed: Optional[int] = None,
                       max_output_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Corpo della richiesta generateContent.

    `system` va in `systemInstruction` (le regole statiche restano separate dai dati del
    record: più aderenza alle istruzioni e prefisso cacheable); `schema` attiva l'output
    strutturato se `cfg.structured_output`.
    """
    gen: Dict[str, Any] = {
        "temperature": cfg.temperature if temperature is None else temperature,
        "seed": cfg.seed if seed is None else seed,
        "maxOutputTokens": cfg.max_output_tokens if max_output_tokens is None else max_output_tokens,
        "responseMimeType": "application/json",   # the prompt already asks for a JSON object
    }
    if schema and cfg.structured_output:
        gen["responseSchema"] = schema
    if cfg.thinking_level:
        gen["thinkingConfig"] = {"thinkingLevel": cfg.thinking_level}
    body: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen,
        "safetySettings": _SAFETY_SETTINGS,
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


@dataclass
class LLMReply:
    text: str
    finish_reason: Optional[str] = None

    @property
    def truncated(self) -> bool:
        """True se la risposta è stata tagliata dal limite di token (JSON quasi certamente rotto)."""
        return self.finish_reason == "MAX_TOKENS"


def extract_reply(data: Dict[str, Any]) -> LLMReply:
    """Return the answer text + finishReason of a generateContent response (thought parts excluded).

    Raises RuntimeError (record-level failure) when the answer was blocked or is empty.
    """
    cands = data.get("candidates") or []
    if not cands:
        block = (data.get("promptFeedback") or {}).get("blockReason")
        raise RuntimeError(
            f"Gemini returned no candidates (blockReason={block or 'n/a'})"
        )
    cand = cands[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(
        p.get("text", "") for p in parts
        if isinstance(p, dict) and not p.get("thought")
    )
    finish = cand.get("finishReason")
    if not text.strip() and finish == "MAX_TOKENS":
        raise RuntimeError(
            "Gemini hit MAX_TOKENS before writing any answer (thinking consumed the whole "
            "budget): raise --max-output-tokens or lower --thinking"
        )
    if not text.strip():
        if finish in _BLOCK_FINISH_REASONS:
            raise RuntimeError(f"Gemini blocked the response (finishReason={finish})")
        raise RuntimeError(f"Gemini returned an empty response (finishReason={finish})")
    return LLMReply(text, finish)


def extract_text(data: Dict[str, Any]) -> str:
    return extract_reply(data).text


def _error_info(body: bytes) -> Tuple[str, str, Dict[str, Any]]:
    """(message, status, error-dict) from a Google API error body."""
    try:
        err = (json.loads(body.decode("utf-8", "replace")) or {}).get("error") or {}
    except (ValueError, AttributeError):
        return body.decode("utf-8", "replace")[:300], "", {}
    return str(err.get("message", "")), str(err.get("status", "")), err


def _retry_delay(err: Dict[str, Any], headers: Any) -> Optional[float]:
    for d in err.get("details") or []:
        rd = d.get("retryDelay") if isinstance(d, dict) else None
        if rd:
            m = re.match(r"^\s*([0-9.]+)\s*s", str(rd))
            if m:
                return float(m.group(1))
    ra = headers.get("Retry-After") if headers else None
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    return None


def _quota_ids(err: Dict[str, Any]) -> str:
    ids = []
    for d in err.get("details") or []:
        if isinstance(d, dict):
            for v in d.get("violations") or []:
                if isinstance(v, dict):
                    ids.append(str(v.get("quotaId", "")))
                    ids.append(str(v.get("quotaMetric", "")))
    return " ".join(ids)


def _classify_429(message: str, err: Dict[str, Any]) -> str:
    """'daily' | 'zero' | 'minute'."""
    ids = _quota_ids(err).lower()
    msg = message.lower()
    if "limit: 0" in msg or "limit\": 0" in msg:
        return "zero"
    if "perday" in ids or "per day" in msg or "daily" in msg or "perday" in msg:
        return "daily"
    return "minute"


def _is_auth_problem(code: int, message: str, err: Dict[str, Any]) -> bool:
    if code in (401, 403):
        return True
    if code == 400:
        low = message.lower()
        if "api key" in low or "api_key" in low:
            return True
        for d in err.get("details") or []:
            if isinstance(d, dict) and d.get("reason") == "API_KEY_INVALID":
                return True
    return False


def backoff_delay(cfg: LLMConfig, attempt: int, server_hint: Optional[float] = None) -> float:
    """Wait before retry number `attempt` (1-based): exponential from ``backoff_initial``, capped
    at ``backoff_max``, with +-jitter; a server ``Retry-After`` hint is a lower bound."""
    base = min(cfg.backoff_max, cfg.backoff_initial * (cfg.backoff_base ** (attempt - 1)))
    wait = base * (1.0 + random.uniform(-cfg.backoff_jitter, cfg.backoff_jitter))
    if server_hint:
        wait = max(wait, server_hint)
    cap = max(cfg.backoff_max, server_hint) if server_hint else cfg.backoff_max
    return max(0.0, min(wait, cap))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def query_llm(prompt: str, cfg: LLMConfig, **kw: Any) -> str:
    """Send `prompt` to Gemini and return the answer text (JSON string)."""
    return query_llm_ex(prompt, cfg, **kw).text


def query_llm_ex(prompt: str, cfg: LLMConfig, *, system: Optional[str] = None,
                 schema: Optional[Dict[str, Any]] = None, temperature: Optional[float] = None,
                 seed: Optional[int] = None, max_output_tokens: Optional[int] = None) -> LLMReply:
    """Come query_llm, ma restituisce anche il finishReason (serve a riconoscere MAX_TOKENS)."""
    _log_first_call(cfg)
    if not cfg.api_key:
        raise LLMFatalError("Gemini API key missing (set GEMINI_API_KEY or GOOGLE_API_KEY)")

    url = (
        f"{cfg.base_url.rstrip('/')}/v1beta/models/"
        f"{cfg.model_name}:generateContent"
    )
    payload = json.dumps(build_request_body(
        prompt, cfg, system=system, schema=schema, temperature=temperature, seed=seed,
        max_output_tokens=max_output_tokens)).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-goog-api-key": cfg.api_key}
    limiter = _get_limiter(cfg)

    attempts = 0          # transient failures consumed
    server_hint: Optional[float] = None   # Retry-After / retryDelay of the last 5xx
    rate_waits = 0        # per-minute 429 waits consumed
    last_exc: Exception | None = None

    while True:
        limiter.acquire()
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return extract_reply(data)

        except urllib.error.HTTPError as exc:
            body = exc.read()
            message, _status, err = _error_info(body)
            code = exc.code

            if code == 429:
                kind = _classify_429(message, err)
                if kind == "zero":
                    raise LLMFatalError(
                        f"Gemini quota limit is 0 for model {cfg.model_name!r} "
                        f"(not available on this key's free tier?): {message}"
                    ) from exc
                if kind == "daily":
                    raise LLMQuotaExhausted(
                        f"Gemini daily quota exhausted for {cfg.model_name!r}: {message}"
                    ) from exc
                rate_waits += 1
                if rate_waits > MAX_RATE_LIMIT_WAITS:
                    raise LLMTransientError(
                        f"Gemini still rate-limited after {MAX_RATE_LIMIT_WAITS} waits: {message}"
                    ) from exc
                delay = _retry_delay(err, exc.headers) or DEFAULT_RATE_LIMIT_SLEEP
                delay = min(delay + 1.0, MAX_RATE_LIMIT_SLEEP)
                limiter.penalize(delay)   # all threads back off together; acquire() at loop top
                print(
                    f"  [gemini] 429 rate limit — waiting {delay:.0f}s "
                    f"({rate_waits}/{MAX_RATE_LIMIT_WAITS})",
                    file=sys.stderr,
                )
                # Do NOT sleep here: limiter.penalize(delay) already pushes the next free slot
                # forward by `delay` seconds; the next limiter.acquire() at the top of the loop
                # will block for that duration. A redundant sleep() here would double the wait.
                continue

            if code == 400 and "thinking" in message.lower():
                raise LLMFatalError(
                    f"Gemini model {cfg.model_name!r} rejected the thinking configuration "
                    f"(thinking_level={cfg.thinking_level!r}): {message}. "
                    f"Use --thinking <level supported by the model> or --thinking default."
                ) from exc
            if _is_auth_problem(code, message, err):
                raise LLMFatalError(f"Gemini auth error (HTTP {code}): {message}") from exc
            if code == 404:
                raise LLMFatalError(
                    f"Gemini model {cfg.model_name!r} not found (HTTP 404): {message}"
                ) from exc
            if code in (408, 500, 502, 503, 504):
                last_exc = RuntimeError(f"HTTP {code}: {message}")
                server_hint = _retry_delay(err, exc.headers)
                limiter.refund()          # overload errors must not eat the rpm budget
            else:
                # e.g. 400 INVALID_ARGUMENT for this specific prompt: record-level failure
                raise RuntimeError(f"Gemini rejected the request (HTTP {code}): {message}") from exc

        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
                http.client.HTTPException, ValueError) as exc:
            # ValueError covers a malformed JSON body (truncated response)
            last_exc = exc

        attempts += 1
        if attempts >= cfg.retries:
            raise LLMTransientError(
                f"Gemini unavailable after {cfg.retries} attempts "
                f"(model={cfg.model_name}): {last_exc}"
            ) from last_exc
        wait = backoff_delay(cfg, attempts, server_hint)
        server_hint = None
        print(
            f"  [gemini] attempt {attempts}/{cfg.retries} failed: {str(last_exc)[:120]}. "
            f"Retrying in {wait:.0f}s…",
            file=sys.stderr,
        )
        # All workers back off together (an overloaded endpoint is not helped by 4 threads
        # retrying in parallel); with rpm=0 there is no limiter slot to push, so sleep here.
        if limiter.interval > 0:
            limiter.penalize(wait)
        else:
            time.sleep(wait)
