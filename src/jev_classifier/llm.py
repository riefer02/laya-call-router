"""Minimal OpenAI client for the evaluation's LLM arm.

Deliberately dependency-free (stdlib `urllib`) and deliberately small: this exists to answer one
question — "what would a cheap generative model do with the same routing task, at what latency and
what price?" — not to be a general provider client.

The key is read from the environment (or `.env`) and is never logged, echoed, or written anywhere.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-5.4-nano"
API_URL = "https://api.openai.com/v1/chat/completions"

# USD per 1M tokens. Approximate, and overridable — treat cost figures as indicative.
PRICES: Dict[str, tuple[float, float]] = {
    "gpt-5.4-nano": (0.05, 0.40),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5.4-mini": (0.25, 2.00),
}

_ENV_LOADED = False


def load_env(path: Optional[Path] = None) -> None:
    """Load `.env` from the repo root into os.environ without overriding anything already set."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_file = path or (ROOT / ".env")
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def model_name() -> str:
    load_env()
    return os.environ.get("JEV_LLM_MODEL", DEFAULT_MODEL)


def available() -> bool:
    load_env()
    return bool(os.environ.get("OPENAI_API_KEY"))


def price(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    rates = PRICES.get(model)
    if not rates:
        return None
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000


class LLMUnavailable(RuntimeError):
    pass


def chat_json(
    system: str,
    user: str,
    schema: Dict[str, Any],
    *,
    model: Optional[str] = None,
    timeout: int = 60,
    retries: int = 3,
) -> Dict[str, Any]:
    """One structured-output call. Returns {data, usage, latency_ms, attempts, model}.

    Raises LLMUnavailable if no key is configured, LLMError after exhausting retries.
    """
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMUnavailable("OPENAI_API_KEY is not set")
    model = model or model_name()

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "routing", "strict": True, "schema": schema},
        },
    }
    payload = json.dumps(body).encode()
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            API_URL,
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_error = exc
            # Retry only on rate limits and server errors; a 4xx schema problem will not fix itself.
            if exc.code in (408, 409, 429) or exc.code >= 500:
                time.sleep(0.8 * attempt)
                continue
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001 - network flake, retry
            last_error = exc
            time.sleep(0.8 * attempt)
            continue

        latency_ms = (time.perf_counter() - started) * 1000
        usage = parsed.get("usage") or {}
        content = parsed["choices"][0]["message"]["content"]
        return {
            "data": json.loads(content),
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
            "latency_ms": latency_ms,
            "attempts": attempt,
            "model": model,
        }

    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_error}")
