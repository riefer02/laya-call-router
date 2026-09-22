"""Minimal multi-provider client for the evaluation's LLM arms and the teacher labelling job.

Deliberately dependency-free (stdlib `urllib`) and small. It exists to answer two questions:
what would a cheap generative model do with the same routing task, and can one be trusted to
label training data.

**Providers differ in how they enforce structure**, and that is the main reason this file is not
three lines:

* OpenAI supports `response_format: json_schema` with `strict: true` — the vocabulary is enforced.
* DeepSeek (V4.1 Flash) rejects `json_schema` ("this response_format type is unavailable now")
  and rejects `tool_choice` in thinking mode, so it is `json_object` only. The vocabulary must be
  enumerated *in the prompt* and every answer validated afterwards. It does not reliably respect
  an enum on its own — measured, it returned `"body shop"` and invented intents like
  `"schedule body work"` where our label is `collision`.

Keys are read from the environment (or `.env`) and are never logged, echoed, or persisted.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    default_model: str
    # USD per 1M tokens: (input cache-miss, input cache-hit, output). Approximate; overridable.
    prices: Dict[str, Tuple[float, float, float]]
    schema_mode: str  # "json_schema" | "json_object"
    supports_thinking: bool = False


PROVIDERS: Dict[str, Provider] = {
    "openai": Provider(
        name="openai",
        base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY",
        default_model="gpt-5.4-nano",
        schema_mode="json_schema",
        prices={
            "gpt-5.4-nano": (0.05, 0.05, 0.40),
            "gpt-4.1-nano": (0.10, 0.10, 0.40),
            "gpt-5.4-mini": (0.25, 0.25, 2.00),
        },
    ),
    "deepseek": Provider(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-flash",
        schema_mode="json_object",
        supports_thinking=True,
        # V4.1 Flash off-peak; peak pricing doubles input and output.
        prices={
            "deepseek-flash": (0.15, 0.003, 0.60),
            "deepseek-v4-pro": (0.66, 0.022, 1.98),
        },
    ),
}

_ENV_LOADED = False


def load_env(path: Optional[Path] = None) -> None:
    """Load `.env` into os.environ without overriding anything already set."""
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


def api_key(provider: str) -> Optional[str]:
    load_env()
    spec = PROVIDERS.get(provider)
    if not spec:
        return None
    return os.environ.get(spec.key_env)


def available(provider: str = "openai") -> bool:
    return bool(api_key(provider))


def parse_model(ref: str) -> Tuple[str, str]:
    """'deepseek:deepseek-flash' -> ('deepseek', 'deepseek-flash'); 'gpt-5.4-nano' -> openai."""
    if ":" in ref:
        provider, model = ref.split(":", 1)
        if provider in PROVIDERS:
            return provider, model
    for name, spec in PROVIDERS.items():
        if ref in spec.prices:
            return name, ref
    return "openai", ref


def price(
    provider: str,
    model: str,
    prompt_miss: int,
    prompt_hit: int,
    completion: int,
) -> Optional[float]:
    spec = PROVIDERS.get(provider)
    if not spec:
        return None
    rates = spec.prices.get(model)
    if not rates:
        return None
    miss_rate, hit_rate, out_rate = rates
    return (prompt_miss * miss_rate + prompt_hit * hit_rate + completion * out_rate) / 1_000_000


class LLMUnavailable(RuntimeError):
    pass


def chat_json(
    system: str,
    user: str,
    schema: Dict[str, Any],
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    thinking: bool = True,
    timeout: int = 120,
    retries: int = 3,
) -> Dict[str, Any]:
    """One structured-output call. Returns {data, usage, latency_ms, attempts, provider, model}."""
    load_env()
    if model and not provider:
        provider, model = parse_model(model)
    provider = provider or "openai"
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise LLMUnavailable(f"unknown provider {provider!r}")
    model = model or spec.default_model

    key = api_key(provider)
    if not key:
        raise LLMUnavailable(f"{spec.key_env} is not set")

    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if spec.schema_mode == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "routing", "strict": True, "schema": schema},
        }
    else:
        # Enforced only by the prompt; the caller must validate the answer.
        body["response_format"] = {"type": "json_object"}
        if spec.supports_thinking and not thinking:
            body["thinking"] = {"type": "disabled"}

    payload = json.dumps(body).encode()
    url = f"{spec.base_url}/chat/completions"
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in (408, 409, 429) or exc.code >= 500:
                time.sleep(0.8 * attempt)
                continue
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"{provider} HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001 - network flake, retry
            last_error = exc
            time.sleep(0.8 * attempt)
            continue

        latency_ms = (time.perf_counter() - started) * 1000
        usage = parsed.get("usage") or {}
        content = parsed["choices"][0]["message"]["content"]
        try:
            data = json.loads(content)
        except (TypeError, ValueError) as exc:
            # Providers without a strict schema occasionally emit prose or a truncated object.
            # That is a retryable failure, not a crash.
            last_error = exc
            time.sleep(0.5 * attempt)
            continue
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        hit = int(usage.get("prompt_cache_hit_tokens", 0))
        miss = int(usage.get("prompt_cache_miss_tokens", 0)) or max(0, prompt_tokens - hit)
        return {
            "data": data,
            "raw": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "prompt_cache_hit": hit,
                "prompt_cache_miss": miss,
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            },
            "latency_ms": latency_ms,
            "attempts": attempt,
            "provider": provider,
            "model": model,
        }

    raise RuntimeError(f"{provider} request failed after {retries} attempts: {last_error}")
