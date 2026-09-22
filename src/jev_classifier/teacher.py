"""Teacher labelling: strict prompt, provider call, vocabulary validation, provenance.

One place so the teacher-validation gate and the training-set generator cannot drift apart — if
they used different prompts, the measured agreement would not describe the labels we actually
train on.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import dealership as D
from . import labels, llm

ALL_INTENTS = sorted({name for branch in D.INTENTS.values() for name in branch})

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "department": {"type": "string", "enum": list(D.DEPARTMENTS)},
        "intent": {"type": "string", "enum": ALL_INTENTS},
    },
    "required": ["department", "intent"],
    "additionalProperties": False,
}

# Deliberately blunt. Measured: told only "return JSON with department and intent", DeepSeek
# answered `"body shop"` and invented `"schedule body work"`; the exact-string instruction below
# is what actually moves it into our vocabulary.
_SYSTEM = (
    "You label car-dealership phone calls for a routing classifier.\n\n"
    "Return a JSON object with exactly two keys: department and intent.\n\n"
    "department MUST be exactly one of: " + ", ".join(D.DEPARTMENTS) + ".\n\n"
    "intent MUST be exactly one of the strings listed for the chosen department:\n"
    + "\n".join(f"  {dept}: " + ", ".join(branch) for dept, branch in D.INTENTS.items())
    + "\n\nCopy values exactly as written: lowercase, underscores, no spaces, nothing invented. "
    "Reply with JSON only."
)


def system_prompt() -> str:
    return _SYSTEM


# A second, independently-worded labelling prompt. Two phrasings agreeing is evidence; one
# phrasing twice is not. Used to filter generated training data.
_SYSTEM_ALT = (
    "You are auditing incoming dealership phone calls for the department that should handle them.\n\n"
    "Decide two things and return them as JSON keys department and intent.\n\n"
    "Permitted departments (use the string verbatim): " + ", ".join(D.DEPARTMENTS) + ".\n\n"
    "Permitted intents, grouped by the department they belong to:\n"
    + "\n".join(f"  {dept}: " + ", ".join(branch) for dept, branch in D.INTENTS.items())
    + "\n\nUse the exact strings, lower case, with underscores. Do not add keys or commentary. "
    "Output JSON only."
)


def labelling_prompt(variant: int = 1) -> str:
    return _SYSTEM if variant == 1 else _SYSTEM_ALT


def label(
    text: str,
    *,
    provider: str = "deepseek",
    model: Optional[str] = None,
    thinking: bool = True,
    variant: int = 1,
) -> Dict[str, Any]:
    """Label one utterance. Returns the validated label plus full provenance."""
    call = llm.chat_json(
        labelling_prompt(variant),
        f'Caller: "{text}"',
        SCHEMA,
        provider=provider,
        model=model,
        thinking=thinking,
    )
    raw = call["data"] if isinstance(call.get("data"), dict) else {}
    department, intent, ok = labels.validate_label(raw.get("department"), raw.get("intent"))
    return {
        "department": department,
        "intent": intent,
        "valid": ok,
        "raw_department": raw.get("department"),
        "raw_intent": raw.get("intent"),
        "latency_ms": call["latency_ms"],
        "usage": call["usage"],
        "provider": call["provider"],
        "model": call["model"],
        "variant": variant,
    }


def cost_of(record: Dict[str, Any]) -> Optional[float]:
    usage = record.get("usage") or {}
    return llm.price(
        record.get("provider", "openai"),
        record.get("model", ""),
        int(usage.get("prompt_cache_miss", usage.get("prompt_tokens", 0))),
        int(usage.get("prompt_cache_hit", 0)),
        int(usage.get("completion_tokens", 0)),
    )
