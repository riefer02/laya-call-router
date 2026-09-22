"""Teacher labelling: strict prompt, provider call, vocabulary validation, provenance.

One place so the teacher-validation gate and the training-set generator cannot drift apart — if
they used different prompts, the measured agreement would not describe the labels we actually
train on.

The taxonomy comes from the store profile, so relabelling against a different store needs no code
change here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from . import labels, llm, store_profile as SP

PROFILE = SP.load()
DESTINATIONS = PROFILE.destination_keys
ALL_SUBQUEUES = sorted({s.key for s in PROFILE.subqueues})

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "destination": {"type": "string", "enum": list(DESTINATIONS)},
        "subqueue": {"type": "string", "enum": ALL_SUBQUEUES},
    },
    "required": ["destination", "subqueue"],
    "additionalProperties": False,
}

_SUBQUEUE_MENU = "\n".join(
    f"  {d.key}: " + (", ".join(PROFILE.subqueue_keys(d.key)) or "(no sub-queues; use 'other')")
    for d in PROFILE.destinations
)
_DEST_MENU = "\n".join(f"  {d.key}: {d.description}" for d in PROFILE.destinations)

# Deliberately blunt. Measured: told only "return JSON with department and intent", DeepSeek
# answered `"body shop"` and invented `"schedule body work"`; the exact-string instruction below
# is what actually moves it into our vocabulary.
_SYSTEM = (
    "You label car-dealership phone calls for a routing classifier.\n\n"
    "Return a JSON object with exactly two keys: destination and subqueue.\n\n"
    "destination is which part of the dealership owns the work:\n"
    f"{_DEST_MENU}\n\n"
    "subqueue is the queue within that destination:\n"
    f"{_SUBQUEUE_MENU}\n\n"
    "Important: tyres and detailing are sub-queues of service, NOT destinations. Roadside "
    "assistance is a service sub-queue, not a destination. A caller who is a supplier, a job "
    "applicant or a wrong number is destination non_customer.\n\n"
    "Copy values exactly as written: lowercase, underscores, no spaces, nothing invented. "
    "Reply with JSON only."
)

_SYSTEM_ALT = (
    "You are auditing incoming dealership phone calls for the team that should handle them.\n\n"
    "Decide two things and return them as JSON keys destination and subqueue.\n\n"
    "Which part of the dealership owns the work:\n"
    f"{_DEST_MENU}\n\n"
    "Which queue within it:\n"
    f"{_SUBQUEUE_MENU}\n\n"
    "Use the exact strings, lower case, with underscores. Do not add keys or commentary. "
    "Output JSON only."
)


def system_prompt(variant: int = 1) -> str:
    return _SYSTEM if variant == 1 else _SYSTEM_ALT


def labelling_prompt(variant: int = 1) -> str:
    return system_prompt(variant)


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
    destination, subqueue, ok = labels.validate_label(
        labels.field(raw, "destination", "dept", "department"),
        labels.field(raw, "subqueue", "sub_queue", "intent"),
    )
    return {
        "destination": destination,
        "subqueue": subqueue,
        "valid": ok,
        "raw_destination": labels.field(raw, "destination", "dept", "department"),
        "raw_subqueue": labels.field(raw, "subqueue", "sub_queue", "intent"),
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
