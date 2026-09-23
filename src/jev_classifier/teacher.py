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


# --------------------------------------------------------------------------- yes/no questions
# The noul questions drive dispatch and escalation, and were never trained - they were answered by
# whichever head the base checkpoint happened to ship, with an encoder that had been fine-tuned on
# a different task. `is_safe_to_drive` missed 4 of 18 stranded callers that way.
#
# `json_object` gives no structural guarantee, so the answer is an enumerated string rather than a
# boolean, and it is parsed rather than trusted.
NOUL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["true", "false"]}},
    "required": ["answer"],
    "additionalProperties": False,
}


def noul_system_prompt(key: str, variant: int = 1) -> str:
    q = SP.load().noul_question(key)
    if variant == 1:
        return (
            "You answer one yes/no question about a car-dealership phone call.\n\n"
            f"Question: {q['instructions']}\n\n"
            f"Answer true when: {q['criteria']['true']}\n"
            f"Answer false when: {q['criteria']['false']}\n\n"
            'Reply with JSON only: {"answer": "true"} or {"answer": "false"}.'
        )
    return (
        "Read the caller's message and decide.\n\n"
        f"{q['instructions']}\n"
        f"  true  - {q['criteria']['true']}\n"
        f"  false - {q['criteria']['false']}\n\n"
        'Output JSON only, exactly {"answer": "true"} or {"answer": "false"}.'
    )


def parse_bool(raw: object) -> Optional[bool]:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return None


def label_noul(
    text: str,
    key: str,
    *,
    provider: str = "deepseek",
    model: Optional[str] = None,
    thinking: bool = True,
    variant: int = 1,
) -> Dict[str, Any]:
    """Label one utterance for one yes/no question."""
    call = llm.chat_json(
        noul_system_prompt(key, variant),
        f'Caller: "{text}"',
        NOUL_SCHEMA,
        provider=provider,
        model=model,
        thinking=thinking,
    )
    raw = call["data"] if isinstance(call.get("data"), dict) else {}
    value = parse_bool(labels.field(raw, "answer", "value", key))
    return {
        "key": key,
        "value": value,
        "valid": value is not None,
        "raw": labels.field(raw, "answer", "value", key),
        "latency_ms": call["latency_ms"],
        "usage": call["usage"],
        "provider": call["provider"],
        "model": call["model"],
        "variant": variant,
    }
