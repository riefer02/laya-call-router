"""Car-dealership service domain: the questions the switchboard asks, and how answers map to
queues.

Everything a caller says is turned into a *typed* decision here. Slots (vehicle, location, time)
are `choice` questions over fixed enums rather than free-text extraction, because Laya classifies
— it does not generate or parse. The one exception is the exact appointment time, which is a
clearly-labelled deterministic regex (`extract_time`) and is rendered as a different node kind so
the UI never pretends a model decided it.
"""

from __future__ import annotations

import re
from typing import Dict, List

# --------------------------------------------------------------------------- departments
DEPARTMENTS: Dict[str, str] = {
    "service": "mechanical repair, maintenance, warning lights, noises, engine or starting trouble",
    "body_shop": "collision damage, dents, scratches, paint, glass, after an accident",
    "parts": "ordering or buying a specific part or accessory",
    "tires": "tire replacement, rotation, balancing, flat tires or wheels",
    "detailing": "cleaning, detailing, car wash, interior or paint care",
    "sales": "buying, leasing or appraising a vehicle, inventory questions",
    "finance": "loan or lease terms, insurance, paperwork, payments",
    "towing": "roadside assistance, towing, jump start, lockout or fuel delivery",
    "general": "none of the above, or the caller is unsure",
}

DEPARTMENT_QUESTION = {
    "department": {
        "type": "choice",
        "instructions": "Which department at the dealership should handle this caller?",
        "criteria": DEPARTMENTS,
    }
}

# --------------------------------------------------------------------------- intents
INTENTS: Dict[str, Dict[str, str]] = {
    "service": {
        "no_start": "the vehicle will not start or is dead",
        "warning_light": "a dashboard warning light is on",
        "brakes": "brakes are squealing, grinding or soft",
        "noise": "the vehicle makes an unusual noise or vibration",
        "maintenance": "routine maintenance, oil change or scheduled service",
        "performance": "the vehicle drives poorly, stalls or loses power",
        "other": "some other mechanical problem",
    },
    "body_shop": {
        "collision": "damage from a crash or being hit",
        "dent_scratch": "a dent or scratch without a crash",
        "glass": "broken or cracked window, windscreen or mirror",
        "paint": "paint damage, peeling or a repaint",
        "other": "some other bodywork",
    },
    "parts": {
        "order_part": "wants to order a specific part",
        "availability": "asks whether a part is in stock",
        "accessory": "wants an accessory or add-on",
        "other": "some other parts request",
    },
    "tires": {
        "flat": "has a flat tire or a puncture",
        "replacement": "wants new tires",
        "rotation": "wants a rotation or balancing",
        "alignment": "steering pulls or the vehicle needs an alignment",
        "other": "some other tire or wheel request",
    },
    "detailing": {
        "wash": "wants a wash",
        "full_detail": "wants a full interior and exterior detail",
        "interior": "wants interior cleaning",
        "paint_correction": "wants polishing or paint correction",
        "other": "some other cleaning request",
    },
    "sales": {
        "new_vehicle": "interested in a new vehicle",
        "used_vehicle": "interested in a used vehicle",
        "trade_in": "wants to trade in or appraise a vehicle",
        "inventory": "asks what is in stock",
        "other": "some other sales enquiry",
    },
    "finance": {
        "lease_terms": "asks about lease terms",
        "loan": "asks about financing or a loan",
        "insurance": "asks about insurance",
        "paperwork": "asks about paperwork or a contract",
        "other": "some other finance question",
    },
    "towing": {
        "tow_needed": "needs the vehicle towed",
        "jump_start": "needs a jump start",
        "lockout": "is locked out of the vehicle",
        "fuel": "has run out of fuel",
        "other": "some other roadside request",
    },
    "general": {
        "general_question": "a general question about the dealership",
        "other": "none of the above fits",
    },
}


def intent_question(department: str) -> Dict:
    criteria = INTENTS.get(department, INTENTS["general"])
    return {
        "intent": {
            "type": "choice",
            "instructions": (
                f"This is a {department.replace('_', ' ')} request. What exactly does the "
                "caller want? Pick the single closest option."
            ),
            "criteria": criteria,
        }
    }


def department_question_paraphrase() -> Dict:
    """A second, independently-worded version of the department question.

    Used to *verify* a low-confidence answer rather than to replace it. Measured: re-asking with a
    narrowed option set does not improve accuracy — it re-rolls the answer and inflates confidence
    (a *wrong* `body_shop` at 0.73 replacing a `service` at 0.37). Agreement between two phrasings
    is a real signal; a second roll of the same question is not.
    """
    return {
        "department": {
            "type": "choice",
            "instructions": "What kind of help does this caller need from the dealership?",
            "criteria": DEPARTMENTS,
        }
    }


def intent_question_paraphrase(department: str) -> Dict:
    criteria = INTENTS.get(department, INTENTS["general"])
    return {
        "intent": {
            "type": "choice",
            "instructions": (
                f"What does the caller need from the {department.replace('_', ' ')} department? "
                "Pick the single closest option."
            ),
            "criteria": criteria,
        }
    }


# --------------------------------------------------------------------------- slots
LOCATIONS: Dict[str, str] = {
    "downtown": "the caller said downtown",
    "northside": "the caller said northside",
    "airport": "the caller said airport",
    "westside": "the caller said westside",
    "not_stated": "the caller has not named a location",
}

VEHICLES: Dict[str, str] = {
    "sedan": "the caller said a car or sedan",
    "suv": "the caller said an SUV or crossover",
    "truck": "the caller said a pickup truck",
    "van": "the caller said a van or minivan",
    "ev": "the caller said an electric vehicle",
    "motorcycle": "the caller said a motorcycle",
    "not_stated": "the caller has not said what kind of vehicle it is",
}

TIME_PREFERENCES: Dict[str, str] = {
    "today": "the caller said today or as soon as possible",
    "tomorrow": "the caller said tomorrow",
    "this_week": "the caller said later this week",
    "next_week": "the caller said next week or later",
    "not_stated": "the caller has not said when",
}

# Phrasing measured in scripts/probe_slots.py: the "explicitly mention ... otherwise choose
# not_stated" form scored 23/24 where the plain form scored 19/24 and invented a vehicle
# ("sedan") and a time ("today") from sentences that mentioned neither.
SLOT_QUESTIONS: Dict = {
    "vehicle": {
        "type": "choice",
        "instructions": (
            "What kind of vehicle did the caller explicitly mention? "
            "If no vehicle type was mentioned, choose 'not_stated'."
        ),
        "criteria": VEHICLES,
    },
    "location": {
        "type": "choice",
        "instructions": (
            "Which dealership location did the caller explicitly name? "
            "If they did not name one, choose 'not_stated'."
        ),
        "criteria": LOCATIONS,
    },
    "time_preference": {
        "type": "choice",
        "instructions": (
            "What timing did the caller explicitly state? "
            "If they did not state one, choose 'not_stated'."
        ),
        "criteria": TIME_PREFERENCES,
    },
    "is_safe_to_drive": {
        "type": "noul",
        "instructions": "Does the caller indicate the vehicle is unsafe to drive or stranded?",
    },
    "needs_human": {
        "type": "noul",
        "instructions": (
            "Does this caller need a person rather than the automated booking flow — for "
            "example because the request is unusual, a complaint, or outside routine booking?"
        ),
    },
}

# The first pass each turn: department + slots, all answerable without knowing the branch.
# There is deliberately no model "what should I do next?" question here. Measured, the base
# checkpoints answer it at confidence 0.03 and pick the same option almost regardless of input
# (see scripts/probe_slots.py), so control flow is a deterministic policy instead — see
# next_action_for(). The classifier decides *understanding*; policy decides *control flow*.
PASS1_QUESTIONS: Dict = {
    **DEPARTMENT_QUESTION,
    "vehicle": SLOT_QUESTIONS["vehicle"],
    "location": SLOT_QUESTIONS["location"],
    "time_preference": SLOT_QUESTIONS["time_preference"],
    "is_safe_to_drive": SLOT_QUESTIONS["is_safe_to_drive"],
    "needs_human": SLOT_QUESTIONS["needs_human"],
}

# Run alongside the still-unresolved questions from turn 2 on. If it fires, the facts we had
# pinned (department, intent, urgency) are re-evaluated; if it does not, we skip them. One extra
# question buys the right to skip several.
CHANGE_QUESTION: Dict = {
    "changed": {
        "type": "noul",
        "instructions": (
            "Does the caller's latest message change or add to information the caller gave "
            "earlier in this call?"
        ),
    }
}

CHANGE_FLAG = 0.5

# --------------------------------------------------------------------------- verification
# A second opinion on a low-confidence *classification* question. Measured: re-asking with a
# narrowed option set does not improve accuracy — it re-rolls the answer and inflates confidence
# (a wrong `body_shop` at 0.73 replacing a `service` at 0.37), which is worse than not escalating
# at all. So tier 2 verifies with a paraphrase and never overturns the primary answer.
ESCALATION_MIN_OPTIONS = 4

# A choice answer equal to one of these means "the caller has not said", so it must never be
# pinned — it is exactly the thing a later turn is supposed to resolve.
UNRESOLVED_SENTINELS = {"not_stated"}

# --------------------------------------------------------------------------- next action
# Deterministic control flow. The model is not asked this: it answered at confidence 0.03 and
# effectively ignored the input (scripts/probe_slots.py).
NEXT_ACTION_LABELS: Dict[str, str] = {
    "ask_vehicle": "we do not yet know what kind of vehicle this is",
    "ask_location": "we do not yet know which location the caller wants",
    "ask_time": "we do not yet know when the caller wants to come in",
    "ask_detail": "the problem is too vague to book; ask for more detail",
    "confirm_booking": "we have everything needed to book the appointment",
    "offer_transfer": "this is outside the routine booking flow; hand to a person",
}


def next_action_for(missing: List[str], department: str, unsafe: float) -> str:
    """The switchboard's next step, as policy over the classifier's understanding."""
    if department == "general" or unsafe >= UNSAFE_FLAG:
        return "offer_transfer"
    if not missing:
        return "confirm_booking"
    slot = missing[0]
    return "ask_time" if slot == "time_preference" else "ask_" + slot


# The agent must NOT read out the classifier's option list. Measured: when the switchboard asked
# "today, tomorrow, later this week, or next week?", the slot classifier started answering with
# one of those words even when the caller had said nothing — the question itself was leaking into
# the transcript it classifies. The options still appear as chips in the UI; they are just not
# spoken.
RESPONSES: Dict[str, str] = {
    "ask_vehicle": "Thanks. What kind of vehicle is it?",
    "ask_location": "Got it — which of our locations works best for you?",
    "ask_time": "When would you like to come in?",
    "ask_detail": "I want to make sure we book the right thing — could you tell me a little more about what the vehicle is doing?",
    "confirm_booking": "Perfect, I have everything I need. I'm booking you into {department} at {location} for {time}.",
    "offer_transfer": "Let me get you straight to the right team so nobody has to wait.",
}

REQUIRED_SLOTS = ("vehicle", "location", "time_preference")

# --------------------------------------------------------------------------- routing
QUEUES: Dict[str, str] = {
    "service": "Service Department",
    "body_shop": "Body Shop",
    "parts": "Parts Counter",
    "tires": "Tire Bay",
    "detailing": "Detailing",
    "sales": "Sales Floor",
    "finance": "Finance & Insurance",
    "towing": "Roadside / Towing",
    "general": "Front Desk",
}

# Roadside requests always go to dispatch, whatever the department said.
SPECIAL_QUEUES: Dict[tuple, str] = {
    ("tires", "flat"): "Roadside / Towing",
    ("service", "no_start"): "Roadside / Towing",
}

UNSAFE_FLAG = 0.5
NEEDS_HUMAN_FLAG = 0.5


def decide(answers: Dict, missing: List[str]) -> Dict:
    """Terminal routing outcome from the accumulated answers."""

    def pick(qid):
        ans = answers.get(qid)
        if not isinstance(ans, dict):
            return None
        return ans.get("choice") if ans.get("type") == "choice" else (
            ans.get("noul") if ans.get("type") == "noul" else ans.get("score")
        )

    def prob(qid) -> float:
        v = pick(qid)
        return float(v) if isinstance(v, (int, float)) else 0.0

    department = pick("department") or "general"
    intent = pick("intent") or "other"
    unsafe = prob("is_safe_to_drive")
    needs_human = prob("needs_human")

    flags: List[str] = []
    if unsafe >= UNSAFE_FLAG:
        flags.append("unsafe_to_drive")
    if needs_human >= NEEDS_HUMAN_FLAG:
        flags.append("needs_human")
    if department == "general":
        flags.append("out_of_scope")
    if missing:
        flags.append("missing_info")

    queue = SPECIAL_QUEUES.get((department, intent), QUEUES.get(department, QUEUES["general"]))
    if unsafe >= UNSAFE_FLAG:
        queue = "Roadside / Towing"
        flags.append("dispatch")

    priority = "HIGH" if (unsafe >= UNSAFE_FLAG or needs_human >= NEEDS_HUMAN_FLAG) else "NORMAL"
    handler = "human" if (needs_human >= NEEDS_HUMAN_FLAG or unsafe >= UNSAFE_FLAG) else "auto"

    return {
        "queue": queue,
        "priority": priority,
        "handler": handler,
        "flags": flags,
        "department": department,
        "intent": intent,
        "reasons": [
            f"department={department}, intent={intent}",
            f"is_safe_to_drive={unsafe:.2f}",
            f"needs_human={needs_human:.2f}",
            ("still missing: " + ", ".join(missing)) if missing else "all required slots filled",
        ],
    }


# --------------------------------------------------------------------------- extraction
_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b"
    r"|\b(\d{1,2})\s*o'?clock\b"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def extract_time(text: str) -> str | None:
    """Deterministic clock/day extraction. Labelled as `extract` in the UI, not a model decision."""
    m = _TIME_RE.search(text)
    if not m:
        return None
    return m.group(0).strip()


def display_name(key: str) -> str:
    return key.replace("_", " ")
