"""Car-dealership domain: the questions the switchboard asks, and how answers map to queues.

The taxonomy is **not defined here** — it lives in `config/store_profile.json` and is loaded via
`store_profile`. This module turns that profile into the typed questions the cascade asks and the
policy that converts answers into a queue.

Structure follows how dealerships actually organise (Fixed Operations: service/parts/body shop;
Variable Operations: sales/F&I), with tires and detailing as *Service sub-queues* rather than peer
departments, and roadside as a dispatch flag rather than a place. See `store_profile` for why.

Slots (vehicle, location, time) are `choice` questions over fixed enums, because Laya classifies
rather than parses. The one exception is the exact appointment time, a clearly-labelled
deterministic regex (`extract_time`) rendered as a different node kind so the UI never pretends a
model decided it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import store_profile as SP

PROFILE = SP.load()

# --------------------------------------------------------------------------- taxonomy views
# Kept as plain dicts/strings because that is what the question builders and the UI consume.
DESTINATIONS: Dict[str, str] = {d.key: d.description for d in PROFILE.destinations}
SUBQUEUES: Dict[str, Dict[str, str]] = {
    d.key: {s.key: s.description for s in PROFILE.subqueues_for(d.key)} for d in PROFILE.destinations
}
DESTINATION_QUESTION: Dict = PROFILE.destination_question()

# --------------------------------------------------------------------------- slots
LOCATIONS: Dict[str, str] = {loc["key"]: loc["description"] for loc in PROFILE.locations}

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

# First pass each turn: destination + slots, all answerable without knowing the branch.
PASS1_QUESTIONS: Dict = {
    **DESTINATION_QUESTION,
    "vehicle": SLOT_QUESTIONS["vehicle"],
    "location": SLOT_QUESTIONS["location"],
    "time_preference": SLOT_QUESTIONS["time_preference"],
    "is_safe_to_drive": SLOT_QUESTIONS["is_safe_to_drive"],
    "needs_human": SLOT_QUESTIONS["needs_human"],
}

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

# A choice answer equal to one of these means "the caller has not said", so it must never be
# pinned — it is exactly the thing a later turn is supposed to resolve.
UNRESOLVED_SENTINELS = {"not_stated"}
ESCALATION_MIN_OPTIONS = 4


def destination_question_paraphrase() -> Dict:
    """A second, independently-worded version of the destination question, used to *verify* a
    low-confidence answer rather than to replace it.

    Measured: re-asking with a narrowed option set does not improve accuracy — it re-rolls the
    answer and inflates confidence. Agreement between two phrasings is a signal; a second roll of
    the same question is not.
    """
    return {
        "destination": {
            "type": "choice",
            "instructions": (
                "Where does this caller's work belong — which team actually does it?"
            ),
            "criteria": DESTINATIONS,
        }
    }


def subqueue_question(destination: str) -> Optional[Dict[str, Any]]:
    return PROFILE.subqueue_question(destination)


def subqueue_question_paraphrase(destination: str) -> Optional[Dict[str, Any]]:
    subs = PROFILE.subqueues_for(destination)
    if not subs:
        return None
    dest = PROFILE.destination(destination)
    return {
        "subqueue": {
            "type": "choice",
            "instructions": (
                f"Which {dest.label.lower() if dest else destination} queue should take this?"
            ),
            "criteria": {s.key: s.description for s in subs},
        }
    }


# --------------------------------------------------------------------------- next action
NEXT_ACTION_LABELS: Dict[str, str] = {
    "ask_vehicle": "we do not yet know what kind of vehicle this is",
    "ask_location": "we do not yet know which location the caller wants",
    "ask_time": "we do not yet know when the caller wants to come in",
    "ask_detail": "the problem is too vague to book; ask for more detail",
    "confirm_booking": "we have everything needed to look for a real appointment time",
    "offer_slots": "real times have been read out; waiting for the caller to pick one",
    "ask_which_slot": "the caller's answer did not clearly name one of the offered times",
    "booked": "the caller accepted a time and the appointment is filed",
    "offer_transfer": "this is outside the routine booking flow; hand to a person",
}

# The agent must NOT read out the classifier's option list — measured, the spoken question leaks
# into the transcript the model then classifies and biases the slot answer.
RESPONSES: Dict[str, str] = {
    "ask_vehicle": "Thanks. What kind of vehicle is it?",
    "ask_location": "Got it — which of our locations works best for you?",
    "ask_time": "When would you like to come in?",
    "ask_detail": "I want to make sure we book the right thing — could you tell me a little more about what the vehicle is doing?",
    # Booking is a two-step: offer times that actually exist, then confirm the one they took.
    # "I'm booking you into service for next week" was never an appointment.
    "offer_slots": "I can get you in at {slots}. Which of those works best for you?",
    "ask_which_slot": "Sorry — which of those times did you want?",
    "booked": "You're all set: {booking}. We'll see you then.",
    "confirm_booking": "Perfect, I have everything I need — let me find you a time.",
    "offer_transfer": "Let me get you straight to the right team so nobody has to wait.",
}

REQUIRED_SLOTS = ("vehicle", "location", "time_preference")

# Destinations where booking an appointment is not the outcome: hand to a person instead.
TRANSFER_DESTINATIONS = {"non_customer"}


def slot_applies(destination: Optional[str], slot: str) -> bool:
    """Slots only matter for destinations that end in an appointment."""
    if destination in TRANSFER_DESTINATIONS:
        return False
    return slot in REQUIRED_SLOTS


def next_action_for(missing: List[str], destination: str, unsafe: float) -> str:
    """The switchboard's next step, as policy over the classifier's understanding."""
    if destination == "non_customer" or unsafe >= unsafe_threshold():
        return "offer_transfer"
    if not missing:
        return "confirm_booking"
    slot = missing[0]
    return "ask_time" if slot == "time_preference" else "ask_" + slot


# --------------------------------------------------------------------------- routing policy
UNSAFE_FLAG = 0.5  # fallback only; the live value comes from the store profile
NEEDS_HUMAN_FLAG = 0.5


def unsafe_threshold() -> float:
    """The bar for treating a call as unsafe-to-drive.

    One number, read from the store profile, used by every decision that depends on it: the
    `unsafe_to_drive` flag, the transfer decision, and the roadside dispatch. These were briefly
    three separate thresholds, which produced the contradiction of a caller dispatched as unsafe
    but not flagged as unsafe.
    """
    return float(PROFILE.policy.get("unsafe_threshold", UNSAFE_FLAG))


def decide(answers: Dict, missing: List[str]) -> Dict:
    """Terminal routing outcome: destination + optional sub-queue -> queue, priority, handler."""

    def pick(qid):
        ans = answers.get(qid)
        if not isinstance(ans, dict):
            return None
        if ans.get("type") == "choice":
            return ans.get("choice")
        if ans.get("type") == "noul":
            return ans.get("noul")
        return ans.get("score")

    def prob(qid) -> float:
        v = pick(qid)
        return float(v) if isinstance(v, (int, float)) else 0.0

    destination = pick("destination")
    if destination not in DESTINATIONS:
        destination = None
    subqueue = pick("subqueue")
    if PROFILE.subqueue(destination, subqueue) is None:
        subqueue = None
    unsafe = prob("is_safe_to_drive")
    needs_human = prob("needs_human")

    flags: List[str] = list(PROFILE.flags_for(destination, subqueue))
    if unsafe >= unsafe_threshold():
        flags.append("unsafe_to_drive")
    if needs_human >= NEEDS_HUMAN_FLAG:
        flags.append("needs_human")
    if missing:
        flags.append("missing_info")

    # Roadside is a policy outcome, never a destination: a stranded caller is dispatched whichever
    # department owns the work.
    roadside = PROFILE.policy.get("roadside_flag", "roadside_dispatch")
    if roadside in flags or unsafe >= unsafe_threshold():
        queue = PROFILE.policy.get("roadside_queue", "Roadside / Towing")
        if "dispatch" not in flags:
            flags.append("dispatch")
    else:
        queue = PROFILE.queue_for(destination, subqueue)

    priority = "HIGH" if (unsafe >= unsafe_threshold() or "dispatch" in flags) else "NORMAL"

    sub = PROFILE.subqueue(destination, subqueue)
    dest = PROFILE.destination(destination)
    handler = "auto"
    if needs_human >= PROFILE.policy.get("needs_human_threshold", NEEDS_HUMAN_FLAG):
        handler = "human"
    for obj in (sub, dest):
        if obj is not None and getattr(obj, "handler", "route") == "human":
            handler = "human"
    if unsafe >= unsafe_threshold():
        handler = "human"

    where = destination or "unknown"
    if subqueue:
        where += f"/{subqueue}"
    reasons = [
        f"destination={where}",
        f"is_safe_to_drive={unsafe:.2f}",
        f"needs_human={needs_human:.2f}",
        ("still missing: " + ", ".join(missing)) if missing else "all required slots filled",
    ]

    return {
        "queue": queue,
        "priority": priority,
        "handler": handler,
        "flags": flags,
        "destination": destination,
        "subqueue": subqueue,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- extraction
def extract_contact(text: str) -> dict:
    """Name and number in one pass, for the booking. Missing values stay missing."""
    return {"caller_name": extract_name(text), "callback_number": extract_phone(text)}
_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b"
    r"|\b(\d{1,2})\s*o'?clock\b"
    r"|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

# Deliberately conservative. A phone number is a fact you must not invent, so this matches only
# things that look like a number and nothing else: 7+ digits, optionally grouped, optionally with
# a leading + or an opening bracket. A near-miss (an address, an order number) is left to the
# caller to repeat rather than confidently written into an appointment.
_PHONE_RE = re.compile(r"\(?\+?\d[\d\s().-]{5,}\d")

# `my name is Dana`, `this is Dana`, `I'm Dana`, `Dana speaking`. Kept narrow on purpose: a wrong
# name on a booking is worse than no name, because a person will act on it.
#
# The prefix is case-insensitive but the NAME IS NOT, and that is load-bearing: a blanket
# re.IGNORECASE makes `[A-Z]` match lowercase, which captured "Dana and" out of
# "my name is Dana and my number is...". The capitalisation is the only signal that a word is a
# name rather than the next word of the sentence.
_NAME_RE = re.compile(
    r"\b(?i:my name is|this is|i am|i'm|it's)\s+([A-Z][a-z]+(?:[ '-][A-Z][a-z]+)?)"
)


def extract_time(text: str) -> str | None:
    """Deterministic clock/day extraction. Labelled as `extract` in the UI, not a model decision."""
    m = _TIME_RE.search(text)
    return m.group(0).strip() if m else None


def extract_phone(text: str) -> str | None:
    """Deterministic phone extraction, digits preserved as spoken.

    Not a classifier question: Laya chooses among options, it does not parse spans, and a callback
    number is the one field where a plausible-looking invention does real damage.
    """
    m = _PHONE_RE.search(text)
    if not m:
        return None
    raw = m.group(0).strip()
    return raw if len(re.sub(r"\D", "", raw)) >= 7 else None


def extract_name(text: str) -> str | None:
    """Deterministic name capture. Narrow by design - a wrong name is worse than none."""
    m = _NAME_RE.search(text)
    return m.group(1).strip() if m else None


def display_name(key: str) -> str:
    return key.replace("_", " ")
