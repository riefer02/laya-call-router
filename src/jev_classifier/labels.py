"""Normalise and validate model output against the store profile's vocabulary.

DeepSeek (and any `json_object` provider) gives no structural guarantee, and in practice volunteers
near-misses: `"Body Shop"`, `"body shop"`, `"collision_repair"`, `"schedule body work"`. Everything
that becomes a training label goes through here first, so a label is either provably in the
profile's vocabulary or it is rejected. Silent coercion is how a dataset rots.

The alias tables are deliberately small and hand-checked. An aggressive fuzzy matcher would
"helpfully" turn an unsure teacher answer into a confident-looking label, which is the one failure
mode that poisons training data.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from . import store_profile as SP

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_KEY_NOISE = re.compile(r"[^a-z0-9]")

PROFILE = SP.load()
DESTINATION_KEYS = PROFILE.destination_keys
ALL_SUBQUEUE_KEYS = sorted({s.key for s in PROFILE.subqueues})

# Hand-checked synonyms only. Nothing speculative.
#
# The important ones are the structural corrections: a model that answers "tires" or "detailing" is
# naming a Service *sub-queue*, not a destination, so it maps to `service` — and the sub-queue
# question then decides the bay. Getting this wrong is what produced the old tires-vs-service
# confusion in the first place.
DEST_ALIASES = {
    "bodyshop": "body_shop",
    "body": "body_shop",
    "collision": "body_shop",
    "collision_repair": "body_shop",
    "mechanical": "service",
    "repair": "service",
    "repairs": "service",
    "workshop": "service",
    "service_center": "service",
    # sub-functions of Service that a model may name as if they were departments
    "tires": "service",
    "tire": "service",
    "tyre": "service",
    "tyres": "service",
    "wheel": "service",
    "detailing": "service",
    "detail": "service",
    "valet": "service",
    "wash": "service",
    "quick_lube": "service",
    "express": "service",
    "maintenance": "service",
    "towing": "service",
    "roadside": "service",
    "roadside_assistance": "service",
    "warranty": "service",
    # other
    "purchase": "sales",
    "purchasing": "sales",
    "leasing": "sales",
    "vehicle_sales": "sales",
    "financing": "finance",
    "f_and_i": "finance",
    "fi": "finance",
    "insurance": "finance",
    "operator": "front_desk",
    "reception": "front_desk",
    "general": "front_desk",
    "general_enquiry": "front_desk",
    "enquiry": "front_desk",
    "complaint": "front_desk",
    "other": "front_desk",
    "unknown": "front_desk",
    "vendor": "non_customer",
    "supplier": "non_customer",
    "jobseeker": "non_customer",
    "job_seeker": "non_customer",
    "recruiting": "non_customer",
    "wrong_number": "non_customer",
    "spam": "non_customer",
}

SUBQUEUE_ALIASES = {
    "oil_change": "express_maintenance",
    "scheduled_service": "express_maintenance",
    "routine_maintenance": "express_maintenance",
    "quick_lube": "express_maintenance",
    "tyre": "tires",
    "tyres": "tires",
    "tire": "tires",
    "flat": "tires",
    "puncture": "tires",
    "alignment": "tires",
    "wheel_alignment": "tires",
    "wont_start": "mechanical_diagnostic",
    "will_not_start": "mechanical_diagnostic",
    "no_start": "mechanical_diagnostic",
    "dead_battery": "mechanical_diagnostic",
    "engine_light": "mechanical_diagnostic",
    "warning_light": "mechanical_diagnostic",
    "dashboard_light": "mechanical_diagnostic",
    "brake": "mechanical_diagnostic",
    "brakes": "mechanical_diagnostic",
    "noise": "mechanical_diagnostic",
    "performance": "mechanical_diagnostic",
    "stalling": "mechanical_diagnostic",
    "diagnostics": "mechanical_diagnostic",
    "diagnostic": "mechanical_diagnostic",
    "tow_needed": "roadside_assistance",
    "tow": "roadside_assistance",
    "jump_start": "roadside_assistance",
    "lockout": "roadside_assistance",
    "fuel": "roadside_assistance",
    "recall": "warranty_recall",
    "warranty": "warranty_recall",
    "wash": "detailing",
    "car_wash": "detailing",
    "full_detail": "detailing",
    "interior": "detailing",
    "paint_correction": "detailing",
    "polish": "detailing",
    "order": "order_part",
    "in_stock": "availability",
    "stock": "availability",
    "stock_check": "availability",
    "floor_mats": "accessory",
    "add_on": "accessory",
    "accident": "collision",
    "crash": "collision",
    "rear_ended": "collision",
    "dent": "dent_scratch",
    "scratch": "dent_scratch",
    "windscreen": "glass",
    "windshield": "glass",
    "paintwork": "paint",
    "respray": "paint",
    "new_car": "new_vehicle",
    "used_car": "used_vehicle",
    "second_hand": "used_vehicle",
    "tradein": "trade_in",
    "appraisal": "trade_in",
    "lease": "lease_terms",
    "monthly_payment": "loan",
    "finance_application": "loan",
    "gap_insurance": "insurance",
    "contract": "paperwork",
    "documents": "paperwork",
    "opening_hours": "general_question",
    "location": "general_question",
    "general": "general_question",
    "question": "general_question",
    "complaint": "feedback",
    "compliment": "feedback",
    "job": "jobseeker",
    "hiring": "jobseeker",
    "wrong_number": "wrong_number",
    "supplier": "vendor",
}


def canon(value: object) -> str:
    """Lowercase, collapse anything non-alphanumeric to single underscores."""
    if not isinstance(value, str):
        return ""
    return _NON_ALNUM.sub("_", value.strip().lower()).strip("_")


def field(raw: object, *names: str):
    """Read a key from a model's JSON, tolerating spelling drift.

    Measured: told the key is `subqueue`, DeepSeek answers `sub_queue` — a reasonable
    normalisation of "sub-queue" that a strict reader rejects. Since `json_object` providers give no
    structural guarantee, match keys case- and separator-insensitively rather than discarding a
    correct answer over punctuation.
    """
    if not isinstance(raw, dict):
        return None
    for name in names:
        value = raw.get(name)
        if value not in (None, ""):
            return value
    canonical = {_KEY_NOISE.sub("", str(k).lower()): v for k, v in raw.items()}
    for name in names:
        value = canonical.get(_KEY_NOISE.sub("", name.lower()))
        if value not in (None, ""):
            return value
    return None


def normalise_destination(raw: object) -> Optional[str]:
    c = canon(raw)
    if not c:
        return None
    if c in DESTINATION_KEYS:
        return c
    return DEST_ALIASES.get(c)


def normalise_subqueue(raw: object, destination: Optional[str]) -> Optional[str]:
    c = canon(raw)
    if not c or not destination:
        return None
    branch = PROFILE.subqueue_keys(destination)
    if c in branch:
        return c
    alias = SUBQUEUE_ALIASES.get(c)
    if alias and alias in branch:
        return alias
    return None


def validate_label(
    raw_destination: object, raw_subqueue: object
) -> Tuple[Optional[str], Optional[str], bool]:
    """Return (destination, subqueue, ok).

    `ok` requires a valid destination. A destination with no sub-queues in this profile is fine —
    the destination is the whole answer — so a missing sub-queue is not an error there.
    """
    destination = normalise_destination(raw_destination)
    if destination is None:
        return None, None, False
    if not PROFILE.subqueue_keys(destination):
        return destination, None, True
    subqueue = normalise_subqueue(raw_subqueue, destination)
    return destination, subqueue, subqueue is not None
