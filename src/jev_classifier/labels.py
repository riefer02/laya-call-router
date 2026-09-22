"""Normalise and validate model output against our department/intent vocabulary.

DeepSeek (and any `json_object` provider) gives no structural guarantee, and in practice
volunteers near-misses: `"Body Shop"`, `"body shop"`, `"collision_repair"`, `"schedule body work"`.
Everything that becomes a training label goes through here first, so a label is either
provably in our vocabulary or it is rejected. Silent coercion is how a dataset rots.

The alias tables are deliberately small and hand-checked. An aggressive fuzzy matcher would
"helpfully" turn an unsure teacher answer into a confident-looking label, which is the one
failure mode that poisons training data.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from . import dealership as D

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Hand-checked synonyms only. Nothing speculative.
DEPT_ALIASES = {
    "bodyshop": "body_shop",
    "body": "body_shop",
    "collision": "body_shop",
    "collision_repair": "body_shop",
    "mechanical": "service",
    "repair": "service",
    "repairs": "service",
    "maintenance": "service",
    "workshop": "service",
    "tyre": "tires",
    "tyres": "tires",
    "tire": "tires",
    "wheel": "tires",
    "wheels": "tires",
    "wash": "detailing",
    "valet": "detailing",
    "cleaning": "detailing",
    "detailing_and_cleaning": "detailing",
    "purchase": "sales",
    "purchasing": "sales",
    "leasing": "sales",
    "vehicle_sales": "sales",
    "financing": "finance",
    "f_and_i": "finance",
    "fi": "finance",
    "roadside": "towing",
    "roadside_assistance": "towing",
    "tow": "towing",
    "towing_and_roadside": "towing",
    "front_desk": "general",
    "operator": "general",
    "reception": "general",
    "other": "general",
    "unknown": "general",
}

INTENT_ALIASES = {
    "wont_start": "no_start",
    "will_not_start": "no_start",
    "dead_battery": "no_start",
    "engine_light": "warning_light",
    "dashboard_light": "warning_light",
    "brake": "brakes",
    "brake_issue": "brakes",
    "brake_problem": "brakes",
    "strange_noise": "noise",
    "weird_noise": "noise",
    "rattling": "noise",
    "service": "maintenance",
    "oil_change": "maintenance",
    "scheduled_service": "maintenance",
    "stalling": "performance",
    "loss_of_power": "performance",
    "accident": "collision",
    "crash": "collision",
    "collision_repair": "collision",
    "rear_ended": "collision",
    "dent": "dent_scratch",
    "scratch": "dent_scratch",
    "dents_and_scratches": "dent_scratch",
    "windscreen": "glass",
    "windshield": "glass",
    "cracked_windscreen": "glass",
    "paintwork": "paint",
    "respray": "paint",
    "order": "order_part",
    "part_order": "order_part",
    "in_stock": "availability",
    "stock_check": "availability",
    "part_availability": "availability",
    "floor_mats": "accessory",
    "add_on": "accessory",
    "puncture": "flat",
    "flat_tyre": "flat",
    "new_tires": "replacement",
    "tyre_replacement": "replacement",
    "balance": "rotation",
    "tyre_rotation": "rotation",
    "wheel_alignment": "alignment",
    "tracking": "alignment",
    "car_wash": "wash",
    "interior_clean": "interior",
    "valet": "full_detail",
    "full_detail": "full_detail",
    "polish": "paint_correction",
    "buffing": "paint_correction",
    "new_car": "new_vehicle",
    "used_car": "used_vehicle",
    "second_hand": "used_vehicle",
    "tradein": "trade_in",
    "trade": "trade_in",
    "appraisal": "trade_in",
    "stock": "inventory",
    "lease": "lease_terms",
    "monthly_payment": "loan",
    "finance_application": "loan",
    "gap_insurance": "insurance",
    "contract": "paperwork",
    "documents": "paperwork",
    "tow_truck": "tow_needed",
    "breakdown": "tow_needed",
    "jump": "jump_start",
    "battery_boost": "jump_start",
    "keys_locked_in": "lockout",
    "locked_out": "lockout",
    "out_of_fuel": "fuel",
    "fuel_delivery": "fuel",
    "opening_hours": "general_question",
    "location": "general_question",
    "question": "general_question",
    "complaint": "other",
    "general": "general_question",
}


def canon(value: object) -> str:
    """Lowercase, collapse anything non-alphanumeric to single underscores."""
    if not isinstance(value, str):
        return ""
    return _NON_ALNUM.sub("_", value.strip().lower()).strip("_")


def normalise_department(raw: object) -> Optional[str]:
    c = canon(raw)
    if not c:
        return None
    if c in D.DEPARTMENTS:
        return c
    return DEPT_ALIASES.get(c)


def normalise_intent(raw: object, department: Optional[str]) -> Optional[str]:
    c = canon(raw)
    if not c or not department:
        return None
    branch = D.INTENTS.get(department, {})
    if c in branch:
        return c
    alias = INTENT_ALIASES.get(c)
    if alias and alias in branch:
        return alias
    return None


def validate_label(raw_department: object, raw_intent: object) -> Tuple[Optional[str], Optional[str], bool]:
    """Return (department, intent, ok). ok is False if either could not be placed in-vocabulary."""
    department = normalise_department(raw_department)
    intent = normalise_intent(raw_intent, department)
    return department, intent, department is not None and intent is not None
