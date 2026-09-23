"""Scripted caller turns for the demo.

The caller is scripted so a run is deterministic and records cleanly. The seam is a plain list of
strings, so a real voice/STT caller can replace it later without touching the cascade or the UI.
"""

from __future__ import annotations

from typing import Dict, List

SCENARIOS: List[Dict] = [
    {
        "id": "no_start",
        "label": "Won't start",
        "blurb": "Roadside → Service, unsafe to drive",
        "expect": {"queue": "Roadside / Towing", "completion": "dispatched"},
        "turns": [
            "Hi, my car won't start at all. I think I need service.",
        ],
    },
    {
        "id": "collision",
        "label": "Accident damage",
        "blurb": "Body shop → appointment",
        "expect": {"queue": "Body Shop", "completion": "booked"},
        "turns": [
            "Someone rear-ended me in a parking lot yesterday. I need body work.",
            "It's a Ford F-150 truck.",
            "Northside works for me.",
            "Next week is fine — how about Sunday at 3am?",
            "Sorry, this is Morgan, 555-0136. The first time you offered works.",
        ],
    },
    {
        "id": "flat_tire",
        "label": "Flat tire",
        "blurb": "Roadside dispatch, urgent",
        "expect": {"queue": "Roadside / Towing", "completion": "dispatched"},
        "turns": [
            "I've got a flat tire and I'm stuck on the highway.",
        ],
    },
    {
        "id": "buy_car",
        "label": "Buying a car",
        "blurb": "Sales floor → appointment",
        "expect": {"queue": "Sales Floor", "completion": "booked"},
        "turns": [
            "I'm thinking about buying a new car, maybe an electric one.",
            "I'd like to visit the westside showroom.",
            "This week sometime would work, but I haven't chosen a time yet.",
            "Sure — this is Dana, and my number is 555-0140.",
            "Sunday at 3am works for me.",
            # "the first one" rather than "the 8am one": the offer depends on what is still free, and
            # a scenario that names a fixed time only books on the first run. Measured - after one
            # booking took 8am, the next run offered 9am and this turn was correctly vetoed as
            # naming a time that was not on offer, so nothing was booked.
            "Sorry — yes, the first one please.",
        ],
    },
    {
        "id": "part_order",
        "label": "Order a part",
        "blurb": "Parts counter → handoff",
        "expect": {"queue": "Parts Counter", "completion": "transferred"},
        "turns": [
            "I need to order a replacement side mirror for my van.",
        ],
    },
    {
        "id": "vague",
        "label": "Service diagnostic",
        "blurb": "Drivable car → Service appointment",
        "expect": {"queue": "Service Department", "completion": "booked"},
        "turns": [
            "My car is making a weird noise but still drives normally.",
            "It's a small SUV.",
            "The downtown one.",
            "This is Alex, 555-0177. Tomorrow works, but I haven't chosen a time yet.",
            "The first time you offered works.",
        ],
    },
    {
        "id": "out_of_scope",
        "label": "Wrong number",
        "blurb": "Not a customer → Front Desk",
        "expect": {"queue": "Front Desk", "completion": "transferred"},
        "turns": [
            "Sorry, I have the wrong number. I meant to call the dentist.",
        ],
    },
    {
        "id": "finance_question",
        "label": "Loan payment",
        "blurb": "Finance → handoff",
        "expect": {"queue": "Finance & Insurance", "completion": "transferred"},
        "turns": ["I have a question about my car loan payment."],
    },
    {
        "id": "hours",
        "label": "Opening hours",
        "blurb": "Front Desk → handoff",
        "expect": {"queue": "Front Desk", "completion": "transferred"},
        "turns": ["What time do you close today?"],
    },
    {
        "id": "job_applicant",
        "label": "Job applicant",
        "blurb": "Not a customer → Front Desk",
        "expect": {"queue": "Front Desk", "completion": "transferred"},
        "turns": ["I am applying for the receptionist job you posted."],
    },
    {
        "id": "tire_quote",
        "label": "New tires",
        "blurb": "Drivable SUV → Tire Bay appointment",
        "expect": {"queue": "Tire Bay", "completion": "booked"},
        "turns": [
            "Can I buy a new set of tires for my SUV?",
            "The downtown location, please.",
            "This is Riley, 555-0148. Next week works, but I haven't picked a time yet.",
            "The first time you offered works.",
        ],
    },
    {
        "id": "ambiguous_off_topic",
        "label": "Ambiguous off-topic call (known failure)",
        "blurb": "Safety false positive: should reach Front Desk",
        "expect": {"queue": "Front Desk", "completion": "transferred"},
        "known_issue": "The active model dispatches before the caller can clarify.",
        "turns": [
            "Hi, I'm calling about my neighbour's dog that keeps barking all night.",
            "No, it isn't about a car at all.",
        ],
    },
]

SCENARIO_BY_ID: Dict[str, Dict] = {s["id"]: s for s in SCENARIOS}
