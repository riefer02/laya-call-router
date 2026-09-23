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
        "turns": [
            "Hi, my car won't start at all. I think I need service.",
            "It's a Honda Civic, a sedan.",
            "The downtown location, please.",
            "Tomorrow morning would be great, around 9am.",
        ],
    },
    {
        "id": "collision",
        "label": "Accident damage",
        "blurb": "Body shop, collision repair",
        "turns": [
            "Someone rear-ended me in a parking lot yesterday. I need body work.",
            "It's a Ford F-150 truck.",
            "Northside works for me.",
            "Next week is fine — how about Tuesday at 10am?",
        ],
    },
    {
        "id": "flat_tire",
        "label": "Flat tire",
        "blurb": "Roadside dispatch, urgent",
        "turns": [
            "I've got a flat tire and I'm stuck on the highway.",
            "It's an SUV.",
            "The airport location.",
            "Today, as soon as possible.",
        ],
    },
    {
        "id": "buy_car",
        "label": "Buying a car",
        "blurb": "Sales floor",
        "turns": [
            "I'm thinking about buying a new car, maybe an electric one.",
            "I'd like to visit the westside showroom.",
            "This week sometime would work, maybe Thursday.",
            "Sure — this is Dana, and my number is 555-0140.",
            "Tuesday at 8 works for me.",
        ],
    },
    {
        "id": "part_order",
        "label": "Order a part",
        "blurb": "Parts counter",
        "turns": [
            "I need to order a replacement side mirror for my van.",
            "Downtown is closest.",
            "Next week is fine.",
        ],
    },
    {
        "id": "vague",
        "label": "Vague complaint",
        "blurb": "Ambiguous department — narrows the decision",
        "turns": [
            "Hi, something is not right with my car. There is a weird noise and the steering feels off.",
            "It's a small SUV.",
            "The downtown one.",
            "Tomorrow afternoon works.",
        ],
    },
    {
        "id": "out_of_scope",
        "label": "Not a car matter",
        "blurb": "Out of scope → transfer to a person",
        "turns": [
            "Hi, I'm calling about my neighbour's dog that keeps barking all night.",
            "No, it isn't about a car at all.",
        ],
    },
]

SCENARIO_BY_ID: Dict[str, Dict] = {s["id"]: s for s in SCENARIOS}
