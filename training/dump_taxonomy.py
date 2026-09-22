"""Dump the dealership taxonomy to JSON so the training notebook does not need our package.

Run after changing `dealership.py`; `tests/test_taxonomy.py` fails if the two drift apart.

    uv run python training/dump_taxonomy.py
"""

from __future__ import annotations

import json
from pathlib import Path

from jev_classifier import dealership as D

HERE = Path(__file__).resolve().parent


def main() -> None:
    payload = {
        "_note": "generated from src/jev_classifier/dealership.py — do not edit by hand",
        "departments": D.DEPARTMENTS,
        "intents": D.INTENTS,
        "department_question": {
            "instructions": D.DEPARTMENT_QUESTION["department"]["instructions"]
        },
        # The template uses {department}; underscores are already replaced with spaces by the caller.
        "intent_question_template": (
            "This is a {department} request. What exactly does the caller want? "
            "Pick the single closest option."
        ),
    }
    out = HERE / "taxonomy.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  {len(D.DEPARTMENTS)} departments, "
          f"{sum(len(b) for b in D.INTENTS.values())} intents across {len(D.INTENTS)} branches")


if __name__ == "__main__":
    main()
