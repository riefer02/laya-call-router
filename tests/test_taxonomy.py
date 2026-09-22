"""The training taxonomy must not drift from the one the cascade actually asks.

`training/taxonomy.json` is what the Kaggle notebook reads, and `dealership.py` is what the
running system uses. If they diverge, the fine-tuned model is trained on questions that are not
the questions being asked.
"""

from __future__ import annotations

import json
from pathlib import Path

from jev_classifier import dealership as D

TAXONOMY = Path(__file__).resolve().parents[1] / "training" / "taxonomy.json"


def load() -> dict:
    return json.loads(TAXONOMY.read_text())


def test_departments_match():
    assert load()["departments"] == D.DEPARTMENTS


def test_intents_match():
    assert load()["intents"] == D.INTENTS


def test_department_question_matches():
    assert load()["department_question"]["instructions"] == D.DEPARTMENT_QUESTION["department"]["instructions"]


def test_intent_template_matches_rendered_question():
    item = load()
    template = item["intent_question_template"]
    for department in D.DEPARTMENTS:
        rendered = D.intent_question(department)["intent"]["instructions"]
        assert template.format(department=department.replace("_", " ")) == rendered
