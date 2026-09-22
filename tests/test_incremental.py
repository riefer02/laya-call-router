"""Incremental-evaluation tests: when a fact may be pinned, and when it must be re-checked.

No model runs here — `_pin` is pure bookkeeping, so these pin the *rules* rather than the model's
answers.
"""

from __future__ import annotations

from jev_classifier.call import CallSession

SCENARIO = {"id": "test", "label": "test", "turns": []}


def session() -> CallSession:
    return CallSession(SCENARIO, session_id="test")


def choice(value: str, confidence: float, top: float) -> dict:
    """A choice answer as `_summarize` would present it (answer + summary fields in one dict)."""
    return {
        "type": "choice",
        "primitive": "choice",
        "choice": value,
        "confidence": confidence,
        "top_probability": top,
        "probabilities": {value: top},
    }


def noul(value: float, confidence: float) -> dict:
    return {
        "type": "noul",
        "primitive": "noul",
        "noul": value,
        "confidence": confidence,
        "top_probability": max(value, 1 - value),
    }


def test_not_stated_is_never_pinned():
    """`not_stated` is exactly what a later turn is supposed to resolve, so it must stay open."""
    s = session()
    answer = choice("not_stated", 0.9, 0.9)
    s._pin("vehicle", answer, 1, answer)
    assert s._is_pinned("vehicle") is False


def test_concrete_confident_value_is_pinned():
    s = session()
    answer = choice("truck", 0.86, 0.95)
    s._pin("vehicle", answer, 2, answer)
    assert s._is_pinned("vehicle") is True
    assert s.facts["vehicle"]["turn"] == 2


def test_wide_choice_pins_on_top_probability_not_entropy():
    """A 7-option intent with a clear winner scores only ~0.42 entropy confidence.

    Pinning on entropy confidence alone re-ran the intent question on every single turn; pinning
    on the top probability settles it.
    """
    s = session()
    answer = choice("collision", 0.42, 0.72)
    s._pin("intent", answer, 1, answer)
    assert s._is_pinned("intent") is True


def test_undecided_choice_is_not_pinned():
    s = session()
    answer = choice("billing", 0.10, 0.31)
    s._pin("department", answer, 1, answer)
    assert s._is_pinned("department") is False


def test_noul_pins_on_top_probability():
    s = session()
    answer = noul(0.05, 0.95)
    s._pin("needs_human", answer, 1, answer)
    assert s._is_pinned("needs_human") is True


def test_unknown_fact_is_not_pinned():
    assert session()._is_pinned("location") is False
