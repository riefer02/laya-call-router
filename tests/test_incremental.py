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
    """A 6-option sub-queue with a clear winner scores only ~0.42 entropy confidence.

    Pinning on entropy confidence alone re-ran the sub-queue question on every turn; pinning
    on the top probability settles it.
    """
    s = session()
    answer = choice("collision", 0.42, 0.72)
    s._pin("subqueue", answer, 1, answer)
    assert s._is_pinned("subqueue") is True


def test_undecided_choice_is_not_pinned():
    s = session()
    answer = choice("parts", 0.10, 0.31)
    s._pin("destination", answer, 1, answer)
    assert s._is_pinned("destination") is False


def test_noul_pins_on_top_probability():
    s = session()
    answer = noul(0.05, 0.95)
    s._pin("needs_human", answer, 1, answer)
    assert s._is_pinned("needs_human") is True


def test_unknown_fact_is_not_pinned():
    assert session()._is_pinned("location") is False


def test_guessed_time_is_not_pinned_without_caller_evidence():
    s = session()
    s.exchanges.append({"role": "caller", "text": "I am looking for a new car."})
    answer = choice("next_week", 0.99, 0.99)
    s._pin("time_preference", answer, 1, answer)
    assert s._slot_missing("time_preference")
    s.exchanges.append({"role": "caller", "text": "Next week works."})
    s._pin("time_preference", answer, 2, answer)
    assert not s._slot_missing("time_preference")


def test_exhausted_transcript_marks_route_as_provisional():
    s = session()
    s.answers["destination"] = {"type": "choice", "choice": "service"}
    s.answers["subqueue"] = {"type": "choice", "choice": "mechanical_diagnostic"}
    events = list(s._finish(1, "t1.agent", [], pending=True))
    terminal = next(e for e in events if e["type"] == "node_result")
    assert s.completion == "awaiting_caller"
    assert s.routing["queue"] == "Service Department"
    assert terminal["value"] == "Awaiting caller"
