"""Dealership policy tests. Pure logic — no model, no network."""

from __future__ import annotations

from jev_classifier import dealership as D


def test_asks_for_the_first_missing_slot():
    assert D.next_action_for(["vehicle", "location"], "service", 0.0) == "ask_vehicle"
    assert D.next_action_for(["location"], "service", 0.0) == "ask_location"
    assert D.next_action_for(["time_preference"], "service", 0.0) == "ask_time"


def test_confirms_when_nothing_is_missing():
    assert D.next_action_for([], "body_shop", 0.1) == "confirm_booking"


def test_unsafe_short_circuits_to_a_human():
    # A stranded caller should not be walked through a booking flow.
    assert D.next_action_for(["vehicle"], "service", 0.9) == "offer_transfer"
    assert D.next_action_for([], "tires", 0.8) == "offer_transfer"


def test_general_department_transfers():
    assert D.next_action_for([], "general", 0.0) == "offer_transfer"


def _answers(**kwargs):
    out = {}
    for key, value in kwargs.items():
        if key in ("department", "intent"):
            out[key] = {"type": "choice", "choice": value, "probabilities": {value: 1.0}}
        else:
            out[key] = {"type": "noul", "noul": value}
    return out


def test_no_start_dispatches_to_roadside():
    r = D.decide(_answers(department="service", intent="no_start", is_safe_to_drive=0.8, needs_human=0.7), [])
    assert r["queue"] == "Roadside / Towing"
    assert "dispatch" in r["flags"]
    assert r["priority"] == "HIGH"
    assert r["handler"] == "human"


def test_collision_routes_to_body_shop():
    r = D.decide(_answers(department="body_shop", intent="collision", is_safe_to_drive=0.2, needs_human=0.25), [])
    assert r["queue"] == "Body Shop"
    assert r["priority"] == "NORMAL"
    assert r["handler"] == "auto"


def test_needs_human_raises_handler_not_queue():
    r = D.decide(_answers(department="parts", intent="order_part", is_safe_to_drive=0.0, needs_human=0.8), [])
    assert r["queue"] == "Parts Counter"
    assert r["handler"] == "human"
    assert "needs_human" in r["flags"]


def test_extract_time_finds_clock_and_day():
    assert D.extract_time("around 9am please") == "9am"
    assert D.extract_time("Tuesday works") == "Tuesday"
    assert D.extract_time("no preference") is None
