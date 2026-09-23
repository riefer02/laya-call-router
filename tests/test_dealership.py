"""Dealership policy and profile tests. Pure logic — no model, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_classifier import dealership as D
from jev_classifier import store_profile as SP

ROOT = Path(__file__).resolve().parents[1]


def profile() -> SP.StoreProfile:
    SP.clear_cache()
    return SP.load()


# --------------------------------------------------------------------------- taxonomy structure
def test_tires_and_detailing_are_service_subqueues_not_destinations():
    """The correction that motivated the whole refactor: these are Fixed-Ops adjuncts, so they
    belong under service, and a model naming them must map to service."""
    p = profile()
    assert "tires" not in p.destination_keys
    assert "detailing" not in p.destination_keys
    assert "towing" not in p.destination_keys
    assert "tires" in p.subqueue_keys("service")
    assert "detailing" in p.subqueue_keys("service")
    assert "roadside_assistance" in p.subqueue_keys("service")


def test_destinations_follow_fixed_and_variable_operations():
    keys = set(profile().destination_keys)
    assert {"service", "parts", "body_shop"} <= keys  # Fixed Operations
    assert {"sales", "finance"} <= keys  # Variable Operations
    assert {"front_desk", "non_customer"} <= keys


def test_roadside_is_a_flag_not_a_destination():
    p = profile()
    assert "roadside_dispatch" in p.flags_for("service", "roadside_assistance")
    assert p.queue_for("service", "roadside_assistance") == "Roadside / Towing"


def test_every_subqueue_belongs_to_a_real_destination():
    p = profile()
    for sub in p.subqueues:
        assert p.destination(sub.parent) is not None


# --------------------------------------------------------------------------- next action
def test_asks_for_the_first_missing_slot():
    p = profile()
    assert D.next_action_for(["vehicle", "location"], "service", 0.0) == "ask_vehicle"
    assert D.next_action_for(["location"], "service", 0.0) == "ask_location"
    assert D.next_action_for(["time_preference"], "service", 0.0) == "ask_time"


def test_confirms_when_nothing_is_missing():
    assert D.next_action_for([], "body_shop", 0.1) == "confirm_booking"


def test_unsafe_short_circuits_to_a_human():
    assert D.next_action_for(["vehicle"], "service", 0.9) == "offer_transfer"


def test_non_customer_always_transfers():
    assert D.next_action_for([], "non_customer", 0.0) == "offer_transfer"


def test_slots_do_not_apply_to_transfer_destinations():
    assert D.slot_applies("service", "vehicle") is True
    assert D.slot_applies("non_customer", "vehicle") is False


# --------------------------------------------------------------------------- routing policy
def _choice(qid, value):
    return {qid: {"type": "choice", "choice": value, "probabilities": {value: 1.0}}}


def test_flat_tire_routes_to_the_tire_bay():
    r = D.decide({**_choice("destination", "service"), **_choice("subqueue", "tires")}, [])
    assert r["queue"] == "Tire Bay"
    assert r["destination"] == "service" and r["subqueue"] == "tires"


def test_roadside_subqueue_dispatches():
    r = D.decide(
        {**_choice("destination", "service"), **_choice("subqueue", "roadside_assistance")}, []
    )
    assert r["queue"] == "Roadside / Towing"
    assert "dispatch" in r["flags"]
    assert r["priority"] == "HIGH"


def test_unsafe_drives_dispatch_regardless_of_subqueue():
    r = D.decide(
        {
            **_choice("destination", "service"),
            **_choice("subqueue", "mechanical_diagnostic"),
            "is_safe_to_drive": {"type": "noul", "noul": 0.8},
        },
        [],
    )
    assert r["queue"] == "Roadside / Towing"
    assert "dispatch" in r["flags"]
    assert r["handler"] == "human"


def test_partial_cascade_still_routes_somewhere():
    r = D.decide(_choice("destination", "parts"), [])
    assert r["queue"] == "Parts Counter"


def test_unknown_destination_does_not_crash():
    r = D.decide({"destination": {"type": "choice", "choice": "billing"}}, [])
    assert r["destination"] is None
    assert r["queue"]  # still a queue


def test_invalid_subqueue_is_ignored():
    r = D.decide({**_choice("destination", "finance"), **_choice("subqueue", "tires")}, [])
    assert r["subqueue"] is None
    assert r["queue"] == "Finance & Insurance"


def test_non_customer_is_handled_by_a_human():
    r = D.decide({**_choice("destination", "non_customer"), **_choice("subqueue", "jobseeker")}, [])
    assert r["handler"] == "human"
    assert r["queue"] == "Front Desk"


def test_missing_info_is_flagged():
    r = D.decide({**_choice("destination", "sales"), **_choice("subqueue", "new_vehicle")}, ["vehicle"])
    assert "missing_info" in r["flags"]


# --------------------------------------------------------------------------- profile as config
def test_a_store_can_promote_tires_to_a_destination(tmp_path):
    """A store with its own tyre centre promotes it out of the service sub-queues. No code change."""
    raw = json.loads((ROOT / "config" / "store_profile.json").read_text())
    raw["destinations"].append(
        {
            "key": "tire_centre",
            "label": "Tire centre",
            "description": "the store's own tyre centre",
            "queue": "Tire Centre",
        }
    )
    raw["subqueues"] = [s for s in raw["subqueues"] if s["key"] != "tires"]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))
    SP.clear_cache()
    p = SP.load(str(path))
    assert "tire_centre" in p.destination_keys
    assert "tires" not in p.subqueue_keys("service")
    SP.clear_cache()


def test_shipped_profile_is_valid():
    p = profile()
    p.validate()
    assert len(p.destination_keys) == 7
    assert "other" in p.subqueue_keys("service")  # residual exists so the model can decline


# --------------------------------------------------------------------------- question text
def test_question_templates_live_in_the_profile():
    """Training and inference must ask the identical question.

    The fine-tune learns to answer one exact instruction string. A copy of that string drifted
    once already (training said "service", inference said "Service"), which asks the model a
    question it was never trained on. Both sides now read the profile, so this asserts the
    templates are present and wired to the built questions rather than duplicated in code.
    """
    p = profile()
    assert set(p.questions) >= {"destination", "subqueue"}
    assert p.destination_question()["destination"]["instructions"] == p.question_text("destination")
    assert (
        p.subqueue_question("service")["subqueue"]["instructions"]
        == p.question_text("subqueue", label="Service")
    )


def test_subqueue_instruction_uses_the_destination_label_verbatim():
    """`Service`, not `service`. The label is what the runtime sends, so it is what training sends."""
    p = profile()
    assert "This is a Service call." in p.subqueue_question("service")["subqueue"]["instructions"]
    assert "This is a Body Shop call." in p.subqueue_question("body_shop")["subqueue"]["instructions"]


def test_subqueue_question_needs_a_substitution():
    p = profile()
    assert "{label}" in p.questions["subqueue"]
    assert "{label}" not in p.question_text("subqueue", label="Parts")


def test_a_profile_without_templates_falls_back_to_the_defaults():
    p = profile()
    bare = SP.StoreProfile(
        name="no templates", destinations=p.destinations, subqueues=p.subqueues
    )
    assert bare.destination_question()["destination"]["instructions"] == SP.DEFAULT_QUESTIONS["destination"]
    assert "This is a Service call." in bare.subqueue_question("service")["subqueue"]["instructions"]


# --------------------------------------------------------------------------- contact capture
def test_phone_is_extracted_in_several_formats():
    for text, want in [
        ("my number is 555-0140", "555-0140"),
        ("call me on (555) 019-2837", "(555) 019-2837"),
        ("reach me at +1 555 010 9988", "+1 555 010 9988"),
        ("my cell is 555.0192", "555.0192"),
    ]:
        assert D.extract_phone(text) == want, text


def test_phone_is_not_invented_from_an_address_or_a_mileage():
    """A plausible-looking number written into an appointment is the one field that does damage."""
    assert D.extract_phone("the address is 1400 Riverside Drive") is None
    assert D.extract_phone("a 2019 Honda with 45000 miles") is None
    assert D.extract_phone("my car won't start") is None


def test_name_capture_does_not_swallow_the_next_word():
    """Regression: re.IGNORECASE made `[A-Z]` match lowercase and captured "Dana and"."""
    assert D.extract_name("my name is Dana and my number is 555-0140") == "Dana"
    assert D.extract_name("This is Sam Whitfield, call me") == "Sam Whitfield"
    assert D.extract_name("I'm Alex. Reach me at 9") == "Alex"


def test_name_is_not_invented_when_absent():
    assert D.extract_name("my car won't start") is None
    assert D.extract_name("I need a service") is None


def test_extract_contact_returns_both_and_leaves_missing_missing():
    got = D.extract_contact("Hi, my name is Dana and my number is 555-0140.")
    assert got == {"caller_name": "Dana", "callback_number": "555-0140"}
    assert D.extract_contact("my car won't start") == {"caller_name": None, "callback_number": None}
