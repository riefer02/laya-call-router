"""Mock availability and booking.

Pure logic — a fixed date, a temp store, no model and no network. The date is pinned because the
whole point of this module is that a slot is a real time, so a test that drifted with the calendar
would be testing nothing.
"""

from __future__ import annotations

from datetime import date

import pytest

from jev_classifier import store_profile as SP
from jev_classifier.schedule import (
    Booking,
    Scheduler,
    Slot,
    acceptance_options,
    resolve_acceptance,
)

# 2026-09-22 is a Tuesday, 09-26 a Saturday, 09-27 a Sunday.
TUESDAY = date(2026, 9, 22)
SATURDAY = date(2026, 9, 26)
SUNDAY = date(2026, 9, 27)


@pytest.fixture()
def sched(tmp_path):
    SP.clear_cache()
    return Scheduler(SP.load(), store_path=tmp_path / "bookings.jsonl")


# --------------------------------------------------------------------------- spoken form
def test_slot_label_is_spoken_not_padded():
    assert Slot("2026-09-22", "09:00").label() == "9am"
    assert Slot("2026-09-22", "09:30").label() == "9:30am"
    assert Slot("2026-09-22", "13:30").label() == "1:30pm"
    assert Slot("2026-09-22", "12:00").label() == "12pm"
    assert Slot("2026-09-22", "00:00").label() == "12am"


def test_slot_spoken_names_the_day():
    """A caller cannot act on '2026-09-22T09:00'."""
    spoken = Slot("2026-09-22", "09:00").spoken()
    assert "Tuesday" in spoken and "22 September" in spoken and "9am" in spoken


# --------------------------------------------------------------------------- service shape
def test_duration_comes_from_the_profile():
    assert Scheduler(SP.load()).duration_for("express_maintenance") == 45
    assert Scheduler(SP.load()).duration_for("mechanical_diagnostic") == 240


def test_roadside_is_not_an_appointment(sched):
    """Dispatch is not scheduling: a stranded caller is sent help, not offered a bay."""
    assert sched.duration_for("roadside_assistance") is None
    assert sched.availability("downtown", "roadside_assistance", TUESDAY) == []


def test_unknown_subqueue_falls_back_to_the_default(sched):
    assert sched.duration_for("something_new") == 60


def test_opening_hours_differ_by_day(sched):
    assert sched.opening_hours(TUESDAY) == (8 * 60, 18 * 60)
    assert sched.opening_hours(SATURDAY) == (8 * 60, 16 * 60)
    assert sched.opening_hours(SUNDAY) is None  # closed


# --------------------------------------------------------------------------- availability
def test_slots_stay_inside_opening_hours(sched):
    slots = sched.availability("downtown", "express_maintenance", TUESDAY)
    assert slots[0].time == "08:00"
    # A 45-minute job on a 30-minute grid must finish by close, so the last start is 17:00:
    # 17:30 would run to 18:15, past the 18:00 close.
    assert slots[-1].time == "17:00"
    assert all(s.day == "2026-09-22" for s in slots)


def test_closed_day_offers_nothing(sched):
    assert sched.availability("downtown", "express_maintenance", SUNDAY) == []


def test_a_longer_job_starts_earlier_in_the_day(sched):
    """A 4-hour diagnostic cannot be offered at 5pm if the store shuts at 6."""
    last = sched.availability("downtown", "mechanical_diagnostic", TUESDAY)[-1]
    assert last.time == "14:00"


def test_bays_cap_how_many_jobs_run_at_once(sched):
    """express_maintenance has 3 bays. Filling them must remove the early slots."""
    before = sched.availability("downtown", "express_maintenance", TUESDAY)
    assert Slot("2026-09-22", "08:00") in before
    for _ in range(sched.bays_for("express_maintenance")):
        sched.book(
            location="downtown", destination="service", subqueue="express_maintenance",
            queue="Express Bay", slot=Slot("2026-09-22", "08:00"),
        )
    after = sched.availability("downtown", "express_maintenance", TUESDAY)
    assert Slot("2026-09-22", "08:00") not in after
    # still bookable later the same day - the bay frees up
    assert any(s.time > "12:00" for s in after)


def test_availability_is_per_location(sched):
    sched.book(
        location="downtown", destination="service", subqueue="express_maintenance",
        queue="Express Bay", slot=Slot("2026-09-22", "08:00"),
    )
    assert sched.availability("northside", "express_maintenance", TUESDAY)[0].time == "08:00"


# --------------------------------------------------------------------------- booking
def test_booking_persists_with_an_id(tmp_path):
    SP.clear_cache()
    path = tmp_path / "bookings.jsonl"
    first = Scheduler(SP.load(), store_path=path)
    booking = first.book(
        location="downtown", destination="service", subqueue="tires", queue="Tire Bay",
        slot=Slot("2026-09-22", "10:00"), caller_name="Dana", callback_number="555-0100",
        vehicle="suv",
    )
    assert booking.id.startswith("apt_")
    assert booking.duration_min == 60

    # a fresh scheduler reads it back, which is the whole point of persisting
    second = Scheduler(SP.load(), store_path=path)
    loaded = second.bookings()
    assert len(loaded) == 1 and loaded[0].id == booking.id
    assert loaded[0].caller_name == "Dana"


def test_booking_roadside_is_refused(sched):
    with pytest.raises(ValueError, match="not bookable"):
        sched.book(
            location="downtown", destination="service", subqueue="roadside_assistance",
            queue="Roadside / Towing", slot=Slot("2026-09-22", "10:00"),
        )


def test_cancel_removes_the_booking(sched):
    b = sched.book(
        location="downtown", destination="service", subqueue="tires", queue="Tire Bay",
        slot=Slot("2026-09-22", "10:00"),
    )
    assert sched.cancel(b.id) is True
    assert sched.bookings() == []
    assert sched.cancel(b.id) is False  # idempotent


def test_a_corrupt_line_does_not_take_the_switchboard_down(tmp_path):
    SP.clear_cache()
    path = tmp_path / "bookings.jsonl"
    path.write_text('{"garbage": true}\nnot json at all\n')
    sched = Scheduler(SP.load(), store_path=path)
    assert sched.bookings() == []


def test_summary_is_speakable(sched):
    b = Booking(
        id="apt_x", location="downtown", destination="service", subqueue="tires",
        queue="Tire Bay", slot_day="2026-09-22", slot_time="09:00", duration_min=60,
        caller_name="Dana", callback_number="555-0100", vehicle="suv",
    )
    assert "tires" in b.summary() and "downtown" in b.summary() and "Dana" in b.summary()


# --------------------------------------------------------------------------- offering
def test_offer_searches_forward_across_days(sched):
    """Saturday closes at 4pm, so a 4-hour diagnostic can still be offered on Saturday morning."""
    offered = sched.offer("downtown", "mechanical_diagnostic", count=3, start=SATURDAY)
    assert offered and all(s.day == "2026-09-26" for s in offered)
    assert all(s.minutes() + 240 <= 16 * 60 for s in offered)


def test_offer_skips_a_closed_day(sched):
    offered = sched.offer("downtown", "express_maintenance", count=1, start=SUNDAY)
    assert offered and offered[0].day == "2026-09-28"  # Monday


def test_acceptance_options_name_each_slot(sched):
    offered = sched.offer("downtown", "tires", count=2, start=TUESDAY)
    opts = acceptance_options(offered)
    assert len(offered) == 2
    assert "slot_1" in opts and "slot_2" in opts
    # rejecting and being unclear are real answers, not failures
    assert "none_of_these" in opts and "unclear" in opts
    assert "8am" in opts["slot_1"]


def test_non_customer_and_front_desk_work_is_never_booked(sched):
    """A wrong number is not an appointment. Neither is a question the desk just answers, nor a
    complaint that needs a person. These are transferred or answered, not scheduled."""
    for sub in ("wrong_number", "vendor", "jobseeker", "general_question", "feedback"):
        assert sched.duration_for(sub) is None, sub
        assert sched.availability("downtown", sub, TUESDAY) == [], sub


def test_every_configured_service_can_actually_be_offered(sched):
    """The invariant that caught a real config bug.

    `collision` was configured at 1440 minutes - the length of the repair rather than the length
    of the booked slot - which can never fit inside an 8am-6pm window, so it silently offered
    nothing at all. A store profile that configures work nobody can be booked in for should fail
    loudly here, not at the phone.
    """
    profile = SP.load()
    unofferable = []
    for dest in profile.destinations:
        for sub in profile.subqueue_keys(dest.key):
            if sched.duration_for(sub) is None:
                continue  # dispatched, not scheduled
            if not sched.offer("downtown", sub, count=1):
                unofferable.append(f"{dest.key}/{sub} ({sched.duration_for(sub)} min)")
    assert not unofferable, f"configured but never offerable: {unofferable}"


# --------------------------------------------------------------------------- acceptance
OFFERED = [Slot("2026-09-22", "08:00"), Slot("2026-09-22", "08:30")]


def test_acceptance_maps_a_slot_choice_to_the_right_time():
    assert resolve_acceptance("slot_1", 0.9, OFFERED) == ("accept", 0)
    assert resolve_acceptance("slot_2", 0.9, OFFERED) == ("accept", 1)


def test_a_decisive_rejection_re_offers():
    assert resolve_acceptance("none_of_these", 0.9, OFFERED) == ("reject", None)


def test_an_indecisive_answer_never_books():
    """The bug this exists for.

    The untrained acceptance classifier answered `slot_1` at p=0.41 for "this is Dana, and my
    number is 555-0140" - an utterance that mentions no time at all - and the appointment was
    filed. An argmax of a near-uniform distribution is not a decision.
    """
    assert resolve_acceptance("slot_1", 0.41, OFFERED) == ("clarify", None)
    assert resolve_acceptance("none_of_these", 0.30, OFFERED) == ("clarify", None)
    assert resolve_acceptance("unclear", 0.99, OFFERED) == ("clarify", None)
    assert resolve_acceptance(None, None, OFFERED) == ("clarify", None)


def test_a_choice_naming_no_offered_slot_is_clarified_not_booked():
    assert resolve_acceptance("slot_9", 0.99, OFFERED) == ("clarify", None)
    assert resolve_acceptance("slot_0", 0.99, OFFERED) == ("clarify", None)


def test_acceptance_threshold_is_configurable(sched):
    """Tying the booking bar to the pin threshold keeps one notion of 'decisive' in the system."""
    assert resolve_acceptance("slot_1", 0.55, OFFERED, threshold=0.5) == ("accept", 0)
