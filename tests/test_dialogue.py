"""Spoken lines must reflect actual availability and the store's facts."""

from jev_classifier.dialogue import render_response
from jev_classifier.schedule import Booking, Slot


OFFERED = [
    Slot("2026-09-28", "08:00"),
    Slot("2026-09-28", "08:30"),
    Slot("2026-09-28", "09:00"),
]


def test_sales_question_uses_the_known_vehicle_without_inventing_a_time():
    line = render_response("ask_location", destination="sales", vehicle="ev")
    assert line == "I can help you look at electric cars. Which showroom works for you?"
    assert "Monday" not in line


def test_offer_says_the_shared_date_once_and_keeps_real_times():
    line = render_response("offer_slots", offered=OFFERED, location="westside")
    assert line == "I have Monday 28 September at 8am, 8:30am, or 9am at Westside. Which works for you?"


def test_unoffered_time_is_not_treated_as_an_acceptance():
    line = render_response("ask_which_slot", offered=OFFERED, reply="Sunday at 3am works")
    assert "I don't have that time available" in line
    assert "Monday 28 September at 8am" in line


def test_booking_reads_the_filed_slot_and_name():
    booking = Booking(
        id="apt_test", location="westside", destination="sales", subqueue="new_vehicle",
        queue="Sales Floor", slot_day="2026-09-28", slot_time="08:00", duration_min=60,
        caller_name="Dana", callback_number="555-0140", vehicle="ev",
    )
    assert render_response("booked", booking=booking) == (
        "Dana, you're booked at Westside for Monday 28 September at 8am. See you then."
    )


def test_hours_answer_uses_schedule_or_store_profile():
    assert render_response("answer_hours", reply="What time do you close today?", hours=(480, 1080)) == "We close at 6pm today."
    assert render_response("answer_hours", reply="Are you open today?", hours=None) == "We're closed today."
    assert render_response("answer_hours", reply="What are your hours?", store_hours="Monday to Friday 8am to 6pm") == "Our hours are Monday to Friday 8am to 6pm."


def test_wrong_number_ends_without_a_transfer():
    assert render_response("close_wrong_number") == "No problem. Have a good day."
