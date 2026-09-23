"""Short, grounded switchboard replies for the deterministic call policy.

Laya classifies; it cannot write dialogue. These lines use only the selected action and facts
already supplied by the caller or scheduler. No model call or invented appointment detail.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .schedule import Booking, Slot, reply_names_unoffered_time


def offered_times(slots: Sequence[Slot]) -> str:
    """Say a shared date once instead of reading three full timestamps."""
    if not slots:
        return ""
    labels = [slots[0].spoken()]
    same_day = all(slot.day == slots[0].day for slot in slots)
    labels.extend(slot.label() if same_day else slot.spoken() for slot in slots[1:])
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + ", or " + labels[-1]


def render_response(
    action: str,
    *,
    destination: str = "",
    vehicle: str = "",
    location: str = "",
    queue: str = "",
    offered: Sequence[Slot] = (),
    booking: Booking | None = None,
    contact: Mapping[str, str] | None = None,
    reply: str = "",
    unsafe: bool = False,
    no_slots: bool = False,
    hours: tuple[int, int] | None = None,
    store_hours: str = "",
) -> str:
    """Render one spoken response from confirmed state."""
    name = (contact or {}).get("caller_name", "")
    place = location.replace("_", " ").title() if location and location != "not_stated" else ""
    if action == "ask_vehicle":
        return (
            "What kind of car are you looking for?" if destination == "sales"
            else "What kind of vehicle is it?"
        )
    if action == "ask_location":
        if destination == "sales":
            opening = "I can help you look at electric cars." if vehicle == "ev" else "I can help with that."
            return f"{opening} Which showroom works for you?"
        if destination == "body_shop":
            return "I can help arrange an assessment. Which location works for you?"
        return "I can help get that looked at. Which location works for you?"
    if action == "ask_time":
        return f"{place} works. What day would you like to come in?" if place else "What day works for you?"
    if action == "ask_alternative_time":
        return "Those times don't work. What other day would suit you?"
    if action == "ask_detail":
        return "Could you tell me a little more about what the vehicle is doing?"
    if action == "offer_slots":
        at = f" at {place}" if place else ""
        return f"I have {offered_times(offered)}{at}. Which works for you?"
    if action == "ask_which_slot":
        if reply and reply_names_unoffered_time(reply, offered):
            return f"I don't have that time available. I can offer {offered_times(offered)}. Would one of those work?"
        thanks = f"Thanks, {name}. " if name else ""
        return f"{thanks}Would one of those times work for you? I can offer {offered_times(offered)}."
    if action == "booked" and booking:
        who = f"{booking.caller_name}, " if booking.caller_name else ""
        where = f" at {booking.location.replace('_', ' ').title()}" if booking.location else ""
        return f"{who}you're booked{where} for {booking.slot.spoken()}. See you then."
    if action == "answer_hours":
        if "today" not in reply.lower():
            return f"Our hours are {store_hours}."
        if hours is None:
            return "We're closed today."
        def clock(minutes: int) -> str:
            hour, minute = divmod(minutes, 60)
            return f"{hour % 12 or 12}{':' + str(minute).zfill(2) if minute else ''}{'am' if hour < 12 else 'pm'}"
        if "close" in reply.lower() or "closing" in reply.lower():
            return f"We close at {clock(hours[1])} today."
        if "open" in reply.lower():
            return f"We open at {clock(hours[0])} today."
        return f"We're open from {clock(hours[0])} to {clock(hours[1])} today."
    if action == "close_wrong_number":
        return "No problem. Have a good day."
    if action == "offer_transfer":
        if unsafe or queue == "Roadside / Towing":
            return "I'll connect you with roadside assistance so a person can help."
        if no_slots:
            return "I don't have an online opening for that. I'll connect you with the team to find another option."
        team = {
            "parts": "our parts counter",
            "finance": "our finance team",
            "front_desk": "the front desk",
            "non_customer": "the front desk",
        }.get(destination, "the right team")
        return f"I'll connect you with {team}."
    return "Could you tell me a little more?"
