"""Mock availability and booking, shaped like a dealer management system.

The switchboard currently *says* "I'm booking you into service at downtown for this week" and files
nothing. This is the missing half: real slots, a hold, and a persisted appointment with an id.

Two deliberate choices:

* **Availability is derived, not invented.** Slots come from the store's opening hours and a
  per-service duration table, minus what is already booked, capped by how many bays that kind of
  work has. An express oil change is a 45-minute bay; a collision repair is days. So an offer is a
  real time rather than a promise.
* **The interface is shaped like a real calendar** (`availability` → `hold` → `confirm`/`cancel`)
  so a DMS drops in behind it later. It is a local JSON file today because there is no DMS to talk
  to, and pretending otherwise would be the wrong kind of realism.

Time is handled as `date` + `"HH:MM"` strings rather than `datetime` objects: the profile speaks
in wall-clock strings, and rounding a slot to a minute should never depend on a timezone.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import store_profile as SP

WEEKDAY = "weekday"
SATURDAY = "saturday"
SUNDAY = "sunday"


@dataclass(frozen=True)
class Slot:
    day: str  # YYYY-MM-DD
    time: str  # HH:MM, 24h

    @property
    def key(self) -> str:
        return f"{self.day}T{self.time}"

    def minutes(self) -> int:
        hh, mm = self.time.split(":")
        return int(hh) * 60 + int(mm)

    def label(self) -> str:
        """What the agent says out loud. 12-hour, no leading zero, because it is spoken."""
        hh, mm = (int(x) for x in self.time.split(":"))
        suffix = "am" if hh < 12 else "pm"
        hour = hh % 12 or 12
        return f"{hour}{':' + str(mm).zfill(2) if mm else ''}{suffix}"

    def spoken(self) -> str:
        """'Tuesday 24 September at 9am' — a caller cannot act on '2026-09-24T09:00'."""
        try:
            d = datetime.strptime(self.day, "%Y-%m-%d").date()
            when = d.strftime("%A %-d %B")
        except ValueError:  # pragma: no cover - only on a malformed day
            when = self.day
        return f"{when} at {self.label()}"


@dataclass(frozen=True)
class Booking:
    id: str
    location: str
    destination: str
    subqueue: Optional[str]
    queue: str
    slot_day: str
    slot_time: str
    duration_min: int
    caller_name: str
    callback_number: str
    vehicle: str

    @property
    def slot(self) -> Slot:
        return Slot(self.slot_day, self.slot_time)

    def summary(self) -> str:
        who = f" for {self.caller_name}" if self.caller_name else ""
        what = self.subqueue or self.destination
        return f"{what.replace('_', ' ')} at {self.location} — {self.slot.spoken()}{who}"


def _kind_of(day: date) -> str:
    if day.weekday() == 6:
        return SUNDAY
    if day.weekday() == 5:
        return SATURDAY
    return WEEKDAY


class Scheduler:
    """Availability over the store profile's opening hours and service durations."""

    def __init__(self, profile: Optional[SP.StoreProfile] = None, store_path: Optional[Path] = None):
        self.profile = profile or SP.load()
        self.schedule: Dict[str, Any] = dict(self.profile.schedule or {})
        default = str(Path(__file__).resolve().parents[2] / "data" / "bookings.jsonl")
        self.store_path = Path(store_path or os.environ.get("JEV_BOOKINGS") or default)

    # ------------------------------------------------------------------ service shape
    def duration_for(self, subqueue: Optional[str]) -> Optional[int]:
        """Minutes the job takes, or None if it is not an appointment (roadside is dispatched)."""
        if subqueue and subqueue in (self.schedule.get("no_appointment") or []):
            return None
        services = self.schedule.get("services") or {}
        entry = services.get(subqueue or "") or self.schedule.get("default") or {}
        minutes = int(entry.get("minutes", 60))
        return minutes

    def bays_for(self, subqueue: Optional[str]) -> int:
        services = self.schedule.get("services") or {}
        entry = services.get(subqueue or "") or self.schedule.get("default") or {}
        return max(1, int(entry.get("bays", 1)))

    def opening_hours(self, day: date) -> Optional[tuple]:
        hours = (self.schedule.get("opening_hours") or {}).get(_kind_of(day))
        if not hours:
            return None
        return (int(hours[0][:2]) * 60 + int(hours[0][3:]), int(hours[1][:2]) * 60 + int(hours[1][3:]))

    # ------------------------------------------------------------------ availability
    def availability(
        self, location: str, subqueue: Optional[str], day: date, *, limit: int = 0
    ) -> List[Slot]:
        """Free slots for this work on this day at this location, earliest first.

        A slot is free when fewer than `bays` jobs of the same kind already overlap it. Overlap is
        computed on the duration, not just the start time — two 4-hour diagnostics cannot share a
        morning simply because they start at different half-hours.
        """
        duration = self.duration_for(subqueue)
        if duration is None:
            return []
        window = self.opening_hours(day)
        if window is None:
            return []
        opens, closes = window
        interval = int(self.schedule.get("slot_interval_min", 30))
        bays = self.bays_for(subqueue)
        day_str = day.isoformat()
        taken = [
            b
            for b in self._load()
            if b.location == location and b.subqueue == subqueue and b.slot_day == day_str
        ]

        out: List[Slot] = []
        start = opens
        while start + duration <= closes:
            end = start + duration
            overlapping = 0
            for b in taken:
                b_start = b.slot.minutes()
                b_end = b_start + b.duration_min
                if b_start < end and start < b_end:
                    overlapping += 1
            if overlapping < bays:
                out.append(Slot(day_str, f"{start // 60:02d}:{start % 60:02d}"))
                if limit and len(out) >= limit:
                    break
            start += interval
        return out

    def offer(
        self, location: str, subqueue: Optional[str], *, days_ahead: int = 14, count: int = 3,
        start: Optional[date] = None,
    ) -> List[Slot]:
        """The next few real slots, searching forward. Empty means genuinely nothing available."""
        today = start or date.today()
        found: List[Slot] = []
        for offset in range(days_ahead):
            day = today + timedelta(days=offset)
            for slot in self.availability(location, subqueue, day, limit=count - len(found)):
                found.append(slot)
                if len(found) >= count:
                    return found
        return found

    # ------------------------------------------------------------------ booking
    def book(
        self,
        *,
        location: str,
        destination: str,
        subqueue: Optional[str],
        queue: str,
        slot: Slot,
        caller_name: str = "",
        callback_number: str = "",
        vehicle: str = "",
    ) -> Booking:
        duration = self.duration_for(subqueue)
        if duration is None:
            raise ValueError(f"{subqueue!r} is not bookable (dispatched, not scheduled)")
        booking = Booking(
            id=f"apt_{uuid.uuid4().hex[:10]}",
            location=location,
            destination=destination,
            subqueue=subqueue,
            queue=queue,
            slot_day=slot.day,
            slot_time=slot.time,
            duration_min=duration,
            caller_name=caller_name,
            callback_number=callback_number,
            vehicle=vehicle,
        )
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("a") as fh:
            fh.write(json.dumps(asdict(booking)) + "\n")
        return booking

    def cancel(self, booking_id: str) -> bool:
        rows = self._load()
        keep = [b for b in rows if b.id != booking_id]
        if len(keep) == len(rows):
            return False
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text("".join(json.dumps(asdict(b)) + "\n" for b in keep))
        return True

    def _load(self) -> List[Booking]:
        if not self.store_path.is_file():
            return []
        out: List[Booking] = []
        for line in self.store_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                out.append(Booking(**json.loads(line)))
            except (TypeError, json.JSONDecodeError):
                continue  # a corrupt line must not take the whole switchboard down
        return out

    def bookings(self) -> List[Booking]:
        return sorted(self._load(), key=lambda b: (b.slot_day, b.slot_time))


def acceptance_options(offered: Sequence[Slot]) -> Dict[str, str]:
    """The choice set for "which of these did the caller agree to?".

    A `choice` question rather than free text, because deciding *whether the caller accepted a
    time* is a classification, and it is exactly the kind of thing a generative model would answer
    with invented prose. `none` and `unclear` are real answers, not failures.
    """
    options = {f"slot_{i + 1}": f"the caller accepted {s.spoken()}" for i, s in enumerate(offered)}
    options["none_of_these"] = "the caller rejected every time offered"
    options["unclear"] = "the caller has not clearly accepted or rejected a time"
    return options


def acceptance_question(offered: Sequence[Slot], profile: Optional[SP.StoreProfile] = None) -> Dict[str, Any]:
    """The instruction comes from the store profile, like every other question.

    It is the same reason as the choice questions: the fine-tune learns one exact instruction, so
    training and inference must not hold separate copies. `build_items.py` reads the same string.
    """
    p = profile or SP.load()
    return {
        "acceptance": {
            "type": "choice",
            "instructions": p.question_text("acceptance"),
            "criteria": acceptance_options(offered),
        }
    }


def slot_choice_index(choice: Optional[str], offered: Sequence[Slot]) -> Optional[int]:
    """Map an acceptance answer back to the slot it names, or None if it names none of them."""
    if not choice or not choice.startswith("slot_"):
        return None
    try:
        idx = int(choice.split("_", 1)[1]) - 1
    except (IndexError, ValueError):
        return None
    return idx if 0 <= idx < len(offered) else None


_DAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)
_CLOCK_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)


def reply_names_unoffered_time(reply: str, offered: Sequence[Slot]) -> bool:
    """True when the reply names a day or clock time that none of the offered slots has.

    Measured on the booking scenario: the agent offered three *Wednesday* times, the caller said
    **"Tuesday at 8 works for me"**, and the acceptance classifier answered `slot_1` at p=1.00 —
    matching the hour and ignoring the day — so an appointment was filed for a time nobody asked
    for. Same failure as the original *"this is Dana, and my number is 555-0140"* → `slot_1`, in a
    new disguise: the classifier finds the nearest slot instead of declining.

    The classifier proposes; this disposes. It vetoes only on an explicit contradiction, so a reply
    that names no time at all ("the first one, please") is still the model's to judge.
    """
    days = {d.lower() for d in _DAY_RE.findall(reply)}
    if days:
        offered_days = set()
        for slot in offered:
            try:
                offered_days.add(datetime.strptime(slot.day, "%Y-%m-%d").strftime("%A").lower())
            except ValueError:  # pragma: no cover - only on a malformed day
                continue
        if offered_days and not (days & offered_days):
            return True

    clocks = set()
    for hour, minute, meridiem in _CLOCK_RE.findall(reply):
        hh = int(hour) % 12 + (12 if meridiem.lower() == "pm" else 0)
        clocks.add(f"{hh:02d}:{minute or '00'}")
    if clocks:
        offered_times = {slot.time for slot in offered}
        if offered_times and not (clocks & offered_times):
            return True
    return False


def resolve_acceptance(
    choice: Optional[str],
    top_probability: Optional[float],
    offered: Sequence[Slot],
    *,
    threshold: float = 0.6,
    reply: str = "",
) -> tuple:
    """Decide what the caller's reply to a slot offer actually means.

    Returns `(verdict, index)` where verdict is one of:

      `accept`   they named one of the offered times, decisively
      `reject`   they turned down every time offered, decisively
      `clarify`  anything else - including a confident-sounding answer we cannot trust

    The `clarify` branch is the important one. Measured: the acceptance classifier is untrained, and
    it answered `slot_1` at p=0.41 for an utterance that mentioned no time at all ("this is Dana,
    and my number is 555-0140"). Acting on that files an appointment nobody agreed to. An
    argmax of a near-uniform distribution is not a decision - the same rule that stops a hard stop
    firing on a weak signal elsewhere in this system.

    And a decisive answer is not automatically trustworthy either: trained, it answered `slot_1` at
    p=1.00 for "Tuesday at 8" against three Wednesday offers. So the caller's own words get a veto
    before any confidence is honoured.
    """
    if reply and reply_names_unoffered_time(reply, offered):
        return "clarify", None
    idx = slot_choice_index(choice, offered)
    decisive = float(top_probability or 0.0) >= threshold
    if idx is not None and decisive:
        return "accept", idx
    if choice == "none_of_these" and decisive:
        return "reject", None
    return "clarify", None
