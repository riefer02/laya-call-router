"""Run every demo scenario against the active checkpoint and check its stated ending.

Uses a temporary booking store so the check never changes the presentation's availability.
Known model failures are printed and excluded from the default pass gate; --strict includes them.
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

from jev_classifier.agent import get_router, resolve_checkpoint
from jev_classifier.call import CallSession
from jev_classifier.scenarios import SCENARIOS
from jev_classifier.schedule import Scheduler, Slot


def spoken_path(events: list[dict], expected_actions: list[str] | None = None) -> tuple[bool, str]:
    """Check what the audience hears as well as where the call ends."""
    results = [event for event in events if event.get("type") == "node_result"]
    callers = [event for event in results if event["id"].endswith(".caller")]
    replies = [event for event in results if event["id"].endswith(".agent")]
    actions = [event["value"] for event in results if event["id"].endswith(".next_action")]
    if len(replies) != len(callers):
        return False, f"{len(callers)} caller turns but {len(replies)} spoken replies"
    if any(not isinstance(event.get("value"), str) or not event["value"].strip() for event in replies):
        return False, "an empty spoken reply"
    if expected_actions is not None and actions != expected_actions:
        return False, f"actions {actions}, expected {expected_actions}"
    if any("Let me take a look at that for you" in event["value"] for event in replies):
        return False, "a generic filler reply returned"
    end = events[-1]
    booking = end.get("booking")
    if booking and replies:
        closing = replies[-1]["value"]
        slot = Slot(booking["slot_day"], booking["slot_time"]).spoken()
        if slot not in closing:
            return False, "the spoken confirmation does not match the filed appointment"
        if booking["caller_name"] and booking["caller_name"] not in closing:
            return False, "the spoken confirmation omits the booked caller's name"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict", action="store_true", help="fail on documented model failures too"
    )
    args = parser.parse_args()
    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        print("FAIL: no fine-tuned checkpoint selected; set JEV_MODEL or models/active")
        return 1
    print(f"checkpoint: {checkpoint.resolve()}")
    router = get_router()
    router.preload(["english", "multilingual"])
    failures = 0
    known = 0
    passed = 0
    with tempfile.TemporaryDirectory(prefix="jev-demo-") as tmp:
        for scenario in SCENARIOS:
            expect = scenario["expect"]
            scheduler = Scheduler(store_path=Path(tmp) / f"{scenario['id']}.jsonl")
            events = list(CallSession(scenario, router=router, scheduler=scheduler).advance())
            end = events[-1]
            actual = {
                "queue": (end.get("routing") or {}).get("queue"),
                "completion": end.get("completion"),
            }
            booking = end.get("booking")
            if "subqueue" in expect:
                actual["subqueue"] = (booking or {}).get("subqueue")
            future_booking = not booking or datetime.fromisoformat(
                f"{booking['slot_day']}T{booking['slot_time']}"
            ) > datetime.now()
            speech_ok, speech_note = spoken_path(events, scenario.get("expect_actions"))
            match = actual == expect and future_booking and speech_ok
            is_known = bool(scenario.get("known_issue"))
            label = "PASS" if match else "KNOWN FAIL" if is_known else "FAIL"
            print(f"{label:10} {scenario['id']:24} {actual['queue']} / {actual['completion']}")
            if not match:
                print(f"           expected {expect}")
                if not future_booking:
                    print("           booked slot is already in the past")
                if not speech_ok:
                    print(f"           dialogue: {speech_note}")
                known += int(is_known)
                failures += int(not is_known or args.strict)
            else:
                passed += 1
    print(f"{passed} passed, {known} documented failure(s), {failures} gate failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
