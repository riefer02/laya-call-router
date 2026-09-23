"""Training data for the one question the switchboard asks mid-conversation.

"Which of the times I just offered did the caller agree to?" — it is the only question whose state
is a whole conversation rather than one utterance, and it is currently answered by the base
checkpoint, which has never seen it. Measured consequence: on a live call the caller said "this is
Dana, and my number is 555-0140" — no time at all — and the classifier answered slot_1 at p=0.41,
and the appointment was filed. The confidence floor now stops that, but "asks again about half the
time" is not a product.

The state is the transcript exactly as the cascade builds it, including the agent's offer with the
real spoken times, so the model can match a reply against the times it was given.

Replies come from templates rather than a teacher, deliberately: the label is then exact rather
than inferred, and the option texts contain the slot times, so a teacher's answer would have to be
mapped back to a slot anyway — one more place to be wrong. Variety comes from many phrasings per
class rather than from generation.

    uv run python scripts/generate_acceptance.py --per-class 120
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_classifier import dealership as D, store_profile as SP  # noqa: E402
from jev_classifier.schedule import Scheduler, Slot, acceptance_options  # noqa: E402

OUT = ROOT / "data" / "calls" / "acceptance_train.jsonl"

# What the caller says. `{t}` is the spoken form of the slot they mean, `{other}` one they do not.
ACCEPT = [
    "Yes, {t} works for me.",
    "{t} is perfect, thanks.",
    "The one at {t}, please.",
    "{t} would be great.",
    "Let's do {t}.",
    "I can make {t}.",
    "{t} suits me.",
    "Sure, book me in for {t}.",
    "Yes please, {t}.",
    "{t}, that one works.",
    "Can we do {t}?",
    "I'll take {t}.",
]
REJECT = [
    "None of those work for me, sorry.",
    "I can't do any of those times.",
    "None of those are any good.",
    "I'm not free at any of those.",
    "Sorry, none of those suit.",
    "Do you have anything else? None of those work.",
]
UNCLEAR = [
    "Hmm, let me think about it.",
    "What else do you have?",
    "Could it be later in the day?",
    "Is there anything in the afternoon?",
    "I'm not sure yet.",
    "Maybe — what are the options again?",
    "I'd have to check with my wife.",
    "Can I let you know?",
]
VEHICLES = ["car", "SUV", "truck", "van", "sedan"]
OPENERS = [
    "I'd like to book my {v} in for {what}.",
    "Can I get my {v} booked in for {what}?",
    "I need to bring my {v} in for {what}.",
    "I'd like an appointment for {what} on my {v}.",
]
TIMING = ["Sometime this week.", "Next week would be good.", "As soon as you can.", "Whenever works."]


def build_transcript(rng: random.Random, location: str, subqueue: str, offered) -> str:
    what = (subqueue or "service").replace("_", " ")
    slots = ", ".join(s.spoken() for s in offered)
    return "\n".join(
        [
            f"Caller: {rng.choice(OPENERS).format(v=rng.choice(VEHICLES), what=what)}",
            "Agent: Got it — which of our locations works best for you?",
            f"Caller: {location.title()} please.",
            "Agent: When would you like to come in?",
            f"Caller: {rng.choice(TIMING)}",
            f"Agent: I can get you in at {slots}. Which of those works best for you?",
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100, help="examples per acceptance class")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    SP.clear_cache()
    profile = SP.load()
    scheduler = Scheduler(profile, store_path=Path(args.out).with_suffix(".bookings.jsonl"))
    rng = random.Random(args.seed)

    bookable = [
        s.key
        for d in profile.destinations
        for s in profile.subqueues_for(d.key)
        if s.key != "other" and scheduler.duration_for(s.key) is not None
    ]
    locations = [l["key"] for l in profile.locations if l["key"] != "not_stated"]

    rows = []
    start = date(2026, 9, 22)
    accept_i = 0
    for _ in range(args.per_class):
        for kind in ("accept", "reject", "unclear"):
            location = rng.choice(locations)
            subqueue = rng.choice(bookable)
            offered = scheduler.offer(
                location, subqueue, count=3, start=start + timedelta(days=rng.randint(0, 20))
            )
            if len(offered) < 3:
                continue
            state = build_transcript(rng, location, subqueue, offered)
            if kind == "accept":
                # cycle the slot rather than sampling it, so slot_1/2/3 stay balanced - they differ
                # only by the time text, and an uneven split would teach a positional bias
                idx = accept_i % 3
                accept_i += 1
                reply = rng.choice(ACCEPT).format(t=offered[idx].label())
                choice = f"slot_{idx + 1}"
            elif kind == "reject":
                reply = rng.choice(REJECT)
                choice = "none_of_these"
            else:
                reply = rng.choice(UNCLEAR)
                choice = "unclear"
            rows.append(
                {
                    "text": state + f"\nCaller: {reply}",
                    "choice": choice,
                    "offered": [s.key for s in offered],
                    "reply": reply,
                }
            )

    # the option set is per-row, because it names the times that were actually offered
    for row in rows:
        offered = [Slot(*k.split("T")) for k in row["offered"]]
        row["options"] = acceptance_options(offered)

    out = Path(args.out)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {out} ({len(rows)} examples)")
    print("by class:", dict(Counter(r["choice"] for r in rows)))
    print("\nexample:")
    ex = rows[0]
    print("  state:", ex["text"].replace("\n", " | ")[:150])
    print("  choice:", ex["choice"], "| options:", list(ex["options"])[:3], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
