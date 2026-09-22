"""Probe: make the slot questions answer 'not stated' honestly.

The first cut of the dealership schema had the model invent a vehicle ("sedan") and a time
("today") from a sentence that mentioned neither. These variants test wording that refuses to
guess, and the `next_action` question that has to know what is still missing.

Run:  uv run python scripts/probe_slots.py
"""

from __future__ import annotations

import laya_mlx as laya

# (utterance, expected vehicle, expected location, expected time)
CASES = [
    ("Hi, my car won't start at all. I think I need service.", "not_stated", "not_stated", "not_stated"),
    ("It's a Honda Civic, a sedan.", "sedan", "not_stated", "not_stated"),
    ("The downtown location, please.", "not_stated", "downtown", "not_stated"),
    ("Tomorrow morning would be great, around 9am.", "not_stated", "not_stated", "tomorrow"),
    ("I've got a flat tire and I'm stuck on the highway.", "not_stated", "not_stated", "not_stated"),
    ("It's a Ford F-150 truck.", "truck", "not_stated", "not_stated"),
    ("The airport location.", "not_stated", "airport", "not_stated"),
    ("Today, as soon as possible.", "not_stated", "not_stated", "today"),
]

VARIANT_QUESTIONS = {
    "A_plain": {
        "vehicle": {
            "type": "choice",
            "instructions": "What kind of vehicle is the caller talking about?",
            "criteria": {
                "sedan": "a car or sedan", "suv": "an SUV or crossover", "truck": "a pickup truck",
                "van": "a van or minivan", "ev": "an electric vehicle",
                "not_stated": "the vehicle type is not stated",
            },
        },
        "location": {
            "type": "choice",
            "instructions": "Which dealership location does the caller want, if any?",
            "criteria": {
                "downtown": "the downtown branch", "northside": "the northside branch",
                "airport": "the airport branch", "westside": "the westside branch",
                "not_stated": "the caller has not said",
            },
        },
        "time_preference": {
            "type": "choice",
            "instructions": "When does the caller want to come in?",
            "criteria": {
                "today": "today or as soon as possible", "tomorrow": "tomorrow",
                "this_week": "later this week", "next_week": "next week or later",
                "not_stated": "no stated preference",
            },
        },
    },
    "B_explicit_not_stated": {
        "vehicle": {
            "type": "choice",
            "instructions": (
                "What kind of vehicle did the caller explicitly mention? "
                "If no vehicle type was mentioned, choose 'not_stated'."
            ),
            "criteria": {
                "sedan": "the caller said a car or sedan", "suv": "the caller said SUV or crossover",
                "truck": "the caller said pickup truck", "van": "the caller said van",
                "ev": "the caller said electric vehicle",
                "not_stated": "the caller has not said what kind of vehicle it is",
            },
        },
        "location": {
            "type": "choice",
            "instructions": (
                "Which dealership location did the caller explicitly name? "
                "If they did not name one, choose 'not_stated'."
            ),
            "criteria": {
                "downtown": "the caller said downtown", "northside": "the caller said northside",
                "airport": "the caller said airport", "westside": "the caller said westside",
                "not_stated": "the caller has not named a location",
            },
        },
        "time_preference": {
            "type": "choice",
            "instructions": (
                "What timing did the caller explicitly state? "
                "If they did not state one, choose 'not_stated'."
            ),
            "criteria": {
                "today": "the caller said today or as soon as possible",
                "tomorrow": "the caller said tomorrow",
                "this_week": "the caller said later this week",
                "next_week": "the caller said next week or later",
                "not_stated": "the caller has not said when",
            },
        },
    },
    "C_noul_then_choice": {
        "vehicle": {
            "type": "choice",
            "instructions": (
                "The caller has NOT stated a vehicle type unless they clearly said one. "
                "Choose 'not_stated' when in doubt."
            ),
            "criteria": {
                "not_stated": "no vehicle type was stated — choose this when in doubt",
                "sedan": "a sedan or car", "suv": "an SUV", "truck": "a pickup truck",
                "van": "a van", "ev": "an electric vehicle",
            },
        },
        "location": {
            "type": "choice",
            "instructions": (
                "The caller has NOT named a location unless they clearly said one. "
                "Choose 'not_stated' when in doubt."
            ),
            "criteria": {
                "not_stated": "no location was named — choose this when in doubt",
                "downtown": "downtown", "northside": "northside",
                "airport": "airport", "westside": "westside",
            },
        },
        "time_preference": {
            "type": "choice",
            "instructions": (
                "The caller has NOT stated a time unless they clearly said one. "
                "Choose 'not_stated' when in doubt."
            ),
            "criteria": {
                "not_stated": "no timing was stated — choose this when in doubt",
                "today": "today or asap", "tomorrow": "tomorrow",
                "this_week": "later this week", "next_week": "next week or later",
            },
        },
    },
}

# next_action when 'location' is the only missing slot
NEXT_ACTION_VARIANTS = {
    "no_constraint": {
        "next_action": {
            "type": "choice",
            "instructions": "The switchboard must decide what to do next in this call. Choose the single best next step.",
            "criteria": {
                "ask_location": "we do not yet know which location the caller wants",
                "ask_detail": "the problem is too vague to book; ask for more detail",
                "confirm_booking": "we have everything needed to book the appointment",
                "offer_transfer": "hand to a person",
            },
        }
    },
    "explicit_missing": {
        "next_action": {
            "type": "choice",
            "instructions": (
                "The switchboard still needs the LOCATION before it can book. "
                "It cannot confirm the booking yet. Choose the step that asks for the missing "
                "information."
            ),
            "criteria": {
                "ask_location": "ask which location the caller wants",
                "ask_detail": "ask for more detail about the problem",
                "offer_transfer": "hand to a person",
            },
        }
    },
}

# ------------------------------------------------------------------ scope question
SCOPE_CASES = [
    ("Someone rear-ended me in a parking lot yesterday. I need body work.", True),
    ("Hi, my car won't start at all. I think I need service.", True),
    ("I need to order a replacement side mirror for my van.", True),
    ("I'm thinking about buying a new car, maybe an electric one.", True),
    ("I've got a flat tire and I'm stuck on the highway.", True),
    ("Hi, I'm calling about my neighbour's dog that keeps barking all night.", False),
    ("No, it isn't about a car at all.", False),
]

SCOPE_VARIANTS = {
    "noul_v1": {
        "is_matter": {
            "type": "noul",
            "instructions": "Is this call about a vehicle or the dealership, rather than something unrelated?",
        }
    },
    "noul_v2": {
        "is_matter": {
            "type": "noul",
            "instructions": "Does the caller want help with a vehicle or the dealership?",
            "criteria": {
                "true": "the caller wants vehicle or dealership help",
                "false": "the call is not about a vehicle or the dealership",
            },
        }
    },
    "choice_v1": {
        "is_matter": {
            "type": "choice",
            "instructions": "What is this call about?",
            "criteria": {
                "vehicle_or_dealership": "the caller wants help with a vehicle or the dealership",
                "unrelated": "the call is not about a vehicle or the dealership at all",
            },
        }
    },
}

# ------------------------------------------------------ next_action, single ask option
SINGLE_ASK_CASES = [
    (["vehicle"], "ask_vehicle", "Caller: my car won't start."),
    (["location"], "ask_location", "Caller: my car won't start. Agent: it's a sedan."),
    (
        ["time_preference"],
        "ask_time",
        "Caller: my car won't start. Agent: which location? Caller: downtown. Agent: it's a sedan? Caller: yes.",
    ),
]


def single_ask_question(first_missing: str) -> dict:
    key = "ask_time" if first_missing == "time_preference" else "ask_" + first_missing
    return {
        "next_action": {
            "type": "choice",
            "instructions": (
                "The switchboard still needs the "
                + first_missing.replace("_", " ")
                + " before it can book, so it cannot confirm yet. "
                "Choose the step that asks for that missing information."
            ),
            "criteria": {
                key: "ask for the " + first_missing.replace("_", " "),
                "ask_detail": "ask for more detail about the problem",
                "offer_transfer": "hand to a person",
            },
        }
    }


def main() -> None:
    router = laya.Router(max_loaded=2)
    print("preloading checkpoints ...", flush=True)
    router.preload(["english", "multilingual"])

    for name, questions in VARIANT_QUESTIONS.items():
        print(f"\n### {name}")
        correct = 0
        total = 0
        for text, *expected in CASES:
            res = router.predict({"call": text}, questions)
            got = [res["answers"][k]["choice"] for k in ("vehicle", "location", "time_preference")]
            marks = ["✓" if g == e else "✗" for g, e in zip(got, expected)]
            total += 3
            correct += sum(1 for g, e in zip(got, expected) if g == e)
            print(f"  {marks} {text[:52]:54s} -> v={got[0]} l={got[1]} t={got[2]}")
        print(f"  accuracy {correct}/{total} = {correct / total:.2f}")

    print("\n### next_action (only 'location' missing)")
    for name, questions in NEXT_ACTION_VARIANTS.items():
        res = router.predict(
            {"call": "Caller: my car won't start. Agent: which location?"}, questions
        )
        ans = res["answers"]["next_action"]
        print(
            f"  {name:16s} -> {ans['choice']:20s} conf {ans['confidence']:.2f} "
            f"p {max(ans['probabilities'].values()):.2f}"
        )

    print("\n### scope question")
    for name, questions in SCOPE_VARIANTS.items():
        correct = 0
        for text, expected in SCOPE_CASES:
            res = router.predict({"call": text}, questions)
            ans = res["answers"]["is_matter"]
            got = ans["choice"] == "vehicle_or_dealership" if ans["type"] == "choice" else ans["noul"] >= 0.5
            ok = got == expected
            correct += ok
            print(f"  {'✓' if ok else '✗'} {text[:56]:58s} -> matter={got}")
        print(f"  {name}: {correct}/{len(SCOPE_CASES)} = {correct / len(SCOPE_CASES):.2f}")

    print("\n### next_action with a single ask option")
    for missing, expected, transcript in SINGLE_ASK_CASES:
        res = router.predict({"call": transcript}, single_ask_question(missing[0]))
        ans = res["answers"]["next_action"]
        ok = ans["choice"] == expected
        print(
            f"  {'✓' if ok else '✗'} missing={missing[0]:16s} -> {ans['choice']:16s} "
            f"conf {ans['confidence']:.2f} p {max(ans['probabilities'].values()):.2f}"
        )


if __name__ == "__main__":
    main()
