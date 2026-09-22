"""Per-turn cost of a scripted call: how many questions we ask, over how many tokens, in how long.

This is the "before" number for the incremental-evaluation work. It reports, per turn, the number
of Laya questions evaluated and the input tokens they ran over — the two things that should fall
once we stop re-deciding what we already know.

Run:  uv run python scripts/bench_call.py
      uv run python scripts/bench_call.py --out results/baseline_call_cost.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from jev_classifier.agent import get_router
from jev_classifier.call import CallSession
from jev_classifier.scenarios import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def run_once(scenario: dict, router) -> dict:
    session = CallSession(scenario, router=router)
    events = list(session.advance())
    end = next((e for e in reversed(events) if e["type"] == "call_end"), {})
    return {
        "scenario": scenario["id"],
        "turns": session.turn_stats,
        "total_questions": sum(t["questions"] for t in session.turn_stats),
        "total_input_tokens": end.get("input_tokens"),
        "total_compute_ms": end.get("compute_ms"),
    }


def median_run(scenario: dict, router, repeats: int) -> dict:
    runs = [run_once(scenario, router) for _ in range(repeats)]
    runs.sort(key=lambda r: r["total_compute_ms"] or 0)
    return runs[len(runs) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    router = get_router()
    router.preload(["english", "multilingual"])

    print(f"{'scenario':14s} {'turn':>4s} {'questions':>10s} {'in_tokens':>10s} {'compute_ms':>11s}")
    print("-" * 54)

    report = {"scenarios": [], "repeats": args.repeats}
    for scenario in SCENARIOS:
        r = median_run(scenario, router, args.repeats)
        for t in r["turns"]:
            print(
                f"{r['scenario']:14s} {t['turn']:>4d} {t['questions']:>10d} "
                f"{t['input_tokens']:>10d} {t['compute_ms']:>11.2f}"
            )
        print(
            f"{'':14s} {'all':>4s} {r['total_questions']:>10d} "
            f"{r['total_input_tokens']:>10d} {r['total_compute_ms']:>11.2f}"
        )
        print()
        report["scenarios"].append(r)

    multi = [r for r in report["scenarios"] if len(r["turns"]) > 1]
    if multi:
        later = [t["questions"] for r in multi for t in r["turns"][1:]]
        first = [r["turns"][0]["questions"] for r in multi]
        report["summary"] = {
            "mean_questions_turn1": round(statistics.fmean(first), 2),
            "mean_questions_later_turns": round(statistics.fmean(later), 2),
            "later_turn_questions": later,
        }
        print(
            f"turn 1 asks {statistics.fmean(first):.1f} questions on average; "
            f"later turns ask {statistics.fmean(later):.1f} "
            f"(all already-answered ones repeated)"
        )

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
