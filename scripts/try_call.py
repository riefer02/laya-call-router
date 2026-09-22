"""Run a scripted dealership call and print the decision trace. No browser needed.

Run:  uv run python scripts/try_call.py
      uv run python scripts/try_call.py --scenario collision
"""

from __future__ import annotations

import argparse

from jev_classifier.agent import get_router
from jev_classifier.call import CallSession
from jev_classifier.scenarios import SCENARIO_BY_ID, SCENARIOS


def render(ev: dict) -> None:
    t = ev["type"]
    if t == "call_start":
        print(f"═══ call: {ev['scenario']['label']}")
    elif t == "turn_start":
        print(f"\n── turn {ev['turn']} ─────────────────────────────")
    elif t == "node_result":
        kind = ev.get("status")
        mark = {"ok": "✓", "low_confidence": "⚠", "warn": "!", "rejected": "✗"}.get(kind, "·")
        value = ev.get("value")
        summary = ev.get("summary") or {}
        if summary.get("primitive") == "choice":
            detail = f"{summary['choice']} (conf {summary['confidence']:.2f}, p {summary['top_probability']:.2f})"
        elif summary.get("primitive") == "noul":
            detail = f"P(true)={summary['noul']:.2f}"
        elif value is not None:
            detail = str(value)
        else:
            detail = ""
        label = ev["id"].split(".")[-1]
        meta = f"{ev.get('model') or 'rule'} · {ev.get('latency_ms')} ms" if ev.get("latency_ms") else "rule"
        print(f"  {mark} {label:22s} {detail}")
        if ev.get("note"):
            print(f"      ↳ {ev['note']}  [{meta}]")
    elif t == "edge" and ev.get("kind") in ("branch", "terminal"):
        print(f"      ↳ edge {ev['source']} → {ev['target']}  ({ev.get('label')})")
    elif t == "call_end":
        print(
            f"\n═══ done: {ev['turns']} turns · {ev['compute_ms']} ms compute · "
            f"{ev['tokens_generated']} tokens generated · ${ev['cost_usd']:.2f}"
        )
        if ev.get("routing"):
            r = ev["routing"]
            print(f"    route: {r['queue']}  priority={r['priority']}  flags={r['flags']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="no_start")
    args = ap.parse_args()

    if args.scenario not in SCENARIO_BY_ID:
        print("scenarios:", ", ".join(SCENARIO_BY_ID))
        return
    scenario = SCENARIO_BY_ID[args.scenario]

    get_router().preload(["english", "multilingual"])
    session = CallSession(scenario)
    for ev in session.advance():
        render(ev)


if __name__ == "__main__":
    main()
