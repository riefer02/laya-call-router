"""Headless runner for the triage cascade.

    uv run jev-classify --ticket "I was charged twice, refund me or I cancel"
    uv run jev-classify --persona outage
    uv run jev-classify --list-personas
"""

from __future__ import annotations

import argparse
import sys

from .agent import get_router
from .personas import PERSONA_BY_ID, PERSONAS
from .pipeline import DEFAULT_THRESHOLD, run_to_completion

BAR_FULL = "▓"
BAR_EMPTY = "░"


def _bar(p: float, width: int = 10) -> str:
    filled = int(round(p * width))
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)


def _render_event(event: dict) -> None:
    kind = event.get("type")
    if kind == "stage_start":
        print(f"\n▶ {event['title']}  [{event['primitive']}]")
    elif kind == "stage_result":
        status = event.get("status")
        mark = {"ok": "✓", "low_confidence": "⚠", "rejected": "✗"}.get(status, "·")
        print(f"  {mark} {event.get('model') or 'policy'}  {event.get('latency_ms')} ms")
        for qid, ans in event.get("answers", {}).items():
            prim = ans["primitive"]
            if prim == "choice":
                probs = ans.get("probabilities") or {}
                ranked = sorted(probs.items(), key=lambda kv: -kv[1])
                top = ans["choice"]
                print(f"    {qid}: {top}  (conf {ans['confidence']:.2f}, p {ans['top_probability']:.2f})")
                for label, p in ranked:
                    marker = "→" if label == top else " "
                    print(f"      {marker} {label:20s} {_bar(p)} {p:.2f}")
            elif prim == "score":
                legend = ans.get("legend") or {}
                levels = " | ".join(legend.get(str(i), str(i)) for i in range(len(legend)))
                print(f"    {qid}: {ans['score']:.2f}/{len(legend) - 1}  (conf {ans['confidence']:.2f})")
                print(f"      levels: {levels}")
            else:
                print(f"    {qid}: P(true)={ans['noul']:.2f}  (conf {ans['confidence']:.2f})")
        if event.get("note"):
            print(f"    note: {event['note']}")
    elif kind == "clarify":
        print(f"    ❓ {event['prompt']}  chips={event['chips']}")
    elif kind == "routing":
        print(
            f"\n▶ Routing\n  queue={event['queue']}  priority={event['priority']}  "
            f"handler={event['handler']}"
        )
        print(f"  flags={event['flags']}")
        for reason in event.get("reasons", []):
            print(f"    - {reason}")
    elif kind == "done":
        print(
            f"\n═ done: {event['compute_ms']} ms compute ({event['total_ms']} ms wall), "
            f"{event['stages_run']} stages, {event['tokens_generated']} tokens generated"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-classify", description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticket", help="a support message to route")
    src.add_argument("--persona", help="run a scripted persona by id")
    src.add_argument("--list-personas", action="store_true", help="show personas and exit")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD, help="choice confidence threshold"
    )
    args = parser.parse_args(argv)

    if args.list_personas:
        for p in PERSONAS:
            print(f"{p['id']:16s} {p['label']:24s} {p['expect']}")
        return 0

    if args.persona:
        if args.persona not in PERSONA_BY_ID:
            print(f"unknown persona {args.persona!r}", file=sys.stderr)
            return 2
        message = PERSONA_BY_ID[args.persona]["message"]
    else:
        message = args.ticket

    print(f'ticket: "{message}"')
    # Preload so the first stage's latency reflects inference, not a cold model build.
    router = get_router()
    router.preload(["english", "multilingual"])
    result = run_to_completion(message, confidence_threshold=args.threshold, router=router)
    for event in result["events"]:
        _render_event(event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
