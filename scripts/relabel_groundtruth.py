"""Relabel the ground truth onto the corrected taxonomy, and report the drift.

This is the H3 relabel: the old labels mixed levels (tires and detailing sat beside service as if
they were peers), so a mechanical mapping resolves most of it. A handful of cases need a judgement
call, and those are listed explicitly with the reasoning rather than smeared through a lookup
table — they are exactly the ones the old taxonomy made ambiguous.

Run with --check to see the diff without writing.

    uv run python scripts/relabel_groundtruth.py --check
    uv run python scripts/relabel_groundtruth.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import sys as _sys

ROOT = Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT / "src"))

# --------------------------------------------------------------------------- the rubric, as code
# old service intents collapse onto service sub-queues; the problem-based detail becomes the
# diagnostic queue, scheduled work becomes express.
SERVICE_INTENTS = {
    "no_start": "mechanical_diagnostic",
    "warning_light": "mechanical_diagnostic",
    "brakes": "mechanical_diagnostic",
    "noise": "mechanical_diagnostic",
    "performance": "mechanical_diagnostic",
    "maintenance": "express_maintenance",
    "other": "mechanical_diagnostic",
}

# One old department maps onto a whole service sub-queue.
SUBFUNCTION_DESTINATIONS = {
    "tires": ("service", "tires"),
    "detailing": ("service", "detailing"),
    "towing": ("service", "roadside_assistance"),
}

# Pass-through departments keep their intents.
PASSTHROUGH = {
    "body_shop": "body_shop",
    "parts": "parts",
    "sales": "sales",
    "finance": "finance",
}

# `general` was two different things wearing one name. Each case is decided explicitly.
# "kind" is the reason, kept so the drift report is auditable.
GENERAL_CASES: Dict[str, Tuple[str, Optional[str], str]] = {
    "gen-01": ("front_desk", "general_question", "opening hours: a dealership question no department owns"),
    "gen-02": ("front_desk", "general_question", "location: as above"),
    "gen-03": ("front_desk", "general_question", "facilities: as above"),
    "gen-04": ("front_desk", "feedback", "a complaint about a visit"),
    "gen-05": ("non_customer", "jobseeker", "asked about hiring: not a customer"),
    "gen-06": ("front_desk", "general_question", "asking for the parts desk's NUMBER, not about a part"),
    "gen-07": ("front_desk", "feedback", "wants the manager: an escalation, not part of booking"),
    "gen-08": ("front_desk", "general_question", "policy question about loaners; the service mention is context"),
    "gen-09": ("front_desk", "other", "a dealership/sponsorship enquiry, not a vehicle matter"),
}

# Call-level expectations whose queue legitimately changes, with the reason.
CALL_QUEUE_CHANGES = {
    "call-maintenance": ("Express Bay", "scheduled maintenance is an express-bay service line"),
    "call-trade-in": ("Appraisal Desk", "appraisals are their own desk, not the sales floor"),
    "call-no-start": (
        "Roadside / Towing",
        "roadside is now a dispatch flag on an unsafe-to-drive call, not a department",
    ),
}

# Call-level label judgement calls (the routing-set equivalent is GENERAL_CASES).
CALL_LABEL_OVERRIDES = {
    "call-out-of-scope": (
        "non_customer",
        "wrong_number",
        "the neighbour's dog: not about the dealership at all",
    ),
}


def map_label(department: str, intent: str, case_id: str):
    """Return (destination, subqueue, reason)."""
    if department == "service":
        return "service", SERVICE_INTENTS.get(intent, "mechanical_diagnostic"), "service intent -> sub-queue"
    if department in SUBFUNCTION_DESTINATIONS:
        dest, sub = SUBFUNCTION_DESTINATIONS[department]
        return dest, sub, f"{department} was a service sub-function, not a department"
    if department in PASSTHROUGH:
        return PASSTHROUGH[department], intent, "pass-through"
    if department == "general":
        if case_id in GENERAL_CASES:
            dest, sub, why = GENERAL_CASES[case_id]
            return dest, sub, why
        return "front_desk", "other", "general fallback: decide explicitly"
    raise ValueError(f"unmapped department {department!r} for {case_id}")


def relabel_routing(rows, *, check: bool):
    changed = []
    out = []
    for row in rows:
        dest, sub, why = map_label(row["department"], row["intent"], row["id"])
        new = {"id": row["id"], "text": row["text"], "destination": dest}
        if sub:
            new["subqueue"] = sub
        if (dest, sub) != (row["department"], row["intent"]):
            changed.append({"id": row["id"], "from": f"{row['department']}/{row['intent']}", "to": f"{dest}/{sub}", "why": why})
        out.append(new)
    return out, changed


def relabel_calls(rows, *, check: bool):
    changed = []
    out = []
    for row in rows:
        dest, sub, why = map_label(row["department"], row["intent"], row["id"])
        if row["id"] in CALL_LABEL_OVERRIDES:
            dest, sub, why = CALL_LABEL_OVERRIDES[row["id"]]
        queue = row["queue"]
        if row["id"] in CALL_QUEUE_CHANGES:
            queue, qwhy = CALL_QUEUE_CHANGES[row["id"]]
            why = f"{why}; queue: {qwhy}"
        new = {"id": row["id"], "label": row["label"], "turns": row["turns"], "destination": dest, "queue": queue}
        if sub:
            new["subqueue"] = sub
        if (dest, sub, queue) != (row["department"], row["intent"], row["queue"]):
            changed.append({
                "id": row["id"],
                "from": f"{row['department']}/{row['intent']} -> {row['queue']}",
                "to": f"{dest}/{sub} -> {queue}",
                "why": why,
            })
        out.append(new)
    return out, changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report the drift without writing")
    args = ap.parse_args()

    rpath = ROOT / "data" / "calls" / "routing.jsonl"
    cpath = ROOT / "data" / "calls" / "scripts.jsonl"
    routing = [json.loads(l) for l in rpath.read_text().splitlines() if l.strip()]
    calls = [json.loads(l) for l in cpath.read_text().splitlines() if l.strip()]

    new_routing, rchanged = relabel_routing(routing, check=args.check)
    new_calls, cchanged = relabel_calls(calls, check=args.check)

    total = len(routing) + len(calls)
    print(f"=== relabel drift: {len(rchanged) + len(cchanged)} of {total} cases changed ===\n")
    print(f"routing ({len(rchanged)}/{len(routing)}):")
    for c in rchanged:
        print(f"  {c['id']:9s} {c['from']:24s} -> {c['to']:30s} {c['why']}")
    print(f"\ncalls ({len(cchanged)}/{len(calls)}):")
    for c in cchanged:
        print(f"  {c['id']:18s} {c['from']:34s} -> {c['to']:34s} {c['why']}")

    before = Counter(r["department"] for r in routing)
    after = Counter(r["destination"] for r in new_routing)
    print("\ndestination distribution:")
    for key in sorted(set(before) | set(after)):
        print(f"  {key:12s} {before.get(key, 0):3d} -> {after.get(key, 0):3d}")

    print("\nsub-queue distribution (new):")
    for key, n in sorted(Counter(r.get("subqueue") for r in new_routing).items(), key=lambda kv: str(kv[0])):
        print(f"  {str(key):24s} {n}")

    if args.check:
        print("\n--check: nothing written")
        return

    rpath.write_text("\n".join(json.dumps(r) for r in new_routing) + "\n")
    cpath.write_text("\n".join(json.dumps(r) for r in new_calls) + "\n")
    print(f"\nrewrote {rpath.name} ({len(new_routing)}) and {cpath.name} ({len(new_calls)})")


if __name__ == "__main__":
    main()
