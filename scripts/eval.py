"""Four-arm evaluation: our cascade vs a cheap generative model, on labelled ground truth.

    uv run python scripts/eval.py
    uv run python scripts/eval.py --skip-llm              # cascade arms only, no key needed
    uv run python scripts/eval.py --limit 20              # quick smoke run

Writes results/eval.json. The API key is read from .env and never printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev_classifier import evalharness as H
from jev_classifier import llm
from jev_classifier.agent import get_router

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap the routing cases (0 = all)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--determinism", type=int, default=3, help="repeats for the agreement check")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--out", default="results/eval.json")
    args = ap.parse_args()

    routing = H.load_routing()
    calls = H.load_calls()
    if args.limit:
        routing = routing[: args.limit]

    use_llm = not args.skip_llm and llm.available()
    model = llm.model_name() if use_llm else None

    print(f"ground truth: {len(routing)} routing cases, {len(calls)} calls")
    print(f"llm arm: {'disabled' if not use_llm else model}")
    if not use_llm and not args.skip_llm:
        print("  (no OPENAI_API_KEY found — set it in .env, or pass --skip-llm)")

    router = get_router()
    print("preloading checkpoints ...", flush=True)
    router.preload(["english", "multilingual"])

    report: dict = {"model": model, "routing": {}, "calls": {}}

    # ---- decision level ------------------------------------------------------
    print("\nrunning cascade arm ...", flush=True)
    laya_routing = H.run_laya_routing(routing, router)
    report["routing"]["laya"] = H.score_routing(routing, laya_routing)

    if use_llm:
        print(f"running llm arm ({model}, {args.concurrency} workers) ...", flush=True)
        llm_runs = [
            H.run_llm_routing(routing, concurrency=args.concurrency)
            for _ in range(max(1, args.determinism))
        ]
        report["routing"]["llm"] = H.score_routing(routing, llm_runs[0], model=model)
        report["routing"]["llm"]["determinism"] = H.agreement(llm_runs)
        report["routing"]["laya"]["determinism"] = H.agreement(
            [H.run_laya_routing(routing, router) for _ in range(max(1, args.determinism))]
        )

    # ---- call level ----------------------------------------------------------
    print("\nrunning call arms ...", flush=True)
    report["calls"]["laya-full"] = H.score_calls(
        calls, [H.run_laya_call(c, router, incremental=False, verify=False) for c in calls]
    )
    report["calls"]["laya"] = H.score_calls(
        calls, [H.run_laya_call(c, router) for c in calls]
    )
    if use_llm:
        report["calls"]["hybrid"] = H.score_calls(
            calls, [H.run_hybrid_call(c, router) for c in calls], model=model
        )
        try:
            report["calls"]["llm"] = H.score_calls(
                calls, [H.run_llm_call(c) for c in calls], model=model
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  llm call arm failed: {exc}")

    # ---- hybrid frontier -----------------------------------------------------
    if use_llm and llm_runs:
        frontier = [
            H.simulate_hybrid(routing, laya_routing, llm_runs[0], t, model=model)
            for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
        ]
        report["routing"]["hybrid_frontier"] = frontier
        frows = [
            [
                f"{f['threshold']:.2f}",
                f"{f['accuracy']:.3f}",
                f"{f['flag_rate']:.1%}",
                f"{f['recall']:.2f}" if f["recall"] is not None else "n/a",
                f"${f['cost_per_case']:.6f}",
            ]
            for f in frontier
        ]
        print()
        print(
            H.render(
                "HYBRID FRONTIER — trust the cascade unless confidence < threshold",
                frows,
                ["threshold", "accuracy", "% to llm", "error recall", "cost/case"],
            )
        )
        print(
            f"\n  cascade alone: {report['routing']['laya']['department_accuracy']:.3f} accuracy at $0"
            f"   |   llm alone: {report['routing']['llm']['department_accuracy']:.3f} accuracy at "
            f"${report['routing']['llm']['cost_usd']:.6f}/case"
        )

    # ---- report --------------------------------------------------------------
    print()
    rows = []
    arms = [a for a in ("laya", "llm") if a in report["routing"]]
    for label, key in [
        ("department accuracy", "department_accuracy"),
        ("intent accuracy", "intent_accuracy"),
        ("joint accuracy", "joint_accuracy"),
        ("p50 latency (ms)", None),
        ("p95 latency (ms)", None),
        ("cost per case", "cost_usd"),
        ("errors", "errors"),
        ("determinism", "determinism"),
    ]:
        row = [label]
        for arm in arms:
            s = report["routing"][arm]
            if key == "cost_usd":
                row.append(H.money(s[key]) if arm == "llm" else "$0")
            elif label.startswith("p50"):
                row.append(f"{s['latency_ms']['p50']}")
            elif label.startswith("p95"):
                row.append(f"{s['latency_ms']['p95']}")
            elif key == "determinism":
                row.append(f"{s.get('determinism', 1.0):.2f}")
            elif key in ("department_accuracy", "intent_accuracy", "joint_accuracy"):
                row.append(f"{s[key]:.3f}")
            else:
                row.append(str(s.get(key, "")))
        rows.append(row)
    print(H.render(f"DECISION LEVEL  ({len(routing)} cases: department + intent)", rows, ["metric", *arms]))

    gate = report["routing"]["laya"].get("gate") or {}
    if gate:
        print()
        print(
            f"GATE QUALITY (laya) — escalate when department confidence < {gate.get('threshold')}\n"
            f"  errors                       {gate.get('errors')}\n"
            f"  flagged for escalation       {gate.get('flagged')} ({gate.get('flag_rate'):.1%} of cases)\n"
            f"  of those errors, caught      {gate.get('errors_caught')}  (recall {gate.get('recall')})\n"
            f"  flags that were warranted    {gate.get('precision')}\n"
            f"  accuracy when confident      {gate.get('accuracy_when_confident')}\n"
            f"  accuracy when flagged        {gate.get('accuracy_when_flagged')}"
        )

    print()
    carms = [a for a in ("laya-full", "laya", "hybrid", "llm") if a in report["calls"]]
    crows = []
    for label, key in [
        ("queue accuracy", "queue_accuracy"),
        ("llm calls made", "llm_calls"),
        ("would-escalate (cascade)", "would_escalate"),
        ("questions asked", "questions"),
        ("p50 latency (ms)", None),
        ("cost total", "cost_usd"),
    ]:
        row = [label]
        for arm in carms:
            s = report["calls"][arm]
            if key == "cost_usd":
                row.append(H.money(s[key]) if arm in ("llm", "hybrid") else "$0")
            elif label.startswith("p50"):
                row.append(f"{s['latency_ms']['p50']}")
            elif key == "queue_accuracy":
                row.append(f"{s[key]:.3f}")
            else:
                row.append(str(s.get(key, "")))
        crows.append(row)
    print(H.render(f"CALL LEVEL  ({len(calls)} calls: final queue)", crows, ["metric", *carms]))

    for arm in arms:
        misses = report["routing"][arm]["misses"]
        if misses:
            print(f"\n{arm} department misses ({len(misses)}):")
            for m in misses[:12]:
                print(f"  {m['id']:9s} expected {m['expected']:11s} got {m['got']}")
    for arm in carms:
        misses = report["calls"][arm]["misses"]
        if misses:
            print(f"\n{arm} call misses ({len(misses)}):")
            for m in misses:
                print(f"  {m['id']:18s} expected {m['expected']:22s} got {m['got']}")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
