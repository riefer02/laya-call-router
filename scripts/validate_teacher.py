"""STEP 1 — the teacher gate.

Before generating thousands of labels and committing a GPU slot, measure whether the teacher can
reproduce labels that a human agrees with. Distilling a teacher caps you at its ceiling, so this
number bounds everything downstream.

Runs the teacher over the 81 held-out hand-labelled routing cases and reports agreement against a
bar. Exit code is non-zero if the bar is missed, so it can be used as a real gate.

    uv run python scripts/validate_teacher.py
    uv run python scripts/validate_teacher.py --provider openai --model gpt-5.4-nano
    uv run python scripts/validate_teacher.py --bar-destination 0.92 --bar-subqueue 0.85
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from jev_classifier import evalharness as H
from jev_classifier import llm, teacher
from jev_classifier.labels import canon

ROOT = Path(__file__).resolve().parents[1]


def run(cases, provider: str, model: str, concurrency: int) -> list[dict]:
    def one(case):
        try:
            rec = teacher.label(case.text, provider=provider, model=model)
        except Exception as exc:  # noqa: BLE001
            rec = {"error": str(exc), "valid": False, "destination": None, "subqueue": None}
        rec["id"] = case.id
        rec["expected_destination"] = case.destination
        rec["expected_subqueue"] = case.subqueue
        rec["destination_ok"] = rec.get("destination") == case.destination
        rec["subqueue_ok"] = case.subqueue is None or rec.get("subqueue") == case.subqueue
        return rec

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, cases))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--bar-destination", type=float, default=0.92)
    ap.add_argument("--bar-subqueue", type=float, default=0.85)
    ap.add_argument("--out", default="results/teacher_validation.json")
    args = ap.parse_args()

    provider = args.provider
    model = args.model or llm.PROVIDERS[provider].default_model
    if not llm.available(provider):
        print(f"no API key for provider {provider!r} (expected {llm.PROVIDERS[provider].key_env})")
        return 2

    cases = H.load_routing()
    print(f"teacher gate: {provider}:{model} over {len(cases)} held-out hand-labelled cases")
    print("(the hand labels are the only human judgement in this project — agreement with them")
    print(" is the ceiling anything we distil from this teacher can reach)\n")

    records = run(cases, provider, model, args.concurrency)

    errors = [r for r in records if r.get("error")]
    n = len(records)
    valid = [r for r in records if r.get("valid")]
    dest_ok = sum(1 for r in records if r.get("destination_ok"))
    sub_ok = sum(1 for r in records if r.get("subqueue_ok"))
    invalid = n - len(valid)
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms")]
    cost = sum(teacher.cost_of(r) or 0.0 for r in records if not r.get("error"))

    dest_acc = dest_ok / n
    sub_acc = sub_ok / n

    report = {
        "provider": provider,
        "model": model,
        "n": n,
        "destination_accuracy": round(dest_acc, 4),
        "subqueue_accuracy": round(sub_acc, 4),
        "joint_accuracy": round(sum(1 for r in records if r.get("destination_ok") and r.get("subqueue_ok")) / n, 4),
        "invalid_labels": invalid,
        "invalid_rate": round(invalid / n, 4),
        "errors": len(errors),
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1) if latencies else None,
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
        },
        "cost_usd": round(cost, 6),
        "bars": {"destination": args.bar_destination, "subqueue": args.bar_subqueue},
        "passes": dest_acc >= args.bar_destination and sub_acc >= args.bar_subqueue,
        "retry_worthy": [],   # invalid or department-wrong cases we could repair
        "records": records,
    }

    # what the teacher gets wrong, grouped, so the failure modes are visible
    wrong = [r for r in records if not r.get("destination_ok")]
    report["destination_misses"] = [
        {
            "id": r["id"],
            "expected": r.get("expected_destination"),
            "got": r.get("destination"),
            "raw": r.get("raw_destination"),
        }
        for r in wrong
    ]
    invalid_items = [r for r in records if not r.get("valid")]
    report["invalid_items"] = [
        {"id": r["id"], "raw_destination": r.get("raw_destination"), "raw_subqueue": r.get("raw_subqueue")}
        for r in invalid_items
    ]

    print(f"destination agreement {dest_ok}/{n} = {dest_acc:.3f}   (bar {args.bar_destination:.2f})")
    print(f"sub-queue agreement   {sub_ok}/{n} = {sub_acc:.3f}   (bar {args.bar_subqueue:.2f})")
    print(f"unusable labels       {invalid}/{n} = {invalid / n:.1%}")
    print(f"latency p50           {report['latency_ms']['p50']} ms")
    print(f"cost                  ${cost:.6f}")
    print()

    if wrong:
        print(f"destination misses ({len(wrong)}):")
        for r in wrong[:20]:
            print(
                f"  {r['id']:9s} expected {str(r.get('expected_destination')):12s} "
                f"got {str(r.get('destination')):12s} raw={r.get('raw_destination')!r}"
            )
    if invalid_items:
        print(f"\nlabels that could not be placed in vocabulary ({len(invalid_items)}):")
        for r in invalid_items[:15]:
            print(f"  {r['id']:9s} raw={r.get('raw_destination')!r} / {r.get('raw_subqueue')!r}")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    verdict = "PASS" if report["passes"] else "FAIL"
    print(f"\n{verdict} — teacher gate {'cleared' if report['passes'] else 'NOT cleared'}; wrote {out}")
    if not report["passes"]:
        print("Do not generate a training set from this teacher yet. Options: raise the bar by")
        print("using a stronger teacher (deepseek-v4-pro), or fix the prompt/taxonomy first.")
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
