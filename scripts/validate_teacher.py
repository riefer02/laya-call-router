"""STEP 1 — the teacher gate.

Before generating thousands of labels and committing a GPU slot, measure whether the teacher can
reproduce labels that a human agrees with. Distilling a teacher caps you at its ceiling, so this
number bounds everything downstream.

Runs the teacher over the 81 held-out hand-labelled routing cases and reports agreement against a
bar. Exit code is non-zero if the bar is missed, so it can be used as a real gate.

    uv run python scripts/validate_teacher.py
    uv run python scripts/validate_teacher.py --provider openai --model gpt-5.4-nano
    uv run python scripts/validate_teacher.py --bar-department 0.92 --bar-intent 0.85
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
            rec = {"error": str(exc), "valid": False, "department": None, "intent": None}
        rec["id"] = case.id
        rec["expected_department"] = case.department
        rec["expected_intent"] = case.intent
        rec["department_ok"] = rec.get("department") == case.department
        rec["intent_ok"] = rec.get("intent") == case.intent
        return rec

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, cases))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--bar-department", type=float, default=0.92)
    ap.add_argument("--bar-intent", type=float, default=0.85)
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
    dept_ok = sum(1 for r in records if r.get("department_ok"))
    intent_ok = sum(1 for r in records if r.get("intent_ok"))
    invalid = n - len(valid)
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms")]
    cost = sum(teacher.cost_of(r) or 0.0 for r in records if not r.get("error"))

    dept_acc = dept_ok / n
    intent_acc = intent_ok / n

    report = {
        "provider": provider,
        "model": model,
        "n": n,
        "department_accuracy": round(dept_acc, 4),
        "intent_accuracy": round(intent_acc, 4),
        "joint_accuracy": round(sum(1 for r in records if r.get("department_ok") and r.get("intent_ok")) / n, 4),
        "invalid_labels": invalid,
        "invalid_rate": round(invalid / n, 4),
        "errors": len(errors),
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1) if latencies else None,
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
        },
        "cost_usd": round(cost, 6),
        "bars": {"department": args.bar_department, "intent": args.bar_intent},
        "passes": dept_acc >= args.bar_department and intent_acc >= args.bar_intent,
        "retry_worthy": [],   # invalid or department-wrong cases we could repair
        "records": records,
    }

    # what the teacher gets wrong, grouped, so the failure modes are visible
    wrong = [r for r in records if not r.get("department_ok")]
    report["department_misses"] = [
        {
            "id": r["id"],
            "expected": r.get("expected_department"),
            "got": r.get("department"),
            "raw": r.get("raw_department"),
        }
        for r in wrong
    ]
    invalid_items = [r for r in records if not r.get("valid")]
    report["invalid_items"] = [
        {"id": r["id"], "raw_department": r.get("raw_department"), "raw_intent": r.get("raw_intent")}
        for r in invalid_items
    ]

    print(f"department agreement  {dept_ok}/{n} = {dept_acc:.3f}   (bar {args.bar_department:.2f})")
    print(f"intent agreement      {intent_ok}/{n} = {intent_acc:.3f}   (bar {args.bar_intent:.2f})")
    print(f"unusable labels       {invalid}/{n} = {invalid / n:.1%}")
    print(f"latency p50           {report['latency_ms']['p50']} ms")
    print(f"cost                  ${cost:.6f}")
    print()

    if wrong:
        print(f"department misses ({len(wrong)}):")
        for r in wrong[:20]:
            print(
                f"  {r['id']:9s} expected {str(r.get('expected_department')):11s} "
                f"got {str(r.get('department')):11s} raw={r.get('raw_department')!r}"
            )
    if invalid_items:
        print(f"\nlabels that could not be placed in vocabulary ({len(invalid_items)}):")
        for r in invalid_items[:15]:
            print(f"  {r['id']:9s} raw={r.get('raw_department')!r} / {r.get('raw_intent')!r}")

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
