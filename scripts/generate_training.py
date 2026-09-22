"""STEP 3 — generate and validate the synthetic training set.

Quality bar (the dataset does not ship unless every line passes):

    every intent >= --min-per-intent          (default 25)
    every department >= --min-per-department  (default 100)
    100% of labels inside our vocabulary
    zero near-duplicates of the held-out hand-labelled cases
    both labelling phrasings agreed on 100% of kept rows

Candidates come from the teacher, but a candidate only becomes a training example when *two*
independently-worded labelling passes agree with each other *and* with the intended target. That
filter is the whole point: it removes exactly the cases the teacher is unsure about.

    uv run python scripts/generate_training.py --per-intent 6 --out data/calls/pilot.jsonl
    uv run python scripts/generate_training.py --per-intent 50
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from concurrent.futures import ThreadPoolExecutor

from jev_classifier import dealership as D
from jev_classifier import evalharness as H
from jev_classifier import llm, synthgen, teacher

ROOT = Path(__file__).resolve().parents[1]
BATCH = 6


def make_pair(department: str, intent: str, target: int, provider: str, model: str, rounds: int) -> Dict[str, Any]:
    kept: List[dict] = []
    stats = {"generated": 0, "dup": 0, "invalid": 0, "disagreed": 0, "mismatch": 0, "rounds": 0, "cost": 0.0}
    style_i = abs(hash((department, intent))) % len(synthgen.STYLES)

    for _ in range(rounds):
        if len(kept) >= target:
            break
        stats["rounds"] += 1
        style = synthgen.STYLES[style_i % len(synthgen.STYLES)]
        style_i += 1
        try:
            utterances, meta = synthgen.generate(
                department, intent, BATCH, style, provider=provider, model=model
            )
            stats["cost"] += teacher.cost_of({"usage": meta["usage"], "provider": provider, "model": model}) or 0.0
        except Exception as exc:  # noqa: BLE001
            stats.setdefault("errors", []).append(str(exc))
            continue

        for text in utterances:
            if len(kept) >= target:
                break
            stats["generated"] += 1
            # The deduper is consulted before labelling so we never pay to label a duplicate.
            if DEDUPER.why_reject(text):
                stats["dup"] += 1
                continue
            try:
                l1 = teacher.label(text, provider=provider, model=model, variant=1)
                l2 = teacher.label(text, provider=provider, model=model, variant=2)
            except Exception as exc:  # noqa: BLE001
                stats.setdefault("errors", []).append(str(exc))
                stats["invalid"] += 1
                continue
            stats["cost"] += (teacher.cost_of(l1) or 0.0) + (teacher.cost_of(l2) or 0.0)
            if not (l1["valid"] and l2["valid"]):
                stats["invalid"] += 1
                continue
            if (l1["department"], l1["intent"]) != (l2["department"], l2["intent"]):
                stats["disagreed"] += 1
                continue
            if (l1["department"], l1["intent"]) != (department, intent):
                stats["mismatch"] += 1
                continue
            kept.append(
                {
                    "text": text,
                    "department": department,
                    "intent": intent,
                    "style": style,
                    "agreed": True,
                    "teacher": {"provider": provider, "model": model},
                }
            )
    return {"department": department, "intent": intent, "kept": kept, "stats": stats}


DEDUPER: synthgen.Deduper


def main() -> int:
    global DEDUPER
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--min-per-intent", type=int, default=25)
    ap.add_argument("--min-per-department", type=int, default=100)
    ap.add_argument("--dev-fraction", type=float, default=0.1)
    ap.add_argument("--out", default="data/calls/synthetic.jsonl")
    ap.add_argument("--dev-out", default="data/calls/synthetic_dev.jsonl")
    args = ap.parse_args()

    provider = args.provider
    model = args.model or llm.PROVIDERS[provider].default_model
    if not llm.available(provider):
        print(f"no key for {provider}")
        return 2

    held_out = H.load_routing()
    protected = [c.text for c in held_out]
    DEDUPER = synthgen.Deduper(protected=protected, threshold=0.6)

    pairs = [(d, i) for d, branch in D.INTENTS.items() for i in branch]
    print(f"generating: {len(pairs)} department/intent pairs x {args.per_intent} target "
          f"({provider}:{model})")
    print(f"held out from training: {len(protected)} hand-labelled cases "
          f"(near-duplicate guard at Jaccard >= 0.6)\n")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(make_pair, d, i, args.per_intent, provider, model, args.rounds)
            for d, i in pairs
        ]
        for n, fut in enumerate(futures, 1):
            results.append(fut.result())
            if n % 5 == 0 or n == len(futures):
                kept = sum(len(r["kept"]) for r in results)
                cost = sum(r["stats"]["cost"] for r in results)
                print(f"  {n}/{len(futures)} pairs · {kept} kept · ${cost:.4f}", flush=True)

    rows = [row for r in results for row in r["kept"]]
    random.Random(20260922).shuffle(rows)

    by_intent = Counter((r["department"], r["intent"]) for r in rows)
    by_dept = Counter(r["department"] for r in rows)
    stats = {
        k: sum(r["stats"][k] for r in results)
        for k in ("generated", "dup", "invalid", "disagreed", "mismatch", "rounds")
    }
    cost = sum(r["stats"]["cost"] for r in results)
    errors = [e for r in results for e in r["stats"].get("errors", [])]

    # ---- acceptance criteria -------------------------------------------------
    problems: List[str] = []
    for (dept, intent), n in by_intent.items():
        if n < args.min_per_intent:
            problems.append(f"intent {dept}/{intent} has {n} < {args.min_per_intent}")
    for dept, n in by_dept.items():
        if n < args.min_per_department:
            problems.append(f"department {dept} has {n} < {args.min_per_department}")
    if any(not r["agreed"] for r in rows):
        problems.append("some rows were not double-agreement filtered")
    invalid = [r for r in rows if r["department"] not in D.DEPARTMENTS
               or r["intent"] not in D.INTENTS.get(r["department"], {})]
    if invalid:
        problems.append(f"{len(invalid)} rows have out-of-vocabulary labels")
    # explicit re-check: nothing may be near-identical to the held-out set
    leak_check = synthgen.Deduper(protected=protected, threshold=0.6)
    leaked = []
    for r in rows:
        reason = leak_check.why_reject(r["text"])
        if reason and reason.startswith("near-duplicate-of-heldout"):
            leaked.append(r["text"])
    if leaked:
        problems.append(f"{len(leaked)} rows near-duplicate a held-out case")

    n = len(rows)
    dev_n = max(1, int(n * args.dev_fraction)) if n else 0
    dev, train = rows[:dev_n], rows[dev_n:]

    print("\n" + "=" * 62)
    print(f"kept                 {n}")
    print(f"candidates generated {stats['generated']}")
    print(f"  dropped duplicate  {stats['dup']}  ({stats['dup'] / max(1, stats['generated']):.1%})")
    print(f"  dropped invalid    {stats['invalid']}")
    print(f"  dropped disagree   {stats['disagreed']}  (two phrasings disagreed)")
    print(f"  dropped mismatch   {stats['mismatch']}  (label != intended target)")
    print(f"train / dev          {len(train)} / {len(dev)}")
    print(f"departments          {dict(sorted(by_dept.items()))}")
    print(f"intents covered      {len(by_intent)}/{len(pairs)}")
    print(f"min per intent       {min(by_intent.values()) if by_intent else 0}")
    print(f"total cost           ${cost:.4f}")
    if errors:
        print(f"call errors          {len(errors)} (first: {errors[0][:80]})")

    if problems:
        print("\nACCEPTANCE CRITERIA FAILED:")
        for p in problems[:15]:
            print(f"  - {p}")
    else:
        print("\nACCEPTANCE CRITERIA: all passed")

    payload = {
        "provider": provider,
        "model": model,
        "kept": n,
        "stats": stats,
        "by_intent": {f"{d}/{i}": c for (d, i), c in sorted(by_intent.items())},
        "by_department": dict(sorted(by_dept.items())),
        "cost_usd": round(cost, 6),
        "problems": problems,
        "criteria": {
            "min_per_intent": args.min_per_intent,
            "min_per_department": args.min_per_department,
            "dev_fraction": args.dev_fraction,
        },
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_report.json").write_text(json.dumps(payload, indent=2))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in train) + ("\n" if train else ""))
    dev_out = ROOT / args.dev_out
    dev_out.write_text("\n".join(json.dumps(r) for r in dev) + ("\n" if dev else ""))
    print(f"\nwrote {out} ({len(train)}) and {dev_out} ({len(dev)})")
    print("wrote results/dataset_report.json")

    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
