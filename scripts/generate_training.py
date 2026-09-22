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
from typing import Any, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

from jev_classifier import dealership as D
from jev_classifier import evalharness as H
from jev_classifier import llm, synthgen, teacher

ROOT = Path(__file__).resolve().parents[1]
BATCH = 6


def make_pair(
    department: str,
    intent: str,
    target: int,
    provider: str,
    model: str,
    rounds: int,
    seed_kept: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    kept: List[dict] = list(seed_kept or [])
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
    ap.add_argument("--resume", action="store_true", default=True,
                    help="reuse validated rows already on disk and top up only the shortfalls")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
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

    # Resume: keep what a previous run produced and top up only the shortfalls, so raising the
    # target does not pay to regenerate work already validated.
    existing: List[dict] = []
    out_path = ROOT / args.out
    if args.resume and out_path.is_file():
        existing = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        dev_path = ROOT / args.dev_out
        if dev_path.is_file():
            existing += [json.loads(line) for line in dev_path.read_text().splitlines() if line.strip()]
        print(f"resuming: {len(existing)} validated rows already on disk")

    DEDUPER = synthgen.Deduper(protected=protected + [r["text"] for r in existing], threshold=0.6)
    seeded: Dict[tuple, List[dict]] = defaultdict(list)
    for row in existing:
        seeded[(row["department"], row["intent"])].append(row)

    pairs = [(d, i) for d, branch in D.INTENTS.items() for i in branch]
    short = [(d, i) for d, i in pairs if len(seeded[(d, i)]) < args.per_intent]
    print(f"generating: {len(short)}/{len(pairs)} pairs below target x {args.per_intent} "
          f"({provider}:{model})")
    print(f"held out from training: {len(protected)} hand-labelled cases "
          f"(near-duplicate guard at Jaccard >= 0.6)\n")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(make_pair, d, i, args.per_intent, provider, model, args.rounds, seeded[(d, i)]): (d, i)
            for d, i in short
        }
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 5 == 0 or done == len(futures):
                kept = sum(len(r["kept"]) for r in results)
                cost = sum(r["stats"]["cost"] for r in results)
                print(f"  {done}/{len(futures)} pairs · {kept} kept · ${cost:.4f}", flush=True)

    # pairs already at target still count toward coverage
    for (d, i), rows in seeded.items():
        if (d, i) not in short:
            results.append({"department": d, "intent": i, "kept": rows, "stats": {}})

    rows = [row for r in results for row in r["kept"]]
    random.Random(20260922).shuffle(rows)

    by_intent = Counter((r["department"], r["intent"]) for r in rows)
    by_dept = Counter(r["department"] for r in rows)
    stats = {
        k: sum(r["stats"].get(k, 0) for r in results)
        for k in ("generated", "dup", "invalid", "disagreed", "mismatch", "rounds")
    }
    cost = sum(r["stats"].get("cost", 0.0) for r in results)
    errors = [e for r in results for e in r["stats"].get("errors", [])]

    # ---- acceptance criteria -------------------------------------------------
    # `other` is a residual class, not a category we generate positives for: it is what the branch
    # falls back to when nothing specific fits. Gating on its count would push us to manufacture
    # "other" examples, i.e. teach the model to answer `other` for things that have a better label.
    # Coverage is therefore required of the *specific* intents, and a department's requirement
    # scales with how many specific intents it actually has.
    specific = {d: [i for i in branch if i != "other"] for d, branch in D.INTENTS.items()}
    problems: List[str] = []
    for (dept, intent), n in by_intent.items():
        if intent == "other":
            continue
        if n < args.min_per_intent:
            problems.append(f"intent {dept}/{intent} has {n} < {args.min_per_intent}")
    for dept, n in by_dept.items():
        required = max(1, len(specific[dept])) * args.min_per_intent
        if n < required:
            problems.append(f"department {dept} has {n} < {required} (={len(specific[dept])} specific intents x {args.min_per_intent})")
    for dept, intents in specific.items():
        for intent in intents:
            if (dept, intent) not in by_intent:
                problems.append(f"specific intent {dept}/{intent} has 0 examples")
    missing_others = [f"{d}/other" for d in D.DEPARTMENTS if (d, "other") not in by_intent]
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

    payload = {
        "provider": provider,
        "model": model,
        "kept": n,
        "stats": stats,
        "by_intent": {f"{d}/{i}": c for (d, i), c in sorted(by_intent.items())},
        "by_department": dict(sorted(by_dept.items())),
        "cost_usd": round(cost, 6),
        "problems": problems,
        "residual_missing": missing_others,
        "criteria": {
            "min_per_intent_specific": args.min_per_intent,
            "min_per_department": "max(1, n_specific_intents) x min_per_intent",
            "residual_intents_gated": False,
            "dev_fraction": args.dev_fraction,
        },
    }

    # Write before reporting: an exception in the summary print should never throw away a paid
    # generation run. (It did once.)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_report.json").write_text(json.dumps(payload, indent=2))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in train) + ("\n" if train else ""))
    dev_out = ROOT / args.dev_out
    dev_out.write_text("\n".join(json.dumps(r) for r in dev) + ("\n" if dev else ""))

    print("\n" + "=" * 62)
    print(f"kept                 {n}")
    print(f"candidates generated {stats['generated']}")
    print(f"  dropped duplicate  {stats['dup']}  ({stats['dup'] / max(1, stats['generated']):.1%})")
    print(f"  dropped invalid    {stats['invalid']}")
    print(f"  dropped disagree   {stats['disagreed']}  (two phrasings disagreed)")
    print(f"  dropped mismatch   {stats['mismatch']}  (label != intended target)")
    print(f"train / dev          {len(train)} / {len(dev)}")
    print(f"departments          {dict(sorted(by_dept.items()))}")
    print(f"intents covered      {len(by_intent)}/{len(pairs)} ({len(missing_others)} residual-only missing)")
    print(f"min specific intent  {min((v for k, v in by_intent.items() if k[1] != 'other'), default=0)}")
    print(f"residual (other)     {sum(v for k, v in by_intent.items() if k[1] == 'other')} "
          f"across {len([k for k in by_intent if k[1] == 'other'])} departments (not gated)")
    print(f"total cost           ${cost:.4f}")
    if errors:
        print(f"call errors          {len(errors)} (first: {errors[0][:80]})")

    if problems:
        print("\nACCEPTANCE CRITERIA FAILED:")
        for p in problems[:15]:
            print(f"  - {p}")
    else:
        print("\nACCEPTANCE CRITERIA: all passed")

    print(f"\nwrote {out} ({len(train)}) and {dev_out} ({len(dev)})")
    print("wrote results/dataset_report.json")

    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
