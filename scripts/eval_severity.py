"""Does the switchboard know when a caller is stranded, and when they need a person?

Two `noul` questions drive real decisions and had never been measured:

  is_safe_to_drive   dispatches roadside assistance, and sets priority to HIGH
  needs_human        hands the call to a person instead of the automated flow

**The errors are not symmetric, so accuracy alone would be the wrong number.** Missing a stranded
caller leaves someone at the side of a road; a false alarm sends a truck to someone who was fine.
The first is the failure worth designing against, so this reports recall on the positive class and
sweeps the threshold, which is the knob that trades the two.

    uv run python scripts/eval_severity.py
    uv run python scripts/eval_severity.py --finetuned models/kaggle-out-v4/laya-dealership-routing
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import laya_mlx as laya  # noqa: E402

from jev_classifier import dealership as D  # noqa: E402
from jev_classifier import evalharness as EH  # noqa: E402
from jev_classifier.agent import get_router  # noqa: E402

DATA = ROOT / "data" / "calls" / "severity.jsonl"
QUESTIONS = ("is_safe_to_drive", "needs_human")


def load() -> List[dict]:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no labelled cases in {DATA}")
    return rows


def run(rows: List[dict], router) -> List[dict]:
    questions = {q: D.SLOT_QUESTIONS[q] for q in QUESTIONS}
    out = []
    for row in rows:
        started = time.perf_counter()
        result = router.predict({"call": row["text"]}, questions)
        ms = (time.perf_counter() - started) * 1000
        got = {q: float(result["answers"][q].get("noul") or 0.0) for q in QUESTIONS}
        out.append({**row, "p": got, "latency_ms": ms})
    return out


# The severity set is enriched on purpose: 18 of 45 positives, so that recall can be measured at all.
# A real switchboard is nowhere near that, and precision depends on the base rate - so the precision
# this script has always printed describes a world that does not exist. `precision_by_base_rate`
# (in evalharness, with the maths and the test) is what to read instead.
DEPLOYMENT_BASE_RATES = EH.DEPLOYMENT_BASE_RATES


def at_threshold(rows: List[dict], question: str, threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        want = bool(row[question])
        got = row["p"][question] >= threshold
        tp += want and got
        fp += (not want) and got
        tn += (not want) and (not got)
        fn += want and (not got)
    n = len(rows)
    pos = tp + fn
    neg = tn + fp
    sens = tp / pos if pos else 0.0
    spec = tn / neg if neg else 1.0
    return {
        "threshold": threshold,
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "recall": round(sens, 4) if pos else None,  # of the unsafe callers, how many caught
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        # Precision depends on the base rate, and this set is enriched so that recall can be
        # measured at all. Read these instead when the question is "how often does a truck roll for
        # nothing" - sensitivity and specificity are the classifier's properties, precision is not.
        "specificity": round(spec, 4) if neg else None,
        "precision_by_base_rate": EH.precision_by_base_rate(sens, spec),
        "false_alarms": fp,
        "missed": fn,  # the costly error: a stranded caller treated as fine
        "missed_ids": [r["id"] for r in rows if r[question] and r["p"][question] < threshold],
        "positives": pos,
        "negatives": neg,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", default="", help="path to a fine-tuned Laya checkpoint")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/severity.json")
    args = ap.parse_args()

    rows = load()
    pos_counts = {q: sum(1 for r in rows if r[q]) for q in QUESTIONS}
    print(f"severity set: {len(rows)} labelled utterances")
    print(f"  positives: " + ", ".join(f"{q}={n}" for q, n in pos_counts.items()))
    print("  (a missed unsafe caller is the error worth designing against, so recall leads)\n")

    router = get_router()
    arms = {"base": router}
    if args.finetuned:
        path = Path(args.finetuned)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            print(f"--finetuned path does not exist: {path}")
            return 2
        ft = laya.Router(models={"english": (str(path), None)}, max_loaded=2)
        arms["fine-tuned"] = ft

    report: Dict[str, object] = {"n": len(rows), "positives": pos_counts, "arms": {}}
    for name, r in arms.items():
        r.preload(["english", "multilingual"])
        scored = run(rows, r)
        lat = [s["latency_ms"] for s in scored]
        report["arms"][name] = {
            "at_default": {q: at_threshold(scored, q, args.threshold) for q in QUESTIONS},
            "sweep": {
                q: [at_threshold(scored, q, t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
                for q in QUESTIONS
            },
            "latency_ms": {"p50": round(statistics.median(lat), 1)},
            "rows": [
                {"id": s["id"], "text": s["text"], **{f"p_{q}": round(s["p"][q], 3) for q in QUESTIONS},
                 **{q: int(s[q]) for q in QUESTIONS}, "why": s["why"]}
                for s in scored
            ],
        }
        print(f"=== {name} (p50 {report['arms'][name]['latency_ms']['p50']} ms) ===")
        for q in QUESTIONS:
            res = report["arms"][name]["at_default"][q]
            print(
                f"  {q:18s} at {args.threshold:.1f}: accuracy {res['accuracy']:.3f}  "
                f"recall {res['recall']}  precision {res['precision']}  "
                f"missed {res['missed']}/{res['positives']}"
            )
            if res["missed_ids"]:
                print(f"      MISSED: {', '.join(res['missed_ids'])}")
        print()

    print("precision by base rate — what a truck actually rolls for nothing")
    print("  (the set above is enriched; these are the same classifiers at deployment rates)")
    header = "  " + f"{'arm':14s}" + "".join(f"{r:>8.0%}" for r in DEPLOYMENT_BASE_RATES)
    print(header)
    for name in arms:
        res = report["arms"][name]["at_default"]["is_safe_to_drive"]
        cells = "".join(f"{res['precision_by_base_rate'][f'{r:.0%}']:>8.3f}" for r in DEPLOYMENT_BASE_RATES)
        print(f"  {name:14s}{cells}   (specificity {res['specificity']:.3f})")
    print()

    print("threshold sweep — is_safe_to_drive (recall of unsafe callers vs false alarms)")
    for name in arms:
        print(f"  {name}:")
        for row in report["arms"][name]["sweep"]["is_safe_to_drive"]:
            print(
                f"    thr {row['threshold']:.1f}  recall {row['recall']}  "
                f"precision {row['precision']}  missed {row['missed']}  false alarms {row['false_alarms']}"
            )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
