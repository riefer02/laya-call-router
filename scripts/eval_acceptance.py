"""Measure the one question the switchboard asks mid-conversation.

"Which of the times I just offered did the caller agree to?" It was untrained, and the failure was
specific rather than diffuse: the model never once answered `unclear`. Every "Hmm, let me think
about it" was forced into a slot or a rejection, which is why a booking was filed for a caller who
had agreed to nothing, and why the demo books on some runs and asks again on others.

    uv run python scripts/eval_acceptance.py
    uv run python scripts/eval_acceptance.py --finetuned models/kaggle-out-v5/laya-dealership-routing
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import laya_mlx as laya  # noqa: E402

from jev_classifier import store_profile as SP  # noqa: E402

DATA = ROOT / "data" / "calls" / "acceptance_train.jsonl"


def load() -> List[dict]:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no examples in {DATA}")
    return rows


def run(rows: List[dict], router, profile) -> List[dict]:
    instructions = profile.question_text("acceptance")
    out = []
    for row in rows:
        q = {
            "acceptance": {
                "type": "choice",
                "instructions": instructions,
                "criteria": row["options"],
            }
        }
        started = time.perf_counter()
        ans = router.predict({"call": row["text"]}, q)["answers"]["acceptance"]
        out.append(
            {
                "want": row["choice"],
                "got": ans.get("choice"),
                "top_probability": ans.get("top_probability"),
                "ms": (time.perf_counter() - started) * 1000,
            }
        )
    return out


def score(scored: List[dict]) -> Dict[str, object]:
    per: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    ok = 0
    for s in scored:
        good = s["got"] == s["want"]
        ok += good
        per[s["want"]][0] += good
        per[s["want"]][1] += 1
    n = len(scored)
    return {
        "n": n,
        "accuracy": round(ok / n, 4) if n else 0.0,
        "per_class": {
            k: {"correct": g, "n": t, "accuracy": round(g / t, 4)} for k, (g, t) in sorted(per.items())
        },
        "abstain_rate": round(
            sum(1 for s in scored if s["got"] == "unclear") / n, 4
        )
        if n
        else 0.0,
        "latency_ms_p50": round(statistics.median([s["ms"] for s in scored]), 1) if n else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", default="")
    ap.add_argument("--out", default="results/acceptance.json")
    args = ap.parse_args()

    SP.clear_cache()
    profile = SP.load()
    rows = load()
    print(f"acceptance set: {len(rows)} examples")
    print("  the question the switchboard asks after offering times\n")

    routers = {"base": laya.Router(max_loaded=2)}
    if args.finetuned:
        path = Path(args.finetuned)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            print(f"--finetuned path does not exist: {path}")
            return 2
        routers["fine-tuned"] = laya.Router(
            models={"english": (str(path), None)}, max_loaded=2
        )

    report: Dict[str, object] = {"n": len(rows), "arms": {}}
    for name, router in routers.items():
        router.preload(["english", "multilingual"])
        scored = run(rows, router, profile)
        report["arms"][name] = score(scored)
        s = report["arms"][name]
        print(f"=== {name} — accuracy {s['accuracy']:.3f}  (p50 {s['latency_ms_p50']} ms) ===")
        for cls, row in s["per_class"].items():
            flag = "  <-- never answered" if row["correct"] == 0 else ""
            print(f"    {cls:16s} {row['correct']:>4}/{row['n']:<4} {row['accuracy']:.3f}{flag}")
        print()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
