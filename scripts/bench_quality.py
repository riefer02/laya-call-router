"""Benchmark: accuracy and calibration on a small hand-labelled ticket set.

This is deliberately small (18 tickets) and honest: it measures the base checkpoints zero-shot
on *my* labels, which is exactly the setting the Laya docs warn is weak. The point is to show
the size of the gap, not to reproduce the maintainer's benchmark.

Run:  uv run python scripts/bench_quality.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laya_mlx as laya

from jev_classifier import schemas as S

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tickets" / "labelled.jsonl"
RESULTS = ROOT / "results"

QUESTIONS = {
    **S.DEPARTMENT_QUESTIONS,
    "churn_risk": S.SEVERITY_QUESTIONS["churn_risk"],
    "refund_requested": S.SEVERITY_QUESTIONS["refund_requested"],
}


def load() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def ece(pairs: list[tuple[float, bool]], n_bins: int = 8) -> float:
    """Expected calibration error: bins by reported confidence, compares to observed accuracy."""
    if not pairs:
        return 0.0
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        conf = max(p, 1.0 - p)
        correct = 1.0 if (p >= 0.5) == bool(y) else 0.0
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx].append((conf, correct))
    total = len(pairs)
    err = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(a for _, a in b) / len(b)
        err += (len(b) / total) * abs(acc - conf)
    return round(err, 4)


def evaluate(router, model: str, rows: list[dict]) -> dict:
    dept_ok = 0
    churn_pairs, refund_pairs = [], []
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        res = router.predict({"message": row["message"]}, QUESTIONS, model=model)
        ans = res["answers"]
        got = ans["department"]["choice"]
        want = row["department"]
        confusion.setdefault(want, {}).setdefault(got, 0)
        confusion[want][got] += 1
        dept_ok += int(got == want)
        churn_pairs.append((float(ans["churn_risk"]["noul"]), bool(row.get("churn"))))
        refund_pairs.append((float(ans["refund_requested"]["noul"]), bool(row.get("refund"))))

    n = len(rows)
    return {
        "n": n,
        "department_accuracy": round(dept_ok / n, 4),
        "churn_accuracy": round(sum((p >= 0.5) == y for p, y in churn_pairs) / n, 4),
        "refund_accuracy": round(sum((p >= 0.5) == y for p, y in refund_pairs) / n, 4),
        "churn_brier": round(sum((p - y) ** 2 for p, y in churn_pairs) / n, 4),
        "refund_brier": round(sum((p - y) ** 2 for p, y in refund_pairs) / n, 4),
        "noul_ece": ece(churn_pairs + refund_pairs),
        "confusion": confusion,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["english", "multilingual"])
    args = ap.parse_args()

    rows = load()
    router = laya.Router(max_loaded=3)
    router.preload(args.checkpoints)

    report: dict = {"n_tickets": len(rows), "checkpoints": {}}
    print(f"evaluating {len(rows)} labelled tickets\n")
    header = f"{'checkpoint':16s} {'dept acc':>9s} {'churn acc':>10s} {'refund acc':>11s} {'noul ECE':>9s}"
    print(header)
    print("-" * len(header))
    for name in args.checkpoints:
        res = evaluate(router, name, rows)
        report["checkpoints"][name] = res
        print(
            f"{name:16s} {res['department_accuracy']:>9.2f} {res['churn_accuracy']:>10.2f} "
            f"{res['refund_accuracy']:>11.2f} {res['noul_ece']:>9.3f}"
        )

    print("\nchance baselines: department = 0.28 (5 options), churn = 0.17 (3/18), refund = 0.28 (5/18)")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "quality.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
