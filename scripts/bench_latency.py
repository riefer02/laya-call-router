"""Benchmark: latency and throughput of the triage schema on this machine.

Measures per-checkpoint, on the actual schema the cascade uses, so the numbers mean something
for this project rather than for a synthetic one-question call.

Run:  uv run python scripts/bench_latency.py
      uv run python scripts/bench_latency.py --checkpoints english multilingual
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import laya_mlx as laya

from jev_classifier import schemas as S

STATE = {
    "message": (
        "Hi, we were billed twice for March on invoice #4411. Please refund the "
        "duplicate today or we'll have to cancel our plan."
    )
}

ONE_QUESTION = {"department": S.DEPARTMENT_QUESTIONS["department"]}

FULL_SCHEMA = {
    **S.DEPARTMENT_QUESTIONS,
    **S.sub_intent_questions("billing"),
    **S.SEVERITY_QUESTIONS,
}

RESULTS = Path(__file__).resolve().parents[1] / "results"


def quantiles(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "p50": round(statistics.median(s), 2),
        "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2),
        "mean": round(statistics.fmean(s), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
    }


def time_call(router, state, questions, model: str, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        router.predict(state, questions, model=model)
        out.append((time.perf_counter() - t0) * 1000)
    return out


def peak_memory_mb() -> float | None:
    try:
        import mlx.core as mx

        return round(mx.get_peak_memory() / (1024 * 1024), 1)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        default=["english", "multilingual", "typed-decisions"],
    )
    ap.add_argument("--iterations", type=int, default=50)
    args = ap.parse_args()

    router = laya.Router(max_loaded=3)
    print(f"preloading {args.checkpoints} ...")
    router.preload(args.checkpoints)

    report = {"device": "Apple Silicon (MLX)", "iterations": args.iterations, "checkpoints": {}}
    for name in args.checkpoints:
        print(f"\n=== {name} ===")
        # warm up
        router.predict(STATE, ONE_QUESTION, model=name)
        router.predict(STATE, FULL_SCHEMA, model=name)

        one = time_call(router, STATE, ONE_QUESTION, name, args.iterations)
        full = time_call(router, STATE, FULL_SCHEMA, name, args.iterations)

        n_questions = len(FULL_SCHEMA)
        entry = {
            "one_question_ms": quantiles(one),
            "full_schema_ms": quantiles(full),
            "full_schema_questions": n_questions,
            "questions_per_second": round(n_questions * 1000 / statistics.fmean(full), 1),
        }
        report["checkpoints"][name] = entry

        print(f"  1 question   p50 {entry['one_question_ms']['p50']:>7.2f} ms   p95 {entry['one_question_ms']['p95']:>7.2f} ms")
        print(f"  {n_questions} questions p50 {entry['full_schema_ms']['p50']:>7.2f} ms   p95 {entry['full_schema_ms']['p95']:>7.2f} ms")
        print(f"  throughput   {entry['questions_per_second']:>7.1f} questions/s (batched in one pass)")

    report["peak_mlx_memory_mb"] = peak_memory_mb()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "latency.json"
    out.write_text(json.dumps(report, indent=2))
    peak = report["peak_mlx_memory_mb"]
    print(f"\npeak MLX memory: {peak} MiB" if peak else "")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
