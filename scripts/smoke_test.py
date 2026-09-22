"""Phase 1 smoke test.

Confirms the Laya-MLX primitives (``choice`` / ``score`` / ``noul``) and the ``Router``
work end to end on this machine, and records first latency numbers.

Run:  uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import time

import laya_mlx as laya

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds, duplicate charges",
            "technical": "bugs, outages, system errors, integrations",
            "sales": "pricing, new contracts, upgrades",
            "account": "login, access, profile, cancellation",
            "other": "none of the above fits",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the customer threaten to cancel their plan or stop being a customer?",
    },
}

STATES = {
    "en": {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": (
            "Hi, we were billed twice for March. Please refund the duplicate today "
            "or we will cancel our plan."
        ),
    },
    "zh": {"body": "发票被重复扣款，请今天退款，否则我们将取消订阅。"},
}


def _fmt(answer: dict) -> str:
    kind = answer["type"]
    conf = answer["confidence"]
    if kind == "choice":
        dist = " ".join(
            f"{k}={v:.2f}" for k, v in sorted(
                answer["probabilities"].items(), key=lambda kv: -kv[1]
            )
        )
        return f"choice={answer['choice']!r} conf={conf:.2f} | {dist}"
    if kind == "score":
        legend = answer["legend"]
        return (
            f"score={answer['score']:.2f}/{len(legend) - 1} conf={conf:.2f} | "
            f"levels={[legend[str(i)] for i in range(len(legend))]}"
        )
    return f"noul=P(true)={answer['noul']:.2f} conf={conf:.2f}"


def main() -> None:
    print(f"laya-mlx {getattr(laya, '__version__', '?')}")
    print("preloading English + multilingual checkpoints (first run downloads them)...")
    t0 = time.perf_counter()
    router = laya.Router(max_loaded=2)
    router.preload(["english", "multilingual"])
    print(f"checkpoints resident in {time.perf_counter() - t0:.1f}s")

    # Warm up both checkpoints so the numbers below are inference, not a cold model build.
    for state in STATES.values():
        router.predict(state, QUESTIONS)

    for tag, state in STATES.items():
        print("=" * 78)
        print(f"STATE [{tag}]: {state.get('subject') or state.get('body')}")
        result = None
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            result = router.predict(state, QUESTIONS)
            best = min(best, (time.perf_counter() - t0) * 1000)
        routing = result["routing"]
        print(f"  routed -> {routing['model']}  ({routing['reason']})")
        for qid, answer in result["answers"].items():
            print(f"  {qid:10s} {_fmt(answer)}")
        usage = result["usage"]
        print(
            f"  best of 3: {best:.1f} ms | input_tokens={usage['input_tokens']} "
            f"output_tokens={usage['output_tokens']}"
        )


if __name__ == "__main__":
    main()
