"""Probe: find question phrasings the base checkpoints actually answer well.

Laya's base checkpoints are weak zero-shot and highly sensitive to wording, so the schemas in
`schemas.py` are chosen from measurements, not intuition. This script is the evidence.

Run:  uv run python scripts/probe_questions.py
"""

from __future__ import annotations

import laya_mlx as laya

# Each case: (id, message, expected-if-the-model-is-right)
CASES = [
    ("billing", "Hi, we were billed twice for March on invoice #4411. Please refund the duplicate today or we'll have to cancel our plan.", {"dept": "billing", "churn": True, "refund": True}),
    ("outage", "Your API has been returning 503s for 40 minutes and our checkout is completely down. This is blocking all our customers right now.", {"dept": "technical", "churn": False}),
    ("login", "I can't log in any more. My password reset email never arrives. Can you help me get back in?", {"dept": "account"}),
    ("sales", "We're a 40-person team evaluating your product. Could you send pricing for the business tier and arrange a demo next week?", {"dept": "sales"}),
    ("spam", "CONGRATULATIONS!! You have WON a $1000 gift card. Click here to claim your prize now >> bit.ly/not-a-real-link", {"dept": "other"}),
]

VARIANTS = {
    "gate_noul_broad": {"is_support_request": {"type": "noul", "instructions": "Is the customer conversation a genuine request for help with a product or service, rather than spam, marketing or an automated blast?"}},
    "gate_noul_message": {"is_support_request": {"type": "noul", "instructions": "Is `message` a genuine customer support request rather than spam or marketing?"}},
    "gate_noul_plain": {"is_support_request": {"type": "noul", "instructions": "Is this a real customer asking for help with their account or product?"}},
    "gate_choice": {"is_support_request": {"type": "choice", "instructions": "What kind of message is this?", "criteria": {"genuine_support": "a real customer asking for help with their account or product", "spam_or_marketing": "spam, advertising or a scam", "other": "none of the above"}}},
    "churn_noul_1": {"churn_risk": {"type": "noul", "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?"}},
    "churn_noul_2": {"churn_risk": {"type": "noul", "instructions": "Does the customer say they will cancel, stop paying or leave for a competitor?"}},
    "churn_noul_3": {"churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel their plan or stop being a customer?"}},
    "urgency_noul": {"urgency": {"type": "noul", "instructions": "Does the customer need this resolved today or is it blocking their work?"}},
    "urgency_score": {"urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["no time pressure", "needs attention soon", "blocking issue or hard deadline"]}},
    "refund_noul": {"refund_requested": {"type": "noul", "instructions": "Does the customer explicitly ask for money back?"}},
    "dept_choice": {"department": {"type": "choice", "instructions": "Which department should handle this customer request?", "criteria": {"billing": "invoices, payments, refunds, duplicate charges, plan changes", "technical": "bugs, outages, errors, integrations, performance problems", "sales": "pricing, new contracts, upgrades, demos", "account": "login, access, profile details, cancellation", "other": "none of the other options fits"}}},
}


def main() -> None:
    router = laya.Router(max_loaded=2)

    for name, questions in VARIANTS.items():
        print(f"\n### {name}")
        for cid, message, expected in CASES:
            res = router.predict({"message": message}, questions)
            qid = next(iter(questions))
            ans = res["answers"][qid]
            if ans["type"] == "choice":
                got = ans["choice"]
                detail = " ".join(f"{k}={v:.2f}" for k, v in sorted(ans["probabilities"].items(), key=lambda kv: -kv[1])[:3])
            elif ans["type"] == "score":
                got = round(ans["score"], 2)
                detail = f"conf={ans['confidence']:.2f}"
            else:
                got = round(ans["noul"], 2)
                detail = f"conf={ans['confidence']:.2f}"
            flag = ""
            if "dept" in expected and got == expected["dept"]:
                flag = "  ✓ expected"
            elif qid == "churn_risk" and "churn" in expected:
                flag = "  ✓ expected" if (got >= 0.5) == expected["churn"] else "  ✗ expected"
            elif qid == "refund_requested" and "refund" in expected:
                flag = "  ✓ expected" if (got >= 0.5) == expected["refund"] else "  ✗ expected"
            print(f"  {cid:8s} -> {got!r:20s} ({detail}){flag}")


if __name__ == "__main__":
    main()
