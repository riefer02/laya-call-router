"""Stage 6: turn the cascade's typed decisions into a concrete routing outcome.

Deliberately deterministic — this is ordinary business logic, not a model call. The point of
the cascade is that everything upstream of here is a calibrated probability; the routing
policy itself should be readable and auditable.
"""

from __future__ import annotations

from typing import Any, Dict, List

QUEUES: Dict[str, str] = {
    "billing": "Billing Team",
    "technical": "Tier-2 Engineering",
    "sales": "Sales Desk",
    "account": "Account & Access",
    "other": "General Support",
}

# Sub-intents that override the default department queue.
SPECIAL_QUEUES: Dict[tuple, str] = {
    ("billing", "refund"): "Refund Desk",
    ("billing", "duplicate_charge"): "Refund Desk",
    ("technical", "outage"): "Incident Response",
    ("account", "cancellation"): "Retention Specialist",
    ("sales", "new_contract"): "Sales Desk (priority)",
}

# Thresholds for the policy. Kept as named constants so they are easy to tune.
HIGH_BLOCKING = 0.6
HIGH_FRUSTRATION = 2.5
CHURN_FLAG = 0.5
REFUND_FLAG = 0.6
ABUSE_FLAG = 0.5


def decide(answers: Dict[str, Any], low_confidence_stages: List[str]) -> Dict[str, Any]:
    """Build the routing outcome from the cascade's aggregated answers.

    ``answers`` maps question id -> the raw Laya answer dict. Missing keys are treated as
    "unknown" so a partial cascade still routes somewhere.
    """

    def pick(qid: str) -> Any:
        ans = answers.get(qid)
        if not isinstance(ans, dict):
            return None
        if ans.get("type") == "choice":
            return ans.get("choice")
        if ans.get("type") == "score":
            return ans.get("score")
        if ans.get("type") == "noul":
            return ans.get("noul")
        return None

    def prob(qid: str) -> float:
        val = pick(qid)
        return float(val) if isinstance(val, (int, float)) else 0.0

    department = pick("department") or "other"
    sub_intent = pick("sub_intent") or "other"
    is_blocking = prob("is_blocking")
    frustration = prob("frustration")
    churn = prob("churn_risk")
    refund = prob("refund_requested")
    abusive = prob("is_abusive")

    flags: List[str] = []
    if churn >= CHURN_FLAG:
        flags.append("churn_risk")
    if refund >= REFUND_FLAG:
        flags.append("refund_requested")
    if abusive >= ABUSE_FLAG:
        flags.append("abusive_language")
    if sub_intent == "outage":
        flags.append("incident")
    if low_confidence_stages:
        flags.append("low_confidence")

    # Priority: any strong signal lifts it, otherwise it follows whether the issue is blocking.
    if is_blocking >= HIGH_BLOCKING or churn >= CHURN_FLAG or frustration >= HIGH_FRUSTRATION:
        priority = "HIGH"
    elif is_blocking < 0.3 and churn < 0.2 and frustration < 1.0:
        priority = "LOW"
    else:
        priority = "NORMAL"

    if abusive >= ABUSE_FLAG:
        queue = "Trust & Safety"
    else:
        queue = SPECIAL_QUEUES.get((department, sub_intent), QUEUES.get(department, QUEUES["other"]))

    # Escalate to a human on anything high-risk or low-confidence; otherwise it can be handled
    # by the automated queue with a canned response.
    escalate = (
        priority == "HIGH"
        or churn >= CHURN_FLAG
        or refund >= REFUND_FLAG
        or abusive >= ABUSE_FLAG
        or bool(low_confidence_stages)
    )
    handler = "human" if escalate else "auto"

    reasons: List[str] = []
    reasons.append(f"department={department}, sub_intent={sub_intent}")
    if priority != "LOW":
        reasons.append(f"is_blocking={is_blocking:.2f}")
    if churn >= CHURN_FLAG:
        reasons.append(f"churn_risk={churn:.2f}")
    if refund >= REFUND_FLAG:
        reasons.append(f"refund_requested={refund:.2f}")
    if low_confidence_stages:
        reasons.append("unresolved low confidence: " + ", ".join(low_confidence_stages))
    if not reasons:
        reasons.append("no elevated signals")

    return {
        "queue": queue,
        "priority": priority,
        "handler": handler,
        "flags": flags,
        "reasons": reasons,
    }
