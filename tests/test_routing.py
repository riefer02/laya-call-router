"""Stage-6 routing policy tests. Pure logic — no model, no network."""

from __future__ import annotations

from jev_classifier import routing


def _answers(**kwargs):
    """Build a minimal answers dict; values are the primitive's natural payload."""
    out = {}
    for key, value in kwargs.items():
        if key in ("department", "sub_intent"):
            out[key] = {"type": "choice", "choice": value, "probabilities": {value: 1.0}}
        elif key == "frustration":
            out[key] = {"type": "score", "score": value}
        else:
            out[key] = {"type": "noul", "noul": value}
    return out


def test_billing_duplicate_charge_escalates_to_refund_desk():
    r = routing.decide(
        _answers(
            department="billing",
            sub_intent="duplicate_charge",
            is_blocking=0.8,
            churn_risk=0.7,
            refund_requested=0.8,
            frustration=1.0,
        ),
        [],
    )
    assert r["queue"] == "Refund Desk"
    assert r["priority"] == "HIGH"
    assert r["handler"] == "human"
    assert "churn_risk" in r["flags"]
    assert "refund_requested" in r["flags"]


def test_outage_goes_to_incident_response():
    r = routing.decide(
        _answers(department="technical", sub_intent="outage", is_blocking=0.9, churn_risk=0.05, refund_requested=0.1),
        [],
    )
    assert r["queue"] == "Incident Response"
    assert "incident" in r["flags"]
    assert r["priority"] == "HIGH"


def test_routine_login_is_auto_handled():
    r = routing.decide(
        _answers(department="account", sub_intent="login_access", is_blocking=0.1, churn_risk=0.05, refund_requested=0.02, frustration=0.2),
        [],
    )
    assert r["queue"] == "Account & Access"
    assert r["handler"] == "auto"
    assert r["priority"] == "LOW"


def test_cancellation_routes_to_retention():
    r = routing.decide(
        _answers(department="account", sub_intent="cancellation", is_blocking=0.2, churn_risk=0.9, refund_requested=0.1),
        [],
    )
    assert r["queue"] == "Retention Specialist"
    assert r["handler"] == "human"


def test_abuse_overrides_department():
    r = routing.decide(
        _answers(department="sales", sub_intent="pricing", is_blocking=0.0, churn_risk=0.0, refund_requested=0.0, is_abusive=0.9),
        [],
    )
    assert r["queue"] == "Trust & Safety"
    assert "abusive_language" in r["flags"]
    assert r["handler"] == "human"


def test_low_confidence_forces_human_review():
    r = routing.decide(
        _answers(department="other", sub_intent="general", is_blocking=0.1, churn_risk=0.0, refund_requested=0.0),
        ["department"],
    )
    assert r["handler"] == "human"
    assert "low_confidence" in r["flags"]
    assert any("department" in reason for reason in r["reasons"])


def test_missing_answers_still_route_somewhere():
    r = routing.decide({}, [])
    assert r["queue"] == routing.QUEUES["other"]
    assert r["handler"] in ("auto", "human")
