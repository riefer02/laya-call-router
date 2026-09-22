"""Scripted example support calls, used by the web demo's persona picker and by the CLI.

Each entry is a first-person message that a human customer might send, chosen to exercise a
different branch of the cascade.
"""

from __future__ import annotations

from typing import Dict, List

PERSONAS: List[Dict[str, str]] = [
    {
        "id": "billing_dispute",
        "label": "Billing dispute",
        "message": (
            "Hi, we were billed twice for March on invoice #4411. Please refund the "
            "duplicate today or we'll have to cancel our plan."
        ),
        "expect": "billing / duplicate_charge, high churn + refund flags",
    },
    {
        "id": "outage",
        "label": "Service outage",
        "message": (
            "Your API has been returning 503s for the last 40 minutes and our checkout is "
            "completely down. This is blocking all our customers right now. Help!"
        ),
        "expect": "technical / outage -> Incident Response, HIGH priority",
    },
    {
        "id": "login",
        "label": "Locked out",
        "message": (
            "I can't log in to my account any more. My password reset email never arrives. "
            "Can you help me get back in?"
        ),
        "expect": "account / login_access, normal priority",
    },
    {
        "id": "sales",
        "label": "Pricing enquiry",
        "message": (
            "We're a 40-person team and evaluating your product. Could you send pricing for "
            "the business tier and maybe arrange a quick demo next week?"
        ),
        "expect": "sales / pricing or demo_request",
    },
    {
        "id": "multilingual",
        "label": "Non-English (Chinese)",
        "message": "发票被重复扣款，请今天退款，否则我们将取消订阅。",
        "expect": "Router switches to multilingual checkpoint",
    },
    {
        "id": "not_support",
        "label": "Not a support request",
        "message": (
            "CONGRATULATIONS!! You have WON a $1000 gift card. Click here to claim your "
            "prize now >> bit.ly/not-a-real-link"
        ),
        "expect": "gate rejects: not a genuine support request",
    },
]

PERSONA_BY_ID: Dict[str, Dict[str, str]] = {p["id"]: p for p in PERSONAS}
