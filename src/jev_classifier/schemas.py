"""Typed-question schemas for the staged support-triage cascade.

Every stage is one or more Laya questions over the same state. Question ids and the option
sets here are the vocabulary the deterministic router in :mod:`jev_classifier.routing` keys on,
so keep them in sync.

Primitives (Laya's decision space):
  choice  -> probabilities over named options, plus a pick
  score   -> expected level on an ordinal rubric (0-based)
  noul    -> calibrated P(true), i.e. "yes/no/unknown" as a number
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------------- stage 1: gate
# Measured (scripts/probe_questions.py): a `choice` gate separates spam from genuine requests
# reliably (spam 0.94 on the spam classification) where every `noul` phrasing tried did not —
# the base checkpoint answered a "is this genuine support?" noul at 0.13 on an obvious billing
# request. So the gate is a choice question.
GATE_ACCEPT_LABEL = "genuine_support"
# The gate is a hard stop, so it must only fire on a *confident* spam signal, not on a 3-way
# argmax. Measured: the multilingual checkpoint scores a genuine Chinese ticket
# spam 0.37 / other 0.33 / genuine 0.30 (entropy confidence 0.00) — an argmax rule rejects it,
# while real spam scores 0.94-0.96. So rejection requires p(spam) above this threshold.
GATE_REJECT_LABEL = "spam_or_marketing"
GATE_REJECT_THRESHOLD = 0.6

GATE_QUESTIONS: Dict = {
    "is_support_request": {
        "type": "choice",
        "instructions": "What kind of message is this?",
        "criteria": {
            "genuine_support": "a real customer asking for help with their account or product",
            "spam_or_marketing": "spam, advertising or a scam",
            "other": "none of the above",
        },
    },
    "is_abusive": {
        "type": "noul",
        "instructions": (
            "Does the customer conversation contain abuse, threats or harassment directed "
            "at staff?"
        ),
    },
}

# ----------------------------------------------------------------------- stage 3: department
DEPARTMENTS: Dict[str, str] = {
    "billing": "invoices, payments, refunds, duplicate charges, plan changes",
    "technical": "bugs, outages, errors, integrations, performance problems",
    "sales": "pricing, new contracts, upgrades, demos",
    "account": "login, access, profile details, cancellation",
    "other": "none of the other options fits",
}

DEPARTMENT_QUESTIONS: Dict = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this customer request?",
        "criteria": DEPARTMENTS,
    },
}

# ----------------------------------------------------------------------- stage 4: sub-intent
SUB_INTENTS: Dict[str, Dict[str, str]] = {
    "billing": {
        "refund": "asks for money back or a refund",
        "duplicate_charge": "was charged twice or sees a duplicate charge",
        "invoice_question": "a question about an invoice, plan or payment method",
        "payment_failed": "a payment failed or a card was declined",
        "other": "some other billing or payment matter",
    },
    "technical": {
        "outage": "the service is down or unavailable",
        "bug_report": "something is broken or behaving incorrectly",
        "setup_help": "needs help setting something up or integrating",
        "performance": "slow or degraded performance",
        "other": "some other technical matter",
    },
    "sales": {
        "pricing": "asks about price or which plan fits",
        "new_contract": "wants to buy, sign or upgrade",
        "demo_request": "wants a demo or a walkthrough",
        "other": "some other sales matter",
    },
    "account": {
        "login_access": "cannot log in or has lost access",
        "cancellation": "wants to cancel or downgrade",
        "profile_change": "wants to change account details or seats",
        "other": "some other account matter",
    },
    "other": {
        "general": "a general question or comment",
        "other": "none of the above fits",
    },
}


def sub_intent_questions(department: str) -> Dict:
    """The stage-4 question for a given department branch."""
    criteria = SUB_INTENTS.get(department, SUB_INTENTS["other"])
    return {
        "sub_intent": {
            "type": "choice",
            "instructions": (
                f"Within {department} support, what does the customer want? "
                "Pick the single closest option."
            ),
            "criteria": criteria,
        }
    }


# ------------------------------------------------------------------------- stage 5: severity
# Phrasing chosen from measurement (scripts/probe_questions.py). Notably the churn question has
# to say "threaten to cancel their plan or stop being a customer": a softer "may leave for a
# competitor" variant scored 0.11 on a message that literally said "we'll have to cancel", while
# an "or stop paying" variant false-positived on an outage report.
SEVERITY_QUESTIONS: Dict = {
    "is_blocking": {
        "type": "noul",
        "instructions": "Does the customer need this resolved today or is it blocking their work?",
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer sound?",
        "criteria": [
            "calm and neutral",
            "concerned but civil",
            "clearly annoyed",
            "very angry or using strong language",
        ],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": (
            "Does the customer threaten to cancel their plan or stop being a customer?"
        ),
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly ask for money back?",
    },
}

# ------------------------------------------------------------------------- stage metadata
STAGE_TITLES = {
    "gate": "Gate",
    "language": "Language",
    "department": "Department",
    "sub_intent": "Sub-intent",
    "severity": "Severity",
    "routing": "Routing",
}

# Templated follow-ups (Laya never generates text, so clarifications are fixed prompts whose
# options come straight from the schema above).
CLARIFY_PROMPTS = {
    "department": "I want to route you correctly — which area is this about?",
    "sub_intent": "Which of these is closest to what you need?",
}


def clarify_for(stage: str, department: str | None = None) -> Dict:
    """The templated clarifying question for a low-confidence stage."""
    if stage == "department":
        options = list(DEPARTMENTS)
    elif stage == "sub_intent" and department:
        options = list(SUB_INTENTS.get(department, SUB_INTENTS["other"]))
    else:
        options = []
    return {
        "stage": stage,
        "prompt": CLARIFY_PROMPTS.get(stage, "Could you tell me a bit more?"),
        "chips": [o.replace("_", " ") for o in options],
    }


def display_name(key: str) -> str:
    return key.replace("_", " ")


def stage_question_text(stage: str, department: str | None = None) -> str:
    if stage == "department":
        return DEPARTMENT_QUESTIONS["department"]["instructions"]
    if stage == "sub_intent":
        return sub_intent_questions(department or "other")["sub_intent"]["instructions"]
    return ""


def option_map(stage: str, department: str | None = None) -> Dict[str, str]:
    if stage == "department":
        return dict(DEPARTMENTS)
    if stage == "sub_intent":
        return dict(SUB_INTENTS.get(department or "other", SUB_INTENTS["other"]))
    return {}


def options_for(stage: str, department: str | None = None) -> List[str]:
    return list(option_map(stage, department))
