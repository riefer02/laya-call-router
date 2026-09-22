"""The staged support-triage cascade.

A :class:`TriageSession` is a resumable state machine. Call :meth:`advance` to run stages until
the cascade either finishes or needs to ask the customer a clarifying question; if it needs to
ask, feed the reply into :meth:`provide_answer` and advance again.

The model never generates text. Clarifying questions are templated (:mod:`schemas`) and the
customer's reply is appended to the transcript, which is re-sent as the model's state so the
next forward pass is better informed.

Every stage emits events as plain dicts so the same stream drives a terminal or an SSE endpoint.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional

import laya_mlx as laya

from . import routing as routing_policy
from . import schemas as S
from .agent import get_router

STAGES = ["gate", "language", "department", "sub_intent", "severity", "routing"]

# Default confidence below which a choice stage asks the customer a follow-up. Note that Laya's
# `confidence` is entropy-normalised (1 - H/log k), not the top probability, so it is
# option-count sensitive: a decisive 5-way choice tops out around 0.9-0.95. 0.85 is too strict
# there (a clean technical-vs-rest decision scores ~0.77), so the default is 0.75.
DEFAULT_THRESHOLD = 0.75


def _summarize(answer: Dict[str, Any]) -> Dict[str, Any]:
    """A JSON-friendly view of one Laya answer, including both top-prob and entropy confidence."""
    kind = answer.get("type")
    out: Dict[str, Any] = {
        "primitive": kind,
        "confidence": answer.get("confidence"),
        "action_probability": (answer.get("action") or {}).get("act_probability"),
    }
    if kind == "choice":
        probs = answer.get("probabilities") or {}
        out["choice"] = answer.get("choice")
        out["probabilities"] = probs
        out["top_probability"] = round(max(probs.values()), 4) if probs else None
    elif kind == "score":
        probs = answer.get("probabilities") or {}
        legend = answer.get("legend") or {}
        out["score"] = answer.get("score")
        out["legend"] = legend
        out["probabilities"] = probs
        out["top_probability"] = round(max(probs.values()), 4) if probs else None
    elif kind == "noul":
        out["noul"] = answer.get("noul")
        p = answer.get("noul")
        out["top_probability"] = round(max(p, 1.0 - p), 4) if isinstance(p, (int, float)) else None
    return out


class TriageSession:
    """One customer conversation moving through the cascade."""

    def __init__(
        self,
        message: str,
        router: Optional["laya.Router"] = None,
        confidence_threshold: float = DEFAULT_THRESHOLD,
        max_clarifications_per_stage: int = 1,
        session_id: str = "local",
    ) -> None:
        self.session_id = session_id
        self.router = router or get_router()
        self.threshold = float(confidence_threshold)
        self.max_clarify = int(max_clarifications_per_stage)

        self.turns: List[Dict[str, str]] = [{"role": "customer", "content": message}]
        self.answers: Dict[str, Dict[str, Any]] = {}
        self.stage_outputs: Dict[str, Dict[str, Any]] = {}
        self.low_confidence: List[str] = []
        self._clarify_counts: Dict[str, int] = {}

        self._stage_i = 0
        self.waiting: Optional[Dict[str, Any]] = None
        self.done = False
        self.rejected = False
        self.routing: Optional[Dict[str, Any]] = None
        self.output_tokens = 0
        self.input_tokens = 0
        self.compute_ms = 0.0
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ helpers
    def _transcript(self) -> str:
        if len(self.turns) == 1:
            return self.turns[0]["content"]
        lines = []
        for t in self.turns:
            who = "Customer" if t["role"] == "customer" else "Agent"
            lines.append(f"{who}: {t['content']}")
        return "\n".join(lines)

    @property
    def original_message(self) -> str:
        return self.turns[0]["content"]

    def _run(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        """One routed forward pass over the transcript; accumulates token usage."""
        state = self._transcript()
        result = self.router.predict(state, questions)
        usage = result.get("usage") or {}
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        return result

    def _stage_result_event(
        self,
        stage: str,
        result: Dict[str, Any],
        latency_ms: float,
        status: str = "ok",
        note: str = "",
    ) -> Dict[str, Any]:
        summarized = {qid: _summarize(a) for qid, a in result["answers"].items()}
        self.answers.update(result["answers"])
        self.compute_ms += latency_ms
        route = result.get("routing") or {}
        event = {
            "type": "stage_result",
            "stage": stage,
            "title": S.STAGE_TITLES.get(stage, stage),
            "status": status,
            "note": note,
            "answers": summarized,
            "latency_ms": round(latency_ms, 2),
            "model": route.get("model"),
            "routing_reason": route.get("reason"),
        }
        self.stage_outputs[stage] = event
        return event

    @staticmethod
    def _stage_start(stage: str, primitive: str) -> Dict[str, Any]:
        return {
            "type": "stage_start",
            "stage": stage,
            "title": S.STAGE_TITLES.get(stage, stage),
            "primitive": primitive,
        }

    # ------------------------------------------------------------------ stages
    def _stage_gate(self) -> Iterator[Dict[str, Any]]:
        yield self._stage_start("gate", "choice+noul")
        t0 = time.perf_counter()
        result = self._run(S.GATE_QUESTIONS)
        ms = (time.perf_counter() - t0) * 1000
        kind = result["answers"]["is_support_request"]["choice"]
        probs = result["answers"]["is_support_request"].get("probabilities") or {}
        p_spam = float(probs.get(S.GATE_REJECT_LABEL, 0.0))
        p_abuse = float(result["answers"]["is_abusive"]["noul"])

        notes = []
        if p_abuse >= routing_policy.ABUSE_FLAG:
            notes.append("abusive language detected")
        if p_spam >= S.GATE_REJECT_THRESHOLD:
            status = "rejected"
            notes.append(
                f"confident spam/marketing (p={p_spam:.2f}) — not routed"
            )
            self.rejected = True
        else:
            status = "ok"
            notes.append(f"accepted, p(spam)={p_spam:.2f}" + (f", argmax={kind}" if kind != S.GATE_ACCEPT_LABEL else ""))
        yield self._stage_result_event("gate", result, ms, status=status, note="; ".join(notes))

    def _stage_language(self) -> Iterator[Dict[str, Any]]:
        yield self._stage_start("language", "router")
        t0 = time.perf_counter()
        state = self._transcript()
        detection = laya.detect_language(state)
        decision = self.router.route(state, {})
        ms = (time.perf_counter() - t0) * 1000
        event = {
            "type": "stage_result",
            "stage": "language",
            "title": S.STAGE_TITLES["language"],
            "status": "ok",
            "note": decision.get("reason", ""),
            "answers": {},
            "latency_ms": round(ms, 2),
            "model": decision.get("model"),
            "routing_reason": decision.get("reason"),
            "detection": {
                "script": detection.get("script"),
                "language": detection.get("language"),
                "is_english": detection.get("is_english"),
            },
        }
        self.stage_outputs["language"] = event
        yield event

    def _choice_stage(self, stage: str, questions: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        primitive = "choice"
        yield self._stage_start(stage, primitive)
        t0 = time.perf_counter()
        result = self._run(questions)
        ms = (time.perf_counter() - t0) * 1000
        qid = next(iter(questions))
        answer = result["answers"][qid]
        confidence = float(answer.get("confidence", 0.0))
        choice = answer.get("choice")

        if confidence < self.threshold:
            used = self._clarify_counts.get(stage, 0)
            if used < self.max_clarify:
                department = self.answers.get("department", {}).get("choice")
                clarify = S.clarify_for(stage, department)
                clarify["type"] = "clarify"
                clarify["confidence"] = round(confidence, 4)
                clarify["provisional_choice"] = choice
                clarify["session_id"] = self.session_id
                # Record what we asked so the transcript is coherent on resume.
                self.turns.append({"role": "agent", "content": clarify["prompt"]})
                self.waiting = clarify
                yield self._stage_result_event(
                    stage,
                    result,
                    ms,
                    status="low_confidence",
                    note=(
                        f"confidence {confidence:.2f} < {self.threshold:.2f}; "
                        f"asking the customer (provisional: {S.display_name(choice)})"
                    ),
                )
                yield clarify
                return
            # Already clarified and still unsure: accept and flag.
            self.low_confidence.append(stage)
            yield self._stage_result_event(
                stage,
                result,
                ms,
                status="low_confidence",
                note=(
                    f"still {confidence:.2f} after clarification; accepting "
                    f"{S.display_name(choice)} and flagging for human review"
                ),
            )
            return

        yield self._stage_result_event(
            stage, result, ms, status="ok", note=f"{S.display_name(choice)} ({confidence:.2f})"
        )

    def _stage_severity(self) -> Iterator[Dict[str, Any]]:
        yield self._stage_start("severity", "score+noul")
        t0 = time.perf_counter()
        result = self._run(S.SEVERITY_QUESTIONS)
        ms = (time.perf_counter() - t0) * 1000
        yield self._stage_result_event("severity", result, ms)

    def _stage_routing(self) -> Iterator[Dict[str, Any]]:
        # No stage_start: the routing node renders from this single event.
        outcome = routing_policy.decide(self.answers, list(self.low_confidence))
        self.routing = outcome
        self.stage_outputs["routing"] = outcome
        yield {"type": "routing", "stage": "routing", "title": S.STAGE_TITLES["routing"], **outcome}

    # ------------------------------------------------------------------ driver
    def advance(self) -> Iterator[Dict[str, Any]]:
        """Run stages until the cascade finishes or blocks on a clarifying question."""
        if self.done:
            return
        while self._stage_i < len(STAGES):
            stage = STAGES[self._stage_i]
            if stage == "gate":
                gen = self._stage_gate()
            elif stage == "language":
                gen = self._stage_language()
            elif stage == "department":
                gen = self._choice_stage("department", S.DEPARTMENT_QUESTIONS)
            elif stage == "sub_intent":
                department = self.answers.get("department", {}).get("choice") or "other"
                gen = self._choice_stage("sub_intent", S.sub_intent_questions(department))
            elif stage == "severity":
                gen = self._stage_severity()
            else:
                gen = self._stage_routing()

            for event in gen:
                yield event

            if self.waiting is not None:
                return  # blocked on the customer
            if self.rejected:
                yield self._finish_event()
                return
            self._stage_i += 1

        yield self._finish_event()

    def _finish_event(self) -> Dict[str, Any]:
        self.done = True
        total_ms = (time.perf_counter() - self._t0) * 1000
        return {
            "type": "done",
            "total_ms": round(total_ms, 2),
            "compute_ms": round(self.compute_ms, 2),
            "stages_run": len(self.stage_outputs),
            "tokens_generated": 0,
            "output_tokens": self.output_tokens,
            "input_tokens": self.input_tokens,
            "rejected": self.rejected,
            "routing": self.routing,
        }

    def provide_answer(self, text: str) -> None:
        """Feed the customer's reply to a clarifying question, resuming the blocked stage."""
        if self.waiting is None:
            raise RuntimeError("session is not waiting for a clarification")
        stage = self.waiting["stage"]
        self.turns.append({"role": "customer", "content": text})
        self._clarify_counts[stage] = self._clarify_counts.get(stage, 0) + 1
        self.waiting = None


def run_to_completion(message: str, **kwargs) -> Dict[str, Any]:
    """Convenience: run a session headless, auto-answering any clarification with the
    provisional choice (useful for CLI/tests where there is no human to ask)."""
    session = TriageSession(message, **kwargs)
    events: List[Dict[str, Any]] = []
    while not session.done:
        for event in session.advance():
            events.append(event)
        if session.waiting is not None:
            provisional = session.waiting.get("provisional_choice") or ""
            session.provide_answer(str(provisional))
    return {"session": session, "events": events}
