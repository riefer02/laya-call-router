"""The call driver: alternates caller turns with a classifier-driven switchboard.

Each turn re-reads the whole conversation and runs two batched forward passes:

  pass 1  department + slots            (no branch knowledge needed)
  pass 2  intent (branched) + next step (depends on pass 1)

Emitting the same columns every turn is deliberate: the graph is a stable grid, and the visible
change between turns *is* the accumulation of state (confidence rising, slots filling).

The caller is scripted (`scenarios.py`); the switchboard's decisions are all Laya typed questions.
Response text is templated — Laya never generates.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional

import laya_mlx as laya

from . import dealership as D
from .agent import get_router

# Fixed column per node key so the frontend can lay out deterministically.
COLUMNS: Dict[str, int] = {
    "caller": 0,
    "department": 1,
    "vehicle": 2,
    "location": 3,
    "time_preference": 4,
    "is_safe_to_drive": 5,
    "needs_human": 6,
    "extract_time": 7,
    "missing_slots": 8,
    "intent": 9,
    "next_action": 10,
    "agent": 11,
    "terminal": 12,
}

TITLES: Dict[str, str] = {
    "caller": "Caller",
    "department": "Department",
    "vehicle": "Vehicle",
    "location": "Location",
    "time_preference": "When",
    "is_safe_to_drive": "Unsafe?",
    "needs_human": "Needs human?",
    "extract_time": "Time (regex)",
    "missing_slots": "What's missing",
    "intent": "Intent",
    "next_action": "Next step",
    "agent": "Switchboard",
    "terminal": "Route",
}

DEFAULT_THRESHOLD = 0.75


def _summarize(answer: Dict[str, Any]) -> Dict[str, Any]:
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
    elif kind == "noul":
        p = answer.get("noul")
        out["noul"] = p
        out["top_probability"] = round(max(p, 1 - p), 4) if isinstance(p, (int, float)) else None
    return out


class CallSession:
    def __init__(
        self,
        scenario: Dict[str, Any],
        router: Optional["laya.Router"] = None,
        session_id: str = "call",
        confidence_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.scenario = scenario
        self.session_id = session_id
        self.router = router or get_router()
        self.threshold = confidence_threshold

        self.exchanges: List[Dict[str, str]] = []  # {"role": caller|agent, "text": ...}
        self.answers: Dict[str, Dict[str, Any]] = {}
        self.asked: set[str] = set()
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.decisions = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.compute_ms = 0.0
        self.finished = False
        self.routing: Optional[Dict[str, Any]] = None
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ transcript
    def _transcript(self) -> str:
        return "\n".join(
            f"{'Caller' if e['role'] == 'caller' else 'Agent'}: {e['text']}"
            for e in self.exchanges
        )

    def _run(self, questions: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = self.router.predict(self._transcript(), questions)
        ms = (time.perf_counter() - t0) * 1000
        usage = result.get("usage") or {}
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.compute_ms += ms
        result["_ms"] = ms
        return result

    # ------------------------------------------------------------------ graph plumbing
    def _edge(self, source: str, target: str, label: str = "", kind: str = "flow") -> Dict[str, Any]:
        eid = f"{source}->{target}"
        if not any(e["id"] == eid for e in self.edges):
            self.edges.append({"id": eid, "source": source, "target": target, "label": label, "kind": kind})
        return {"type": "edge", "id": eid, "source": source, "target": target, "label": label, "kind": kind}

    def _node(
        self,
        node_id: str,
        turn: int,
        key: str,
        kind: str,
        stage: str = "",
        primitive: str = "",
    ) -> Dict[str, Any]:
        node = {
            "type": "node",
            "id": node_id,
            "turn": turn,
            "col": COLUMNS[key],
            "key": key,
            "title": TITLES[key],
            "kind": kind,
            "stage": stage,
            "primitive": primitive,
            "status": "running",
        }
        self.nodes.append(node)
        if kind == "decision":
            self.decisions += 1
        return node

    def _result(
        self,
        node_id: str,
        turn: int,
        *,
        status: str = "ok",
        summary: Optional[Dict[str, Any]] = None,
        question: str = "",
        options: Optional[List[str]] = None,
        model: Optional[str] = None,
        routing_reason: str = "",
        latency_ms: float = 0.0,
        batch_size: int = 1,
        note: str = "",
        value: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "type": "node_result",
            "id": node_id,
            "turn": turn,
            "status": status,
            "summary": summary or {},
            "question": question,
            "options": options or [],
            "model": model,
            "routing_reason": routing_reason,
            "latency_ms": round(latency_ms, 2),
            "batch_size": batch_size,
            "note": note,
            "value": value,
            **(extra or {}),
        }

    # ------------------------------------------------------------------ turn
    def _run_turn(self, turn: int, utterance: str) -> Iterator[Dict[str, Any]]:
        self.exchanges.append({"role": "caller", "text": utterance})
        yield {"type": "turn_start", "turn": turn, "speaker": "caller"}

        caller_id = f"t{turn}.caller"
        yield self._node(caller_id, turn, "caller", "utterance")
        yield self._result(caller_id, turn, value=utterance, note="caller speaks")

        prev_id = caller_id

        # ---- pass 1: department + slots -----------------------------------------
        result = self._run(D.PASS1_QUESTIONS)
        ms = result.pop("_ms")
        routing = result.get("routing") or {}
        questions = D.PASS1_QUESTIONS
        n = len(questions)
        for key, answer in result["answers"].items():
            node_id = f"t{turn}.{key}"
            yield self._node(node_id, turn, key, "decision", primitive=answer["type"])
            status = "ok"
            if answer["type"] == "choice":
                summary = _summarize(answer)
                if summary["confidence"] is not None and summary["confidence"] < self.threshold:
                    status = "low_confidence"
            else:
                summary = _summarize(answer)
                if key == "is_safe_to_drive" and float(answer["noul"]) >= 0.5:
                    status = "warn"
            self.answers[key] = answer
            yield self._result(
                node_id,
                turn,
                status=status,
                summary=summary,
                question=questions[key]["instructions"],
                options=list(questions[key].get("criteria") or []),
                model=routing.get("model"),
                routing_reason=routing.get("reason", ""),
                latency_ms=ms,
                batch_size=n,
            )
            yield self._edge(prev_id, node_id)
            prev_id = node_id

        # ---- deterministic extraction -------------------------------------------
        extracted = D.extract_time(utterance)
        extract_id = f"t{turn}.extract_time"
        if extracted:
            yield self._node(extract_id, turn, "extract_time", "extract")
            yield self._result(
                extract_id, turn, value=extracted, note="regex, not a model decision"
            )
            yield self._edge(prev_id, extract_id, label=extracted, kind="extract")

        # ---- policy: what is still missing --------------------------------------
        missing = [s for s in D.REQUIRED_SLOTS if self._slot_missing(s) and s not in self.asked]
        policy_id = f"t{turn}.missing_slots"
        yield self._node(policy_id, turn, "missing_slots", "policy")
        yield self._result(
            policy_id,
            turn,
            value=missing,
            note="deterministic rule over the slot answers",
            extra={"missing": missing, "asked": sorted(self.asked)},
        )
        yield self._edge(prev_id, policy_id, kind="policy")

        department = self.answers.get("department", {}).get("choice") or "general"
        unsafe = float(self.answers.get("is_safe_to_drive", {}).get("noul", 0.0))

        # ---- pass 2: branched intent --------------------------------------------
        result2 = self._run(D.intent_question(department))
        ms2 = result2.pop("_ms")
        routing2 = result2.get("routing") or {}
        intent_ans = result2["answers"]["intent"]
        intent_id = f"t{turn}.intent"
        yield self._node(intent_id, turn, "intent", "decision", primitive="choice")
        yield self._result(
            intent_id,
            turn,
            summary=_summarize(intent_ans),
            question=D.intent_question(department)["intent"]["instructions"],
            options=list(D.INTENTS.get(department, D.INTENTS["general"])),
            model=routing2.get("model"),
            routing_reason=routing2.get("reason", ""),
            latency_ms=ms2,
            batch_size=len(result2["answers"]),
        )
        self.answers["intent"] = intent_ans
        yield self._edge(prev_id, intent_id, label=f"department = {department}", kind="branch")
        prev_id = intent_id

        # ---- next action: deterministic policy ----------------------------------
        action = D.next_action_for(missing, department, unsafe)
        action_id = f"t{turn}.next_action"
        yield self._node(action_id, turn, "next_action", "policy")
        yield self._result(
            action_id,
            turn,
            value=action,
            note="policy over the classifier's slots, not a model decision",
            extra={
                "missing": missing,
                "reason": D.NEXT_ACTION_LABELS.get(action, ""),
                "inputs": {"missing": missing, "unsafe": unsafe, "department": department},
            },
        )
        yield self._edge(prev_id, action_id, kind="policy")

        # remember that we asked, so a "no preference" answer is not asked forever
        if action == "ask_vehicle":
            self.asked.add("vehicle")
        elif action == "ask_location":
            self.asked.add("location")
        elif action == "ask_time":
            self.asked.add("time_preference")

        # ---- switchboard responds ------------------------------------------------
        text = self._response_text(action)
        self.exchanges.append({"role": "agent", "text": text})
        agent_id = f"t{turn}.agent"
        yield self._node(agent_id, turn, "agent", "utterance")
        yield self._result(
            agent_id,
            turn,
            value=text,
            note=f"templated response for next_action={action}",
            extra={"template_id": action, "driven_by": [action_id]},
        )
        yield self._edge(prev_id, agent_id, label=action)

        # ---- terminal ------------------------------------------------------------
        if action in ("confirm_booking", "offer_transfer"):
            yield from self._finish(turn, agent_id, missing)

    def _slot_missing(self, slot: str) -> bool:
        ans = self.answers.get(slot)
        if not ans:
            return True
        return ans.get("choice") in (None, "not_stated")

    def _response_text(self, action: str) -> str:
        template = D.RESPONSES.get(action, D.RESPONSES["ask_detail"])
        department = D.display_name(self.answers.get("department", {}).get("choice") or "general")
        location = D.display_name(self.answers.get("location", {}).get("choice") or "not_stated")
        time_pref = D.display_name(
            self.answers.get("time_preference", {}).get("choice") or "not_stated"
        )
        return template.format(department=department, location=location, time=time_pref)

    def _finish(self, turn: int, prev_id: str, missing: List[str]) -> Iterator[Dict[str, Any]]:
        outcome = D.decide(self.answers, [s for s in D.REQUIRED_SLOTS if self._slot_missing(s)])
        self.routing = outcome
        node_id = f"t{turn}.terminal"
        yield self._node(node_id, turn, "terminal", "terminal")
        yield self._result(node_id, turn, value=outcome["queue"], extra={"routing": outcome})
        yield self._edge(prev_id, node_id, label=outcome["queue"], kind="terminal")
        self.finished = True

    # ------------------------------------------------------------------ driver
    def advance(self) -> Iterator[Dict[str, Any]]:
        yield {
            "type": "call_start",
            "call_id": self.session_id,
            "scenario": {"id": self.scenario["id"], "label": self.scenario["label"]},
            "columns": TITLES,
        }
        for i, utterance in enumerate(self.scenario["turns"], start=1):
            if self.finished:
                break
            yield from self._run_turn(i, utterance)
        if not self.finished:
            # ran out of scripted turns: route on what we have
            yield from self._finish(len(self.scenario["turns"]), f"t{len(self.scenario['turns'])}.agent", [])
        yield {
            "type": "call_end",
            "call_id": self.session_id,
            "turns": len([e for e in self.exchanges if e["role"] == "caller"]),
            "decisions": self.decisions,
            "compute_ms": round(self.compute_ms, 2),
            "total_ms": round((time.perf_counter() - self._t0) * 1000, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_generated": 0,
            "cost_usd": 0.0,
            "routing": self.routing,
        }
