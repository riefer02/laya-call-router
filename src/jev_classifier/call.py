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
    "ack": 1,
    "changed": 2,
    "department": 3,
    "vehicle": 4,
    "location": 5,
    "time_preference": 6,
    "is_safe_to_drive": 7,
    "needs_human": 8,
    "extract_time": 9,
    "missing_slots": 10,
    "intent": 11,
    "next_action": 12,
    "agent": 13,
    "terminal": 14,
}

TITLES: Dict[str, str] = {
    "caller": "Caller",
    "ack": "Switchboard (ack)",
    "changed": "Anything new?",
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

# Spoken the moment the caller stops talking, before any classification runs — the "speak sooner"
# half of latency. It is a fixed phrase, not generated: Laya cannot write, and we do not want it to.
ACK_PHRASES = [
    "Let me take a look at that for you.",
    "One moment while I pull that up.",
    "Got it — let me check that.",
]

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
        pin_threshold: float = 0.6,
        verify_threshold: float = 0.75,
    ) -> None:
        self.scenario = scenario
        self.session_id = session_id
        self.router = router or get_router()
        # Three different questions, three different numbers:
        #   confidence_threshold — "should a human look at this?" (entropy confidence, 0.75)
        #   pin_threshold        — "can we stop re-deciding this?" (top probability, 0.6)
        #   verify_threshold     — "is a second opinion worth ~10 ms?" (top probability, 0.75)
        # Pinning on entropy confidence alone never settles a wide choice: intent answers at
        # p=0.72 over 7 options score only 0.42, so it was re-run every single turn.
        self.threshold = confidence_threshold
        self.pin_threshold = pin_threshold
        self.verify_threshold = verify_threshold

        self.exchanges: List[Dict[str, str]] = []  # {"role": caller|agent, "text": ...}
        self.answers: Dict[str, Dict[str, Any]] = {}
        # Settled facts: what we already know, with the confidence and the turn that settled it.
        # A fact is only pinned when it is a concrete value answered confidently; "not_stated"
        # answers are never pinned, because resolving them is the whole point of a later turn.
        self.facts: Dict[str, Dict[str, Any]] = {}
        self.asked: set[str] = set()
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.decisions = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.compute_ms = 0.0
        self.turn_stats: List[Dict[str, Any]] = []
        self.escalations = 0
        self.llm_escalations = 0
        self._turn_q = 0
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
        self._turn_q += len(questions)
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
        self._turn_q = 0
        turn_t0 = time.perf_counter()
        turn_ms0 = self.compute_ms
        turn_tok0 = self.input_tokens

        self.exchanges.append({"role": "caller", "text": utterance})
        yield {"type": "turn_start", "turn": turn, "speaker": "caller"}

        caller_id = f"t{turn}.caller"
        yield self._node(caller_id, turn, "caller", "utterance")
        yield self._result(caller_id, turn, value=utterance, note="caller speaks")

        # Acknowledge before classifying. The voice channel needs *something* within a few hundred
        # milliseconds; this fixed phrase is emitted before any forward pass so the caller is never
        # met with silence while the cascade runs.
        ack_text = ACK_PHRASES[(turn - 1) % len(ACK_PHRASES)]
        ack_id = f"t{turn}.ack"
        yield self._node(ack_id, turn, "ack", "utterance")
        yield self._result(
            ack_id,
            turn,
            value=ack_text,
            note="acknowledgement — spoken before any classification",
            extra={"template_id": "ack"},
        )
        yield self._edge(caller_id, ack_id)

        prev_id = ack_id

        # ---- choose what actually needs evaluating this turn --------------------
        # Turn 1 evaluates everything. Later turns evaluate the still-unresolved questions plus a
        # single change-detector; facts that are settled and confident are skipped entirely.
        pass1 = dict(D.PASS1_QUESTIONS)
        to_run: Dict[str, Any] = {}
        changed_fired = False

        if turn == 1 or not self.facts:
            to_run = dict(pass1)
        else:
            to_run = dict(D.CHANGE_QUESTION)
            to_run.update({k: q for k, q in pass1.items() if not self._is_pinned(k)})

        result: Dict[str, Any] = {}
        ms = 0.0
        routing: Dict[str, Any] = {}
        if to_run:
            result = self._run(to_run)
            ms = result.pop("_ms")
            routing = result.get("routing") or {}
            if "changed" in result["answers"]:
                changed_fired = float(result["answers"]["changed"]["noul"]) >= D.CHANGE_FLAG

        # If something genuinely changed, re-evaluate the facts we would otherwise have skipped.
        if changed_fired:
            revisit = {k: q for k, q in pass1.items() if self._is_pinned(k)}
            if revisit:
                extra = self._run(revisit)
                extra.pop("_ms", None)
                result.setdefault("answers", {}).update(extra["answers"])

        answers = result.get("answers") or {}
        n = len(to_run)

        # Emit in column order: change-detector first, then each pass-1 question, whether it was
        # evaluated or skipped.
        emit_keys = (["changed"] if turn > 1 and self.facts else []) + list(pass1)
        for key in emit_keys:
            node_id = f"t{turn}.{key}"
            if key in answers:
                answer = answers[key]
                yield self._node(node_id, turn, key, "decision", primitive=answer["type"])
                status = "ok"
                summary = _summarize(answer)
                if answer["type"] == "choice":
                    if summary["confidence"] is not None and summary["confidence"] < self.threshold:
                        status = "low_confidence"
                elif key == "is_safe_to_drive" and float(answer["noul"]) >= 0.5:
                    status = "warn"
                self.answers[key] = answer
                self._pin(key, answer, turn, summary)
                yield self._result(
                    node_id,
                    turn,
                    status=status,
                    summary=summary,
                    question=(D.CHANGE_QUESTION if key == "changed" else pass1)[key]["instructions"],
                    options=list(((D.CHANGE_QUESTION if key == "changed" else pass1)[key]).get("criteria") or []),
                    model=routing.get("model"),
                    routing_reason=routing.get("reason", ""),
                    latency_ms=ms,
                    batch_size=n,
                )
                if key == "department" and answer["type"] == "choice":
                    yield from self._verify(
                        turn, key, D.department_question_paraphrase(), answer, summary
                    )
            elif key in self.facts:
                fact = self.facts[key]
                yield self._node(node_id, turn, key, "decision", primitive=fact.get("type", "choice"))
                yield self._result(
                    node_id,
                    turn,
                    status="skipped",
                    summary=fact.get("summary"),
                    question=pass1[key]["instructions"],
                    options=list(pass1[key].get("criteria") or []),
                    note=f"already known — not re-computed (settled turn {fact.get('turn')})",
                    extra={"pinned": True, "from_turn": fact.get("turn")},
                )
            else:
                continue
            yield self._edge(prev_id, node_id, kind="skip" if key in self.facts and key not in answers else "flow")
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

        # ---- branched intent: re-run only if it is not settled -------------------
        intent_id = f"t{turn}.intent"
        if self._is_pinned("intent") and not changed_fired:
            fact = self.facts["intent"]
            yield self._node(intent_id, turn, "intent", "decision", primitive="choice")
            yield self._result(
                intent_id,
                turn,
                status="skipped",
                summary=fact.get("summary"),
                question=D.intent_question(department)["intent"]["instructions"],
                options=list(D.INTENTS.get(department, D.INTENTS["general"])),
                note=f"already known — not re-computed (settled turn {fact.get('turn')})",
                extra={"pinned": True, "from_turn": fact.get("turn")},
            )
            yield self._edge(prev_id, intent_id, kind="skip")
        else:
            result2 = self._run(D.intent_question(department))
            ms2 = result2.pop("_ms")
            routing2 = result2.get("routing") or {}
            intent_ans = result2["answers"]["intent"]
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
            self._pin("intent", intent_ans, turn, _summarize(intent_ans))
            yield self._edge(prev_id, intent_id, label=f"department = {department}", kind="branch")
            yield from self._verify(
                turn,
                "intent",
                D.intent_question_paraphrase(department),
                intent_ans,
                _summarize(intent_ans),
            )
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

        # per-turn cost accounting: this is the number the optimisation has to move
        self.turn_stats.append(
            {
                "turn": turn,
                "questions": self._turn_q,
                "input_tokens": self.input_tokens - turn_tok0,
                "compute_ms": round(self.compute_ms - turn_ms0, 2),
                "wall_ms": round((time.perf_counter() - turn_t0) * 1000, 2),
            }
        )

        # ---- terminal ------------------------------------------------------------
        if action in ("confirm_booking", "offer_transfer"):
            yield from self._finish(turn, agent_id, missing)

    # ------------------------------------------------------------------ settled facts
    def _pin(self, key: str, answer: Dict[str, Any], turn: int, summary: Dict[str, Any]) -> None:
        """Record a fact as settled if it is concrete and decisive enough to rely on.

        Decisiveness is judged on the *top probability*, not the entropy confidence, so a wide
        choice with a clear winner still settles (see the constructor note).
        """
        conf = float(answer.get("confidence") or 0.0)
        top = summary.get("top_probability")
        kind = answer.get("type")
        concrete = True
        if kind == "choice":
            concrete = answer.get("choice") not in D.UNRESOLVED_SENTINELS
        pinned = concrete and top is not None and float(top) >= self.pin_threshold
        self.facts[key] = {
            "type": kind,
            "value": answer.get("choice") if kind == "choice" else answer.get("noul"),
            "confidence": conf,
            "top_probability": top,
            "turn": turn,
            "summary": summary,
            "pinned": pinned,
        }

    def _is_pinned(self, key: str) -> bool:
        fact = self.facts.get(key)
        return bool(fact and fact.get("pinned"))

    # ------------------------------------------------------------------ verification
    def _needs_verification(self, summary: Dict[str, Any]) -> bool:
        """Worth a second opinion when a classification question is not decisive."""
        if summary.get("primitive") != "choice":
            return False
        probs = summary.get("probabilities") or {}
        if len(probs) < D.ESCALATION_MIN_OPTIONS:
            return False
        top_label, top_p = max(probs.items(), key=lambda kv: kv[1])
        if top_label in D.UNRESOLVED_SENTINELS:
            return False
        return float(top_p) < self.verify_threshold

    def _verify(
        self,
        turn: int,
        key: str,
        paraphrase: Dict[str, Any],
        answer: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Iterator[Dict[str, Any]]:
        """Tier 2: ask again in different words and check the two phrasings agree.

        Verification only — the primary answer is never overturned by the second phrasing. Two
        independently-worded questions agreeing is evidence; a second roll of the same question
        is not, and was measured to *inflate* confidence on wrong answers (see
        `dealership.department_question_paraphrase`).
        """
        if not paraphrase or not self._needs_verification(summary):
            return
        try:
            res = self._run(paraphrase)
        except ValueError:
            return
        ms = res.pop("_ms")
        alt = res["answers"][key]
        alt_summary = _summarize(alt)
        self.escalations += 1
        agrees = alt.get("choice") == answer.get("choice")
        if not agrees:
            self.llm_escalations += 1
        else:
            # Two phrasings agreeing is enough to stop re-deciding it, even below the pin bar.
            fact = self.facts.get(key)
            if fact is not None:
                fact["pinned"] = True
                fact["verified"] = True
        route = res.get("routing") or {}
        yield self._result(
            f"t{turn}.{key}",
            turn,
            status="verified" if agrees else "uncertain",
            summary=summary,
            question=paraphrase[key]["instructions"],
            options=list(paraphrase[key].get("criteria") or []),
            model=route.get("model"),
            routing_reason=route.get("reason", ""),
            latency_ms=ms,
            batch_size=1,
            note=(
                f"second phrasing agrees ({alt.get('choice')})"
                if agrees
                else (
                    f"second phrasing disagrees ({alt.get('choice')} vs {answer.get('choice')}) "
                    "— kept the first answer, would escalate to an LLM"
                )
            ),
            extra={
                "verification": {
                    "tier": 2,
                    "agrees": agrees,
                    "primary": {
                        "choice": answer.get("choice"),
                        "top_probability": summary.get("top_probability"),
                    },
                    "paraphrase": {
                        "choice": alt.get("choice"),
                        "top_probability": alt_summary.get("top_probability"),
                    },
                    "llm_escalation": not agrees,
                }
            },
        )

    def _slot_missing(self, slot: str) -> bool:
        """A slot is missing unless we can rely on it.

        Not merely "the answer isn't the not_stated sentinel": a low-confidence guess is not a
        known value either. The vague-complaint scenario had the model infer `sedan` at p=0.52 for
        a caller who had said nothing about a vehicle, and the old rule then never asked.
        """
        fact = self.facts.get(slot)
        if not fact:
            return True
        return not fact.get("pinned")

    def _response_text(self, action: str) -> str:
        template = D.RESPONSES.get(action, D.RESPONSES["ask_detail"])
        department = D.display_name(self.answers.get("department", {}).get("choice") or "general")
        location = D.display_name(self.answers.get("location", {}).get("choice") or "not_stated")
        time_pref = D.display_name(
            self.answers.get("time_preference", {}).get("choice") or "not_stated"
        )
        return template.format(department=department, location=location, time=time_pref)

    def _finish(self, turn: int, prev_id: str, missing: List[str]) -> Iterator[Dict[str, Any]]:
        # "Missing" for routing means the caller never provided it. A shaky but present answer is
        # already visible as a low-confidence node, and should not read as an unasked question.
        unaddressed = [s for s in D.REQUIRED_SLOTS if self._slot_missing(s) and s not in self.asked]
        outcome = D.decide(self.answers, unaddressed)
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
            "turn_stats": self.turn_stats,
            "escalations": self.escalations,
            "llm_escalations": self.llm_escalations,
        }
