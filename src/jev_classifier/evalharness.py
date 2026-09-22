"""Four-arm evaluation: our classifier cascade against a small generative model.

  laya        our cascade (incremental + verification)
  laya-full   ablation: incremental and verification disabled
  llm         a cheap structured-output model, same task, same vocabulary
  hybrid      laya, escalating to the llm only when verification disagrees

Methodology, stated so it can be argued with:

* **Ground truth is hand-authored once** (`data/calls/*.jsonl`) and not tuned after seeing
  results.
* **The LLM gets the same information we give Laya** — the same department descriptions and the
  same intent vocabulary — and returns strict JSON. It is not strawmanned.
* **Intent is scored against the ground-truth department's branch**, so a right-intent /
  wrong-department case is not double-counted.
* **At call level the LLM sees the whole transcript at once** (one shot), which is *more*
  context than our turn-by-turn cascade gets. That is generous to the LLM by design.
* **Routing is compared as the queue**, and for the LLM the queue is produced by the *same*
  deterministic policy we use (`dealership.decide`) applied to its answer — so we are comparing
  the understanding, not the routing table.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from concurrent.futures import ThreadPoolExecutor

from . import dealership as D
from . import llm
from .agent import get_router
from .call import CallSession

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "calls"

DEPARTMENT_LIST = list(D.DEPARTMENTS)
ALL_INTENTS = sorted({name for branch in D.INTENTS.values() for name in branch})


# --------------------------------------------------------------------------- ground truth
@dataclass
class RoutingCase:
    id: str
    text: str
    department: str
    intent: str


@dataclass
class CallCase:
    id: str
    label: str
    turns: List[str]
    department: str
    intent: str
    queue: str


def load_routing(path: Optional[Path] = None) -> List[RoutingCase]:
    rows = _read_jsonl(path or DATA / "routing.jsonl")
    _validate(rows, {"id", "text", "department", "intent"}, "routing")
    return [RoutingCase(r["id"], r["text"], r["department"], r["intent"]) for r in rows]


def load_calls(path: Optional[Path] = None) -> List[CallCase]:
    rows = _read_jsonl(path or DATA / "scripts.jsonl")
    _validate(rows, {"id", "label", "turns", "department", "intent", "queue"}, "calls")
    return [CallCase(r["id"], r["label"], r["turns"], r["department"], r["intent"], r["queue"]) for r in rows]


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"ground truth not found: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate(rows: Sequence[dict], required: set, what: str) -> None:
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{what} case {row.get('id')!r} is missing {sorted(missing)}")
        if "department" in row and row["department"] not in D.DEPARTMENTS:
            raise ValueError(f"{what} case {row['id']!r} has unknown department {row['department']!r}")
        if "intent" in row and row["intent"] not in D.INTENTS.get(row["department"], {}):
            raise ValueError(
                f"{what} case {row['id']!r}: intent {row['intent']!r} is not in the "
                f"{row['department']!r} branch"
            )


# --------------------------------------------------------------------------- llm arm
def _routing_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "department": {"type": "string", "enum": DEPARTMENT_LIST},
            "intent": {"type": "string", "enum": ALL_INTENTS},
        },
        "required": ["department", "intent"],
        "additionalProperties": False,
    }


def _intent_menu() -> str:
    lines = []
    for dept, intents in D.INTENTS.items():
        lines.append(f"  {dept}: " + ", ".join(intents))
    return "\n".join(lines)


def _system_prompt() -> str:
    depts = "\n".join(f"  {k}: {v}" for k, v in D.DEPARTMENTS.items())
    return (
        "You are the switchboard for a car dealership. Read the caller's message and decide which "
        "department should handle it, and what they want.\n\n"
        f"Departments:\n{depts}\n\n"
        "Valid intents per department:\n"
        f"{_intent_menu()}\n\n"
        "Choose the single closest department and the single closest intent for that department. "
        "Reply with JSON only."
    )


def run_llm_routing(cases: Sequence[RoutingCase], *, concurrency: int = 4) -> List[dict]:
    schema, system = _routing_schema(), _system_prompt()

    def one(case: RoutingCase) -> dict:
        try:
            result = llm.chat_json(system, f'Caller: "{case.text}"', schema)
        except Exception as exc:  # noqa: BLE001 - recorded as a failure, not a crash
            return {"id": case.id, "error": str(exc), "department": None, "intent": None}
        data = result["data"]
        return {
            "id": case.id,
            "department": data.get("department"),
            "intent": data.get("intent"),
            "latency_ms": result["latency_ms"],
            "usage": result["usage"],
            "model": result["model"],
        }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, cases))


def run_llm_call(case: CallCase, *, schema: Optional[Dict[str, Any]] = None) -> dict:
    system = _system_prompt()
    transcript = "\n".join(f"Caller: {t}" for t in case.turns)
    result = llm.chat_json(system, transcript, schema or _routing_schema())
    data = result["data"]
    return {
        "id": case.id,
        "department": data.get("department"),
        "intent": data.get("intent"),
        "queue": queue_from(data.get("department"), data.get("intent")),
        "latency_ms": result["latency_ms"],
        "usage": result["usage"],
        "model": result["model"],
        "llm_calls": 1,
        "would_escalate": 1,
    }


# --------------------------------------------------------------------------- laya arm
def run_laya_routing(cases: Sequence[RoutingCase], router) -> List[dict]:
    import time

    out = []
    for case in cases:
        try:
            started = time.perf_counter()
            first = router.predict({"call": case.text}, D.DEPARTMENT_QUESTION)
            dept = first["answers"]["department"]["choice"]
            branch = D.intent_question(dept)
            second = router.predict({"call": case.text}, branch)
            ms = (time.perf_counter() - started) * 1000
            intent = second["answers"]["intent"]["choice"]
            out.append(
                {
                    "id": case.id,
                    "department": dept,
                    "intent": intent,
                    "latency_ms": ms,
                    "department_confidence": first["answers"]["department"].get("confidence"),
                    "usage": {
                        "prompt_tokens": first["usage"]["input_tokens"] + second["usage"]["input_tokens"],
                        "completion_tokens": 0,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append({"id": case.id, "error": str(exc), "department": None, "intent": None})
    return out


def run_laya_call(case: CallCase, router, *, incremental: bool = True, verify: bool = True) -> dict:
    session = CallSession(
        {"id": case.id, "label": case.label, "turns": case.turns},
        router=router,
        incremental=incremental,
        verify=verify,
    )
    events = list(session.advance())
    end = next((e for e in reversed(events) if e["type"] == "call_end"), {})
    return {
        "id": case.id,
        "department": (session.routing or {}).get("department"),
        "intent": (session.routing or {}).get("intent"),
        "queue": (session.routing or {}).get("queue"),
        "latency_ms": end.get("compute_ms", 0.0),
        "questions": sum(t["questions"] for t in session.turn_stats),
        "would_escalate": session.llm_escalations,
        "llm_calls": 0,
        # Laya's own token counts are deliberately NOT reported as LLM usage: they cost nothing,
        # and counting them would bill the cascade at generative prices.
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "laya_compute_ms": end.get("compute_ms", 0.0),
        "laya_questions": sum(t["questions"] for t in session.turn_stats),
    }


def run_hybrid_call(case: CallCase, router, *, schema: Optional[Dict[str, Any]] = None) -> dict:
    """Laya first; only ask the LLM when verification disagreed somewhere in the call."""
    base = run_laya_call(case, router)
    if base.get("would_escalate", 0) > 0:
        try:
            escalated = run_llm_call(case, schema=schema)
        except Exception as exc:  # noqa: BLE001
            base["escalation_error"] = str(exc)
            base["escalated"] = True
            return base
        escalated["escalated"] = True
        escalated["laya_queue"] = base.get("queue")
        escalated["would_escalate"] = base.get("would_escalate", 0)
        # Total wall time is the cascade plus the LLM call it had to make.
        escalated["latency_ms"] = float(escalated.get("latency_ms") or 0.0) + float(
            base.get("laya_compute_ms") or 0.0
        )
        return escalated
    base["escalated"] = False
    return base


# --------------------------------------------------------------------------- queues
def queue_from(department: Optional[str], intent: Optional[str]) -> Optional[str]:
    """Apply our deterministic routing policy to any arm's (department, intent) answer."""
    if not department:
        return None
    return D.decide(
        {
            "department": {"type": "choice", "choice": department, "probabilities": {department: 1.0}},
            "intent": {"type": "choice", "choice": intent or "other", "probabilities": {}},
        },
        [],
    )["queue"]


# --------------------------------------------------------------------------- scoring
def simulate_hybrid(
    cases: Sequence[RoutingCase],
    laya_results: Sequence[dict],
    llm_results: Sequence[dict],
    threshold: float,
    *,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay the decisions using a real escalation rule: trust the cascade unless its
    department confidence is below `threshold`, otherwise take the LLM's answer.

    This is a genuine simulation over recorded predictions, not an estimate — so the frontier
    below is what the system would actually have scored.
    """
    laya_by = {r["id"]: r for r in laya_results}
    llm_by = {r["id"]: r for r in llm_results}
    ok = flagged = errors_caught = errors_total = 0
    prompt_tokens = completion_tokens = 0

    for case in cases:
        laya = laya_by.get(case.id) or {}
        alt = llm_by.get(case.id) or {}
        conf = laya.get("department_confidence")
        escalate = conf is None or conf < threshold
        got = alt.get("department") if escalate else laya.get("department")
        correct = got == case.department
        ok += correct
        if laya.get("department") != case.department:
            errors_total += 1
            if escalate:
                errors_caught += 1
        if escalate:
            flagged += 1
            usage = alt.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))

    n = len(cases)
    cost = llm.price(model, prompt_tokens, completion_tokens) if model else None
    return {
        "threshold": threshold,
        "accuracy": round(ok / n, 4),
        "flag_rate": round(flagged / n, 4),
        "recall": round(errors_caught / errors_total, 4) if errors_total else None,
        "cost_per_case": round(cost / n, 8) if cost is not None else None,
        "cost_total": round(cost, 8) if cost is not None else None,
    }


def _pct(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    s = sorted(values)
    return {
        "p50": round(statistics.median(s), 1),
        "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 1),
        "mean": round(statistics.fmean(s), 1),
    }


def gate_stats(rows: Sequence[tuple], threshold: float) -> Dict[str, Any]:
    """Does our confidence actually identify the cases we get wrong?

    This is the number the whole hybrid design rests on: if a cheap tier's uncertainty signal does
    not separate right from wrong, escalating on it cannot help. `recall` is the share of errors
    that would have been sent upward; `precision` is how often a flag was warranted.
    """
    usable = [(c, ok) for c, ok in rows if c is not None]
    if not usable:
        return {}
    errors = [ok for _, ok in usable if not ok]
    flagged = [(c, ok) for c, ok in usable if c < threshold]
    caught = [ok for _, ok in flagged if not ok]
    confident = [(c, ok) for c, ok in usable if c >= threshold]
    return {
        "threshold": threshold,
        "flagged": len(flagged),
        "flag_rate": round(len(flagged) / len(usable), 4),
        "errors": len(errors),
        "errors_caught": len(caught),
        "recall": round(len(caught) / len(errors), 4) if errors else None,
        "precision": round(len(caught) / len(flagged), 4) if flagged else None,
        "accuracy_when_confident": (
            round(sum(1 for _, ok in confident if ok) / len(confident), 4) if confident else None
        ),
        "accuracy_when_flagged": (
            round(sum(1 for _, ok in flagged if ok) / len(flagged), 4) if flagged else None
        ),
    }


def score_routing(cases: Sequence[RoutingCase], results: Sequence[dict], *, model: Optional[str] = None) -> dict:
    by_id = {r["id"]: r for r in results}
    dept_ok = intent_ok = joint_ok = 0
    errors = 0
    latencies: List[float] = []
    prompt_tokens = completion_tokens = 0
    misses: List[dict] = []
    gate_rows: List[tuple] = []

    for case in cases:
        row = by_id.get(case.id) or {}
        if row.get("error") or not row.get("department"):
            errors += 1
            misses.append({"id": case.id, "text": case.text, "expected": case.department, "got": "ERROR"})
            continue
        latencies.append(float(row.get("latency_ms") or 0.0))
        usage = row.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens", 0))
        completion_tokens += int(usage.get("completion_tokens", 0))
        d_ok = row["department"] == case.department
        i_ok = row.get("intent") == case.intent
        dept_ok += d_ok
        intent_ok += i_ok
        joint_ok += d_ok and i_ok
        gate_rows.append((row.get("department_confidence"), d_ok))
        if not d_ok:
            misses.append(
                {"id": case.id, "text": case.text, "expected": case.department, "got": row["department"]}
            )

    n = len(cases)
    cost = llm.price(model, prompt_tokens, completion_tokens) if model else 0.0
    return {
        "n": n,
        "department_accuracy": round(dept_ok / n, 4),
        "intent_accuracy": round(intent_ok / n, 4),
        "joint_accuracy": round(joint_ok / n, 4),
        "errors": errors,
        "latency_ms": _pct(latencies),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 8) if cost is not None else None,
        "cost_model": model,
        "gate": gate_stats(gate_rows, 0.75),
        "misses": misses,
    }


def score_calls(cases: Sequence[CallCase], results: Sequence[dict], *, model: Optional[str] = None) -> dict:
    by_id = {r["id"]: r for r in results}
    ok = 0
    latencies: List[float] = []
    prompt_tokens = completion_tokens = 0
    llm_calls = would_escalate = 0
    questions = 0
    misses: List[dict] = []
    for case in cases:
        row = by_id.get(case.id) or {}
        if row.get("queue") == case.queue:
            ok += 1
        else:
            misses.append({"id": case.id, "expected": case.queue, "got": row.get("queue")})
        latencies.append(float(row.get("latency_ms") or 0.0))
        # Only real LLM calls are billed; a row with no LLM call contributes nothing.
        if int(row.get("llm_calls", 0)) > 0:
            usage = row.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
        llm_calls += int(row.get("llm_calls", 0))
        would_escalate += int(row.get("would_escalate", 0))
        questions += int(row.get("questions", 0))
    n = len(cases)
    cost = llm.price(model, prompt_tokens, completion_tokens) if model else 0.0
    return {
        "n": n,
        "queue_accuracy": round(ok / n, 4),
        "llm_calls": llm_calls,
        "would_escalate": would_escalate,
        "questions": questions,
        "latency_ms": _pct(latencies),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 8) if cost is not None else None,
        "misses": misses,
    }


def agreement(runs: Sequence[Sequence[dict]], key: str = "department") -> float:
    """Fraction of cases where every repeat produced the same answer."""
    if len(runs) < 2:
        return 1.0
    by_id = [{r["id"]: r.get(key) for r in run} for run in runs]
    ids = [r["id"] for r in runs[0]]
    same = sum(1 for i in ids if len({b.get(i) for b in by_id}) == 1)
    return round(same / max(1, len(ids)), 4)


# --------------------------------------------------------------------------- report
def render(title: str, rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    out = [title, "-" * len(line), line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def money(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "$0"
    return f"${value:.6f}"
