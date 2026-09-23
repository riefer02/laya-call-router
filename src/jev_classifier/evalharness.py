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
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from concurrent.futures import ThreadPoolExecutor

from . import dealership as D
from . import labels, llm
from . import store_profile as SP
from .agent import get_router
from .call import CallSession

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "calls"

PROFILE = SP.load()
DESTINATION_LIST = list(PROFILE.destination_keys)
ALL_SUBQUEUES = sorted({s.key for s in PROFILE.subqueues})


# --------------------------------------------------------------------------- ground truth
@dataclass
class RoutingCase:
    id: str
    text: str
    destination: str
    subqueue: Optional[str] = None


@dataclass
class CallCase:
    id: str
    label: str
    turns: List[str]
    destination: str
    queue: str
    subqueue: Optional[str] = None


def load_routing(path: Optional[Path] = None) -> List[RoutingCase]:
    rows = _read_jsonl(path or DATA / "routing.jsonl")
    _validate(rows, {"id", "text", "destination"}, "routing")
    return [RoutingCase(r["id"], r["text"], r["destination"], r.get("subqueue")) for r in rows]


def load_calls(path: Optional[Path] = None) -> List[CallCase]:
    rows = _read_jsonl(path or DATA / "scripts.jsonl")
    _validate(rows, {"id", "label", "turns", "destination", "queue"}, "calls")
    return [
        CallCase(r["id"], r["label"], r["turns"], r["destination"], r["queue"], r.get("subqueue"))
        for r in rows
    ]


def _read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"ground truth not found: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate(rows: Sequence[dict], required: set, what: str) -> None:
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{what} case {row.get('id')!r} is missing {sorted(missing)}")
        if "destination" in row and row["destination"] not in PROFILE.destination_keys:
            raise ValueError(
                f"{what} case {row['id']!r} has unknown destination {row['destination']!r}"
            )
        sub = row.get("subqueue")
        if sub is not None and PROFILE.subqueue(row.get("destination"), sub) is None:
            raise ValueError(
                f"{what} case {row['id']!r}: sub-queue {sub!r} is not under "
                f"{row.get('destination')!r}"
            )


# --------------------------------------------------------------------------- llm arm
def _routing_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "enum": DESTINATION_LIST},
            "subqueue": {"type": "string", "enum": ALL_SUBQUEUES},
        },
        "required": ["destination", "subqueue"],
        "additionalProperties": False,
    }


def _subqueue_menu() -> str:
    lines = []
    for dest in PROFILE.destinations:
        subs = PROFILE.subqueue_keys(dest.key)
        lines.append(f"  {dest.key}: " + (", ".join(subs) if subs else "(none)"))
    return "\n".join(lines)


def _system_prompt() -> str:
    depts = "\n".join(f"  {d.key}: {d.description}" for d in PROFILE.destinations)
    return (
        "You are the switchboard for a car dealership. Read the caller's message and decide which "
        "part of the dealership owns the work, and which queue should take it.\n\n"
        f"Destinations:\n{depts}\n\n"
        "Sub-queues per destination:\n"
        f"{_subqueue_menu()}\n\n"
        "Tyres, detailing and roadside assistance are sub-queues of service, not destinations. "
        "Choose the closest destination and the closest sub-queue within it. Reply with JSON only."
    )


def run_llm_routing(
    cases: Sequence[RoutingCase], *, model_ref: str = "openai:gpt-5.4-nano", concurrency: int = 4
) -> List[dict]:
    schema, system = _routing_schema(), _system_prompt()
    provider, model = llm.parse_model(model_ref)

    def one(case: RoutingCase) -> dict:
        try:
            result = llm.chat_json(
                system, f'Caller: "{case.text}"', schema, provider=provider, model=model
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a failure, not a crash
            return {"id": case.id, "error": str(exc), "destination": None, "subqueue": None}
        raw = result["data"] if isinstance(result.get("data"), dict) else {}
        destination, subqueue, ok = labels.validate_label(
            labels.field(raw, "destination", "dept", "department"),
            labels.field(raw, "subqueue", "sub_queue", "intent"),
        )
        return {
            "id": case.id,
            "destination": destination,
            "subqueue": subqueue,
            "valid": ok,
            "raw_destination": labels.field(raw, "destination", "dept", "department"),
            "raw_subqueue": labels.field(raw, "subqueue", "sub_queue", "intent"),
            "latency_ms": result["latency_ms"],
            "usage": result["usage"],
            "provider": result["provider"],
            "model": result["model"],
        }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(one, cases))


def run_llm_call(
    case: CallCase, *, model_ref: str = "openai:gpt-5.4-nano", schema: Optional[Dict[str, Any]] = None
) -> dict:
    system = _system_prompt()
    transcript = "\n".join(f"Caller: {t}" for t in case.turns)
    provider, model = llm.parse_model(model_ref)
    result = llm.chat_json(
        system, transcript, schema or _routing_schema(), provider=provider, model=model
    )
    raw = result["data"] if isinstance(result.get("data"), dict) else {}
    destination, subqueue, _ = labels.validate_label(
        labels.field(raw, "destination", "dept", "department"),
        labels.field(raw, "subqueue", "sub_queue", "intent"),
    )
    return {
        "id": case.id,
        "destination": destination,
        "subqueue": subqueue,
        "queue": queue_from(destination, subqueue),
        "latency_ms": result["latency_ms"],
        "usage": result["usage"],
        "provider": result["provider"],
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
            first = router.predict({"call": case.text}, D.DESTINATION_QUESTION)
            dest = first["answers"]["destination"]["choice"]
            ptr = first["usage"]["input_tokens"]
            sub = None
            sub_q = D.subqueue_question(dest)
            if sub_q is not None:
                second = router.predict({"call": case.text}, sub_q)
                sub = second["answers"]["subqueue"]["choice"]
                ptr += second["usage"]["input_tokens"]
            ms = (time.perf_counter() - started) * 1000
            out.append(
                {
                    "id": case.id,
                    "destination": dest,
                    "subqueue": sub,
                    "latency_ms": ms,
                    "destination_confidence": first["answers"]["destination"].get("confidence"),
                    "usage": {"prompt_tokens": ptr, "completion_tokens": 0},
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append({"id": case.id, "error": str(exc), "destination": None, "subqueue": None})
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
        "destination": (session.routing or {}).get("destination"),
        "subqueue": (session.routing or {}).get("subqueue"),
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


def run_hybrid_call(
    case: CallCase,
    router,
    *,
    model_ref: str = "openai:gpt-5.4-nano",
    schema: Optional[Dict[str, Any]] = None,
) -> dict:
    """Laya first; only ask the LLM when verification disagreed somewhere in the call."""
    base = run_laya_call(case, router)
    if base.get("would_escalate", 0) > 0:
        try:
            escalated = run_llm_call(case, model_ref=model_ref, schema=schema)
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
def queue_from(destination: Optional[str], subqueue: Optional[str]) -> Optional[str]:
    """Apply our deterministic routing policy to any arm's answer."""
    if not destination:
        return None
    answers: Dict[str, Any] = {
        "destination": {"type": "choice", "choice": destination, "probabilities": {destination: 1.0}}
    }
    if subqueue:
        answers["subqueue"] = {"type": "choice", "choice": subqueue, "probabilities": {subqueue: 1.0}}
    return D.decide(answers, [])["queue"]


# --------------------------------------------------------------------------- scoring
def simulate_hybrid(
    cases: Sequence[RoutingCase],
    laya_results: Sequence[dict],
    llm_results: Sequence[dict],
    threshold: float,
    *,
    model_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay the decisions using a real escalation rule: trust the cascade unless its
    department confidence is below `threshold`, otherwise take the LLM's answer.

    This is a genuine simulation over recorded predictions, not an estimate — so the frontier
    below is what the system would actually have scored.
    """
    laya_by = {r["id"]: r for r in laya_results}
    llm_by = {r["id"]: r for r in llm_results}
    ok = flagged = errors_caught = errors_total = 0
    escalated_rows: List[dict] = []

    for case in cases:
        laya = laya_by.get(case.id) or {}
        alt = llm_by.get(case.id) or {}
        conf = laya.get("destination_confidence")
        escalate = conf is None or conf < threshold
        got = alt.get("destination") if escalate else laya.get("destination")
        correct = got == case.destination
        ok += correct
        if laya.get("destination") != case.destination:
            errors_total += 1
            if escalate:
                errors_caught += 1
        if escalate:
            flagged += 1
            escalated_rows.append(alt)

    n = len(cases)
    totals = token_totals(escalated_rows)
    cost = cost_of(model_ref, totals) if escalated_rows else 0.0
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


def token_totals(rows: Sequence[dict]) -> Dict[str, int]:
    """Sum real token usage. Rows with no LLM call contribute nothing — a cascade row is not
    billed at generative prices just because it also produced tokens internally."""
    tot = {"prompt_tokens": 0, "prompt_cache_hit": 0, "prompt_cache_miss": 0, "completion_tokens": 0}
    for row in rows:
        if "llm_calls" in row and int(row.get("llm_calls", 0)) < 1:
            continue
        usage = row.get("usage") or {}
        tot["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        tot["prompt_cache_hit"] += int(usage.get("prompt_cache_hit", 0))
        tot["prompt_cache_miss"] += int(usage.get("prompt_cache_miss", 0))
        tot["completion_tokens"] += int(usage.get("completion_tokens", 0))
    if not tot["prompt_cache_hit"] and not tot["prompt_cache_miss"]:
        tot["prompt_cache_miss"] = tot["prompt_tokens"]
    return tot


def cost_of(model_ref: Optional[str], totals: Dict[str, int]) -> Optional[float]:
    if not model_ref:
        return None
    provider, model = llm.parse_model(model_ref)
    return llm.price(
        provider,
        model,
        totals.get("prompt_cache_miss", 0),
        totals.get("prompt_cache_hit", 0),
        totals.get("completion_tokens", 0),
    )


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Dict[str, float]:
    """Wilson score interval for a proportion.

    Reported because our test set is 81 cases: one case moves a percentage by 1.23 points, so an
    unqualified "0.654 vs 0.728" invites a conclusion the sample cannot support. The interval is
    the honest unit of comparison.
    """
    if n <= 0:
        return {"lo": 0.0, "hi": 0.0, "half_width": 0.0}
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return {"lo": round(lo, 4), "hi": round(hi, 4), "half_width": round((hi - lo) / 2.0, 4)}


def _confusion(pairs: Iterable[tuple]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for expected, got in pairs:
        out.setdefault(str(expected), {})
        key = str(got)
        out[str(expected)][key] = out[str(expected)].get(key, 0) + 1
    return {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in sorted(out.items())}


def _calibration(rows: Iterable[tuple], buckets: int = 5) -> Dict[str, Dict[str, Any]]:
    """Accuracy by confidence band.

    This is the question the escalation gate actually depends on: does the model's confidence
    separate its right answers from its wrong ones? A model that is confidently wrong everywhere
    has a flat curve, and no threshold can rescue it - which is what fine-tuning produced, with
    0 of 81 cases flagged. A model whose low-confidence band is much worse than its high band has
    a usable gate.
    """
    vals = [(float(c), bool(ok)) for c, ok in rows if c is not None]
    if not vals:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(buckets):
        lo, hi = i / buckets, (i + 1) / buckets
        band = [ok for c, ok in vals if (lo <= c < hi) or (i == buckets - 1 and c >= hi)]
        if band:
            out[f"{lo:.1f}-{hi:.1f}"] = {
                "n": len(band),
                "accuracy": round(sum(band) / len(band), 3),
            }
    return out


def score_routing(
    cases: Sequence[RoutingCase], results: Sequence[dict], *, model_ref: Optional[str] = None
) -> dict:
    by_id = {r["id"]: r for r in results}
    dest_ok = sub_ok = joint_ok = 0
    errors = 0
    latencies: List[float] = []
    misses: List[dict] = []
    gate_rows: List[tuple] = []
    invalid = 0
    dest_pairs: List[tuple] = []
    sub_pairs: List[tuple] = []
    n_sub_predicted = n_sub_scored = n_sub_other = 0
    queue_ok = 0
    queue_pairs: List[tuple] = []

    for case in cases:
        row = by_id.get(case.id) or {}
        if row.get("error") or not row.get("destination"):
            errors += 1
            misses.append({"id": case.id, "text": case.text, "expected": case.destination, "got": "ERROR"})
            continue
        latencies.append(float(row.get("latency_ms") or 0.0))
        if row.get("valid") is False:
            invalid += 1
        d_ok = row["destination"] == case.destination
        s_ok = case.subqueue is None or row.get("subqueue") == case.subqueue
        dest_ok += d_ok
        sub_ok += s_ok
        joint_ok += d_ok and s_ok
        # The label is internal; the queue is what the caller experiences. Two different labels can
        # route to the same place (`front_desk` and `non_customer` both go to Front Desk), so a
        # label miss is not automatically a routing miss. Reported alongside, never instead of.
        want_queue = queue_from(case.destination, case.subqueue)
        got_queue = queue_from(row["destination"], row.get("subqueue"))
        q_ok = bool(want_queue) and got_queue == want_queue
        queue_ok += q_ok
        queue_pairs.append((want_queue, got_queue))
        dest_pairs.append((case.destination, row["destination"]))
        if case.subqueue is not None:
            n_sub_scored += 1
            sub_pairs.append((case.subqueue, row.get("subqueue")))
            if row.get("subqueue") == "other":
                n_sub_other += 1
        if row.get("subqueue") is not None:
            n_sub_predicted += 1
        gate_rows.append((row.get("destination_confidence"), d_ok))
        if not d_ok:
            misses.append(
                {"id": case.id, "text": case.text, "expected": case.destination, "got": row["destination"]}
            )

    n = len(cases)
    totals = token_totals(results)
    cost = cost_of(model_ref, totals)
    sub_misses = sum(1 for e, g in sub_pairs if e != g)
    other_among_misses = sum(1 for e, g in sub_pairs if e != g and g == "other")
    return {
        "n": n,
        "destination_accuracy": round(dest_ok / n, 4),
        "subqueue_accuracy": round(sub_ok / n, 4),
        "joint_accuracy": round(joint_ok / n, 4),
        "destination_ci95": wilson_ci(dest_ok, n),
        "subqueue_ci95": wilson_ci(sub_ok, n),
        "joint_ci95": wilson_ci(joint_ok, n),
        "queue_accuracy": round(queue_ok / n, 4),
        "queue_ci95": wilson_ci(queue_ok, n),
        "destination_confusion": _confusion(dest_pairs),
        "subqueue_confusion": _confusion(sub_pairs),
        "queue_confusion": _confusion(queue_pairs),
        "other": {
            "predicted_other": n_sub_other,
            "rate_of_subqueue_predictions": (
                round(n_sub_predicted and n_sub_other / n_sub_predicted, 4)
            ),
            "subqueue_misses": sub_misses,
            "miss_absorbed_by_other": other_among_misses,
            "share_of_misses": round(other_among_misses / sub_misses, 4) if sub_misses else None,
        },
        "errors": errors,
        "invalid_labels": invalid,
        "latency_ms": _pct(latencies),
        "prompt_tokens": totals["prompt_tokens"],
        "prompt_cache_hit_tokens": totals["prompt_cache_hit"],
        "completion_tokens": totals["completion_tokens"],
        "cost_usd": round(cost, 8) if cost is not None else None,
        "cost_model": model_ref,
        "gate": gate_stats(gate_rows, 0.75),
        "calibration": _calibration(gate_rows),
        "misses": misses,
    }


def score_calls(
    cases: Sequence[CallCase], results: Sequence[dict], *, model_ref: Optional[str] = None
) -> dict:
    by_id = {r["id"]: r for r in results}
    ok = 0
    latencies: List[float] = []
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
        llm_calls += int(row.get("llm_calls", 0))
        would_escalate += int(row.get("would_escalate", 0))
        questions += int(row.get("questions", 0))
    n = len(cases)
    totals = token_totals(results)
    cost = cost_of(model_ref, totals)
    return {
        "n": n,
        "queue_accuracy": round(ok / n, 4),
        "llm_calls": llm_calls,
        "would_escalate": would_escalate,
        "questions": questions,
        "latency_ms": _pct(latencies),
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "cost_usd": round(cost, 8) if cost is not None else None,
        "misses": misses,
    }


def agreement(runs: Sequence[Sequence[dict]], key: str = "destination") -> float:
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
