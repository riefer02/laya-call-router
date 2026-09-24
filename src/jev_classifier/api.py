"""FastAPI app.

A call is executed eagerly, recorded to `results/runs/<id>.jsonl`, and returned as a complete
event list. The browser owns playback (play / pause / step / speed), because for a demo tool
deterministic client-side playback beats live streaming — you can scrub, replay, and record
cleanly without depending on inference latency.

Endpoints
  GET  /                  -> the React app (web/dist when built)
  GET  /api/health
  GET  /api/scenarios     -> scripted caller scenarios
  POST /api/call          -> run a scenario, record it, return every event + summary
  GET  /api/runs          -> recorded runs
  GET  /api/runs/{id}     -> one recorded run's events
  POST /api/classify      -> generic support-triage, headless, no UI
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from . import runs as runs_store
from .agent import get_router
from .call import DEFAULT_THRESHOLD, CallSession
from .scenarios import SCENARIO_BY_ID, SCENARIOS

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"

app = FastAPI(title="Laya Call Router", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm() -> None:
    import threading

    # The app has no authentication and, once the LLM/STT arms land, fronts a billable API key.
    # Binding to anything but loopback would expose both to the network.
    argv = " ".join(sys.argv)
    if any(bad in argv for bad in ("--host 0.0.0.0", "--host ::", "--host 0.0.0.0/0")):
        print(
            "\n  !! Laya Call Router is bound to a non-loopback address.\n"
            "     This service has no auth and will proxy key-backed calls. Use --host 127.0.0.1.\n"
        )

    def _load() -> None:
        router = get_router()
        router.preload(["english", "multilingual"])
        # Warm the graph, not just the weights. Loading a checkpoint is not the same as running it:
        # measured, the first call after startup took 287 ms where the same turn takes 69 ms warm, so
        # the very first thing anyone sees is the slowest this system will ever be. One throwaway
        # predict pays that cost before the audience arrives.
        try:
            router.predict(
                "Caller: hello",
                {
                    "destination": {
                        "type": "choice",
                        "instructions": "Which department should handle this?",
                        "criteria": {"service": "service", "sales": "sales"},
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001 - a warmup must never stop the server starting
            print(f"warmup skipped: {exc}")

    threading.Thread(target=_load, name="laya-preload", daemon=True).start()


# ----------------------------------------------------------------------------- scenarios
@app.get("/api/health")
def health() -> Dict[str, Any]:
    from .agent import resolve_checkpoint

    checkpoint = resolve_checkpoint()
    return {
        "ok": True,
        "version": __version__,
        "runs_dir": str(runs_store.RUNS_DIR),
        "model": "fine-tuned" if checkpoint else "base",
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
    }


@app.get("/api/scenarios")
def scenarios() -> Dict[str, Any]:
    return {"scenarios": SCENARIOS}


# ----------------------------------------------------------------------------- calls
class CallRequest(BaseModel):
    scenario_id: Optional[str] = None
    turns: Optional[List[str]] = None
    label: str = "Custom call"
    threshold: float = DEFAULT_THRESHOLD


def _summary(session: CallSession, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    end = next((e for e in reversed(events) if e["type"] == "call_end"), {})
    return {
        "call_id": session.session_id,
        "scenario": session.scenario.get("label"),
        "turns": end.get("turns"),
        "decisions": end.get("decisions"),
        "compute_ms": end.get("compute_ms"),
        "input_tokens": end.get("input_tokens"),
        "output_tokens": end.get("output_tokens"),
        "tokens_generated": 0,
        "cost_usd": 0.0,
        "routing": session.routing,
        "completion": end.get("completion"),
        "booking": end.get("booking"),
        "contact": end.get("contact"),
    }


# ----------------------------------------------------------------------------- evidence
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
ARM_LABELS = {
    "laya": "base",
    "cascade-ft": "fine-tuned",
    "gpt-5.4-nano": "gpt-5.4-nano",
    "deepseek-flash": "deepseek-flash",
}


def _results_tag() -> str:
    """The results files belonging to the checkpoint actually being served.

    The Evidence tab used to read hardcoded `eval_v4.json` and `severity.json`, so it showed v4's
    routing numbers and the *pre-training* safety numbers while the app served v7 - a demo whose
    evidence contradicts its own behaviour. The tag is derived from the served checkpoint instead,
    so the two cannot drift.
    """
    from .agent import resolve_checkpoint

    ckpt = resolve_checkpoint()
    if ckpt is None:
        return ""
    match = re.search(r"kaggle-out-(v\d+)", str(ckpt.resolve()))
    return match.group(1) if match else ""


def _read_json(name: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/api/results")
def results() -> Dict[str, Any]:
    """The measurements behind the design, so the app can show its own evidence.

    Read straight off `results/*.json` rather than baked into the bundle: a number in the UI that
    cannot be traced to a report is a number nobody can check.
    """
    from . import store_profile as SP

    profile = SP.load()
    taxonomy = [
        {
            "key": d.key,
            "label": d.label,
            "description": d.description,
            "queue": d.queue,
            "subqueues": [
                {
                    "key": s.key,
                    "label": s.label,
                    "description": s.description,
                    "queue": s.queue,
                    "handler": s.handler,
                }
                for s in profile.subqueues_for(d.key)
            ],
        }
        for d in profile.destinations
    ]

    tag = _results_tag()
    # A custom checkpoint may have no matching report, so name the reference report actually
    # shown and flag when its numbers do not belong to the served model.
    eval_candidates = ([f"eval_{tag}.json"] if tag else []) + ["eval_v7.json", "eval_v4.json"]
    eval_name = next((name for name in eval_candidates if _read_json(name) is not None), "")
    evaluation = _read_json(eval_name) or {}
    routing = evaluation.get("routing") or {}
    arms = []
    for key, score in routing.items():
        if not isinstance(score, dict) or "destination_accuracy" not in score:
            continue
        arms.append(
            {
                "key": key,
                "label": ARM_LABELS.get(key, key),
                "destination": score.get("destination_accuracy"),
                "destination_ci": (score.get("destination_ci95") or {}).get("half_width"),
                "subqueue": score.get("subqueue_accuracy"),
                "joint": score.get("joint_accuracy"),
                "queue": score.get("queue_accuracy"),
                "latency_p50": (score.get("latency_ms") or {}).get("p50"),
                # `cost_usd` is the arm's total across all cases. Renaming it here to
                # `cost_per_case` told the evidence table that deepseek costs $0.010 a call when the
                # measured figure is $0.000124 - a 81x overstatement, and it propagated into the
                # pitch deck before anyone checked.
                "cost_per_case": (
                    (score.get("cost_usd") or 0) / score["n"]
                    if score.get("cost_usd") and score.get("n")
                    else score.get("cost_usd")
                ),
                "cost_total": score.get("cost_usd"),
                "determinism": score.get("determinism"),
                "calibration": score.get("calibration") or {},
                "other_rate": (score.get("other") or {}).get("rate_of_subqueue_predictions"),
                "gate": score.get("gate") or {},
            }
        )
    order = {"base": 0, "fine-tuned": 1, "gpt-5.4-nano": 2, "deepseek-flash": 3}
    arms.sort(key=lambda a: order.get(a["label"], 9))

    calls = []
    for key, score in (evaluation.get("calls") or {}).items():
        if not isinstance(score, dict):
            continue
        calls.append(
            {
                "label": ARM_LABELS.get(key, key),
                "n": score.get("n"),
                "queue": score.get("queue_accuracy"),
                "questions": score.get("questions"),
                "latency_p50": (score.get("latency_ms") or {}).get("p50"),
                "cost": score.get("cost_usd"),
            }
        )

    sev_candidates = ([f"severity_{tag}.json"] if tag else []) + ["severity_v7.json", "severity.json"]
    sev_name = next((name for name in sev_candidates if _read_json(name) is not None), "")
    severity = _read_json(sev_name) or {}
    sev_rows = []
    for arm, blob in (severity.get("arms") or {}).items():
        at = blob.get("at_default") or {}
        sev_rows.append(
            {
                "arm": arm,
                "safe": at.get("is_safe_to_drive"),
                "human": at.get("needs_human"),
                "sweep": (blob.get("sweep") or {}).get("is_safe_to_drive") or [],
            }
        )

    dataset = _read_json("dataset_report.json") or {}
    generality = _read_json("generality_e8.json") or {}

    # The sample size comes from a scored arm, not from a miss count. It was briefly read off the
    # base arm's `misses` list, which told the UI there were 28 cases instead of 81.
    n_cases = next(
        (s.get("n") for s in routing.values() if isinstance(s, dict) and s.get("n")),
        None,
    )

    return {
        # Which reports these numbers came from, so the UI can say - a measurement in the UI that
        # cannot be traced to a file is one nobody can check.
        "sources": {
            "routing": eval_name,
            "severity": sev_name,
            "checkpoint": tag or "base",
            "matches_served_model": bool(tag and eval_name == f"eval_{tag}.json"),
        },
        "taxonomy": taxonomy,
        "n_cases": n_cases,
        "arms": arms,
        "calls": calls,
        "calibration": (routing.get("cascade-ft") or {}).get("calibration") or {},
        "severity": sev_rows,
        "dataset": {
            "kept": dataset.get("kept"),
            "by_destination": dataset.get("by_destination") or {},
            "cost_usd": dataset.get("cost_usd"),
            "criteria": dataset.get("criteria") or {},
        },
        "generality": {
            "suites": generality.get("suites") or {},
            "verdict": (generality.get("verdict") or {}).get("text"),
        },
        "policy": profile.policy,
        "facts": profile.facts,
    }


@app.post("/api/call")
def run_call(req: CallRequest) -> Dict[str, Any]:
    if req.turns:
        scenario = {"id": "custom", "label": req.label, "turns": req.turns}
    elif req.scenario_id:
        scenario = SCENARIO_BY_ID.get(req.scenario_id)
        if scenario is None:
            raise HTTPException(status_code=404, detail=f"unknown scenario {req.scenario_id!r}")
    else:
        raise HTTPException(status_code=400, detail="provide scenario_id or turns")

    call_id = uuid.uuid4().hex[:12]
    session = CallSession(scenario, session_id=call_id, confidence_threshold=req.threshold)
    events = list(session.advance())
    runs_store.save(call_id, events)
    return {"events": events, "summary": _summary(session, events)}


@app.get("/api/runs")
def list_runs() -> Dict[str, Any]:
    return {"runs": runs_store.list_runs()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str = PathParam(..., pattern=r"^[A-Za-z0-9_-]{1,64}$")) -> Dict[str, Any]:
    try:
        events = runs_store.load_events(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="unknown run")
    end = next((e for e in reversed(events) if e["type"] == "call_end"), {})
    return {"events": events, "summary": {"call_id": run_id, **end}}


# ----------------------------------------------------------------------------- generic triage
class ClassifyRequest(BaseModel):
    message: str
    threshold: float = DEFAULT_THRESHOLD


@app.post("/api/classify")
def classify(req: ClassifyRequest) -> Dict[str, Any]:
    """Headless single-shot: route one utterance through the *dealership* cascade.

    This used to call `pipeline.py`, the original generic support-triage cascade, which does not read
    the store profile at all - so "I need a quote for four new tyres" came back as `Sales Desk`. Two
    cascades behind one API is a trap: the endpoint sits beside `/api/call` and looks like the same
    thing with fewer turns. It is now the same cascade, one turn.
    """
    scenario = {"id": "single", "label": "Single utterance", "turns": [req.message]}
    session = CallSession(scenario, confidence_threshold=req.threshold)
    events = list(session.advance())
    end = next((e for e in reversed(events) if e["type"] == "call_end"), {})
    return {
        "message": req.message,
        "routing": end.get("routing") or session.routing,
        "rejected": False,
        "events": events,
    }


# ----------------------------------------------------------------------------- static app
if WEB_DIST.is_dir():
    # Vite replaces dist/assets during a rebuild. The API must remain importable in that brief
    # window; StaticFiles will serve the directory once the build finishes.
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets"), check_dir=False), name="assets")


@app.get("/")
def index():
    if (WEB_DIST / "index.html").is_file():
        return FileResponse(WEB_DIST / "index.html")
    return JSONResponse(
        {"detail": "Frontend not built. Run `cd web && npm install && npm run dev`."},
        status_code=200,
    )
