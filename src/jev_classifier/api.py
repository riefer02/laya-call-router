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

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="jev-classifier", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm() -> None:
    import threading

    def _load() -> None:
        get_router().preload(["english", "multilingual"])

    threading.Thread(target=_load, name="laya-preload", daemon=True).start()


# ----------------------------------------------------------------------------- scenarios
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "version": __version__, "runs_dir": str(runs_store.RUNS_DIR)}


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
def get_run(run_id: str) -> Dict[str, Any]:
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
    """Headless single-shot triage (the original support cascade), auto-answering clarifications."""
    from .pipeline import run_to_completion

    result = run_to_completion(req.message, confidence_threshold=req.threshold)
    session = result["session"]
    return {
        "message": req.message,
        "routing": session.routing,
        "rejected": session.rejected,
        "events": result["events"],
    }


# ----------------------------------------------------------------------------- static app
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets")), name="assets")


@app.get("/")
def index():
    if (WEB_DIST / "index.html").is_file():
        return FileResponse(WEB_DIST / "index.html")
    return JSONResponse(
        {"detail": "Frontend not built. Run `cd web && npm install && npm run dev`."},
        status_code=200,
    )
