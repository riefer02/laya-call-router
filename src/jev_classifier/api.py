"""FastAPI app: streams each cascade stage to the browser over Server-Sent Events.

Endpoints
  GET  /                       -> the single-page demo
  GET  /api/personas           -> scripted example calls
  POST /api/session            -> start a session {"message": "..."} -> {"session_id": "..."}
  GET  /api/session/{id}/stream-> SSE: stage_start / stage_result / clarify / routing / done
  POST /api/session/{id}/answer-> answer a clarifying question {"text": "..."}
  POST /api/classify           -> headless: run to completion, return all events as JSON
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .personas import PERSONAS
from .pipeline import DEFAULT_THRESHOLD, TriageSession

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
STAGE_DELAY = float(os.environ.get("JEV_STAGE_DELAY", "0.35"))

app = FastAPI(title="jev-classifier", version=__version__)

_sessions: Dict[str, TriageSession] = {}
_sessions_lock = threading.Lock()


@app.on_event("startup")
def _warm() -> None:
    """Preload the checkpoints the cascade can route to, off the request path."""
    def _load() -> None:
        from .agent import get_router

        get_router().preload(["english", "multilingual"])

    threading.Thread(target=_load, name="laya-preload", daemon=True).start()


def _store(session: TriageSession) -> str:
    with _sessions_lock:
        _sessions[session.session_id] = session
    return session.session_id


def _get(session_id: str) -> TriageSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return session


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


class StartRequest(BaseModel):
    message: str
    threshold: float = DEFAULT_THRESHOLD
    pace: bool = True


class AnswerRequest(BaseModel):
    text: str


@app.get("/api/personas")
def personas() -> Dict[str, Any]:
    return {"personas": PERSONAS, "stage_delay": STAGE_DELAY}


@app.post("/api/session")
def start_session(req: StartRequest) -> Dict[str, str]:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is empty")
    session = TriageSession(
        req.message.strip(),
        confidence_threshold=req.threshold,
        session_id=uuid.uuid4().hex[:12],
    )
    session.pace = bool(req.pace)  # type: ignore[attr-defined]
    return {"session_id": _store(session)}


def _stream(session_id: str) -> Iterator[str]:
    session = _get(session_id)
    pace = getattr(session, "pace", True)
    try:
        for event in session.advance():
            yield _sse(event)
            if pace and event.get("type") != "done":
                time.sleep(STAGE_DELAY)
        yield "event: close\ndata: {}\n\n"
    except Exception as exc:  # surface model errors to the UI instead of hanging the stream
        yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


@app.get("/api/session/{session_id}/stream")
def stream(session_id: str) -> StreamingResponse:
    return StreamingResponse(
        _stream(session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/session/{session_id}/answer")
def answer(session_id: str, req: AnswerRequest) -> Dict[str, Any]:
    session = _get(session_id)
    if session.waiting is None:
        raise HTTPException(status_code=409, detail="session is not waiting for an answer")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="answer is empty")
    session.provide_answer(req.text.strip())
    return {"ok": True}


class ClassifyRequest(BaseModel):
    message: str
    threshold: float = DEFAULT_THRESHOLD


@app.post("/api/classify")
def classify(req: ClassifyRequest) -> Dict[str, Any]:
    """Headless run: auto-answers clarifications with the provisional choice."""
    from .pipeline import run_to_completion

    result = run_to_completion(req.message, confidence_threshold=req.threshold)
    session: TriageSession = result["session"]
    return {
        "message": req.message,
        "routing": session.routing,
        "rejected": session.rejected,
        "events": result["events"],
        "total_ms": result["events"][-1].get("total_ms") if result["events"] else None,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
