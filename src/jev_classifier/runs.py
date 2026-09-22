"""Record and replay call runs.

A run is just its event stream, so recording is appending JSON lines and replay is reading them
back. Replay makes a screen recording reproducible without depending on model latency — and it is
the foundation any future "recover a dropped call" feature would build on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "results" / "runs"


def save(session_id: str, events: List[Dict[str, Any]]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{session_id}.jsonl"
    with path.open("w") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")
    return path


def list_runs() -> List[Dict[str, Any]]:
    if not RUNS_DIR.is_dir():
        return []
    out = []
    for path in sorted(RUNS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            first = json.loads(path.read_text().splitlines()[0])
        except Exception:
            continue
        if first.get("type") != "call_start":
            continue
        out.append(
            {
                "id": path.stem,
                "scenario": first.get("scenario", {}),
                "modified": path.stat().st_mtime,
                "bytes": path.stat().st_size,
            }
        )
    return out


def load_events(session_id: str) -> List[Dict[str, Any]]:
    path = RUNS_DIR / f"{session_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(session_id)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replay(session_id: str, delay: float = 0.3) -> Iterator[Dict[str, Any]]:
    """Yield a recorded run's events with the same pacing, for a clean screen recording."""
    for event in load_events(session_id):
        yield event
        if delay and event.get("type") not in ("call_end",):
            time.sleep(delay)


def latest_id() -> Optional[str]:
    runs = list_runs()
    return runs[0]["id"] if runs else None
