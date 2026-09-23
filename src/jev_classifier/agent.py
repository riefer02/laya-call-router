"""Shared Laya-MLX router.

All three checkpoints together are ~1.16B parameters, so the process keeps one Router and
preloads the checkpoints it serves. `preload=True` pays the cold-load cost once at startup
instead of on the first request that switches language.

**The fine-tune has to be selected explicitly, or the app serves the base model.** `laya.Router()`
with no `models` argument falls back to `convaiinnovations/laya` - the stock checkpoint that scores
0.654 destination / 0.518 sub-queue. That is the weak model this project exists to replace, and
nothing about the app looks wrong when it is serving it. Resolved in order:

  JEV_MODEL          an explicit checkpoint directory (used by the eval scripts)
  models/active      a symlink or directory naming the checkpoint this install should serve
  otherwise          the base checkpoint, with a warning

`models/active` exists so switching the demo to a new fine-tune is one command, and so a
half-downloaded `kaggle-out-*` directory cannot be picked up by accident.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import laya_mlx as laya

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "models" / "active"

_router: Optional[laya.Router] = None
_lock = threading.Lock()


def resolve_checkpoint() -> Optional[Path]:
    """The fine-tuned checkpoint to serve, or None to serve the base model."""
    env = os.environ.get("JEV_MODEL")
    if env:
        path = Path(env)
        if not path.is_absolute():
            path = ROOT / path
        if (path / "rl_agent_config.json").is_file():
            return path
        raise SystemExit(f"JEV_MODEL={env} is not a checkpoint directory (no rl_agent_config.json)")
    if (ACTIVE / "rl_agent_config.json").is_file():
        return ACTIVE
    return None


def new_base_router(max_loaded: int = 2) -> "laya.Router":
    """An explicit, separate stock router for evaluation baseline arms.

    The app's get_router() follows models/active. Reusing it for an eval arm named `base`
    silently scores the fine-tune twice whenever a demo checkpoint is selected.
    """
    return laya.Router(max_loaded=max_loaded)


def get_router(
    preload: bool = False,
    dtype: str = "float16",
    max_loaded: int = 3,
    device: Optional[str] = None,
) -> "laya.Router":
    """Return the process-wide Router, building it on first call."""
    global _router
    with _lock:
        if _router is None:
            checkpoint = resolve_checkpoint()
            if checkpoint is None:
                print(
                    "laya: serving the BASE checkpoint. Routing quality is the base model's, not "
                    "this project's fine-tune. Point models/active (or JEV_MODEL) at a "
                    "kaggle-out-*/laya-dealership-routing directory to serve the fine-tune."
                )
                _router = laya.Router(max_loaded=max_loaded, dtype=dtype, device=device)
            else:
                try:
                    shown = checkpoint.relative_to(ROOT)
                except ValueError:
                    shown = checkpoint  # JEV_MODEL may point outside the repo
                print(f"laya: serving fine-tuned checkpoint {shown}")
                _router = laya.Router(
                    models={"english": (str(checkpoint), None)},
                    max_loaded=max_loaded,
                    dtype=dtype,
                    device=device,
                )
            if preload:
                _router.preload()
        return _router


def reset_router() -> None:
    """Drop the cached Router (tests / reconfiguration)."""
    global _router
    with _lock:
        _router = None
