"""Shared Laya-MLX router.

All three checkpoints together are ~1.16B parameters, so the process keeps one Router and
preloads the checkpoints it serves. `preload=True` pays the cold-load cost once at startup
instead of on the first request that switches language.
"""

from __future__ import annotations

import threading
from typing import Optional

import laya_mlx as laya

_router: Optional[laya.Router] = None
_lock = threading.Lock()


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
            _router = laya.Router(max_loaded=max_loaded, dtype=dtype, device=device)
            if preload:
                _router.preload()
        return _router


def reset_router() -> None:
    """Drop the cached Router (tests / reconfiguration)."""
    global _router
    with _lock:
        _router = None
