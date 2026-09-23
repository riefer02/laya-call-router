"""Which checkpoint the app actually serves.

`laya.Router()` with no `models` argument falls back to `convaiinnovations/laya` — the stock
checkpoint that scores 0.654 destination against our fine-tune's 0.963. The app served it, the
visualizer looked fine, and nothing anywhere said so. Found while checking whether the demo would
work, which is a bad time to find it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_classifier import agent

ROOT = Path(__file__).resolve().parents[1]


def _fake_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "rl_agent_config.json").write_text(json.dumps({"encoder": "x", "head_layers": 2}))
    return path


def test_an_explicit_model_wins(tmp_path, monkeypatch):
    ckpt = _fake_checkpoint(tmp_path / "ckpt")
    monkeypatch.setenv("JEV_MODEL", str(ckpt))
    assert agent.resolve_checkpoint() == ckpt


def test_a_relative_model_path_resolves_against_the_repo(tmp_path, monkeypatch):
    ckpt = _fake_checkpoint(ROOT / "models" / "_test_ckpt")
    monkeypatch.setenv("JEV_MODEL", "models/_test_ckpt")
    try:
        assert agent.resolve_checkpoint() == ckpt
    finally:
        (ckpt / "rl_agent_config.json").unlink()
        ckpt.rmdir()


def test_a_model_path_that_is_not_a_checkpoint_fails_loudly(tmp_path, monkeypatch):
    """Silently falling back to the base model is the failure this whole module exists to prevent."""
    monkeypatch.setenv("JEV_MODEL", str(tmp_path))
    with pytest.raises(SystemExit):
        agent.resolve_checkpoint()


def test_active_is_used_when_nothing_is_named(tmp_path, monkeypatch):
    ckpt = _fake_checkpoint(tmp_path / "active")
    monkeypatch.delenv("JEV_MODEL", raising=False)
    monkeypatch.setattr(agent, "ACTIVE", ckpt)
    assert agent.resolve_checkpoint() == ckpt


def test_with_neither_it_returns_none_so_the_caller_can_warn(tmp_path, monkeypatch):
    monkeypatch.delenv("JEV_MODEL", raising=False)
    monkeypatch.setattr(agent, "ACTIVE", tmp_path / "nothing-here")
    assert agent.resolve_checkpoint() is None


def test_the_router_is_pointed_at_the_checkpoint_when_one_resolves(monkeypatch):
    """The regression that matters: a Router built without `models` serves the base checkpoint."""
    import laya_mlx as laya

    seen = {}

    class FakeRouter:
        def __init__(self, models=None, **kw):
            seen["models"] = models

    monkeypatch.setattr(laya, "Router", FakeRouter)
    monkeypatch.setattr(agent, "resolve_checkpoint", lambda: Path("/tmp/pretend-checkpoint"))
    agent.reset_router()
    agent.get_router()
    agent.reset_router()
    assert seen["models"] == {"english": ("/tmp/pretend-checkpoint", None)}
