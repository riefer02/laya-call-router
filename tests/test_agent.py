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


def test_base_evaluation_router_ignores_the_active_finetune(monkeypatch):
    import laya_mlx as laya

    seen = []

    class FakeRouter:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    monkeypatch.setattr(laya, "Router", FakeRouter)
    monkeypatch.setenv("JEV_MODEL", "models/active")
    agent.new_base_router()
    assert seen == [{"max_loaded": 2}]


def test_the_evidence_tab_reads_the_served_checkpoints_reports():
    """It read hardcoded `eval_v4.json` and `severity.json`.

    So it showed v4's routing numbers and the *pre-training* safety numbers while the app served v7 -
    a demo whose evidence contradicts its own behaviour, and nothing flagged it because both files
    exist and both parse. The tag now comes from the served checkpoint.
    """
    import json as _json
    from pathlib import Path as _Path

    from jev_classifier import agent, api

    ckpt = agent.resolve_checkpoint()
    if ckpt is None:
        pytest.skip("no checkpoint selected")
    tag = api._results_tag()
    if not tag:
        pytest.skip(f"served checkpoint {ckpt} is not a kaggle-out-vN run")

    # Whatever the tab resolves to, it must be that checkpoint's report if one exists.
    results = _Path(__file__).resolve().parents[1] / "results"
    expected = results / f"eval_{tag}.json"
    if expected.is_file():
        payload = api.results()
        assert payload["sources"]["routing"] == f"eval_{tag}.json"
        assert payload["sources"]["checkpoint"] == tag
        reported = next(a for a in payload["arms"] if a["key"] == "cascade-ft")
        on_disk = _json.loads(expected.read_text())["routing"]["cascade-ft"]
        assert abs(reported["joint"] - on_disk["joint_accuracy"]) < 1e-9, (
            "the Evidence tab and the served checkpoint's report disagree"
        )


def test_reference_report_is_not_presented_as_the_loaded_model(monkeypatch):
    """The stock base model must not inherit the bundled fine-tune's report."""
    from jev_classifier import api

    monkeypatch.setattr(api, "_results_tag", lambda: "")
    payload = api.results()
    assert payload["sources"]["checkpoint"] == "base"
    assert payload["sources"]["matches_served_model"] is False
    assert payload["sources"]["routing"] == "eval_v7.json"
