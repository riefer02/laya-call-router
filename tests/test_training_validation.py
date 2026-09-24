"""Tests for pure validation/checkpoint-selection rules."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "jev_training_validation", Path(__file__).resolve().parents[1] / "training" / "validation.py"
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
check_calibration_boundary = _module.check_calibration_boundary
macro_score = _module.macro_score
select_checkpoint = _module.select_checkpoint
summarise_rows = _module.summarise_rows


def test_summary_and_macro_score_weight_task_families_equally():
    rows = [
        {"task": "destination", "correct": 1, "nll": 1.0, "brier": 0.2},
        {"task": "destination", "correct": 0, "nll": 3.0, "brier": 0.8},
        {"task": "acceptance", "correct": 1, "nll": 0.0, "brier": 0.0},
    ]
    by_task = summarise_rows(rows)
    assert by_task["destination"]["accuracy"] == 0.5
    assert by_task["acceptance"]["nll"] == 0.0
    assert macro_score(by_task) == 1.0


def test_checkpoint_selection_uses_validation_score_and_task_guard():
    history = [
        {"epoch": 1, "macro_nll": 0.2, "by_task": {"destination": {}, "acceptance": {}}},
        {"epoch": 2, "macro_nll": 0.1, "by_task": {"destination": {}}},
        {"epoch": 3, "macro_nll": 0.3, "by_task": {"destination": {}, "acceptance": {}}},
    ]
    chosen = select_checkpoint(history, required_tasks=("destination", "acceptance"))
    assert chosen["epoch"] == 1


def test_calibration_overlap_is_rejected():
    item = {"ids": [1, 2, 3], "task": "destination"}
    with pytest.raises(ValueError, match="overlaps"):
        check_calibration_boundary([item], [item])
