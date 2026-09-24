"""Pure checkpoint-selection helpers for the Kaggle trainer.

The GPU loop records per-task validation rows; this module makes the selection rule explicit and
unit-testable without importing Torch. It deliberately selects on validation proper loss, not on
training loss or ECE alone.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def summarise_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-task validation rows.

    Each row must contain ``task``, ``correct`` (0/1), ``nll`` and ``brier``. A task with no rows
    is omitted rather than represented by an optimistic zero.
    """
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task"]), []).append(row)
    out: Dict[str, Dict[str, float]] = {}
    for task, values in sorted(grouped.items()):
        n = len(values)
        out[task] = {
            "n": n,
            "accuracy": sum(float(v["correct"]) for v in values) / n,
            "nll": sum(float(v["nll"]) for v in values) / n,
            "brier": sum(float(v["brier"]) for v in values) / n,
        }
    return out


def macro_score(by_task: Mapping[str, Mapping[str, float]], metric: str = "nll") -> Optional[float]:
    """Equal-weight mean across task families, not across expanded row counts."""
    values = [float(task[metric]) for task in by_task.values() if metric in task and task.get("n")]
    return sum(values) / len(values) if values else None


def select_checkpoint(
    history: Sequence[Mapping[str, Any]],
    *,
    metric: str = "macro_nll",
    lower_is_better: bool = True,
    required_tasks: Sequence[str] = (),
) -> Optional[Mapping[str, Any]]:
    """Select the best eligible epoch, returning the original history row.

    ``required_tasks`` is a guardrail for multi-task training: an epoch missing a task is not
    eligible even if its aggregate score looks good.
    """
    eligible = []
    required = set(required_tasks)
    for row in history:
        tasks = row.get("by_task") or {}
        if required and not required <= set(tasks):
            continue
        value = row.get(metric)
        if value is None:
            continue
        eligible.append(row)
    if not eligible:
        return None
    return min(eligible, key=lambda row: float(row[metric]) if lower_is_better else -float(row[metric]))


def check_calibration_boundary(
    train_items: Sequence[Mapping[str, Any]], calibration_items: Sequence[Mapping[str, Any]]
) -> None:
    """Reject an accidental calibration set that overlaps training rows."""
    def key(item: Mapping[str, Any]) -> Tuple[Any, ...]:
        ids = item.get("ids")
        return (tuple(ids) if ids is not None else item.get("case_id"), item.get("task"))

    train_keys = {key(item) for item in train_items}
    overlap = train_keys & {key(item) for item in calibration_items}
    if overlap:
        raise ValueError(f"calibration data overlaps training data in {len(overlap)} item(s)")
