"""Measure raw-logit quality of a local Laya checkpoint without retraining.

This is a Phase-A diagnostic, not a final benchmark. It reports probability quality on the legacy
routing set, option-order stability, and what the current checkpoint does with the new scope
question. It never calls a hosted model.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import laya_mlx as laya

from jev_classifier import dealership as D
from jev_classifier import evalharness as H
from jev_classifier.diagnostics import (
    classification_metrics,
    free_energy,
    logit_margin,
    permutation_predictions,
    raw_question_logits,
    softmax,
)

ROOT = Path(__file__).resolve().parents[1]


def _portable_path(value: str | Path) -> str:
    """Prefer repository-relative paths in generated provenance reports."""
    path = Path(value)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _labels(question: Dict[str, Any]) -> List[str]:
    return list(question.get("criteria", {}))


def _metrics(rows):
    return classification_metrics(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "calls" / "routing.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "diagnostics_logits.json"))
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    cases = H.load_routing(Path(args.data))
    if args.limit:
        cases = cases[: args.limit]

    agent = laya.load(str(checkpoint), dtype="float16")
    destination_question = D.DESTINATION_QUESTION["destination"]
    destination_labels = _labels(destination_question)
    destination_rows = []
    subqueue_rows = []
    end_to_end_subqueue_correct = []
    joint_correct = []
    permutation_failures = []
    scope_counts = Counter()
    margins = []
    energies = []

    for case in cases:
        state = {"call": case.text}
        destination_logits = raw_question_logits(agent, state, destination_question)
        destination_p = softmax(destination_logits)
        gold = destination_labels.index(case.destination)
        predicted_destination = destination_labels[int(destination_p.argmax())]
        destination_rows.append((destination_p, gold))
        margins.append(logit_margin(destination_logits))
        energies.append(free_energy(destination_logits))

        question = D.subqueue_question(case.destination)
        if question and case.subqueue:
            sub_labels = _labels(question["subqueue"])
            sub_logits = raw_question_logits(agent, state, question["subqueue"])
            subqueue_rows.append((softmax(sub_logits), sub_labels.index(case.subqueue)))

        predicted_question = D.subqueue_question(predicted_destination)
        if predicted_question and case.subqueue:
            predicted_labels = _labels(predicted_question["subqueue"])
            predicted_logits = raw_question_logits(agent, state, predicted_question["subqueue"])
            predicted_subqueue = predicted_labels[int(softmax(predicted_logits).argmax())]
            subqueue_ok = predicted_subqueue == case.subqueue
            end_to_end_subqueue_correct.append(subqueue_ok)
            joint_correct.append(predicted_destination == case.destination and subqueue_ok)
        elif case.subqueue is None:
            joint_correct.append(predicted_destination == case.destination)

        # The Phase-A question is not in the v7 training set. We record its answer rather than
        # pretending it is a valid v7 metric; this tells us whether a future run has a sane starting point.
        scope_logits = raw_question_logits(agent, state, D.SCOPE_QUESTION["scope"])
        scope_labels = _labels(D.SCOPE_QUESTION["scope"])
        scope_counts[scope_labels[int(softmax(scope_logits).argmax())]] += 1

        n = len(destination_labels)
        identity = list(range(n))
        reverse = list(reversed(identity))
        permuted = permutation_predictions(agent, state, destination_question, [identity, reverse])
        predictions = {row["prediction"] for row in permuted}
        if len(predictions) > 1:
            permutation_failures.append(
                {"id": case.id, "expected": case.destination, "predictions": sorted(predictions)}
            )

    report = {
        "checkpoint": _portable_path(checkpoint),
        "data": _portable_path(args.data),
        "n_cases": len(cases),
        "destination": _metrics(destination_rows),
        "subqueue_teacher_forced_gold_destination": _metrics(subqueue_rows) if subqueue_rows else {"n": 0},
        "end_to_end_subqueue_accuracy": (
            sum(end_to_end_subqueue_correct) / len(end_to_end_subqueue_correct)
            if end_to_end_subqueue_correct
            else None
        ),
        "joint_accuracy": sum(joint_correct) / len(joint_correct) if joint_correct else None,
        "mean_logit_margin": sum(margins) / len(margins) if margins else None,
        "mean_free_energy": sum(energies) / len(energies) if energies else None,
        "option_order": {
            "destination_cases_changed": len(permutation_failures),
            "destination_consistency": (
                1 - len(permutation_failures) / len(cases) if cases else None
            ),
            "failures": permutation_failures[:20],
        },
        "scope_question_on_legacy_checkpoint": {
            "counts": dict(scope_counts),
            "warning": "v7 was not trained on scope; this is a diagnostic, not a score.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
