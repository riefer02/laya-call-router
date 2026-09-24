"""Pure tests for the no-training checkpoint diagnostics."""

from __future__ import annotations

import math

import numpy as np

from jev_classifier.diagnostics import (
    answer_confidence,
    classification_metrics,
    fit_temperature,
    free_energy,
    logit_margin,
    permutation_predictions,
    softmax,
)


def test_softmax_and_confidence_use_max_probability():
    p = softmax([3.0, 1.0, 0.0])
    assert math.isclose(float(p.sum()), 1.0)
    assert answer_confidence(p) == float(p.max())
    assert answer_confidence([0.2, 0.8]) == 0.8


def test_margin_and_energy_are_diagnostic_scores():
    assert logit_margin([4.0, 1.0, 0.0]) == 3.0
    typical = free_energy([4.0, 1.0, 0.0])
    odd = free_energy([0.0, 0.0, 0.0])
    assert isinstance(typical, float)
    assert isinstance(odd, float)


def test_temperature_fit_can_sharpen_a_deterministic_logit_set():
    rows = [([2.0, 0.0], 0), ([0.0, 2.0], 1)] * 20
    assert fit_temperature(rows) < 1.0


def test_classification_metrics_report_nll_brier_and_ece():
    got = classification_metrics([([0.9, 0.1], 0), ([0.2, 0.8], 1)], bins=2)
    assert got["n"] == 2
    assert got["accuracy"] == 1.0
    assert got["nll"] > 0
    assert got["brier"] > 0
    assert 0 <= got["ece"] <= 1


def test_permutation_diagnostic_maps_answers_back_to_original_labels():
    class FakeAgent:
        class Tok:
            pad_token_id = 0

        tok = Tok()
        pad_to_multiple = None
        cfg = {"max_len": 512}

        def prepare(self, state, questions):
            question = next(iter(questions.values()))
            labels = list(question["criteria"])
            return [{"ids": [1], "markers": [0] * len(labels), "qtype": 0}], [question]

        def forward(self, batch):
            # The first option in each supplied ordering gets the highest logit.
            n = batch["marker_mask"].shape[0]
            k = batch["marker_mask"].shape[1]
            logits = np.zeros((n, k), dtype=np.float32)
            logits[:, 0] = 4.0
            return logits, np.zeros((n, 1), dtype=np.float32)

    rows = permutation_predictions(
        FakeAgent(),
        "text",
        {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}},
        [[0, 1], [1, 0]],
    )
    assert [row["prediction"] for row in rows] == ["a", "b"]
    assert rows[0]["probabilities"]["a"] > rows[0]["probabilities"]["b"]
    assert rows[1]["probabilities"]["b"] > rows[1]["probabilities"]["a"]
