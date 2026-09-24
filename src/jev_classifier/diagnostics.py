"""Local, no-training diagnostics for typed-decision checkpoints.

The public Laya MLX API rounds probabilities and returns a legacy confidence field. These helpers
use the already-computed logits to measure the properties the application actually needs: proper
scores, calibration, error ranking, option-order stability, and simple OOD signals.

Nothing here calls a hosted model or changes the router. It is intentionally usable against the
bundled v7 checkpoint before spending a GPU run.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def softmax(logits: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - np.max(z)
    p = np.exp(z)
    return p / p.sum()


def answer_confidence(probabilities: Sequence[float]) -> float:
    """The upstream calibration quantity: max(p), not legacy entropy confidence."""
    return float(np.max(np.asarray(probabilities, dtype=np.float64)))


def logit_margin(logits: Sequence[float]) -> float:
    z = np.sort(np.asarray(logits, dtype=np.float64))[-2:]
    return float(z[1] - z[0]) if len(z) > 1 else 0.0


def free_energy(logits: Sequence[float], temperature: float = 1.0) -> float:
    """Standard free-energy OOD score; larger means less typical under the model."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    m = np.max(z)
    return float(-float(temperature) * (m + math.log(np.exp(z - m).sum())))


def ece(probabilities: Sequence[float], correct: bool, bins: int = 10) -> float:
    """Top-label expected calibration error for one observation.

    This helper is used with a fixed binning rule by the batch metric below. It is kept separate
    so the per-example report can be checked without relying on a plotting library.
    """
    p = answer_confidence(probabilities)
    idx = min(bins - 1, max(0, int(p * bins)))
    # The caller supplies the bin boundaries through `bins`; this single-example form is only a
    # convenience for tests. Batch calibration is implemented in `classification_metrics`.
    return abs(p - float(correct))


def fit_temperature(
    rows: Iterable[Tuple[Sequence[float], int]], lo: float = 0.2, hi: float = 10.0, steps: int = 160
) -> float:
    """Fit one scalar temperature by held-out NLL using a deterministic grid.

    The upstream notebook uses a differentiable fit. This small version is for local diagnostics
    and tests; it makes no claim that the grid is the production calibrator.
    """
    materialised = [(np.asarray(z, dtype=np.float64), int(g)) for z, g in rows]
    if len(materialised) < 25:
        return 1.0
    best_t, best_nll = 1.0, float("inf")
    for temperature in np.geomspace(lo, hi, steps):
        nll = 0.0
        for logits, gold in materialised:
            p = softmax(logits, temperature)
            nll -= math.log(max(float(p[gold]), 1e-12))
        if nll < best_nll:
            best_nll, best_t = nll, float(temperature)
    return round(best_t, 4)


def classification_metrics(
    rows: Iterable[Tuple[Sequence[float], int]], bins: int = 10
) -> Dict[str, float]:
    """Compute classification and probability-quality metrics from raw probability rows.

    ``rows`` contains ``(probabilities, gold_index)``. Metrics that need no gold include margin and
    energy, but those are calculated from logits by :func:`raw_question_logits`; this function is
    intentionally independent of Laya so it is cheap to test.
    """
    materialised = [(np.asarray(p, dtype=np.float64), int(gold)) for p, gold in rows]
    if not materialised:
        return {"n": 0}
    n = len(materialised)
    accuracy = sum(int(np.argmax(p) == g) for p, g in materialised) / n
    nll = -sum(math.log(max(float(p[g]), 1e-12)) for p, g in materialised) / n
    brier = sum(float(np.sum((p - np.eye(len(p))[g]) ** 2)) for p, g in materialised) / n
    conf = np.asarray([answer_confidence(p) for p, _ in materialised])
    correct = np.asarray([int(np.argmax(p) == g) for p, g in materialised], dtype=float)
    # Equal-width top-label bins, with the final bin including confidence == 1.
    ece_value = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        mask = (conf >= lo) & ((conf < hi) if index < bins - 1 else (conf <= hi))
        if mask.any():
            ece_value += float(mask.mean()) * abs(float(conf[mask].mean()) - float(correct[mask].mean()))
    return {
        "n": n,
        "accuracy": round(accuracy, 6),
        "nll": round(nll, 6),
        "brier": round(brier, 6),
        "ece": round(ece_value, 6),
        "mean_confidence": round(float(conf.mean()), 6),
    }


def raw_question_logits(agent: Any, state: Any, question: Mapping[str, Any], qid: str = "q") -> np.ndarray:
    """Return the finite option logits for one question from an ``laya_mlx.Agent``.

    ``Agent.forward`` already returns logits internally; the public ``system_one`` method converts
    them to rounded probabilities. This deliberately uses the internal preparation path so the
    diagnostic measures the model's ranking, not the presentation rounding.
    """
    from laya_mlx.agent import collate_items

    prepared, _ = agent.prepare(state, {qid: dict(question)})
    if len(prepared) != 1:
        raise ValueError("raw_question_logits expects exactly one question")
    batch = collate_items(
        prepared,
        agent.tok.pad_token_id,
        pad_to_multiple=getattr(agent, "pad_to_multiple", None),
        max_length=agent.cfg.get("max_len", 512),
    )
    logits, _ = agent.forward(batch)
    values = np.asarray(logits, dtype=np.float64)[0]
    k = len(prepared[0]["markers"])
    return values[:k]


def permutation_predictions(
    agent: Any,
    state: Any,
    question: Mapping[str, Any],
    permutations: Sequence[Sequence[int]],
) -> List[Dict[str, Any]]:
    """Run a choice question under option-order permutations and map answers back to labels."""
    labels = list(question.get("criteria", {}))
    if not labels:
        raise ValueError("permutation diagnostics require a choice question")
    rows: List[Dict[str, Any]] = []
    for permutation in permutations:
        if sorted(permutation) != list(range(len(labels))):
            raise ValueError("each option permutation must contain every option exactly once")
        ordered = {labels[i]: question["criteria"][labels[i]] for i in permutation}
        permuted = dict(question)
        permuted["criteria"] = ordered
        logits = raw_question_logits(agent, state, permuted)
        probs = softmax(logits)
        mapped = {labels[int(i)]: float(probs[pos]) for pos, i in enumerate(permutation)}
        rows.append(
            {
                "permutation": list(permutation),
                "prediction": max(mapped, key=mapped.get),
                "probabilities": mapped,
                "margin": logit_margin(logits),
            }
        )
    return rows
