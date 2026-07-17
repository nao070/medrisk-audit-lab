"""Uncertainty-based abstention.

The system may answer "no reliable prediction" when the calibrated
probability falls inside an uncertainty band around the frozen threshold.
The band is selected on the validation split only: among candidate bands
with coverage >= MIN_COVERAGE, pick the one maximizing selective F1
(docs/DATA_AND_MODEL_PLAN.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from medrisk.evaluation.metrics import compute_classification_metrics

MIN_COVERAGE = 0.80
BAND_GRID = tuple(np.round(np.linspace(0.0, 0.10, 21), 4))
SELECTION_RULE = f"max selective F1 subject to coverage >= {MIN_COVERAGE}"


def abstain_mask(y_prob: np.ndarray, threshold: float, band: float) -> np.ndarray:
    """True where the model abstains."""
    return np.abs(y_prob - threshold) < band


def band_sweep(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for band in BAND_GRID:
        answered = ~abstain_mask(y_prob, threshold, float(band))
        coverage = float(answered.mean())
        if answered.sum() == 0 or len(np.unique(y_true[answered])) < 2:
            continue
        m = compute_classification_metrics(y_true[answered], y_prob[answered], threshold)
        rows.append(
            {
                "band": float(band),
                "coverage": coverage,
                "abstention_rate": 1.0 - coverage,
                "selective_f1": m["f1"],
                "selective_recall": m["recall"],
                "selective_precision": m["precision"],
                "selective_specificity": m["specificity"],
                "selective_auroc": m["auroc"],
                "selective_brier": m["brier_score"],
            }
        )
    return rows


def select_band(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    sweep = band_sweep(y_true, y_prob, threshold)
    eligible = [r for r in sweep if r["coverage"] >= MIN_COVERAGE]
    best = max(eligible, key=lambda r: r["selective_f1"])
    return {"rule": SELECTION_RULE, "selected": best, "sweep": sweep}
