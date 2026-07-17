"""Threshold analysis and selection.

The operating threshold is chosen on the validation split only and frozen
before any final-test use (docs/DATA_AND_MODEL_PLAN.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np

SELECTION_CRITERION = "max_f1_on_validation"


def threshold_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_points: int = 99
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.01, 0.99, n_points):
        y_pred = (y_prob >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "f1": 2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0,
                "flagged_rate": float((y_pred == 1).mean()),
            }
        )
    return rows


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    curve = threshold_curve(y_true, y_prob)
    best = max(curve, key=lambda r: r["f1"])
    return {
        "criterion": SELECTION_CRITERION,
        "selected_threshold": best["threshold"],
        "at_selection": best,
    }
