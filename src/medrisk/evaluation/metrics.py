"""Classification metrics (v0.1: point estimates on the validation split).

Bootstrap confidence intervals, calibration metrics, and threshold analysis
arrive in v0.2 per docs/IMPLEMENTATION_ROADMAP.md. The final test split is
never evaluated in v0.1; the one-shot final-test entry point is a v1.0
deliverable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

PROVISIONAL_THRESHOLD = 0.5


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = PROVISIONAL_THRESHOLD
) -> dict[str, Any]:
    """Compute the v0.1 metric set at a provisional threshold.

    The threshold is provisional: operating-threshold selection happens on
    the validation split in v0.2 and is frozen before any final-test use.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": len(y_true),
        "prevalence": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "threshold": {
            "value": float(threshold),
            "status": "provisional_default_v0_1",
        },
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
