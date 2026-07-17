"""Calibration quality: ECE, reliability curves, slope and intercept."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

N_BINS = 10
_EPS = 1e-6


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> list[dict[str, float]]:
    from itertools import pairwise

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    for lo, hi in pairwise(edges):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        bins.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "n": float(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
            }
        )
    return bins


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> float:
    bins = reliability_bins(y_true, y_prob, n_bins)
    total = sum(b["n"] for b in bins)
    return float(sum(b["n"] * abs(b["mean_predicted"] - b["observed_rate"]) for b in bins) / total)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Logistic recalibration fit: slope 1 / intercept 0 means well calibrated."""
    logit = np.log(np.clip(y_prob, _EPS, 1 - _EPS) / np.clip(1 - y_prob, _EPS, 1 - _EPS))
    # C=np.inf == unpenalized fit; replaces penalty=None (deprecated in sklearn 1.8)
    model = LogisticRegression(C=np.inf, max_iter=1000)
    model.fit(logit.reshape(-1, 1), y_true)
    return {
        "slope": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
    }


def calibration_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    return {
        "ece": expected_calibration_error(y_true, y_prob),
        **calibration_slope_intercept(y_true, y_prob),
        "reliability_bins": reliability_bins(y_true, y_prob),
    }
