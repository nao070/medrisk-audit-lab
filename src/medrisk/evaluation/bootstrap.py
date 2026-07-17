"""Patient-level bootstrap confidence intervals.

Patients (not encounters) are resampled with replacement, respecting
within-patient correlation (docs/DATA_AND_MODEL_PLAN.md).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

N_REPLICATES = 1000
CI_LEVEL = 0.95


def patient_bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], Mapping[str, float]],
    n_replicates: int = N_REPLICATES,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Bootstrap CIs for every scalar returned by ``metric_fn``.

    ``df`` must contain a ``patient_nbr`` column; ``metric_fn`` receives a
    resampled frame and returns a flat mapping of metric name to value.
    Replicates where a metric is undefined (single-class resample) are
    skipped for that metric.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby("patient_nbr", sort=True).indices
    patient_ids = np.array(sorted(int(pid) for pid in groups))  # type: ignore[call-overload]
    row_index = {int(pid): np.asarray(rows) for pid, rows in groups.items()}  # type: ignore[call-overload]

    samples: dict[str, list[float]] = {}
    for _ in range(n_replicates):
        chosen = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        rows = np.concatenate([row_index[pid] for pid in chosen])
        try:
            values = metric_fn(df.iloc[rows])
        except ValueError:
            continue
        for name, value in values.items():
            if np.isfinite(value):
                samples.setdefault(name, []).append(float(value))

    alpha = (1.0 - CI_LEVEL) / 2.0
    point = metric_fn(df)
    out: dict[str, dict[str, float]] = {}
    for name, value in point.items():
        draws = samples.get(name, [])
        entry: dict[str, float] = {"value": float(value)}
        if len(draws) >= 100:
            entry["ci_low"] = float(np.quantile(draws, alpha))
            entry["ci_high"] = float(np.quantile(draws, 1.0 - alpha))
            entry["n_bootstrap"] = float(len(draws))
        out[name] = entry
    return out


def flat_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    """Scalar-only view of the standard metric set, for bootstrapping."""
    from medrisk.evaluation.calibration import expected_calibration_error
    from medrisk.evaluation.metrics import compute_classification_metrics

    if len(np.unique(y_true)) < 2:
        raise ValueError("resample contains a single class")
    m = compute_classification_metrics(y_true, y_prob, threshold)
    return {
        "auroc": m["auroc"],
        "auprc": m["auprc"],
        "brier_score": m["brier_score"],
        "recall": m["recall"],
        "specificity": m["specificity"],
        "precision": m["precision"],
        "f1": m["f1"],
        "ece": expected_calibration_error(y_true, y_prob),
    }


def metrics_with_ci(
    df: pd.DataFrame,
    prob_column: str,
    target_column: str,
    threshold: float,
    seed: int,
    n_replicates: int = N_REPLICATES,
) -> dict[str, Any]:
    def metric_fn(frame: pd.DataFrame) -> Mapping[str, float]:
        return flat_metrics(
            frame[target_column].to_numpy(), frame[prob_column].to_numpy(), threshold
        )

    return patient_bootstrap_ci(df, metric_fn, n_replicates=n_replicates, seed=seed)
