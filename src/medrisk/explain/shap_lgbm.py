"""SHAP attributions for the LightGBM main model.

Uses LightGBM's built-in TreeSHAP (``pred_contrib=True``) — identical
algorithm to the `shap` package's TreeExplainer, without extra dependencies.

SHAP values are model attribution, not causal or clinical evidence. Every
surface that shows them must repeat that wording
(docs/DATA_AND_MODEL_PLAN.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from medrisk.models.lgbm import FEATURE_COLUMNS, to_model_frame

ATTRIBUTION_WORDING = (
    "SHAP values are model attributions on this dataset. They are not causal or clinical evidence."
)


def shap_matrix(
    model: LGBMClassifier,
    df: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Per-row SHAP values in log-odds space; last column (base value) dropped."""
    X = to_model_frame(df, feature_columns)
    contrib = np.asarray(model.predict_proba(X, pred_contrib=True))
    return contrib[:, :-1]


def global_importance(
    model: LGBMClassifier,
    df: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> list[dict[str, float]]:
    values = np.abs(shap_matrix(model, df, feature_columns)).mean(axis=0)
    order = np.argsort(values)[::-1]
    return [{"feature": feature_columns[i], "mean_abs_shap": float(values[i])} for i in order]


def top_attributions_for_row(
    shap_row: np.ndarray,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    k: int = 5,
) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(shap_row))[::-1][:k]
    return [
        {
            "feature": feature_columns[i],
            "direction": "increases" if shap_row[i] > 0 else "decreases",
            "rank": rank + 1,
        }
        for rank, i in enumerate(order)
    ]
