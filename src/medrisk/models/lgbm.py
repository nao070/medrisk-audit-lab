"""LightGBM main model (v0.2).

Hyperparameter search uses grouped cross-validation by ``patient_nbr``
within the train split only. Categorical features use LightGBM's native
handling; the category vocabulary is the fixed value set of each column
(schema knowledge, not a distribution fitted to outcomes).
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from medrisk.data.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

# Small fixed grid (versioned here; changing it is a preprocessing-adjacent change).
SEARCH_GRID: dict[str, list[Any]] = {
    "num_leaves": [31, 63],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
}
FIXED_PARAMS: dict[str, Any] = {
    "min_child_samples": 50,
    "class_weight": "balanced",
    "verbosity": -1,
}
CV_FOLDS = 3


def to_model_frame(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Cast categorical features to pandas category dtype for LightGBM."""
    X = df[list(feature_columns)].copy()
    for column in feature_columns:
        if column in CATEGORICAL_FEATURES:
            X[column] = X[column].astype(str).astype("category")
    return X


def search_lgbm(
    train_df: pd.DataFrame,
    seed: int,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Grouped-CV grid search on the train split; returns best params and CV log."""
    X = to_model_frame(train_df, feature_columns)
    y = train_df[TARGET_COLUMN].to_numpy()
    groups = train_df["patient_nbr"].to_numpy()

    results: list[dict[str, Any]] = []
    names = list(SEARCH_GRID)
    for combo in product(*(SEARCH_GRID[k] for k in names)):
        params = dict(zip(names, combo, strict=True))
        fold_aucs: list[float] = []
        for train_idx, valid_idx in GroupKFold(n_splits=CV_FOLDS).split(X, y, groups):
            model = LGBMClassifier(random_state=seed, **FIXED_PARAMS, **params)
            model.fit(X.iloc[train_idx], y[train_idx])
            prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
            fold_aucs.append(float(roc_auc_score(y[valid_idx], prob)))
        results.append(
            {"params": params, "cv_auroc_mean": float(np.mean(fold_aucs)), "cv_auroc": fold_aucs}
        )
        logger.info("CV %s -> AUROC %.4f", params, results[-1]["cv_auroc_mean"])

    best = max(results, key=lambda r: r["cv_auroc_mean"])
    return dict(best["params"]), results


def fit_lgbm(
    train_df: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> LGBMClassifier:
    model = LGBMClassifier(random_state=seed, **FIXED_PARAMS, **params)
    model.fit(to_model_frame(train_df, feature_columns), train_df[TARGET_COLUMN])
    return model


def predict_lgbm(
    model: LGBMClassifier,
    df: pd.DataFrame,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> np.ndarray:
    return np.asarray(model.predict_proba(to_model_frame(df, feature_columns))[:, 1])
