"""Logistic-regression baseline (v0.1).

Fitted on the train split only. Calibration, threshold selection, and the
main LightGBM model arrive in v0.2 per docs/IMPLEMENTATION_ROADMAP.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from medrisk.data.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN

FEATURE_COLUMNS: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


def build_baseline_pipeline(seed: int, max_iter: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_baseline(train_df: pd.DataFrame, seed: int, max_iter: int) -> Pipeline:
    pipeline = build_baseline_pipeline(seed=seed, max_iter=max_iter)
    pipeline.fit(train_df[list(FEATURE_COLUMNS)], train_df[TARGET_COLUMN])
    return pipeline


def predict_probabilities(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipeline.predict_proba(df[list(FEATURE_COLUMNS)])[:, 1])
