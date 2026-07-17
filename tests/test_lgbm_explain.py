import numpy as np
import pandas as pd

from conftest import make_raw_frame
from medrisk.data.preprocess import TARGET_COLUMN, build_cohort, encode_features
from medrisk.explain.shap_lgbm import (
    global_importance,
    shap_matrix,
    top_attributions_for_row,
)
from medrisk.models.lgbm import FEATURE_COLUMNS, fit_lgbm, predict_lgbm

TEST_PARAMS = {"num_leaves": 7, "learning_rate": 0.2, "n_estimators": 20}


def _cohort() -> pd.DataFrame:
    cohort, _ = build_cohort(make_raw_frame(n_rows=600, seed=5))
    return encode_features(cohort)


def test_lgbm_fits_and_predicts_probabilities() -> None:
    df = _cohort()
    model = fit_lgbm(df, TEST_PARAMS, seed=0)
    prob = predict_lgbm(model, df)
    assert prob.shape == (len(df),)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_shap_matrix_shape_and_additivity() -> None:
    df = _cohort()
    model = fit_lgbm(df, TEST_PARAMS, seed=0)
    values = shap_matrix(model, df)
    assert values.shape == (len(df), len(FEATURE_COLUMNS))
    # TreeSHAP contributions plus base value reproduce the raw score.
    from medrisk.models.lgbm import to_model_frame

    contrib = np.asarray(
        model.predict_proba(to_model_frame(df, FEATURE_COLUMNS), pred_contrib=True)
    )
    raw = contrib.sum(axis=1)
    prob = predict_lgbm(model, df)
    np.testing.assert_allclose(1 / (1 + np.exp(-raw)), prob, atol=1e-6)


def test_global_importance_is_sorted_and_covers_features() -> None:
    df = _cohort()
    model = fit_lgbm(df, TEST_PARAMS, seed=0)
    importance = global_importance(model, df)
    values = [i["mean_abs_shap"] for i in importance]
    assert values == sorted(values, reverse=True)
    assert {i["feature"] for i in importance} == set(FEATURE_COLUMNS)


def test_top_attributions_have_rank_and_direction_only() -> None:
    row = np.array([0.5, -1.2, 0.1, 0.0, 2.0])
    top = top_attributions_for_row(row, feature_columns=("a", "b", "c", "d", "e"), k=3)
    assert [t["feature"] for t in top] == ["e", "b", "a"]
    assert top[0]["direction"] == "increases"
    assert top[1]["direction"] == "decreases"
    # No raw SHAP magnitudes are exposed (the LLM later receives ranks only).
    assert all(set(t) == {"feature", "direction", "rank"} for t in top)


def test_ablation_feature_can_be_dropped() -> None:
    df = _cohort()
    reduced = tuple(f for f in FEATURE_COLUMNS if f != "discharge_disposition_id")
    model = fit_lgbm(df, TEST_PARAMS, seed=0, feature_columns=reduced)
    prob = predict_lgbm(model, df, feature_columns=reduced)
    assert prob.shape == (len(df),)


def test_target_never_in_lgbm_features() -> None:
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert "patient_nbr" not in FEATURE_COLUMNS
    assert "encounter_id" not in FEATURE_COLUMNS
