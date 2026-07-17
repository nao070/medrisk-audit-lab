"""v0.2 stages: main-model training, calibration, ablation, explainability.

Split usage (docs/DATA_AND_MODEL_PLAN.md):
- train: grouped-CV hyperparameter search and final fit
- calibration: calibrator fitting only
- validation: calibration-method selection, threshold selection, reporting
- final test: never touched here
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from medrisk.config import AppConfig
from medrisk.data.download import sha256_of_file
from medrisk.data.preprocess import PREPROCESSING_VERSION, TARGET_COLUMN
from medrisk.data.splits import assign_rows, load_splits, split_hash
from medrisk.evaluation.bootstrap import metrics_with_ci
from medrisk.evaluation.calibration import calibration_report, expected_calibration_error
from medrisk.evaluation.metrics import compute_classification_metrics
from medrisk.evaluation.thresholds import select_threshold, threshold_curve
from medrisk.explain.shap_lgbm import (
    ATTRIBUTION_WORDING,
    global_importance,
    shap_matrix,
    top_attributions_for_row,
)
from medrisk.models.calibrate import METHODS, Calibrator, fit_calibrator
from medrisk.models.lgbm import FEATURE_COLUMNS, fit_lgbm, predict_lgbm, search_lgbm
from medrisk.reporting.manifest import build_manifest, write_manifest
from medrisk.reporting.plots import (
    save_bar,
    save_reliability,
    save_roc_pr,
    save_threshold_curves,
)

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("artifacts")
MODEL_ARTIFACT = ARTIFACT_DIR / "main_model_v1.joblib"
ABLATION_FEATURE = "discharge_disposition_id"


def _load_frames(config: AppConfig) -> tuple[pd.DataFrame, dict[str, list[int]], pd.Series]:
    processed = pd.read_csv(config.processed_dir / "cohort_v1.csv")
    splits = load_splits(config.split.out_dir)
    return processed, splits, assign_rows(processed, splits)


def _dataset_sha(config: AppConfig) -> str:
    archive = config.dataset.raw_dir / "diabetes_130_us_hospitals_1999_2008.zip"
    return sha256_of_file(archive) if archive.exists() else "absent"


def run_train_main(config: AppConfig) -> None:
    processed, splits, assignment = _load_frames(config)
    train_df = processed.loc[assignment == "train"]
    calib_df = processed.loc[assignment == "calibration"]
    valid_df = processed.loc[assignment == "validation"].reset_index(drop=True)

    best_params, cv_log = search_lgbm(train_df, seed=config.model.seed)
    model = fit_lgbm(train_df, best_params, seed=config.model.seed)

    prob_calib_raw = predict_lgbm(model, calib_df)
    prob_valid_raw = predict_lgbm(model, valid_df)
    y_calib = calib_df[TARGET_COLUMN].to_numpy()
    y_valid = valid_df[TARGET_COLUMN].to_numpy()

    # Calibration-method selection on validation (Brier, tie -> ECE).
    candidates: dict[str, Calibrator] = {
        method: fit_calibrator(method, prob_calib_raw, y_calib) for method in METHODS
    }
    scores: dict[str, dict[str, float]] = {}
    for method, calibrator in candidates.items():
        prob = calibrator.transform(prob_valid_raw)
        scores[method] = {
            "brier": float(np.mean((prob - y_valid) ** 2)),
            "ece": expected_calibration_error(y_valid, prob),
        }
    chosen_method = min(scores, key=lambda m: (scores[m]["brier"], scores[m]["ece"]))
    calibrator = candidates[chosen_method]
    prob_valid = calibrator.transform(prob_valid_raw)

    threshold_info = select_threshold(y_valid, prob_valid)
    threshold = float(threshold_info["selected_threshold"])

    valid_eval = valid_df[["patient_nbr"]].copy()
    valid_eval["y"] = y_valid
    valid_eval["prob"] = prob_valid
    ci = metrics_with_ci(valid_eval, "prob", "y", threshold=threshold, seed=config.split.seed)
    point = compute_classification_metrics(y_valid, prob_valid, threshold)
    point["threshold"] = {"value": threshold, "status": "frozen_on_validation_v0_2"}

    # Model comparison (threshold-free metrics) against the v0.1 LR baseline.
    from medrisk.models.baseline import predict_probabilities, train_baseline

    lr = train_baseline(train_df, seed=config.model.seed, max_iter=config.model.max_iter)
    prob_valid_lr = predict_probabilities(lr, valid_df)
    comparison = {
        "logistic_regression": _threshold_free(y_valid, prob_valid_lr),
        "lgbm_uncalibrated": _threshold_free(y_valid, prob_valid_raw),
        f"lgbm_{chosen_method}": _threshold_free(y_valid, prob_valid),
    }
    # Calibration is monotone (AUROC-preserving), so the calibrated LGBM is
    # preferred whenever LGBM beats the LR baseline on discrimination.
    lgbm_name = f"lgbm_{chosen_method}"
    main_model_name = (
        lgbm_name
        if comparison[lgbm_name]["auroc"] >= comparison["logistic_regression"]["auroc"]
        else "logistic_regression"
    )

    reports = config.reports_dir
    (reports / "model_selection").mkdir(parents=True, exist_ok=True)
    (reports / "metrics").mkdir(parents=True, exist_ok=True)
    selection = {
        "main_model": main_model_name,
        "lgbm_params": best_params,
        "cv_log": cv_log,
        "calibration_method": chosen_method,
        "calibration_scores_validation": scores,
        "threshold": threshold_info,
        "split_sha256": split_hash(splits),
        "seed": config.model.seed,
        "selection_data": "validation split only; final test untouched",
    }
    (reports / "model_selection" / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    (reports / "metrics" / "validation_main_v0_2.json").write_text(
        json.dumps(
            {
                "model": main_model_name,
                "point_estimates": point,
                "bootstrap_ci_95": ci,
                "calibration": calibration_report(y_valid, prob_valid),
            },
            indent=2,
        )
        + "\n"
    )
    (reports / "metrics" / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )

    figures = reports / "figures"
    save_roc_pr(y_valid, prob_valid, figures / "roc_pr_main.png", "(validation, calibrated)")
    save_reliability(
        calibration_report(y_valid, prob_valid_raw)["reliability_bins"],
        calibration_report(y_valid, prob_valid)["reliability_bins"],
        figures / "reliability_main.png",
        labels=("uncalibrated", chosen_method),
    )
    save_threshold_curves(
        threshold_curve(y_valid, prob_valid), threshold, figures / "threshold_curves.png"
    )

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "threshold": threshold}, MODEL_ARTIFACT)

    manifest = build_manifest(
        stage="train_main_v0_2",
        dataset_sha256=_dataset_sha(config),
        preprocessing_version=PREPROCESSING_VERSION,
        split_sha256=split_hash(splits),
        config_sha256=config.config_sha256,
        seed=config.model.seed,
        extra={
            "model": main_model_name,
            "evaluated_split": "validation",
            "calibration_method": chosen_method,
            "threshold": threshold,
        },
    )
    write_manifest(manifest, reports)
    logger.info(
        "Main model: %s, calibration=%s, threshold=%.3f", main_model_name, chosen_method, threshold
    )


def _threshold_free(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    return {
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "brier_score": float(brier_score_loss(y, prob)),
        "ece": expected_calibration_error(y, prob),
    }


def run_ablation(config: AppConfig) -> None:
    """With/without ablation of discharge_disposition_id (docs/DATA_CARD.md)."""
    processed, _, assignment = _load_frames(config)
    train_df = processed.loc[assignment == "train"]
    valid_df = processed.loc[assignment == "validation"]
    y_valid = valid_df[TARGET_COLUMN].to_numpy()

    selection = json.loads((config.reports_dir / "model_selection" / "selection.json").read_text())
    params = selection["lgbm_params"]

    reduced = tuple(f for f in FEATURE_COLUMNS if f != ABLATION_FEATURE)
    results: dict[str, Any] = {}
    for label, features in (("with_feature", FEATURE_COLUMNS), ("without_feature", reduced)):
        model = fit_lgbm(train_df, params, seed=config.model.seed, feature_columns=features)
        prob = predict_lgbm(model, valid_df, feature_columns=features)
        results[label] = _threshold_free(y_valid, prob)

    delta_auroc = results["with_feature"]["auroc"] - results["without_feature"]["auroc"]
    report = {
        "feature": ABLATION_FEATURE,
        "note": (
            "Discharge destination is known at the prediction point (discharge) but can "
            "proxy unmeasured severity; this ablation informs the keep/exclude decision "
            "recorded in docs/DATA_CARD.md."
        ),
        "results": results,
        "delta_auroc_with_minus_without": delta_auroc,
        "evaluated_split": "validation",
    }
    out_dir = config.reports_dir / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "discharge_disposition.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Ablation delta AUROC (with - without) = %+.4f", delta_auroc)


def run_explain(config: AppConfig) -> None:
    processed, _, assignment = _load_frames(config)
    valid_df = processed.loc[assignment == "validation"].reset_index(drop=True)
    artifact = joblib.load(MODEL_ARTIFACT)
    model, calibrator, threshold = (
        artifact["model"],
        artifact["calibrator"],
        artifact["threshold"],
    )

    importance = global_importance(model, valid_df)
    shap_values = shap_matrix(model, valid_df)
    prob = calibrator.transform(predict_lgbm(model, valid_df))
    y = valid_df[TARGET_COLUMN].to_numpy()

    # Most confident errors at the frozen threshold.
    errors: list[dict[str, Any]] = []
    fp_order = np.argsort(np.where(y == 0, prob, -1.0))[::-1][:5]
    fn_order = np.argsort(np.where(y == 1, prob, 2.0))[:5]
    for kind, indices in (("false_positive", fp_order), ("false_negative", fn_order)):
        for i in indices:
            errors.append(
                {
                    "kind": kind,
                    "encounter_id": int(valid_df["encounter_id"].iloc[int(i)]),
                    "calibrated_probability": float(prob[int(i)]),
                    "label": int(y[int(i)]),
                    "top_attributions": top_attributions_for_row(shap_values[int(i)]),
                }
            )

    out_dir = config.reports_dir / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "global_importance.json").write_text(
        json.dumps({"wording": ATTRIBUTION_WORDING, "importance": importance}, indent=2) + "\n"
    )
    (out_dir / "error_analysis.json").write_text(
        json.dumps(
            {
                "wording": ATTRIBUTION_WORDING,
                "threshold": threshold,
                "n_cases": len(errors),
                "cases": errors,
            },
            indent=2,
        )
        + "\n"
    )
    save_bar(
        importance,
        "mean_abs_shap",
        "feature",
        config.reports_dir / "figures" / "shap_global_importance.png",
        "Mean |SHAP| (validation) — attribution, not causation",
    )
    logger.info("Explainability reports written to %s", out_dir)
