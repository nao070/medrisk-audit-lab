"""v1.0 stage: the one-shot final-test evaluation.

This is the ONLY code path in the project that reads the final test split.
Everything it uses — model hyperparameters, calibration method, operating
threshold, abstention band — was frozen on the validation split beforehand
(docs/DATA_AND_MODEL_PLAN.md). A completed-run guard refuses to evaluate
twice for the same release.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from medrisk.config import AppConfig
from medrisk.data.preprocess import PREPROCESSING_VERSION, TARGET_COLUMN
from medrisk.data.splits import assign_rows, load_splits, split_hash
from medrisk.evaluation.abstention import abstain_mask
from medrisk.evaluation.bootstrap import metrics_with_ci
from medrisk.evaluation.calibration import calibration_report
from medrisk.evaluation.metrics import compute_classification_metrics
from medrisk.evaluation.subgroup import audit_all_attributes
from medrisk.models.calibrate import fit_calibrator
from medrisk.models.lgbm import fit_lgbm, predict_lgbm
from medrisk.reporting.manifest import build_manifest, write_manifest
from medrisk.reporting.plots import save_reliability, save_roc_pr
from medrisk.stages.main_model import _dataset_sha

logger = logging.getLogger(__name__)

FINAL_DIR = Path("reports/final")
FINAL_METRICS = FINAL_DIR / "final_test_metrics.json"


class FinalTestAlreadyRunError(RuntimeError):
    """The one-shot final-test evaluation was already executed for this release."""


def run_final_test(config: AppConfig) -> None:
    if FINAL_METRICS.exists():
        raise FinalTestAlreadyRunError(
            f"{FINAL_METRICS} already exists. The final test set is evaluated exactly "
            "once per release; delete the report only as part of a new, documented "
            "release cycle."
        )

    selection = json.loads((config.reports_dir / "model_selection" / "selection.json").read_text())
    abstention_sel = json.loads(
        (config.reports_dir / "abstention" / "validation_abstention.json").read_text()
    )
    threshold = float(selection["threshold"]["selected_threshold"])
    band = float(abstention_sel["selected"]["band"])
    method = selection["calibration_method"]

    processed = pd.read_csv(config.processed_dir / "cohort_v1.csv")
    splits = load_splits(config.split.out_dir)
    assignment = assign_rows(processed, splits)
    train_df = processed.loc[assignment == "train"]
    calib_df = processed.loc[assignment == "calibration"]
    test_df = processed.loc[assignment == "test"].reset_index(drop=True)

    logger.info("FINAL TEST: one-shot evaluation on %d encounters", len(test_df))
    model = fit_lgbm(train_df, selection["lgbm_params"], seed=config.model.seed)
    calibrator = fit_calibrator(
        method, predict_lgbm(model, calib_df), calib_df[TARGET_COLUMN].to_numpy()
    )
    prob = calibrator.transform(predict_lgbm(model, test_df))
    y = test_df[TARGET_COLUMN].to_numpy()

    eval_df = test_df[["patient_nbr", "race", "gender", "age"]].copy()
    eval_df["y"] = y
    eval_df["prob"] = prob

    ci = metrics_with_ci(eval_df, "prob", "y", threshold=threshold, seed=config.split.seed)
    point = compute_classification_metrics(y, prob, threshold)
    point["threshold"] = {"value": threshold, "status": "frozen_on_validation"}

    answered = ~abstain_mask(prob, threshold, band)
    selective = compute_classification_metrics(y[answered], prob[answered], threshold)
    abstention_result = {
        "band": band,
        "coverage": float(answered.mean()),
        "abstention_rate": float(1 - answered.mean()),
        "selective_metrics": selective,
    }

    subgroups = audit_all_attributes(
        eval_df, "prob", "y", threshold=threshold, seed=config.split.seed
    )

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_METRICS.write_text(
        json.dumps(
            {
                "split": "final_test (one-shot)",
                "n_encounters": len(test_df),
                "n_patients": int(test_df["patient_nbr"].nunique()),
                "prevalence": float(np.mean(y)),
                "point_estimates": point,
                "bootstrap_ci_95": ci,
                "calibration": calibration_report(y, prob),
                "abstention": abstention_result,
            },
            indent=2,
        )
        + "\n"
    )
    (FINAL_DIR / "final_test_subgroups.json").write_text(
        json.dumps({"threshold": threshold, "results": subgroups}, indent=2) + "\n"
    )
    save_roc_pr(y, prob, FINAL_DIR / "final_roc_pr.png", "(final test, calibrated)")
    save_reliability(
        calibration_report(y, calibrator.transform(predict_lgbm(model, test_df)))[
            "reliability_bins"
        ],
        calibration_report(y, prob)["reliability_bins"],
        FINAL_DIR / "final_reliability.png",
        labels=(method, method),
    )

    manifest = build_manifest(
        stage="final_test_v1_0",
        dataset_sha256=_dataset_sha(config),
        preprocessing_version=PREPROCESSING_VERSION,
        split_sha256=split_hash(splits),
        config_sha256=config.config_sha256,
        seed=config.model.seed,
        extra={"evaluated_split": "test"},
    )
    manifest["final_test_evaluations"] = 1
    write_manifest(manifest, config.reports_dir)
    logger.info(
        "Final test complete: AUROC %.4f, Brier %.4f (reports/final/)",
        point["auroc"],
        point["brier_score"],
    )
