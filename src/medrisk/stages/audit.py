"""v0.3 stage: subgroup fairness audit and abstention selection.

Runs entirely on the validation split with the frozen v0.2 model artifact.
The final test split is never touched here.
"""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from medrisk.config import AppConfig
from medrisk.data.preprocess import PREPROCESSING_VERSION, TARGET_COLUMN
from medrisk.data.splits import assign_rows, load_splits, split_hash
from medrisk.evaluation.abstention import select_band
from medrisk.evaluation.subgroup import MIN_PATIENTS, audit_all_attributes
from medrisk.models.lgbm import predict_lgbm
from medrisk.reporting.manifest import build_manifest, write_manifest
from medrisk.reporting.plots import save_coverage_curve
from medrisk.stages.main_model import MODEL_ARTIFACT, _dataset_sha

logger = logging.getLogger(__name__)


def run_audit(config: AppConfig) -> None:
    processed = pd.read_csv(config.processed_dir / "cohort_v1.csv")
    splits = load_splits(config.split.out_dir)
    assignment = assign_rows(processed, splits)
    valid_df = processed.loc[assignment == "validation"].reset_index(drop=True)

    artifact = joblib.load(MODEL_ARTIFACT)
    model, calibrator, threshold = (
        artifact["model"],
        artifact["calibrator"],
        artifact["threshold"],
    )
    prob = calibrator.transform(predict_lgbm(model, valid_df))

    eval_df = valid_df[["patient_nbr", "race", "gender", "age"]].copy()
    eval_df["y"] = valid_df[TARGET_COLUMN].to_numpy()
    eval_df["prob"] = prob

    subgroups = audit_all_attributes(
        eval_df, "prob", "y", threshold=threshold, seed=config.split.seed
    )
    abstention = select_band(eval_df["y"].to_numpy(), prob, threshold)

    reports = config.reports_dir
    (reports / "subgroup").mkdir(parents=True, exist_ok=True)
    (reports / "abstention").mkdir(parents=True, exist_ok=True)
    (reports / "subgroup" / "validation_subgroups.json").write_text(
        json.dumps(
            {
                "policy": {
                    "min_patients_for_conclusions": MIN_PATIENTS,
                    "note": (
                        "Observational audit; metrics are always paired with group size "
                        "and 95% patient-bootstrap CIs. Missing/unknown categories are "
                        "groups of their own, never imputed."
                    ),
                },
                "threshold": threshold,
                "results": subgroups,
            },
            indent=2,
        )
        + "\n"
    )
    (reports / "abstention" / "validation_abstention.json").write_text(
        json.dumps({"threshold": threshold, **abstention}, indent=2) + "\n"
    )
    save_coverage_curve(
        abstention["sweep"],
        abstention["selected"],
        reports / "figures" / "abstention_coverage.png",
    )

    manifest = build_manifest(
        stage="audit_v0_3",
        dataset_sha256=_dataset_sha(config),
        preprocessing_version=PREPROCESSING_VERSION,
        split_sha256=split_hash(splits),
        config_sha256=config.config_sha256,
        seed=config.split.seed,
        extra={
            "evaluated_split": "validation",
            "abstention_band": abstention["selected"]["band"],
            "abstention_coverage": abstention["selected"]["coverage"],
        },
    )
    write_manifest(manifest, reports)
    logger.info(
        "Audit complete: abstention band %.3f (coverage %.3f)",
        abstention["selected"]["band"],
        abstention["selected"]["coverage"],
    )
