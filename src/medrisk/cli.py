"""MedRisk Audit Lab CLI.

Every stage runs without any API key. v0.1 stages: download, preprocess,
split, train-baseline, and pipeline (all of the above in order).

The final test split is created by ``split`` but no v0.1 command evaluates
it; the one-shot final-test entry point is a v1.0 deliverable.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from medrisk.config import AppConfig, load_config

logger = logging.getLogger("medrisk")

DEFAULT_CONFIG = Path("configs/default.yaml")
PROCESSED_NAME = "cohort_v1.csv"
COHORT_REPORT_NAME = "cohort_report.json"


def _processed_path(config: AppConfig) -> Path:
    return config.processed_dir / PROCESSED_NAME


def cmd_download(config: AppConfig) -> None:
    from medrisk.data.download import ensure_dataset

    ensure_dataset(
        url=config.dataset.url,
        expected_sha256=config.dataset.sha256,
        raw_dir=config.dataset.raw_dir,
        csv_name=config.dataset.csv_name,
    )


def cmd_preprocess(config: AppConfig) -> None:
    from medrisk.data.preprocess import preprocess

    csv_path = config.dataset.raw_dir / config.dataset.csv_name
    if not csv_path.exists():
        raise SystemExit(f"Raw dataset not found at {csv_path}. Run 'medrisk download' first.")
    processed, report = preprocess(str(csv_path))
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    processed.to_csv(_processed_path(config), index=False)
    (config.processed_dir / COHORT_REPORT_NAME).write_text(
        json.dumps(report.__dict__, indent=2) + "\n"
    )
    logger.info("Processed cohort written to %s", _processed_path(config))


def cmd_split(config: AppConfig) -> None:
    import pandas as pd

    from medrisk.data.splits import save_splits, split_patients

    processed = pd.read_csv(_processed_path(config), usecols=["patient_nbr"])
    splits = split_patients(
        processed["patient_nbr"].tolist(),
        fractions=config.split.fractions,
        seed=config.split.seed,
    )
    save_splits(
        splits,
        out_dir=config.split.out_dir,
        seed=config.split.seed,
        fractions=config.split.fractions,
    )


def cmd_train_baseline(config: AppConfig) -> None:
    import pandas as pd

    from medrisk.data.download import sha256_of_file
    from medrisk.data.preprocess import PREPROCESSING_VERSION, TARGET_COLUMN
    from medrisk.data.splits import assign_rows, load_splits, split_hash
    from medrisk.evaluation.metrics import compute_classification_metrics
    from medrisk.models.baseline import predict_probabilities, train_baseline
    from medrisk.reporting.manifest import build_manifest, write_manifest

    processed = pd.read_csv(_processed_path(config))
    splits = load_splits(config.split.out_dir)
    assignment = assign_rows(processed, splits)

    train_df = processed.loc[assignment == "train"]
    validation_df = processed.loc[assignment == "validation"]
    logger.info(
        "Training baseline on %d rows; evaluating on %d validation rows",
        len(train_df),
        len(validation_df),
    )

    pipeline = train_baseline(train_df, seed=config.model.seed, max_iter=config.model.max_iter)
    y_prob = predict_probabilities(pipeline, validation_df)
    metrics = compute_classification_metrics(validation_df[TARGET_COLUMN].to_numpy(), y_prob)

    archive = config.dataset.raw_dir / "diabetes_130_us_hospitals_1999_2008.zip"
    manifest = build_manifest(
        stage="train_baseline_v0_1",
        dataset_sha256=sha256_of_file(archive) if archive.exists() else "absent",
        preprocessing_version=PREPROCESSING_VERSION,
        split_sha256=split_hash(splits),
        config_sha256=config.config_sha256,
        seed=config.model.seed,
        extra={
            "model": "logistic_regression_balanced",
            "evaluated_split": "validation",
            "metrics_validation": metrics,
        },
    )
    manifest_path = write_manifest(manifest, config.reports_dir)

    metrics_path = manifest_path.parent / "metrics_validation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    logger.info("Validation metrics written to %s", metrics_path)
    logger.info("Run manifest written to %s", manifest_path)


def cmd_train_main(config: AppConfig) -> None:
    from medrisk.stages.main_model import run_train_main

    run_train_main(config)


def cmd_ablation(config: AppConfig) -> None:
    from medrisk.stages.main_model import run_ablation

    run_ablation(config)


def cmd_explain(config: AppConfig) -> None:
    from medrisk.stages.main_model import run_explain

    run_explain(config)


COMMANDS = {
    "download": cmd_download,
    "preprocess": cmd_preprocess,
    "split": cmd_split,
    "train-baseline": cmd_train_baseline,
    "train-main": cmd_train_main,
    "ablation": cmd_ablation,
    "explain": cmd_explain,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="medrisk",
        description=(
            "MedRisk Audit Lab pipeline. Research/education/portfolio use only; "
            "not a medical device."
        ),
    )
    parser.add_argument(
        "command", choices=[*COMMANDS, "pipeline"], help="pipeline runs all stages in order"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    stages = list(COMMANDS) if args.command == "pipeline" else [args.command]
    for stage in stages:
        logger.info("=== stage: %s ===", stage)
        COMMANDS[stage](config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
