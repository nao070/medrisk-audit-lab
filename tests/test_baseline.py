import pandas as pd

from medrisk.data.preprocess import TARGET_COLUMN, build_cohort, encode_features
from medrisk.data.splits import assign_rows, split_patients
from medrisk.models.baseline import predict_probabilities, train_baseline

FRACTIONS = {"train": 0.60, "calibration": 0.10, "validation": 0.15, "test": 0.15}


def test_baseline_trains_and_predicts_on_synthetic_cohort(raw_frame: pd.DataFrame) -> None:
    cohort, _ = build_cohort(raw_frame)
    encoded = encode_features(cohort)
    splits = split_patients(encoded["patient_nbr"].tolist(), FRACTIONS, seed=11)
    assignment = assign_rows(encoded, splits)

    train_df = encoded.loc[assignment == "train"]
    validation_df = encoded.loc[assignment == "validation"]
    assert len(train_df) > 0 and len(validation_df) > 0

    pipeline = train_baseline(train_df, seed=11, max_iter=500)
    probs = predict_probabilities(pipeline, validation_df)
    assert probs.shape == (len(validation_df),)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_baseline_never_sees_target_or_identifiers_as_features() -> None:
    from medrisk.models.baseline import FEATURE_COLUMNS

    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert "encounter_id" not in FEATURE_COLUMNS
    assert "patient_nbr" not in FEATURE_COLUMNS
    assert "readmitted" not in FEATURE_COLUMNS


def test_core_ml_mode_is_independent_of_llm_providers() -> None:
    """Core ML Mode must not import any LLM provider (docs/ARCHITECTURE.md)."""
    import sys

    import medrisk.cli
    import medrisk.models.baseline  # noqa: F401

    assert not any(m.startswith("anthropic") for m in sys.modules)
