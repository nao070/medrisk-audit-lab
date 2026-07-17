import pandas as pd

from medrisk.data.preprocess import (
    CATEGORICAL_FEATURES,
    EXCLUDED_DISCHARGE_DISPOSITIONS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_cohort,
    encode_features,
    group_icd9,
)


def test_target_binarization(raw_frame: pd.DataFrame) -> None:
    cohort, _ = build_cohort(raw_frame)
    positives = cohort[cohort["readmitted"] == "<30"]
    negatives = cohort[cohort["readmitted"].isin([">30", "NO"])]
    assert (positives[TARGET_COLUMN] == 1).all()
    assert (negatives[TARGET_COLUMN] == 0).all()
    assert len(positives) + len(negatives) == len(cohort)


def test_death_and_hospice_encounters_are_excluded(raw_frame: pd.DataFrame) -> None:
    cohort, report = build_cohort(raw_frame)
    kept = pd.to_numeric(cohort["discharge_disposition_id"])
    assert not kept.isin(sorted(EXCLUDED_DISCHARGE_DISPOSITIONS)).any()
    assert report.rows_raw == len(raw_frame)
    assert report.rows_excluded_death_hospice + report.rows_final == report.rows_raw
    assert report.rows_excluded_death_hospice > 0  # fixture contains 11 and 13


def test_excluded_features_and_identifiers_are_not_features(raw_frame: pd.DataFrame) -> None:
    cohort, _ = build_cohort(raw_frame)
    encoded = encode_features(cohort)
    feature_columns = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert {"weight", "examide", "citoglipton", "readmitted"}.isdisjoint(encoded.columns)
    assert "encounter_id" not in feature_columns
    assert "patient_nbr" not in feature_columns
    # identifiers are retained as key columns for splitting, target for training
    assert {"encounter_id", "patient_nbr", TARGET_COLUMN} <= set(encoded.columns)


def test_missing_markers_become_explicit_categories(raw_frame: pd.DataFrame) -> None:
    cohort, _ = build_cohort(raw_frame)
    encoded = encode_features(cohort)
    assert "?" not in set(encoded["race"])
    assert "Unknown" in set(encoded["race"])  # race '?' -> Unknown, never imputed
    assert "?" not in set(encoded["payer_code"])
    assert "?" not in set(encoded["medical_specialty"])


def test_gender_unknown_invalid_is_retained(raw_frame: pd.DataFrame) -> None:
    cohort, _ = build_cohort(raw_frame)
    encoded = encode_features(cohort)
    if (raw_frame["gender"] == "Unknown/Invalid").any():
        assert "Unknown/Invalid" in set(encoded["gender"])


def test_icd9_grouping() -> None:
    assert group_icd9("250.13") == "Diabetes"
    assert group_icd9("428") == "Circulatory"
    assert group_icd9("786") == "Respiratory"
    assert group_icd9("540") == "Digestive"
    assert group_icd9("585") == "Genitourinary"
    assert group_icd9("805") == "Injury"
    assert group_icd9("715") == "Musculoskeletal"
    assert group_icd9("174") == "Neoplasms"
    assert group_icd9("V45") == "Other"
    assert group_icd9("E909") == "Other"
    assert group_icd9("?") == "Missing"
    assert group_icd9("300") == "Other"
