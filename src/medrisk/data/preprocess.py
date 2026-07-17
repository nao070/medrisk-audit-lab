"""Preprocessing v1.

Implements the cohort definition and the feature audit fixed in
docs/DATA_AND_MODEL_PLAN.md. Every keep/exclude decision here is mirrored,
with its reason, in docs/DATA_CARD.md. All rules are fixed (not fitted to
data); any future data-fitted transform must be fitted on the train split
only.

Prediction time point: discharge from the index encounter. No feature below
uses information that becomes available only after discharge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

PREPROCESSING_VERSION = "v1"

TARGET_SOURCE = "readmitted"
TARGET_COLUMN = "readmitted_within_30d"

ID_COLUMNS = ("encounter_id", "patient_nbr")

# Encounters ending in death or discharge to hospice: readmission is
# undefined, and keeping them would inject outcome information.
EXCLUDED_DISCHARGE_DISPOSITIONS = frozenset({11, 13, 14, 19, 20, 21})

# Feature-level exclusions (docs/DATA_CARD.md carries the full audit table).
EXCLUDED_FEATURES: dict[str, str] = {
    "weight": "~97% missing; too sparse to be informative",
    "examide": "constant value in the entire dataset",
    "citoglipton": "constant value in the entire dataset",
    # ID_COLUMNS are keys, never features; TARGET_SOURCE is the outcome.
}

NUMERIC_FEATURES: tuple[str, ...] = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)

MEDICATION_FEATURES: tuple[str, ...] = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
)

DIAGNOSIS_COLUMNS: tuple[str, ...] = ("diag_1", "diag_2", "diag_3")

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    *MEDICATION_FEATURES,
    *(f"{c}_group" for c in DIAGNOSIS_COLUMNS),
)


@dataclass(frozen=True)
class CohortReport:
    """Counts recorded in the Data Card."""

    rows_raw: int
    rows_excluded_death_hospice: int
    rows_final: int
    patients_final: int
    positive_rate: float


def group_icd9(code: str) -> str:
    """Map a raw ICD-9 string to the disease groups of Strack et al. (2014)."""
    if code in ("?", "", "Missing"):
        return "Missing"
    if code.startswith(("E", "V")):
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"
    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, na_filter=False)
    for column in (*NUMERIC_FEATURES, *ID_COLUMNS):
        df[column] = pd.to_numeric(df[column])
    return df


def build_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, CohortReport]:
    """Apply cohort exclusions and build the binary target."""
    rows_raw = len(df)
    disposition = pd.to_numeric(df["discharge_disposition_id"])
    excluded_mask = disposition.isin(sorted(EXCLUDED_DISCHARGE_DISPOSITIONS))
    cohort = df.loc[~excluded_mask].copy()

    # Positive = readmitted within 30 days; ">30" and "NO" are negative.
    cohort[TARGET_COLUMN] = (cohort[TARGET_SOURCE] == "<30").astype(int)

    report = CohortReport(
        rows_raw=rows_raw,
        rows_excluded_death_hospice=int(excluded_mask.sum()),
        rows_final=len(cohort),
        patients_final=int(cohort["patient_nbr"].nunique()),
        positive_rate=float(cohort[TARGET_COLUMN].mean()),
    )
    logger.info(
        "Cohort built: %d raw rows, %d excluded (death/hospice), %d kept, %d patients",
        report.rows_raw,
        report.rows_excluded_death_hospice,
        report.rows_final,
        report.patients_final,
    )
    return cohort, report


def encode_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed v1 encodings; returns ids + features + target."""
    out = cohort.copy()

    # '?' becomes an explicit category; protected attributes are never imputed.
    out["race"] = out["race"].replace("?", "Unknown")
    for column in ("payer_code", "medical_specialty"):
        out[column] = out[column].replace("?", "Missing")

    for column in DIAGNOSIS_COLUMNS:
        out[f"{column}_group"] = out[column].map(group_icd9)

    kept = [*ID_COLUMNS, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET_COLUMN]
    return out[kept]


def preprocess(csv_path: str) -> tuple[pd.DataFrame, CohortReport]:
    df = load_raw(csv_path)
    cohort, report = build_cohort(df)
    return encode_features(cohort), report
