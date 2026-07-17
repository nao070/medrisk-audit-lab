"""Shared fixtures: a synthetic frame mimicking the raw dataset schema.

No real dataset rows appear here; values are fabricated and the real data is
never required for the test suite (docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medrisk.data.preprocess import (
    DIAGNOSIS_COLUMNS,
    MEDICATION_FEATURES,
    NUMERIC_FEATURES,
)

RAW_ONLY_COLUMNS = ("weight", "examide", "citoglipton")


def make_raw_frame(n_rows: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_patients = max(2, n_rows // 2)  # some patients get multiple encounters
    patient_ids = rng.choice(np.arange(1000, 1000 + n_patients), size=n_rows)

    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(1, n_rows + 1),
            "patient_nbr": patient_ids,
            "race": rng.choice(["Caucasian", "AfricanAmerican", "Asian", "?"], n_rows),
            "gender": rng.choice(
                ["Female", "Male", "Unknown/Invalid"], n_rows, p=[0.5, 0.49, 0.01]
            ),
            "age": rng.choice(["[50-60)", "[60-70)", "[70-80)"], n_rows),
            "weight": np.full(n_rows, "?"),
            "admission_type_id": rng.choice(["1", "2", "3"], n_rows),
            "discharge_disposition_id": rng.choice(["1", "3", "6", "11", "13"], n_rows),
            "admission_source_id": rng.choice(["1", "7"], n_rows),
            "payer_code": rng.choice(["MC", "HM", "?"], n_rows),
            "medical_specialty": rng.choice(["InternalMedicine", "?"], n_rows),
            "max_glu_serum": rng.choice(["None", "Norm", ">200"], n_rows),
            "A1Cresult": rng.choice(["None", "Norm", ">7"], n_rows),
            "change": rng.choice(["No", "Ch"], n_rows),
            "diabetesMed": rng.choice(["Yes", "No"], n_rows),
            "readmitted": rng.choice(["<30", ">30", "NO"], n_rows, p=[0.15, 0.35, 0.5]),
        }
    )
    for column in NUMERIC_FEATURES:
        frame[column] = rng.integers(0, 20, n_rows)
    for column in DIAGNOSIS_COLUMNS:
        frame[column] = rng.choice(["250.13", "428", "V45", "E909", "786", "?"], n_rows)
    for column in MEDICATION_FEATURES:
        frame[column] = rng.choice(["No", "Steady", "Up", "Down"], n_rows)
    frame["examide"] = "No"
    frame["citoglipton"] = "No"
    return frame


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()
