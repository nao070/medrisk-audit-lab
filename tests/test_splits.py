from pathlib import Path

import pandas as pd
import pytest

from medrisk.data.splits import (
    SPLIT_NAMES,
    SplitOverlapError,
    assert_no_overlap,
    assign_rows,
    load_splits,
    save_splits,
    split_hash,
    split_patients,
)

FRACTIONS = {"train": 0.60, "calibration": 0.10, "validation": 0.15, "test": 0.15}

COMMITTED_SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "splits" / "v1"


def test_patient_level_split_has_no_overlap() -> None:
    ids = list(range(1, 1001))
    splits = split_patients(ids, FRACTIONS, seed=42)
    assert_no_overlap(splits)  # raises on failure
    assert sum(len(v) for v in splits.values()) == len(ids)
    assert set(splits) == set(SPLIT_NAMES)


def test_split_is_deterministic_for_same_seed() -> None:
    ids = list(range(1, 501))
    a = split_patients(ids, FRACTIONS, seed=123)
    b = split_patients(ids, FRACTIONS, seed=123)
    assert a == b
    assert split_hash(a) == split_hash(b)


def test_split_changes_with_seed() -> None:
    ids = list(range(1, 501))
    assert split_hash(split_patients(ids, FRACTIONS, seed=1)) != split_hash(
        split_patients(ids, FRACTIONS, seed=2)
    )


def test_duplicate_encounters_of_one_patient_stay_together() -> None:
    ids = [10, 10, 10, 20, 20, 30, 40, 50, 60, 70]  # duplicates collapse to unique patients
    splits = split_patients(ids, FRACTIONS, seed=7)
    assert sum(len(v) for v in splits.values()) == 7


def test_assert_no_overlap_detects_leak() -> None:
    with pytest.raises(SplitOverlapError):
        assert_no_overlap({"train": [1, 2], "calibration": [3], "validation": [2], "test": [4]})


def test_fractions_must_sum_to_one() -> None:
    bad = dict(FRACTIONS, test=0.99)
    with pytest.raises(ValueError):
        split_patients([1, 2, 3], bad, seed=0)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    splits = split_patients(list(range(100)), FRACTIONS, seed=5)
    digest = save_splits(splits, tmp_path, seed=5, fractions=FRACTIONS)
    loaded = load_splits(tmp_path)
    assert loaded == splits
    assert split_hash(loaded) == digest


def test_assign_rows_maps_every_encounter_to_one_split() -> None:
    splits = {"train": [1, 2], "calibration": [3], "validation": [4], "test": [5]}
    df = pd.DataFrame({"patient_nbr": [1, 1, 2, 3, 4, 5, 5]})
    assigned = assign_rows(df, splits)
    assert list(assigned[df["patient_nbr"] == 1]) == ["train", "train"]
    assert set(assigned.unique()) <= set(SPLIT_NAMES)


def test_assign_rows_rejects_unlisted_patient() -> None:
    splits = {"train": [1], "calibration": [2], "validation": [3], "test": [4]}
    df = pd.DataFrame({"patient_nbr": [1, 99]})
    with pytest.raises(ValueError):
        assign_rows(df, splits)


@pytest.mark.skipif(
    not COMMITTED_SPLIT_DIR.exists(), reason="committed split lists not yet generated"
)
def test_committed_split_lists_have_no_overlap() -> None:
    """The versioned real-data split lists must be pairwise disjoint (CI-enforced)."""
    splits = load_splits(COMMITTED_SPLIT_DIR)
    assert_no_overlap(splits)
    assert all(len(splits[name]) > 0 for name in SPLIT_NAMES)
