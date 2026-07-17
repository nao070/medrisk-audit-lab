import numpy as np
import pandas as pd

from medrisk.evaluation.abstention import abstain_mask, band_sweep, select_band
from medrisk.evaluation.subgroup import (
    INSUFFICIENT_LABEL,
    MIN_PATIENTS,
    subgroup_report,
)


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 1200
    race = rng.choice(["A", "B", "tiny"], n, p=[0.6, 0.37, 0.03])
    y = rng.integers(0, 2, n)
    prob = np.clip(0.5 * y + 0.5 * rng.random(n), 0, 1)
    return pd.DataFrame({"patient_nbr": np.arange(n), "race": race, "y": y, "prob": prob})


def test_subgroup_report_pairs_metrics_with_n_and_ci() -> None:
    df = _frame()
    rows = subgroup_report(df, "race", "prob", "y", threshold=0.5, seed=1, n_replicates=100)
    assert {r["group"] for r in rows} == {"A", "B", "tiny"}
    for row in rows:
        assert row["n_patients"] > 0
        if row["metrics"] is not None:
            assert "auroc" in row["metrics"]
            assert "value" in row["metrics"]["auroc"]


def test_small_groups_are_labeled_inconclusive() -> None:
    df = _frame()
    rows = subgroup_report(df, "race", "prob", "y", threshold=0.5, seed=1, n_replicates=50)
    tiny = next(r for r in rows if r["group"] == "tiny")
    assert tiny["n_patients"] < MIN_PATIENTS
    assert tiny["conclusive"] is False
    assert INSUFFICIENT_LABEL in tiny["note"]
    large = next(r for r in rows if r["group"] == "A")
    assert large["conclusive"] is True


def test_abstain_mask_is_band_around_threshold() -> None:
    prob = np.array([0.10, 0.14, 0.16, 0.20, 0.30])
    mask = abstain_mask(prob, threshold=0.16, band=0.05)
    assert mask.tolist() == [False, True, True, True, False]


def test_band_sweep_coverage_decreases_with_band() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 3000)
    prob = np.clip(0.4 * y + 0.6 * rng.random(3000), 0, 1)
    sweep = band_sweep(y, prob, threshold=0.5)
    coverages = [r["coverage"] for r in sweep]
    assert coverages == sorted(coverages, reverse=True)
    assert coverages[0] == 1.0  # zero-width band answers everything


def test_select_band_respects_min_coverage_and_improves_f1() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 3000)
    # noisy probabilities near the threshold -> abstention should help
    prob = np.clip(0.35 * y + 0.65 * rng.random(3000), 0, 1)
    result = select_band(y, prob, threshold=0.5)
    selected = result["selected"]
    assert selected["coverage"] >= 0.80
    no_abstention = next(r for r in result["sweep"] if r["band"] == 0.0)
    assert selected["selective_f1"] >= no_abstention["selective_f1"]
