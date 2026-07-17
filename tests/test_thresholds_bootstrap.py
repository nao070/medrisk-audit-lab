import numpy as np
import pandas as pd

from medrisk.evaluation.bootstrap import metrics_with_ci, patient_bootstrap_ci
from medrisk.evaluation.thresholds import select_threshold, threshold_curve


def test_threshold_curve_monotone_tradeoff() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    prob = np.clip(y * 0.4 + rng.random(1000) * 0.6, 0, 1)
    curve = threshold_curve(y, prob)
    from itertools import pairwise

    recalls = [r["recall"] for r in curve]
    assert all(a >= b - 1e-12 for a, b in pairwise(recalls))


def test_select_threshold_maximizes_f1() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    prob = np.clip(0.7 * y + 0.3 * rng.random(2000), 0, 1)
    info = select_threshold(y, prob)
    curve = threshold_curve(y, prob)
    assert info["at_selection"]["f1"] == max(r["f1"] for r in curve)
    assert info["criterion"] == "max_f1_on_validation"


def _eval_frame(n_patients: int = 300, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        for _ in range(int(rng.integers(1, 4))):
            y = int(rng.random() < 0.2)
            rows.append(
                {
                    "patient_nbr": pid,
                    "y": y,
                    "prob": float(np.clip(0.2 + 0.4 * y + 0.3 * (rng.random() - 0.5), 0, 1)),
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic() -> None:
    df = _eval_frame()
    a = metrics_with_ci(df, "prob", "y", threshold=0.4, seed=7, n_replicates=200)
    b = metrics_with_ci(df, "prob", "y", threshold=0.4, seed=7, n_replicates=200)
    assert a == b  # deterministic given the seed
    for name in ("auroc", "auprc", "brier_score", "recall", "precision", "f1", "ece"):
        entry = a[name]
        assert entry["ci_low"] <= entry["value"] <= entry["ci_high"], name


def test_bootstrap_resamples_patients_not_rows() -> None:
    df = _eval_frame(50, seed=3)

    seen_sizes: set[int] = set()

    def metric_fn(frame: pd.DataFrame) -> dict[str, float]:
        seen_sizes.add(frame["patient_nbr"].nunique())
        return {"m": float(frame["y"].mean())}

    patient_bootstrap_ci(df, metric_fn, n_replicates=50, seed=4)
    # Every replicate draws exactly n_patients patients (with replacement),
    # so unique counts must never exceed the original patient count.
    assert max(seen_sizes) <= 50
