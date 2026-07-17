import numpy as np

from medrisk.evaluation.calibration import (
    calibration_report,
    calibration_slope_intercept,
    expected_calibration_error,
    reliability_bins,
)
from medrisk.models.calibrate import fit_calibrator


def _miscalibrated_sample(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.6, n)
    y = (rng.random(n) < true_p).astype(int)
    overconfident = np.clip(true_p * 1.8, 0.01, 0.99)  # systematically too high
    return y, overconfident


def test_ece_is_zero_for_perfectly_calibrated_bins() -> None:
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1] * 50)
    prob = np.full(500, 0.2)
    assert expected_calibration_error(y, prob) < 1e-9


def test_ece_detects_overconfidence() -> None:
    y, prob = _miscalibrated_sample()
    assert expected_calibration_error(y, prob) > 0.05


def test_platt_and_isotonic_reduce_brier_on_held_out_data() -> None:
    y_fit, prob_fit = _miscalibrated_sample(seed=1)
    y_eval, prob_eval = _miscalibrated_sample(seed=2)
    brier_before = float(np.mean((prob_eval - y_eval) ** 2))
    for method in ("platt", "isotonic"):
        calibrator = fit_calibrator(method, prob_fit, y_fit)
        calibrated = calibrator.transform(prob_eval)
        brier_after = float(np.mean((calibrated - y_eval) ** 2))
        assert brier_after < brier_before, method
        assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_slope_near_one_for_calibrated_probabilities() -> None:
    rng = np.random.default_rng(3)
    prob = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < prob).astype(int)
    fit = calibration_slope_intercept(y, prob)
    assert abs(fit["slope"] - 1.0) < 0.1
    assert abs(fit["intercept"]) < 0.1


def test_reliability_bins_and_report_structure() -> None:
    y, prob = _miscalibrated_sample()
    bins = reliability_bins(y, prob)
    assert sum(b["n"] for b in bins) == len(y)
    report = calibration_report(y, prob)
    assert {"ece", "slope", "intercept", "reliability_bins"} <= set(report)
