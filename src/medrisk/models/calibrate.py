"""Probability calibration.

Calibrators are fitted only on the calibration split; the method (Platt vs
isotonic) is selected on the validation split (docs/DATA_AND_MODEL_PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6

METHODS = ("platt", "isotonic")


@dataclass
class Calibrator:
    method: str
    _platt: LogisticRegression | None = None
    _isotonic: IsotonicRegression | None = None

    def transform(self, prob: np.ndarray) -> np.ndarray:
        if self.method == "identity":
            return prob
        if self.method == "platt":
            assert self._platt is not None
            return np.asarray(self._platt.predict_proba(_logit(prob).reshape(-1, 1))[:, 1])
        assert self._isotonic is not None
        return np.clip(np.asarray(self._isotonic.predict(prob)), 0.0, 1.0)


def _logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, _EPS, 1 - _EPS)
    return np.asarray(np.log(p / (1 - p)))


def fit_calibrator(method: str, prob_calib: np.ndarray, y_calib: np.ndarray) -> Calibrator:
    if method == "platt":
        model = LogisticRegression(max_iter=1000)
        model.fit(_logit(prob_calib).reshape(-1, 1), y_calib)
        return Calibrator(method="platt", _platt=model)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(prob_calib, y_calib)
        return Calibrator(method="isotonic", _isotonic=iso)
    raise ValueError(f"unknown calibration method: {method}")
