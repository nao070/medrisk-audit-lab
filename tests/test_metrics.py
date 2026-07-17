import numpy as np

from medrisk.evaluation.metrics import compute_classification_metrics


def test_perfect_classifier_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_classification_metrics(y_true, y_prob)
    assert m["auroc"] == 1.0
    assert m["recall"] == 1.0
    assert m["specificity"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_metric_keys_cover_required_set() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_prob = rng.random(200)
    m = compute_classification_metrics(y_true, y_prob)
    required = {
        "auroc",
        "auprc",
        "recall",
        "specificity",
        "precision",
        "f1",
        "brier_score",
        "confusion_matrix",
        "prevalence",
        "n",
        "threshold",
    }
    assert required <= set(m)
    assert m["threshold"]["status"] == "provisional_default_v0_1"


def test_degenerate_all_negative_predictions_do_not_crash() -> None:
    y_true = np.array([0, 0, 0, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0])
    m = compute_classification_metrics(y_true, y_prob)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
