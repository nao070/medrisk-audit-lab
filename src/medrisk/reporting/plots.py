"""Figure generation for committed reports (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def save_roc_pr(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title_suffix: str = ""
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC {title_suffix}")
    ax2.plot(recall, precision)
    ax2.axhline(float(np.mean(y_true)), linestyle="--", linewidth=0.8)
    ax2.set(xlabel="Recall", ylabel="Precision", title=f"PR {title_suffix}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_reliability(
    bins_before: list[dict[str, float]],
    bins_after: list[dict[str, float]],
    out_path: Path,
    labels: tuple[str, str] = ("uncalibrated", "calibrated"),
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="gray")
    for bins, label, marker in zip((bins_before, bins_after), labels, ("o", "s"), strict=True):
        ax.plot(
            [b["mean_predicted"] for b in bins],
            [b["observed_rate"] for b in bins],
            marker=marker,
            label=label,
        )
    ax.set(xlabel="Mean predicted probability", ylabel="Observed rate", title="Reliability")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_threshold_curves(curve: list[dict[str, float]], selected: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    thresholds = [r["threshold"] for r in curve]
    for key in ("precision", "recall", "specificity", "f1"):
        ax.plot(thresholds, [r[key] for r in curve], label=key)
    ax.axvline(selected, linestyle="--", color="gray", label=f"selected={selected:.2f}")
    ax.set(xlabel="Threshold", ylabel="Value", title="Threshold analysis (validation)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_bar(
    items: list[dict[str, Any]],
    value_key: str,
    label_key: str,
    out_path: Path,
    title: str,
    top_k: int = 20,
) -> None:
    rows = items[:top_k][::-1]
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.3 * len(rows))))
    ax.barh([str(r[label_key]) for r in rows], [float(r[value_key]) for r in rows])
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_coverage_curve(
    sweep: list[dict[str, float]], selected_band: dict[str, float], out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([r["coverage"] for r in sweep], [r["selective_f1"] for r in sweep], marker="o")
    ax.axvline(selected_band["coverage"], linestyle="--", color="gray")
    ax.set(
        xlabel="Coverage (fraction answered)",
        ylabel="Selective F1 (answered cases)",
        title="Abstention: coverage vs selective performance (validation)",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
