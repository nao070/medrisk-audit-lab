"""Subgroup (fairness) audit.

Observational audit only: metrics per protected-attribute group, always
paired with group size and CI. Groups under MIN_PATIENTS are labeled
inconclusive and no fairness claim is made about them
(docs/DATA_AND_MODEL_PLAN.md). Missing/unknown categories are reported as
their own groups, never imputed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from medrisk.evaluation.bootstrap import metrics_with_ci

PROTECTED_ATTRIBUTES = ("race", "gender", "age")
MIN_PATIENTS = 50
INSUFFICIENT_LABEL = "insufficient sample for conclusions"


def subgroup_report(
    df: pd.DataFrame,
    attribute: str,
    prob_column: str,
    target_column: str,
    threshold: float,
    seed: int,
    n_replicates: int = 1000,
) -> list[dict[str, Any]]:
    """Per-group metrics for one protected attribute on an evaluation frame."""
    rows: list[dict[str, Any]] = []
    for value, group in df.groupby(attribute, observed=True):
        n_patients = int(group["patient_nbr"].nunique())
        entry: dict[str, Any] = {
            "attribute": attribute,
            "group": str(value),
            "n_patients": n_patients,
            "n_encounters": len(group),
            "prevalence": float(group[target_column].mean()),
            "conclusive": n_patients >= MIN_PATIENTS,
        }
        if not entry["conclusive"]:
            entry["note"] = INSUFFICIENT_LABEL
        try:
            entry["metrics"] = metrics_with_ci(
                group,
                prob_column,
                target_column,
                threshold=threshold,
                seed=seed,
                n_replicates=n_replicates,
            )
        except ValueError:
            entry["metrics"] = None
            entry["note"] = f"metrics undefined (single-class group); {INSUFFICIENT_LABEL}"
            entry["conclusive"] = False
        rows.append(entry)
    rows.sort(key=lambda r: -r["n_patients"])
    return rows


def audit_all_attributes(
    df: pd.DataFrame,
    prob_column: str,
    target_column: str,
    threshold: float,
    seed: int,
    n_replicates: int = 1000,
) -> dict[str, list[dict[str, Any]]]:
    return {
        attribute: subgroup_report(
            df, attribute, prob_column, target_column, threshold, seed, n_replicates
        )
        for attribute in PROTECTED_ATTRIBUTES
    }
