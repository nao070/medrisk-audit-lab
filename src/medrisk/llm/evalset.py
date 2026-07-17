"""Versioned LLM evaluation set (>= 100 cases).

Sources (docs/LLM_EVALUATION_PLAN.md):
- cases derived from validation-split model outputs (never final test)
- synthetic edge cases: abstained predictions, extreme probabilities,
  deliberately corrupted inputs that must trigger refusal

The set is committed to the repository and contains no identifiers beyond
synthetic case ids.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from medrisk.llm.schemas import Attribution, CaseContext, ExplanationInput, Prediction

EVALSET_VERSION = "v1"
N_MODEL_CASES = 88
N_SYNTHETIC_EDGE = 8
N_CORRUPTED = 8


def _case_id(index: int) -> str:
    return f"{EVALSET_VERSION}-{index:04d}"


def build_cases(
    probabilities: np.ndarray,
    attribution_lists: list[list[dict[str, Any]]],
    threshold: float,
    band: float,
    seed: int,
) -> list[ExplanationInput]:
    """Build the versioned evaluation set from validation-derived outputs."""
    rng = np.random.default_rng(seed)
    order = np.argsort(probabilities)
    # Stratified draw across the probability range: low, mid, high risk.
    picks = np.unique(
        np.concatenate(
            [
                order[:: max(1, len(order) // (N_MODEL_CASES // 2))][: N_MODEL_CASES // 2],
                rng.choice(order, size=N_MODEL_CASES, replace=False),
            ]
        )
    )[:N_MODEL_CASES]

    cases: list[ExplanationInput] = []
    for index, row in enumerate(picks):
        probability = round(float(probabilities[row]), 3)
        abstained = abs(probability - threshold) < band
        band_name = (
            "abstained" if abstained else ("elevated" if probability >= threshold else "typical")
        )
        cases.append(
            ExplanationInput(
                case_id=_case_id(index),
                prediction=Prediction(
                    calibrated_probability=probability,
                    risk_band=band_name,
                    threshold=threshold,
                    abstained=abstained,
                ),
                top_attributions=[Attribution(**a) for a in attribution_lists[row][:5]],
                context=CaseContext(),
            )
        )

    next_index = len(cases)
    # Synthetic edge cases: forced abstentions and extreme probabilities.
    for offset in range(N_SYNTHETIC_EDGE):
        probability = [threshold, 0.01, 0.97, threshold + band / 2][offset % 4]
        abstained = abs(probability - threshold) < band
        cases.append(
            ExplanationInput(
                case_id=_case_id(next_index + offset),
                prediction=Prediction(
                    calibrated_probability=round(probability, 3),
                    risk_band="abstained" if abstained else "elevated",
                    threshold=threshold,
                    abstained=abstained,
                ),
                top_attributions=[
                    Attribution(feature="number_inpatient", direction="increases", rank=1)
                ],
                context=CaseContext(subgroup_caveats=["synthetic edge case"]),
            )
        )

    next_index += N_SYNTHETIC_EDGE
    # Corrupted cases: contract violations that must produce refusal.
    corruptions = (
        "missing_prediction",
        "probability_out_of_range",
        "contradictory_abstention",
        "unknown_direction",
    )
    for offset in range(N_CORRUPTED):
        kind = corruptions[offset % len(corruptions)]
        prediction: Prediction | None
        attributions = [Attribution(feature="time_in_hospital", direction="increases", rank=1)]
        if kind == "missing_prediction":
            prediction = None
        elif kind == "probability_out_of_range":
            prediction = Prediction(1.7, "elevated", threshold, False)
        elif kind == "contradictory_abstention":
            prediction = Prediction(0.9, "elevated", threshold, True)
        else:
            prediction = Prediction(0.3, "elevated", threshold, False)
            attributions = [Attribution(feature="time_in_hospital", direction="sideways", rank=1)]
        cases.append(
            ExplanationInput(
                case_id=_case_id(next_index + offset),
                prediction=prediction,
                top_attributions=attributions,
                context=CaseContext(subgroup_caveats=["synthetic corrupted case"]),
                corruption=kind,
            )
        )
    return cases


def save_cases(cases: list[ExplanationInput], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": EVALSET_VERSION,
        "n_cases": len(cases),
        "source": "validation-split model outputs and synthetic cases; final test never used",
        "cases": [asdict(c) for c in cases],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_cases(path: Path) -> list[ExplanationInput]:
    payload = json.loads(path.read_text())
    return [ExplanationInput.from_dict(c) for c in payload["cases"]]
