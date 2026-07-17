"""LLM evaluation runner: rates per condition with binomial CIs.

Runs against stored outputs in CI (ReplayClient) — no paid API call. In
Live LLM Mode the same validators run on fresh outputs before display.
"""

from __future__ import annotations

import math
from typing import Any

from medrisk.llm.schemas import ExplanationInput
from medrisk.llm.validators import DIMENSIONS, validate_case

RATED_DIMENSIONS = tuple(d for d in DIMENSIONS if d != "readability")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total == 0:
        return {"rate": 0.0, "ci_low": 0.0, "ci_high": 1.0}
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return {
        "rate": p,
        "ci_low": max(0.0, center - margin),
        "ci_high": min(1.0, center + margin),
    }


def evaluate_condition(
    cases: list[ExplanationInput],
    outputs: dict[str, list[str]],
    generator_label: str,
) -> dict[str, Any]:
    """Validate every case of one condition and aggregate violation rates."""
    per_case: list[dict[str, Any]] = []
    for case in cases:
        verdicts = validate_case(case, outputs[case.case_id])
        per_case.append({"case_id": case.case_id, "verdicts": verdicts})

    summary: dict[str, Any] = {}
    for dimension in RATED_DIMENSIONS:
        failures = sum(1 for record in per_case if not record["verdicts"][dimension]["passed"])
        summary[dimension] = {
            "violations": failures,
            "n": len(per_case),
            **{f"violation_{k}": v for k, v in wilson_interval(failures, len(per_case)).items()},
        }
    readability = [
        record["verdicts"]["readability"]["details"]["flesch_reading_ease"]
        for record in per_case
        if isinstance(record["verdicts"]["readability"]["details"], dict)
    ]
    summary["readability"] = {
        "mean_flesch_reading_ease": (
            round(sum(readability) / len(readability), 1) if readability else None
        ),
        "note": "descriptive only; no pass/fail rule",
    }
    return {"generator": generator_label, "summary": summary, "per_case": per_case}


def collect_failures(
    cases: list[ExplanationInput],
    outputs: dict[str, list[str]],
    evaluation: dict[str, Any],
    max_per_dimension: int = 2,
) -> list[dict[str, Any]]:
    """Published failure examples: at least one per violated dimension."""
    case_index = {c.case_id: c for c in cases}
    failures: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for record in evaluation["per_case"]:
        for dimension in RATED_DIMENSIONS:
            verdict = record["verdicts"][dimension]
            if verdict["passed"] or seen.get(dimension, 0) >= max_per_dimension:
                continue
            seen[dimension] = seen.get(dimension, 0) + 1
            case = case_index[record["case_id"]]
            failures.append(
                {
                    "dimension": dimension,
                    "case_id": case.case_id,
                    "input": case.to_json(),
                    "raw_output": outputs[case.case_id][0],
                    "validator_details": verdict["details"],
                }
            )
    return failures
