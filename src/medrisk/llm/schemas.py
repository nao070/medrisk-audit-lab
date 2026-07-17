"""Structured input/output contract for the LLM explanation layer.

The LLM receives only this structured input — never raw dataset rows, never
free text from users — and must return JSON matching the output schema
(docs/LLM_EVALUATION_PLAN.md). SHAP magnitudes are reduced to ranks and
directions before the LLM sees them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

RISK_BANDS = ("elevated", "typical", "abstained")
DIRECTIONS = ("increases", "decreases")

OUTPUT_REQUIRED_KEYS = {
    "explanation": str,
    "key_factors": list,
    "uncertainty_note": str,
    "disclaimer": str,
    "refusal": bool,
}


@dataclass(frozen=True)
class Prediction:
    calibrated_probability: float
    risk_band: str
    threshold: float
    abstained: bool


@dataclass(frozen=True)
class Attribution:
    feature: str
    direction: str
    rank: int


@dataclass(frozen=True)
class CaseContext:
    model_type: str = "LightGBM"
    data_era: str = "US hospitals 1999-2008"
    subgroup_caveats: list[str] = field(default_factory=list)
    disclaimer_required: bool = True


@dataclass(frozen=True)
class ExplanationInput:
    case_id: str
    prediction: Prediction | None
    top_attributions: list[Attribution]
    context: CaseContext
    corruption: str | None = None  # set only on deliberate contract-violation test cases

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ExplanationInput:
        prediction = (
            Prediction(**data["prediction"]) if data.get("prediction") is not None else None
        )
        return ExplanationInput(
            case_id=data["case_id"],
            prediction=prediction,
            top_attributions=[Attribution(**a) for a in data.get("top_attributions", [])],
            context=CaseContext(**data.get("context", {})),
            corruption=data.get("corruption"),
        )


class OutputSchemaError(ValueError):
    """Raised when LLM output does not satisfy the output contract."""


def parse_output(raw_text: str) -> dict[str, Any]:
    """Parse and structurally validate raw LLM output; raise OutputSchemaError."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise OutputSchemaError(f"output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise OutputSchemaError("output JSON must be an object")
    for key, expected_type in OUTPUT_REQUIRED_KEYS.items():
        if key not in data:
            raise OutputSchemaError(f"missing required key: {key}")
        if not isinstance(data[key], expected_type):
            raise OutputSchemaError(f"key {key} must be {expected_type.__name__}")
    if not all(isinstance(item, str) for item in data["key_factors"]):
        raise OutputSchemaError("key_factors must be a list of strings")
    return data
