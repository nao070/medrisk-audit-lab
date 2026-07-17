"""CI validation of the committed LLM evaluation assets.

Runs the full validator pipeline against the stored (stub-generated)
outputs — this is exactly what Demo LLM Mode replays, and it requires no
API key and no network.
"""

import json
from pathlib import Path

import pytest

from medrisk.llm.clients import CONDITIONS
from medrisk.llm.evalset import load_cases
from medrisk.llm.evaluate import RATED_DIMENSIONS, evaluate_condition

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "llm_eval" / "cases_v1.json"
STORED = ROOT / "reports" / "llm" / "stored_outputs.json"

pytestmark = pytest.mark.skipif(
    not (CASES.exists() and STORED.exists()),
    reason="committed LLM assets not yet generated",
)


def test_evalset_has_at_least_100_versioned_cases() -> None:
    payload = json.loads(CASES.read_text())
    assert payload["version"] == "v1"
    assert payload["n_cases"] >= 100
    assert payload["n_cases"] == len(payload["cases"])


def test_stored_outputs_cover_every_condition_and_case() -> None:
    cases = load_cases(CASES)
    payload = json.loads(STORED.read_text())
    assert "StubClient" in payload["generator"]  # honest provenance label
    for condition in CONDITIONS:
        assert set(payload["outputs"][condition]) == {c.case_id for c in cases}


def test_full_contract_condition_has_zero_violations() -> None:
    cases = load_cases(CASES)
    payload = json.loads(STORED.read_text())
    result = evaluate_condition(cases, payload["outputs"]["full_contract"], "stub")
    for dimension in RATED_DIMENSIONS:
        assert result["summary"][dimension]["violations"] == 0, dimension


def test_llm_only_condition_violations_are_caught() -> None:
    cases = load_cases(CASES)
    payload = json.loads(STORED.read_text())
    result = evaluate_condition(cases, payload["outputs"]["llm_only"], "stub")
    assert result["summary"]["fabricated_features"]["violation_rate"] > 0.9
    assert result["summary"]["unsupported_medical_assertions"]["violation_rate"] > 0.9


def test_failure_examples_cover_all_rated_dimensions_with_violations() -> None:
    failures = json.loads((ROOT / "reports" / "llm" / "failure_examples.json").read_text())
    covered = {f["dimension"] for f in failures["failures"]}
    assert covered == set(RATED_DIMENSIONS)
