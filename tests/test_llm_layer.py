"""LLM layer tests. No network, no key, no paid API — ever."""

import json
import sys

import pytest

from medrisk.config import LLM_MODE_ENV, anthropic_api_key, resolve_llm_mode
from medrisk.llm.clients import CONDITIONS, ReplayClient, StubClient
from medrisk.llm.evaluate import evaluate_condition, wilson_interval
from medrisk.llm.prompt import DISCLAIMER_SENTENCE
from medrisk.llm.schemas import (
    Attribution,
    CaseContext,
    ExplanationInput,
    OutputSchemaError,
    Prediction,
    parse_output,
)
from medrisk.llm.validators import validate_case


def _case(
    case_id: str = "v1-0001",
    probability: float = 0.27,
    abstained: bool = False,
    corruption: str | None = None,
) -> ExplanationInput:
    return ExplanationInput(
        case_id=case_id,
        prediction=Prediction(
            calibrated_probability=probability,
            risk_band="abstained" if abstained else "elevated",
            threshold=0.16,
            abstained=abstained,
        ),
        top_attributions=[
            Attribution(feature="number_inpatient", direction="increases", rank=1),
            Attribution(feature="time_in_hospital", direction="decreases", rank=2),
        ],
        context=CaseContext(),
        corruption=corruption,
    )


def test_parse_output_accepts_valid_and_rejects_invalid() -> None:
    valid = json.dumps(
        {
            "explanation": "x",
            "key_factors": ["number_inpatient"],
            "uncertainty_note": "y",
            "disclaimer": DISCLAIMER_SENTENCE,
            "refusal": False,
        }
    )
    assert parse_output(valid)["refusal"] is False
    with pytest.raises(OutputSchemaError):
        parse_output("not json")
    with pytest.raises(OutputSchemaError):
        parse_output(json.dumps({"explanation": "x"}))


def test_stub_full_contract_passes_all_validators() -> None:
    stub = StubClient()
    case = _case()
    raw = [stub.generate("full_contract", case) for _ in range(3)]
    verdicts = validate_case(case, raw)
    for dimension, verdict in verdicts.items():
        assert verdict["passed"], (dimension, verdict)


def test_stub_full_contract_refuses_corrupted_input() -> None:
    stub = StubClient()
    case = _case(corruption="contradictory_abstention")
    output = parse_output(stub.generate("full_contract", case))
    assert output["refusal"] is True
    verdicts = validate_case(case, [stub.generate("full_contract", case)])
    assert verdicts["numeric_alteration"]["passed"]  # refusal on corrupt input is compliant


def test_stub_llm_only_condition_violates_and_is_caught() -> None:
    stub = StubClient()
    case = ExplanationInput(
        case_id="v1-0003",
        prediction=None,
        top_attributions=[],
        context=CaseContext(),
    )
    verdicts = validate_case(case, [stub.generate("llm_only", case)])
    assert not verdicts["numeric_alteration"]["passed"]  # invented probability
    assert not verdicts["fabricated_features"]["passed"]  # cites insulin/A1Cresult
    assert not verdicts["unsupported_medical_assertions"]["passed"]  # "treated"
    assert not verdicts["critical_information_omission"]["passed"]  # no disclaimer


def test_abstention_compliance_validator() -> None:
    case = _case(abstained=True, probability=0.16)
    compliant = json.dumps(
        {
            "explanation": "The model abstains: no reliable prediction for this case.",
            "key_factors": ["number_inpatient"],
            "uncertainty_note": "Inside the uncertainty band.",
            "disclaimer": DISCLAIMER_SENTENCE,
            "refusal": False,
        }
    )
    violating = json.dumps(
        {
            "explanation": "The risk level is elevated for this case.",
            "key_factors": ["number_inpatient"],
            "uncertainty_note": "n/a",
            "disclaimer": DISCLAIMER_SENTENCE,
            "refusal": False,
        }
    )
    assert validate_case(case, [compliant])["abstention_compliance"]["passed"]
    assert not validate_case(case, [violating])["abstention_compliance"]["passed"]


def test_numeric_alteration_catches_new_numbers() -> None:
    case = _case(probability=0.27)
    bad = json.dumps(
        {
            "explanation": "The probability is 0.27, roughly a 42.5 percent chance.",
            "key_factors": ["number_inpatient"],
            "uncertainty_note": "note",
            "disclaimer": DISCLAIMER_SENTENCE,
            "refusal": False,
        }
    )
    verdict = validate_case(case, [bad])["numeric_alteration"]
    assert not verdict["passed"]
    assert "42.5" in verdict["details"]


def test_consistency_detects_divergent_replicates() -> None:
    case = _case()
    a = json.dumps(
        {
            "explanation": "The probability is 0.27.",
            "key_factors": ["number_inpatient"],
            "uncertainty_note": "n",
            "disclaimer": DISCLAIMER_SENTENCE,
            "refusal": False,
        }
    )
    b = a.replace('"number_inpatient"', '"time_in_hospital"')
    assert not validate_case(case, [a, b])["consistency"]["passed"]
    assert validate_case(case, [a, a])["consistency"]["passed"]


def test_replay_client_round_robins_replicates() -> None:
    case = _case(case_id="v1-0009")
    client = ReplayClient({"full_contract": {"v1-0009": ["one", "two"]}})
    assert client.generate("full_contract", case) == "one"
    assert client.generate("full_contract", case) == "two"
    assert client.generate("full_contract", case) == "one"


def test_evaluate_condition_aggregates_rates() -> None:
    stub = StubClient()
    cases = [_case(case_id=f"v1-{i:04d}") for i in range(20)]
    outputs = {c.case_id: [stub.generate("full_contract", c)] for c in cases}
    result = evaluate_condition(cases, outputs, "stub")
    assert result["summary"]["numeric_alteration"]["violations"] == 0
    assert result["summary"]["structured_output"]["n"] == 20


def test_wilson_interval_bounds() -> None:
    ci = wilson_interval(3, 100)
    assert 0.0 <= ci["ci_low"] <= ci["rate"] <= ci["ci_high"] <= 1.0


def test_llm_mode_defaults_to_demo_and_key_absence_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LLM_MODE_ENV, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_llm_mode() == "demo"
    assert anthropic_api_key() is None


def test_live_mode_without_key_falls_back_to_demo(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from medrisk.llm.factory import make_client

    stored = tmp_path / "stored.json"
    stored.write_text(json.dumps({"outputs": {"full_contract": {"v1-0001": ["{}"]}}}))
    monkeypatch.setenv(LLM_MODE_ENV, "live")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, mode = make_client(stored)
    assert mode == "demo"
    assert client.name == "replay"


def test_core_ml_and_demo_mode_never_import_anthropic() -> None:
    import medrisk.llm.clients
    import medrisk.llm.factory  # noqa: F401

    assert not any(m.startswith("anthropic") for m in sys.modules)


def test_all_conditions_are_known() -> None:
    assert CONDITIONS == ("llm_only", "with_model_results", "full_contract")
