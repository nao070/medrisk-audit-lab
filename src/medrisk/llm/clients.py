"""LLM provider abstraction.

Three implementations of one interface (docs/ARCHITECTURE.md):

- StubClient   — deterministic templates; unit tests and stored-demo generation.
  Its condition-specific flaws are *simulated* failure modes so the evaluation
  pipeline demonstrably catches them; every stored artifact is labeled as
  stub-generated.
- ReplayClient — returns stored outputs; powers Demo LLM Mode and CI.
- AnthropicClient — Live LLM Mode only. Constructed only when
  MEDRISK_LLM_MODE=live and a key exists in the local environment. Normal
  tests and CI never construct it and never call any paid API.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from medrisk.llm.prompt import DISCLAIMER_SENTENCE, SYSTEM_PROMPT
from medrisk.llm.schemas import ExplanationInput

CONDITIONS = ("llm_only", "with_model_results", "full_contract")
GENERATOR_LABELS = {
    "stub": "deterministic StubClient (simulated outputs; no live LLM was called)",
    "replay": "stored outputs replayed from the repository",
    "anthropic": "live Anthropic API output",
}


class LLMClient(Protocol):
    name: str

    def generate(self, condition: str, case: ExplanationInput) -> str:
        """Return raw model text (expected to be JSON per the output schema)."""
        ...


def _prob_text(case: ExplanationInput) -> str:
    assert case.prediction is not None
    return f"{case.prediction.calibrated_probability}"


class StubClient:
    """Deterministic template client simulating condition-dependent behavior."""

    name = "stub"

    def generate(self, condition: str, case: ExplanationInput) -> str:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        index = int(case.case_id.rsplit("-", 1)[-1])
        if condition == "llm_only":
            return self._llm_only(case, index)
        if condition == "with_model_results":
            return self._with_results(case, index)
        return self._full_contract(case)

    def _llm_only(self, case: ExplanationInput, index: int) -> str:
        # Simulates an unconstrained LLM asked to be the predictor: it guesses
        # a probability, cites features it was never given, and uses clinical
        # language. Every violation here is intentional and must be caught.
        guessed = round(0.05 + (index % 9) * 0.1, 2)
        return json.dumps(
            {
                "explanation": (
                    f"Based on typical diabetic patients, readmission risk is about {guessed}. "
                    "Poor glucose control and missed insulin doses are likely drivers, so the "
                    "patient should be treated aggressively."
                ),
                "key_factors": ["insulin", "A1Cresult"],
                "uncertainty_note": "Fairly confident in this assessment.",
                "disclaimer": "",
                "refusal": False,
            }
        )

    def _with_results(self, case: ExplanationInput, index: int) -> str:
        # Simulates a partially constrained LLM: numbers are respected, but the
        # disclaimer is sometimes dropped and abstention is sometimes ignored.
        if index % 11 == 0:
            return '{"explanation": "The model predicts'  # truncated JSON
        if case.prediction is None:
            return json.dumps(
                {
                    "explanation": "No model output was provided.",
                    "key_factors": [],
                    "uncertainty_note": "Cannot assess without model results.",
                    "disclaimer": DISCLAIMER_SENTENCE,
                    "refusal": True,
                }
            )
        factors = [a.feature for a in case.top_attributions]
        abstained = case.prediction.abstained
        explanation = f"The model estimates a readmission probability of {_prob_text(case)}." + (
            " The risk level is elevated." if abstained and index % 2 == 0 else ""
        )
        return json.dumps(
            {
                "explanation": explanation,
                "key_factors": factors,
                "uncertainty_note": "The probability is a model estimate on historical data.",
                "disclaimer": "" if index % 7 == 0 else DISCLAIMER_SENTENCE,
                "refusal": False,
            }
        )

    def _full_contract(self, case: ExplanationInput) -> str:
        if case.corruption is not None or case.prediction is None:
            return json.dumps(
                {
                    "explanation": "",
                    "key_factors": [],
                    "uncertainty_note": "",
                    "disclaimer": DISCLAIMER_SENTENCE,
                    "refusal": True,
                }
            )
        pred = case.prediction
        factors = [a.feature for a in case.top_attributions]
        if pred.abstained:
            explanation = (
                "The model does not give a reliable prediction for this case: the "
                f"calibrated probability {_prob_text(case)} lies inside the uncertainty "
                "band around the decision threshold, so the system abstains."
            )
        else:
            comparison = "above" if pred.calibrated_probability >= pred.threshold else "below"
            explanation = (
                f"The model estimates a calibrated 30-day readmission probability of "
                f"{_prob_text(case)}, which is {comparison} the operating threshold "
                f"{pred.threshold}. The strongest model attributions are listed as key "
                "factors; they describe this model on historical data, not causes."
            )
        return json.dumps(
            {
                "explanation": explanation,
                "key_factors": factors,
                "uncertainty_note": (
                    "This probability is a calibrated model estimate on 1999-2008 US "
                    "hospital data; it carries statistical uncertainty and does not "
                    "transfer to other settings."
                ),
                "disclaimer": DISCLAIMER_SENTENCE,
                "refusal": False,
            }
        )


class ReplayClient:
    """Replays stored outputs; the backbone of Demo LLM Mode and CI."""

    name = "replay"

    def __init__(self, stored: dict[str, dict[str, list[str]]]) -> None:
        # stored[condition][case_id] -> list of replicate raw outputs
        self._stored = stored
        self._cursor: dict[tuple[str, str], int] = {}

    @staticmethod
    def from_file(path: Any) -> ReplayClient:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return ReplayClient(payload["outputs"])

    def generate(self, condition: str, case: ExplanationInput) -> str:
        replicates = self._stored[condition][case.case_id]
        key = (condition, case.case_id)
        index = self._cursor.get(key, 0)
        self._cursor[key] = (index + 1) % len(replicates)
        return replicates[index]


class AnthropicClient:
    """Live LLM Mode client. Never constructed in normal tests or CI.

    Requires the optional `live` dependency group (`uv sync --group live`)
    and a key in the local environment. Keys are passed in by the config
    layer and are never logged or stored.
    """

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-5") -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "Live LLM Mode requires the optional 'live' dependency group: uv sync --group live"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, condition: str, case: ExplanationInput) -> str:  # pragma: no cover
        # Paid API call: only reachable in Live LLM Mode with a local key.
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": case.to_json()}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
