"""v0.4 stages: build the LLM evaluation set, generate stored outputs, and
run the evaluation pipeline.

No paid API is called here: stored outputs are produced by the deterministic
StubClient and are labeled as such. Cases derive from validation-split model
outputs only; the final test split is never touched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from medrisk.config import AppConfig
from medrisk.data.preprocess import PREPROCESSING_VERSION
from medrisk.data.splits import assign_rows, load_splits, split_hash
from medrisk.explain.shap_lgbm import shap_matrix, top_attributions_for_row
from medrisk.llm.clients import CONDITIONS, GENERATOR_LABELS, StubClient
from medrisk.llm.evalset import EVALSET_VERSION, build_cases, load_cases, save_cases
from medrisk.llm.evaluate import collect_failures, evaluate_condition
from medrisk.llm.prompt import PROMPT_VERSION
from medrisk.models.lgbm import predict_lgbm
from medrisk.reporting.manifest import build_manifest, write_manifest
from medrisk.stages.main_model import MODEL_ARTIFACT, _dataset_sha

logger = logging.getLogger(__name__)

CASES_PATH = Path("data/llm_eval") / f"cases_{EVALSET_VERSION}.json"
STORED_OUTPUTS_PATH = Path("reports/llm/stored_outputs.json")
RESULTS_PATH = Path("reports/llm/evaluation_results.json")
FAILURES_PATH = Path("reports/llm/failure_examples.json")
RUBRIC_PATH = Path("reports/llm/human_rubric_template.csv")
N_REPLICATES = 3


def run_llm_cases(config: AppConfig) -> None:
    processed = pd.read_csv(config.processed_dir / "cohort_v1.csv")
    splits = load_splits(config.split.out_dir)
    assignment = assign_rows(processed, splits)
    valid_df = processed.loc[assignment == "validation"].reset_index(drop=True)

    artifact = joblib.load(MODEL_ARTIFACT)
    model, calibrator, threshold = (
        artifact["model"],
        artifact["calibrator"],
        artifact["threshold"],
    )
    prob = calibrator.transform(predict_lgbm(model, valid_df))
    shap_values = shap_matrix(model, valid_df)
    attributions = [top_attributions_for_row(row) for row in shap_values]

    abstention = json.loads(
        (config.reports_dir / "abstention" / "validation_abstention.json").read_text()
    )
    band = float(abstention["selected"]["band"])

    cases = build_cases(prob, attributions, threshold, band, seed=config.split.seed)
    save_cases(cases, CASES_PATH)
    logger.info(
        "Wrote %d evaluation cases (version %s) to %s", len(cases), EVALSET_VERSION, CASES_PATH
    )


def run_llm_generate(config: AppConfig) -> None:
    """Generate and store deterministic stub outputs for every condition."""
    cases = load_cases(CASES_PATH)
    stub = StubClient()
    outputs: dict[str, dict[str, list[str]]] = {}
    for condition in CONDITIONS:
        outputs[condition] = {}
        for case in cases:
            case_for_condition = case
            if condition == "llm_only" and case.prediction is not None:
                # LLM-only baseline: the model results are withheld.
                case_for_condition = type(case)(
                    case_id=case.case_id,
                    prediction=None,
                    top_attributions=[],
                    context=case.context,
                    corruption=case.corruption,
                )
            outputs[condition][case.case_id] = [
                stub.generate(condition, case_for_condition) for _ in range(N_REPLICATES)
            ]
    STORED_OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORED_OUTPUTS_PATH.write_text(
        json.dumps(
            {
                "generator": GENERATOR_LABELS["stub"],
                "prompt_version": PROMPT_VERSION,
                "evalset_version": EVALSET_VERSION,
                "n_replicates": N_REPLICATES,
                "outputs": outputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    logger.info("Stored stub outputs for %d conditions at %s", len(CONDITIONS), STORED_OUTPUTS_PATH)


def run_llm_eval(config: AppConfig) -> None:
    cases = load_cases(CASES_PATH)
    payload = json.loads(STORED_OUTPUTS_PATH.read_text())
    results: dict[str, object] = {
        "generator": payload["generator"],
        "prompt_version": payload["prompt_version"],
        "evalset_version": payload["evalset_version"],
        "n_cases": len(cases),
        "conditions": {},
    }
    failures: list[dict[str, object]] = []
    for condition in CONDITIONS:
        evaluation = evaluate_condition(cases, payload["outputs"][condition], payload["generator"])
        results["conditions"][condition] = evaluation  # type: ignore[index]
        failures.extend(
            {"condition": condition, **f}
            for f in collect_failures(cases, payload["outputs"][condition], evaluation)
        )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    FAILURES_PATH.write_text(
        json.dumps(
            {
                "note": (
                    "Published failure examples with validator verdicts. Outputs were "
                    "produced by the deterministic StubClient (simulated failure modes); "
                    "no live LLM was called."
                ),
                "failures": failures,
            },
            indent=2,
        )
        + "\n"
    )
    if not RUBRIC_PATH.exists():
        _write_rubric_template(cases)

    manifest = build_manifest(
        stage="llm_eval_v0_4",
        dataset_sha256=_dataset_sha(config),
        preprocessing_version=PREPROCESSING_VERSION,
        split_sha256=split_hash(load_splits(config.split.out_dir)),
        config_sha256=config.config_sha256,
        seed=config.split.seed,
        extra={
            "evalset_version": EVALSET_VERSION,
            "prompt_version": PROMPT_VERSION,
            "n_cases": len(cases),
            "paid_api_called": False,
        },
    )
    write_manifest(manifest, config.reports_dir)
    logger.info("LLM evaluation written to %s (%d failure examples)", RESULTS_PATH, len(failures))


def _write_rubric_template(cases: list) -> None:
    """Human-rubric scoring sheet (30 fixed cases; scored by a person)."""
    import csv

    RUBRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset = [c.case_id for c in cases][:: max(1, len(cases) // 30)][:30]
    with RUBRIC_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "case_id",
                "condition",
                "faithfulness_0_fail_1_ok_2_good",
                "clarity_0_fail_1_ok_2_good",
                "uncertainty_communication_0_fail_1_ok_2_good",
                "notes",
            ]
        )
        for condition in CONDITIONS:
            for case_id in subset:
                writer.writerow([case_id, condition, "", "", "", ""])
