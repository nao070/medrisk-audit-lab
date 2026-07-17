"""MedRisk Audit Lab — Streamlit demo UI (v0.5).

Reads only committed report artifacts; requires no API key. There is no
free-text medical input anywhere: users select stored cases only
(docs/PRODUCT_REQUIREMENTS.md FR-13/FR-14). Live LLM Mode stays disabled
unless explicitly configured locally, and this app never enables it itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

DISCLAIMER = (
    "Research, education, and portfolio demonstration only — NOT a medical "
    "device. No diagnosis, no treatment decisions, no triage. Never enter "
    "real patient data. Data era: US hospitals 1999-2008; results do not "
    "generalize to current practice or to present-day Japanese healthcare."
)

PAGES = (
    "Overview",
    "Model results",
    "Calibration & thresholds",
    "Subgroup audit",
    "Abstention",
    "LLM explanations (Demo)",
    "LLM evaluation",
)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _figure(name: str, caption: str) -> None:
    path = REPORTS / "figures" / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure not generated yet: {name}")


def page_overview() -> None:
    st.title("MedRisk Audit Lab")
    st.markdown(
        "**An audit platform for a clinical risk model** — 30-day hospital "
        "readmission prediction on the public UCI *Diabetes 130-US Hospitals* "
        "dataset, subjected to a full safety audit: patient-level leak "
        "prevention, calibration, threshold and abstention analysis, subgroup "
        "evaluation, and quantitative evaluation of a constrained LLM "
        "explanation layer."
    )
    st.error(DISCLAIMER)
    selection = _load_json(REPORTS / "model_selection" / "selection.json")
    if selection:
        st.subheader("Frozen model selection")
        st.json(
            {
                "main_model": selection["main_model"],
                "lgbm_params": selection["lgbm_params"],
                "calibration_method": selection["calibration_method"],
                "threshold": selection["threshold"]["selected_threshold"],
                "selection_data": selection["selection_data"],
            }
        )
    st.markdown("See `docs/` for the full design documents and audit records.")


def page_model_results() -> None:
    st.header("Model results (validation split)")
    st.caption(
        "The final test split is evaluated exactly once at v1.0; everything "
        "here is validation-split evidence."
    )
    comparison = _load_json(REPORTS / "metrics" / "model_comparison.json")
    if comparison:
        st.subheader("Model comparison (threshold-free)")
        st.dataframe(
            [
                {"model": k, **{m: round(v, 4) for m, v in vals.items()}}
                for k, vals in comparison.items()
            ],
            hide_index=True,
        )
    metrics = _load_json(REPORTS / "metrics" / "validation_main_v0_2.json")
    if metrics:
        st.subheader(f"Main model: {metrics['model']} — 95% patient-bootstrap CIs")
        rows = []
        for name, entry in metrics["bootstrap_ci_95"].items():
            rows.append(
                {
                    "metric": name,
                    "value": round(entry["value"], 4),
                    "ci_low": round(entry.get("ci_low", float("nan")), 4),
                    "ci_high": round(entry.get("ci_high", float("nan")), 4),
                }
            )
        st.dataframe(rows, hide_index=True)
    _figure("roc_pr_main.png", "ROC and precision-recall curves (validation, calibrated)")
    _figure(
        "shap_global_importance.png",
        "Mean |SHAP| attribution — model attribution, NOT causal or clinical evidence",
    )


def page_calibration() -> None:
    st.header("Calibration & thresholds")
    metrics = _load_json(REPORTS / "metrics" / "validation_main_v0_2.json")
    if metrics:
        calibration = metrics["calibration"]
        st.metric("ECE (validation, calibrated)", round(calibration["ece"], 4))
        st.write(
            f"Calibration slope {calibration['slope']:.3f}, "
            f"intercept {calibration['intercept']:.3f} (ideal: 1 / 0)."
        )
    _figure("reliability_main.png", "Reliability diagram before/after calibration")
    _figure("threshold_curves.png", "Threshold analysis; operating threshold frozen on validation")


def page_subgroup() -> None:
    st.header("Subgroup audit (observational)")
    data = _load_json(REPORTS / "subgroup" / "validation_subgroups.json")
    if not data:
        st.info("Subgroup report not generated yet.")
        return
    st.caption(data["policy"]["note"])
    for attribute, rows in data["results"].items():
        st.subheader(attribute)
        table = []
        for row in rows:
            metrics = row["metrics"]
            auroc = metrics["auroc"] if metrics else None
            table.append(
                {
                    "group": row["group"],
                    "patients": row["n_patients"],
                    "encounters": row["n_encounters"],
                    "prevalence": round(row["prevalence"], 3),
                    "AUROC": round(auroc["value"], 3) if auroc else None,
                    "AUROC 95% CI": (
                        f"[{auroc['ci_low']:.3f}, {auroc['ci_high']:.3f}]"
                        if auroc and "ci_low" in auroc
                        else "n/a"
                    ),
                    "conclusive": "yes" if row["conclusive"] else f"NO — {row.get('note', '')}",
                }
            )
        st.dataframe(table, hide_index=True)


def page_abstention() -> None:
    st.header("Abstention (no reliable prediction)")
    data = _load_json(REPORTS / "abstention" / "validation_abstention.json")
    if not data:
        st.info("Abstention report not generated yet.")
        return
    selected = data["selected"]
    st.write(f"Rule: {data['rule']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Band (± around threshold)", selected["band"])
    col2.metric("Coverage", f"{selected['coverage']:.1%}")
    col3.metric("Selective F1", round(selected["selective_f1"], 3))
    _figure("abstention_coverage.png", "Coverage vs selective performance (validation)")


def page_llm_demo() -> None:
    st.header("LLM explanations — Demo LLM Mode")
    st.caption(
        "Stored outputs replayed from the repository. Outputs were generated "
        "by a deterministic StubClient (no live LLM call); Live LLM Mode is "
        "disabled by default and requires a locally configured key."
    )
    cases_payload = _load_json(ROOT / "data" / "llm_eval" / "cases_v1.json")
    stored = _load_json(REPORTS / "llm" / "stored_outputs.json")
    if not cases_payload or not stored:
        st.info("Stored LLM assets not generated yet.")
        return
    case_ids = [c["case_id"] for c in cases_payload["cases"]]
    selected_id = st.selectbox("Stored case (no free-text input is accepted)", case_ids)
    condition = st.selectbox("Condition", list(stored["outputs"].keys()), index=2)
    case = next(c for c in cases_payload["cases"] if c["case_id"] == selected_id)
    st.subheader("Structured input (what the LLM sees)")
    st.json(case)
    raw = stored["outputs"][condition][selected_id][0]
    st.subheader("Stored output")
    try:
        st.json(json.loads(raw))
    except json.JSONDecodeError:
        st.warning("This stored output is not valid JSON (a published failure example):")
        st.code(raw)


def page_llm_eval() -> None:
    st.header("LLM evaluation results")
    results = _load_json(REPORTS / "llm" / "evaluation_results.json")
    if not results:
        st.info("LLM evaluation not generated yet.")
        return
    st.caption(f"Generator: {results['generator']} — {results['n_cases']} versioned cases")
    for condition, data in results["conditions"].items():
        st.subheader(condition)
        rows = []
        for dimension, entry in data["summary"].items():
            if dimension == "readability":
                continue
            rows.append(
                {
                    "dimension": dimension,
                    "violations": entry["violations"],
                    "n": entry["n"],
                    "rate": round(entry["violation_rate"], 3),
                    "95% CI": (
                        f"[{entry['violation_ci_low']:.3f}, {entry['violation_ci_high']:.3f}]"
                    ),
                }
            )
        st.dataframe(rows, hide_index=True)
    failures = _load_json(REPORTS / "llm" / "failure_examples.json")
    if failures:
        st.subheader("Published failure examples")
        st.caption(failures["note"])
        for failure in failures["failures"][:8]:
            with st.expander(
                f"{failure['condition']} / {failure['dimension']} / {failure['case_id']}"
            ):
                st.code(failure["raw_output"], language="json")
                st.write("Validator details:", failure["validator_details"])


def main() -> None:
    st.set_page_config(page_title="MedRisk Audit Lab", layout="wide")
    page = st.sidebar.radio("Page", PAGES)
    st.sidebar.warning(DISCLAIMER)
    dispatch = {
        "Overview": page_overview,
        "Model results": page_model_results,
        "Calibration & thresholds": page_calibration,
        "Subgroup audit": page_subgroup,
        "Abstention": page_abstention,
        "LLM explanations (Demo)": page_llm_demo,
        "LLM evaluation": page_llm_eval,
    }
    dispatch[page]()
    st.divider()
    st.caption(DISCLAIMER)


main()
