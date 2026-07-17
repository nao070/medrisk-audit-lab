# Implementation Roadmap

Each version ends with: relevant tests green, `git diff` inspected,
documentation updated in the same change, and the listed exit criteria
answered Yes.

## v0.1 — Data and baseline

Scope: project skeleton (`src/medrisk`, `tests`, `configs`), pinned
dependencies, pre-commit with gitleaks, CI (ruff/mypy/pytest/gitleaks, no
secrets), dataset download with SHA-256 verification, preprocessing v1 with
the feature audit from `DATA_AND_MODEL_PLAN.md`, patient-level 60/10/15/15
split with versioned ID lists, logistic-regression baseline, validation
metrics JSON, run manifest, Data Card first version.

Exit criteria (Yes required):
- Does the split non-overlap test pass in CI?
- Does the pipeline run end to end without any API key?
- Does `git ls-files` contain no data files and no `.env`?
- Does the Data Card record every feature keep/exclude decision?

## v0.2 — Evaluation, calibration, and explainability

Scope: LightGBM with grouped CV on train, Platt/isotonic calibration on the
calibration split, threshold analysis on validation, full metric set with
patient-level bootstrap CIs, `discharge_disposition_id` ablation, SHAP global
and local attributions, misclassification analysis (≥10 cases), report
generation, Model Card draft.

Exit criteria:
- Are all metrics in `DATA_AND_MODEL_PLAN.md` saved with CIs as JSON and plots?
- Was the final test set still never loaded (run manifest shows zero final-test runs)?
- Does the ablation result appear in the Data Card?

## v0.3 — Fairness, uncertainty, and abstention

Scope: subgroup evaluation (race/gender/age) with group sizes and CIs,
small-group labeling rule, abstention band selection on validation,
coverage-vs-selective-performance reporting, determinism test (same seed ⇒
same split hash and metrics).

Exit criteria:
- Do subgroup tables always show n and CI together?
- Are groups with <50 patients labeled inconclusive?
- Does the determinism test pass?

## v0.4 — LLM explanation and evaluation

Scope: `LLMClient` interface, StubClient, ReplayClient, prompt contract v1,
≥100-case versioned evaluation set, all validators, three compared
conditions, stored outputs and evaluation records (successes and failures),
rubric human evaluation, AnthropicClient behind `MEDRISK_LLM_MODE=live`
(building it needs no key; running it live is approval-gated), manual-only
live integration workflow.

Exit criteria:
- Do all validators run in CI against stored outputs with no paid call?
- Are all eight evaluation dimensions measured and published per condition?
- Do failure examples appear in Demo LLM Mode?

## v0.5 — Demo UI

Scope: Streamlit app (pages per `ARCHITECTURE.md`), no free-text medical
input, persistent disclaimer, Demo LLM Mode default, screenshots and demo GIF
for the README. (A REST API is deliberately excluded from v1.0; see Future
extensions.)

Exit criteria:
- Does the app run fully without any API key?
- Is the disclaimer visible on every page?
- Does the app refuse free-text patient input by construction?

## v1.0 — Tests, CI, documentation, and public release

Scope: one-shot final-test evaluation (recorded in the run manifest), Model
Card final, README (EN) and README.ja.md complete, all documents reconciled,
`ACCEPTANCE_CRITERIA.md` fully evaluated with evidence, remote creation +
push + GitHub protections (each human-approved).

Exit criteria:
- Is every mandatory criterion in `ACCEPTANCE_CRITERIA.md` Yes with evidence?
- Was final-test evaluation executed exactly once for this release?

## Future extensions (explicitly not v1.0 dependencies)

- Second audit target using MIMIC-IV (credentialed) or the public MIMIC-IV
  demo subset; requires its own data-governance review. No v0.x/v1.0
  deliverable depends on MIMIC.
- REST API (FastAPI) over the audit artifacts.
- Public demo deployment (Demo LLM Mode only) — separate approval.
- MLflow-based experiment tracking; resampling strategies for imbalance;
  fairness-constrained training.
