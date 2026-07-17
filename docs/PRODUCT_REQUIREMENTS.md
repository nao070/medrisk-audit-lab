# Product Requirements

## Functional requirements

### Data and modeling (Core ML Mode)

- FR-1: A single documented command downloads the dataset from the UCI
  repository and verifies its SHA-256 checksum; a mismatch aborts with a clear
  error.
- FR-2: A single documented command runs preprocessing, patient-level
  splitting, training, calibration, and validation evaluation end to end
  without any API key.
- FR-3: The pipeline produces machine-readable metrics (JSON) and plots for
  every metric listed in `DATA_AND_MODEL_PLAN.md`, with bootstrap confidence
  intervals.
- FR-4: A dedicated, separate command performs the one-shot final-test
  evaluation and records its execution in the run manifest.
- FR-5: Subgroup evaluation outputs per-group metrics, group sizes, and
  confidence intervals together, and labels groups with fewer than 50
  patients as insufficient for conclusions.
- FR-6: The prediction interface supports abstention: inside the frozen
  uncertainty band it returns "no reliable prediction" instead of a class.
- FR-7: SHAP global and local attributions are computed and stored; all UI
  and report wording presents them as attribution, not causation.

### LLM explanation (Demo and Live LLM Modes)

- FR-8: The LLM layer accepts only the structured JSON defined in
  `LLM_EVALUATION_PLAN.md` and produces the structured output defined there.
- FR-9: Demo LLM Mode replays stored LLM outputs and stored evaluation
  results, including failure cases, without any API key.
- FR-10: Live LLM Mode activates only when `MEDRISK_LLM_MODE=live` is set and
  a key is present in the local environment; it is off by default.
- FR-11: Automated validators measure every evaluation dimension listed in
  `LLM_EVALUATION_PLAN.md` and run against stored outputs in CI.

### Interfaces

- FR-12: A CLI exposes each pipeline stage (download, preprocess, split,
  train, calibrate, evaluate, explain, llm-eval, report).
- FR-13: A Streamlit app presents: model results with calibration and
  subgroup views, abstention behavior, LLM explanations (Demo by default),
  and the LLM evaluation results, with the medical disclaimer visible on
  every page.
- FR-14: The Streamlit app never accepts free-text patient data; inputs are
  restricted to selecting stored cases or entering values for the fixed,
  de-identified feature schema, and a notice forbids entering real patient
  data.

## Non-functional requirements

- NFR-1 Reproducibility: identical seed and config reproduce identical split
  hashes and metrics on a clean machine.
- NFR-2 Feasibility: the full Core ML pipeline completes on an Apple M1 Pro
  32 GB laptop without GPU, and is runnable in a Kaggle notebook environment.
- NFR-3 Cost: no paid API call occurs in Core ML Mode, Demo LLM Mode, normal
  tests, or CI.
- NFR-4 Security: requirements in `SECURITY_DESIGN.md` hold at every release.
- NFR-5 Quality gates: lint (ruff), type check (mypy), and tests (pytest)
  pass in CI for every release tag.
- NFR-6 Dependencies: declared with pinned versions in a lock file.

## User flows

1. **Reviewer (no key):** clone → one setup command → run Core ML pipeline →
   open Streamlit app → inspect metrics, calibration, subgroup, abstention,
   stored LLM evaluations.
2. **Reviewer (browse only):** read README → view committed reports and
   screenshots without running anything.
3. **Developer (live, optional):** set `MEDRISK_LLM_MODE=live` and a local
   key in `.env` → generate fresh LLM explanations → run the same validators
   on them.

## API-key-free behavior

- Core ML Mode and Demo LLM Mode are the defaults and require no key.
- Absence of a key never raises an error in any default path; Live-only
  features render as disabled with an explanatory message.

## Live LLM behavior

- Opt-in via `MEDRISK_LLM_MODE=live` plus a key loaded only from the local
  environment (`.env`, gitignored).
- Keys are never logged, never echoed in errors, never stored in outputs.
- Live inputs and outputs are not persisted by default; persisting a stored
  example requires an explicit flag and writes only non-sensitive structured
  content.
- Any public deployment keeps Live LLM Mode disabled unless separately
  approved.

## Error behavior

- Missing dataset: instructs the user to run the download command; no crash.
- Checksum mismatch: aborts with the expected and actual hash (hashes are not
  secrets) and does not proceed to training.
- Missing key in live mode: a clear message naming the variable to set —
  the message never includes any key material; Demo Mode remains available.
- LLM output failing schema validation: the raw output is quarantined in the
  evaluation record as a failure case; the UI shows the failure, never an
  unvalidated explanation.

## Prohibited uses

The software must not be used for, and its documentation must state it is not
for: diagnosis, treatment recommendations, emergency triage, or any decision
about real patients. Real or identifiable patient information must never be
entered. These statements appear in the README and in the UI footer.
