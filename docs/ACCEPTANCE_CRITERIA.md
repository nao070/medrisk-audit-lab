# Acceptance Criteria

All criteria must be objectively answerable with Yes or No. "Verification"
states the concrete check that produces the answer.

## Repository and secrets

- [ ] `.env` does not appear in `git ls-files` — Verification: run
  `git ls-files | grep -x '.env'` returns nothing.
- [ ] Only `.env.example` among env files is committed — Verification:
  `git ls-files | grep '^\.env'` returns exactly `.env.example`.
- [ ] Secret scanning passes — Verification: gitleaks over the full history
  reports zero findings; GitHub secret-scanning alerts are zero.
- [ ] No realistic credential appears in documentation, tests, fixtures, or
  history — Verification: gitleaks history scan plus review of
  `.env.example` placeholders for obviously-fake form.
- [ ] Normal CI does not invoke a paid external API — Verification: normal
  workflows declare no API-key secrets, and grep shows no live-client
  construction outside the `MEDRISK_LLM_MODE=live` gate.
- [ ] No raw dataset file is committed — Verification:
  `git ls-files` contains no files under `data/`.

## API-key-free operation

- [ ] Core ML Mode runs end to end without an API key — Verification: full
  pipeline executes in a shell with no LLM key variables set.
- [ ] Demo LLM Mode works without an API key — Verification: Streamlit LLM
  pages and llm-eval CLI run in the same key-free shell.
- [ ] Live LLM Mode is disabled safely when no key is configured —
  Verification: with `MEDRISK_LLM_MODE=live` and no key, the system prints
  the documented message and falls back to Demo; exit code 0.
- [ ] Missing credentials do not crash unrelated features — Verification:
  the key-free test suite passes.
- [ ] Logs do not reveal credentials — Verification: with a fake key set,
  run the pipeline and grep all logs/outputs for the fake value; zero hits.

## Reproducibility

- [ ] Training and evaluation run through a documented command sequence —
  Verification: README quickstart commands executed verbatim on a clean
  checkout succeed.
- [ ] Dataset acquisition is checksum-verified — Verification: tampering with
  one byte of the archive causes the download step to abort.
- [ ] Dataset splitting is deterministic and versioned — Verification: split
  ID lists with hash are committed; determinism test passes (same seed ⇒
  same split hash and metrics).
- [ ] Machine-readable metrics are saved — Verification: metrics JSON exists
  under `reports/` for the release run.
- [ ] Dependencies are declared and reproducible — Verification: lock file
  committed; clean-environment install from it succeeds.

## ML evaluation

- [ ] AUROC, AUPRC, recall, specificity, precision, F1, and Brier score are
  saved with 95% bootstrap CIs — Verification: fields present in metrics JSON.
- [ ] Calibration results (reliability diagram, ECE, slope/intercept) are
  saved — Verification: files present in `reports/`.
- [ ] Confusion-matrix values at the frozen threshold are saved —
  Verification: field present in metrics JSON.
- [ ] Splits are patient-level with no overlap — Verification: the automated
  non-overlap test passes in CI.
- [ ] The final test set is not used for feature selection, hyperparameter
  tuning, threshold selection, calibration, model selection, or prompt
  tuning — Verification: run manifest shows exactly one final-test
  evaluation for the release, and only the dedicated entry point loads that
  split (grep-verified).
- [ ] At least 10 misclassified cases are analyzed — Verification: the error
  analysis section of the evaluation report lists ≥10 cases.
- [ ] Subgroup results always pair metrics with group size and CI, and groups
  under 50 patients are labeled inconclusive — Verification: subgroup report
  inspection against the rule in `DATA_AND_MODEL_PLAN.md`.
- [ ] Abstention is implemented and its coverage/performance trade-off is
  reported — Verification: abstention artifacts present in `reports/`.

## LLM evaluation

- [ ] At least 100 versioned evaluation cases exist — Verification: case
  count in the committed evaluation set ≥100 with a version identifier.
- [ ] Numeric alteration, fabricated features, unsupported assertions,
  omission, abstention compliance, structured-output success, and
  consistency are each measured — Verification: per-condition rates for all
  dimensions in the committed LLM evaluation records.
- [ ] All three compared conditions are evaluated — Verification: results
  table contains all three conditions.
- [ ] Failed outputs are published alongside successful outputs —
  Verification: `reports/llm/` and Demo Mode contain at least one failure
  example per violated dimension.
- [ ] LLM validators run in CI without paid API calls — Verification: CI
  logs show validator execution against stored outputs; no key secrets in
  normal workflows.

## Documentation and portfolio

- [ ] README states purpose, prohibited uses, disclaimer, and current status
  near the top — Verification: all four appear before the quickstart section.
- [ ] The data-era limitation (1999–2008 US data; no generalization to
  current or Japanese practice) appears in README, Data Card, and Model
  Card — Verification: text present in all three.
- [ ] Data Card exists and records every feature keep/exclude decision with
  reasons — Verification: document inspection against the audit rules.
- [ ] Model Card exists — Verification: file present with intended use,
  metrics, and limitations sections filled.
- [ ] Security documentation exists — Verification: `SECURITY.md` and
  `docs/SECURITY_DESIGN.md` complete, no TBD.
- [ ] Japanese and English README versions exist by v1.0 — Verification:
  `README.md` and `README.ja.md` both present with matching disclaimers.
- [ ] Architecture and evaluation results are visible without running the
  project — Verification: diagrams, metrics, and plots viewable on GitHub.

## Privacy and medical safety

- [ ] No identifiable patient information is used — Verification: only the
  public de-identified dataset and synthetic cases appear in
  `git ls-files`-tracked content.
- [ ] Public demos do not accept real patient data — Verification: UI has no
  free-text medical input and displays the prohibition notice.
- [ ] The project does not claim diagnostic or treatment capability —
  Verification: grep of README/UI text for prohibited claim patterns plus
  manual review.
- [ ] SHAP output is not described as causal evidence — Verification: the
  attribution-not-causation wording appears in Model Card, UI, and LLM
  prompt; no committed text calls SHAP causal.
- [ ] Public Live LLM access does not expose the owner's unrestricted API
  key — Verification: no deployment configuration enables live mode; any
  future deployment proposal re-evaluates this criterion before approval.
