# Model Card — MedRisk Audit Lab main model (v1.0, final)

> Status: final. The final-test evaluation below was executed exactly once,
> with the model, calibration, threshold, and abstention band all frozen on
> the validation split beforehand (run manifest: `reports/runs/`, stage
> `final_test_v1_0`).

## Final-test performance (one-shot; n=15,076 encounters, 10,499 patients, prevalence 0.117)

| Metric | Value | 95% CI (patient bootstrap) |
|---|---|---|
| AUROC | 0.688 | [0.674, 0.702] |
| AUPRC | 0.244 | [0.224, 0.266] |
| Brier score | 0.098 | [0.093, 0.102] |
| Recall (threshold 0.16) | 0.384 | [0.358, 0.412] |
| Specificity | 0.844 | [0.836, 0.852] |
| Precision | 0.245 | [0.228, 0.263] |
| F1 | 0.299 | [0.279, 0.320] |
| ECE | 0.009 | [0.005, 0.015] |

Calibration slope 1.257, intercept 0.516. Abstention (band ±0.015):
coverage 89.8%, selective F1 0.306. Final-test subgroup audit:
`reports/final/final_test_subgroups.json`. Validation AUROC was 0.677
[0.662, 0.690] — the final-test result is consistent with it, indicating
the validation-based selection did not overfit.

## Model summary

- **Task:** binary prediction of hospital readmission within 30 days of
  discharge (see `docs/DATA_CARD.md` for the target definition and limits).
- **Model:** LightGBM (`num_leaves=31`, `learning_rate=0.05`,
  `n_estimators=200`, `min_child_samples=50`, class-weight balanced), selected
  by grouped 3-fold cross-validation on the train split only.
- **Calibration:** Platt scaling, fitted on the dedicated calibration split;
  method chosen on the validation split (Brier 0.204 → 0.092, ECE 0.321 → 0.007).
- **Operating threshold:** 0.16, selected on the validation split by maximum
  F1; frozen before any final-test use.
- **Baseline:** class-weighted logistic regression (validation AUROC 0.664).

## Intended use

Research, education, and portfolio demonstration of evaluation and audit
methodology. **Not a medical device.** Must not be used for diagnosis,
treatment decisions, emergency triage, or any decision about real patients.

## Out-of-scope and prohibited uses

Clinical deployment of any kind; inference on real patient data; use of SHAP
attributions as causal or clinical evidence; extrapolation to current
practice or to healthcare systems outside the source data (US, 1999–2008 —
in particular, results do not transfer to present-day Japanese healthcare).

## Validation performance (v0.2; patient-level bootstrap 95% CI, n=14,790 encounters)

| Metric | Value | 95% CI |
|---|---|---|
| AUROC | 0.677 | [0.662, 0.690] |
| AUPRC (prevalence 0.108) | 0.217 | [0.200, 0.236] |
| Brier score | 0.092 | [0.088, 0.097] |
| Recall (threshold 0.16) | 0.375 | [0.348, 0.400] |
| Specificity | 0.850 | [0.842, 0.857] |
| Precision | 0.233 | [0.216, 0.250] |
| F1 | 0.287 | [0.267, 0.305] |
| ECE | 0.007 | [0.003, 0.012] |

Calibration slope 1.185, intercept 0.319 (post-Platt, validation).
Figures: `reports/figures/`. Full JSON: `reports/metrics/`.

## Explainability

Global and local attributions via TreeSHAP (LightGBM `pred_contrib`).
Top validation features: `number_inpatient` (prior-year inpatient visits),
`discharge_disposition_id`, `medical_specialty`, `diag_1_group`,
`payer_code`. **SHAP values are model attributions on this dataset; they are
not causal or clinical evidence.** Misclassification analysis:
`reports/explain/error_analysis.json`.

## Subgroup audit (v0.3, validation)

Observational audit across `race`, `gender`, `age`; every metric is paired
with group size and a patient-bootstrap 95% CI
(`reports/subgroup/validation_subgroups.json`). AUROC is statistically
indistinguishable across the large race and gender groups (e.g. Caucasian
0.675 [0.657, 0.692], AfricanAmerican 0.675 [0.642, 0.705]; Female 0.682
[0.661, 0.702], Male 0.671 [0.650, 0.691]). Groups under 50 patients (e.g.
gender Unknown/Invalid, n=1) are labeled "insufficient sample for
conclusions" and no claim is made about them. Missing/unknown categories are
reported as groups, never imputed.

## Abstention (v0.3)

The system abstains ("no reliable prediction") when the calibrated
probability lies within ±0.015 of the 0.16 threshold — selected on
validation by maximizing selective F1 subject to coverage ≥ 0.80. Result:
coverage 89.7%, selective F1 0.292 vs 0.287 without abstention
(`reports/abstention/validation_abstention.json`).

## Known limitations (v0.2–v0.3)

- Discrimination is moderate (AUROC ≈ 0.68), consistent with published
  results on this dataset; the project's value is the audit methodology, not
  the score.
- Label capture is incomplete (readmissions elsewhere are unobserved).
- No temporal validation is possible (no timestamps in the source data).
- Subgroup performance and abstention are added in v0.3; final-test results
  in v1.0.
