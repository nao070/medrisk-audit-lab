# Data and Model Plan

## Dataset

**Selected dataset:** "Diabetes 130-US Hospitals for Years 1999–2008"
(UCI Machine Learning Repository, dataset ID 296).

- 101,766 inpatient encounters, 71,518 unique patients, 50 columns.
- Each row is one hospital encounter of a patient with a diabetes diagnosis.
- Contains `patient_nbr` (de-identified patient key), enabling patient-level splits.
- Contains demographic columns (`race`, `gender`, `age`) enabling subgroup evaluation.

### Dataset candidates considered

| Candidate | Reason for decision |
|---|---|
| UCI Diabetes 130-US Hospitals (selected) | Fully public, patient key available, subgroup columns, feasible on M1 Pro and Kaggle |
| MIMIC-IV (rejected for v1.0) | Requires credentialed access and DUA; derived data cannot be published; breaks public reproducibility. Recorded as a post-v1.0 extension candidate in `IMPLEMENTATION_ROADMAP.md` |
| UCI Heart Disease (rejected) | ~300 rows; too small for calibration, subgroup, and CI analysis; weak differentiation |

## License and allowed use

- The UCI repository distributes this dataset under **CC BY 4.0**.
- Allowed: research, education, portfolio use, redistribution with attribution.
- Design decision: **raw data is never committed to the Git repository.**
  A download script fetches the archive from the UCI repository at setup time
  and verifies its SHA-256 checksum. Only the download procedure, the expected
  checksum, and derived aggregate results (metrics, plots) are published.
- The license and citation (Strack et al., 2014, BioMed Research International)
  are recorded in the Data Card.
- Before public release, the license terms are re-verified against the UCI
  page and the verification date is recorded in the Data Card.

## Era and generalization limits (mandatory disclaimer)

The following statements appear in the Data Card, the Model Card, and the README:

- The data covers **US hospitals between 1999 and 2008**. Clinical practice,
  coding systems, drug availability, and patient populations have changed since.
- Results **do not generalize to current clinical settings**, and in particular
  **do not transfer to present-day Japanese healthcare**.
- The project is for **research, education, and portfolio demonstration only**.
  It is not a medical device and must not inform care decisions.

## Prediction target

### Definition

Binary classification per encounter:

- **Positive (1):** `readmitted == "<30"` — readmitted within 30 days of discharge.
- **Negative (0):** `readmitted == ">30"` or `readmitted == "NO"`.

Positive prevalence is approximately 11% before cohort exclusions.

### Validity of this binarization

- 30-day readmission is an established quality metric (for example CMS HRRP),
  so the threshold is externally meaningful rather than arbitrary.
- Grouping `">30"` with `"NO"` matches the operational question
  "does this discharge lead to an early readmission?" and follows the
  original study of this dataset (Strack et al., 2014).

### Limits of this binarization (recorded in the Data Card)

- `"NO"` includes patients readmitted to a **different** hospital system,
  deceased after discharge, or lost to follow-up. Some negatives are therefore
  mislabeled; measured performance is affected by unobservable outcome capture.
- Treating `">30"` as negative discards ordinal information; a 31-day
  readmission counts the same as no readmission.
- The 30-day window starts at discharge, but exact timestamps are absent from
  the dataset, so the window is taken as recorded by the original data providers.

## Prediction time point and cohort definition

**Prediction time point: the moment of discharge from the index encounter.**
Every feature used by the model must be knowable at or before discharge.

Cohort exclusions, applied before splitting:

1. Encounters ending in **death or discharge to hospice**
   (`discharge_disposition_id` in {11, 13, 14, 19, 20, 21}) are excluded:
   readmission is undefined for these encounters.
2. Encounters with `gender == "Unknown/Invalid"` are retained in training data
   but reported separately in subgroup analysis (see Fairness section);
   they are not silently dropped or imputed.

All exclusion counts are recorded in the Data Card.

## Feature audit for temporal leakage and post-discharge information

Because the prediction point is discharge, features are audited one by one.
The full audit table (all 50 columns, keep/exclude, and the reason) lives in
the Data Card; the design rules and the sensitive cases are fixed here.

### Audit rules

1. A feature is admissible only if its value is available at or before discharge.
2. A feature is excluded if it directly or indirectly encodes the outcome.
3. Every keep/exclude decision and its reason is recorded in the Data Card.
4. The audit is re-run whenever preprocessing changes (versioned, see
   Reproducibility).

### Individually audited variables

| Variable | Decision | Reason |
|---|---|---|
| `readmitted` | Target only | Outcome; never a feature |
| `discharge_disposition_id` (death/hospice values 11, 13, 14, 19, 20, 21) | Cohort exclusion | Readmission undefined; keeping them injects outcome information |
| `discharge_disposition_id` (remaining values) | Keep, with sensitivity analysis | Discharge destination is decided at discharge, so it is knowable at the prediction point; however it can proxy unmeasured severity. v0.2 includes a with/without ablation, and the result is recorded in the Data Card |
| `payer_code` | Keep | Known at admission; high missingness handled as explicit category |
| `time_in_hospital` | Keep | Length of stay is known at discharge |
| `number_inpatient`, `number_outpatient`, `number_emergency` | Keep | Counts of visits in the **preceding** year; strictly pre-encounter |
| `encounter_id` | Exclude as feature | Identifier; ordering may correlate with time and leak era effects |
| `patient_nbr` | Exclude as feature | Identifier; used only for splitting |
| Medication change columns, `diag_1..3`, labs, `A1Cresult`, `max_glu_serum` | Keep | Recorded during the stay, available at discharge |
| `weight` | Exclude | ~97% missing; retaining adds noise and a misleading Data Card entry documents this |

### Temporal validation limitation

The dataset has **no admission or discharge timestamps**, so a temporal
(train-on-past, test-on-future) split is impossible. The design uses
patient-level random splits instead, and this limitation is stated in the
Data Card and the Model Card.

## Split strategy

Splits are performed on **unique `patient_nbr` values**, not on rows.
Every encounter of a patient falls into exactly one split.

| Split | Share of patients | Allowed uses |
|---|---|---|
| train | 60% | Model fitting, feature selection, cross-validated hyperparameter search |
| calibration | 10% | Probability calibration only |
| validation | 15% | Model selection, hyperparameter confirmation, threshold selection, abstention-band selection, prompt/format iteration for the LLM layer |
| final test | 15% | **One** final evaluation per release; nothing else |

- The split is produced with a fixed seed and saved as versioned patient-ID
  lists with a SHA-256 hash, so the split is reproducible byte-for-byte.
- An automated test asserts that the pairwise intersection of `patient_nbr`
  sets across the four splits is empty. This test runs in CI on the versioned
  split lists (the raw data itself is not present in CI).

## Final-test-set isolation (hard rule)

The final test set is **never** used for:

- feature selection
- hyperparameter tuning
- threshold selection
- probability calibration
- model selection
- prompt tuning or LLM format iteration

Enforcement:

- Code loads the final test split only inside a dedicated evaluation entry
  point that writes a timestamped record into the run manifest each time it
  executes; the count of final-test evaluations per release must be 1.
- Reviews of any PR touching `data` or `evaluation` modules check this rule
  (stated in `ACCEPTANCE_CRITERIA.md`).

## Feature and missingness handling

- Missing values are encoded as `"?"` in the raw data; they are mapped to an
  explicit `Missing` category for categorical features. **No imputation of
  demographic or protected attributes is performed.**
- `race == "?"` becomes the explicit category `Unknown`; it is never guessed.
- High-cardinality diagnosis codes (`diag_1..3`, ICD-9) are grouped into the
  disease categories defined by Strack et al. (2014) to keep the model
  auditable.
- All encoding decisions are versioned (see Reproducibility) and listed in the
  Data Card.
- The encodings above are fixed rules, not fitted to data. If any data-fitted
  transform (scaling, target/frequency encoding, learned imputation) is ever
  introduced, it must be fitted on the **train split only** and applied
  unchanged to the other splits.

## Baselines and candidate models

| Model | Role |
|---|---|
| Majority-class and prevalence predictor | Sanity floor for AUPRC/Brier |
| Logistic regression (L2) | Interpretable baseline |
| LightGBM | Main model |

- Hyperparameter search uses cross-validation **within the train split only**,
  with grouped folds by `patient_nbr`.
- Class imbalance (~11% positive) is handled by class weighting; resampling
  methods are out of scope for v1.0 (recorded as a possible extension).

## Metrics

Reported for every model, on validation during development and on final test
once per release:

- AUROC
- AUPRC
- recall (sensitivity)
- specificity
- precision
- F1
- Brier score
- confusion matrix at the selected threshold
- calibration: reliability diagram, expected calibration error (ECE),
  calibration slope and intercept

All scalar metrics carry **95% confidence intervals from patient-level
bootstrap** (resampling patients, not encounters, to respect within-patient
correlation), with at least 1,000 replicates.

## Calibration and threshold analysis

- Calibration methods compared: Platt scaling and isotonic regression, both
  fitted **only on the calibration split**.
- Threshold analysis: precision/recall/specificity as functions of threshold,
  plus decision-curve-style net-benefit reporting; the operating threshold is
  chosen **on the validation split** and frozen before final-test evaluation.

## Explainability

- Global and local explanations via SHAP on the main model.
- SHAP values are presented as **model attribution, not causal evidence**;
  this wording constraint is repeated in the Model Card, the UI, and the LLM
  prompt contract.

## Fairness and subgroup analysis

Protected attributes: `race`, `gender`, `age` (10-year brackets).

- Data-quality and ethical limits are assessed first: `race` is missing for
  roughly 2% of encounters, `gender` contains a tiny `Unknown/Invalid` group,
  and `age` is bracketed. These constraints are stated with the results.
- Missing or unknown categories are **reported as their own groups, never
  imputed**.
- For each group: sample size (patients and encounters), AUROC, recall,
  specificity, precision, Brier score, and calibration, each with patient-level
  bootstrap CIs. **Group size and CI are always displayed together.**
- Groups below a minimum size (fewer than 50 patients in the evaluation split)
  are shown but explicitly labeled "insufficient sample for conclusions";
  no fairness claim, positive or negative, is made about them.
- No demographic feature is used to tune performance for a specific group;
  the analysis is observational audit, not constraint optimization (recorded
  as a possible extension).

## Uncertainty and abstention

- The deployed decision rule may **abstain**: if the calibrated probability
  falls inside an uncertainty band, the system returns "no reliable
  prediction" instead of a class.
- The band is selected on the **validation split** by sweeping band widths and
  reporting coverage (fraction of cases answered) against selective
  performance (metrics on answered cases).
- Final-test reporting includes the abstention rate and selective metrics at
  the frozen band.

## Error analysis

- At least 10 misclassified validation cases (false positives and false
  negatives) are analyzed with their SHAP attributions and documented in the
  evaluation report, using only feature values from the public dataset.

## Reproducibility

- **Acquisition:** a script downloads the dataset from the UCI repository and
  verifies a pinned SHA-256 checksum; a mismatch aborts the pipeline.
- **Versioning:** preprocessing code carries an explicit version string;
  every run manifest records: dataset checksum, preprocessing version, split
  hash, config hash, random seeds, package versions, and git commit.
- **No raw data in Git:** `data/` is gitignored; only split ID lists,
  checksums, configs, and aggregate reports are committed.
- **Determinism:** identical seeds and config must reproduce identical split
  hashes and metrics (asserted by an automated test).
