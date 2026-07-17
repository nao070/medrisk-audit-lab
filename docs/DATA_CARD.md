# Data Card — Diabetes 130-US Hospitals (v0.1)

## Source and license

- **Dataset:** "Diabetes 130-US Hospitals for Years 1999–2008",
  UCI Machine Learning Repository, dataset ID 296.
- **License:** CC BY 4.0 (verified on the UCI dataset page, 2026-07-17).
  Redistribution is permitted with attribution; nevertheless this repository
  does **not** commit raw data — only the acquisition procedure, the pinned
  checksum, split ID lists, and aggregate results.
- **Citation:** Strack B. et al., "Impact of HbA1c Measurement on Hospital
  Readmission Rates: Analysis of 70,000 Clinical Database Patient Records",
  BioMed Research International, 2014.
- **Acquisition:** `medrisk download` fetches the archive and verifies its
  SHA-256; a mismatch aborts the pipeline.
  - Archive SHA-256: `f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a`
    (verified 2026-07-17; archive size 3,170,254 bytes)
- **Preprocessing version:** `v1` (`src/medrisk/data/preprocess.py`)

## Era and generalization limits

- US hospital encounters, **1999–2008**. Clinical practice, coding, drugs,
  and populations have changed since.
- Results **do not generalize to current clinical settings** and **do not
  transfer to present-day Japanese healthcare**.
- Research, education, and portfolio demonstration only. Not a medical
  device; never for diagnosis, treatment, or triage.

## Prediction target

- Positive (1): `readmitted == "<30"` — readmission within 30 days of discharge.
- Negative (0): `readmitted == ">30"` or `"NO"`.
- Known label limits: negatives include readmissions to other hospital
  systems, post-discharge deaths, and loss to follow-up; the ordinal
  difference between ">30" and "NO" is discarded. See
  `docs/DATA_AND_MODEL_PLAN.md`.

## Prediction time point

Discharge from the index encounter. Every kept feature is knowable at or
before discharge. The dataset has no admission/discharge timestamps, so a
temporal (past→future) validation split is impossible; patient-level random
splits are used instead. This is a stated limitation.

## Cohort (v1, measured on the verified archive)

| Step | Count |
|---|---|
| Raw encounters | 101,766 |
| Excluded: death or hospice discharge (`discharge_disposition_id` ∈ {11, 13, 14, 19, 20, 21}) | 2,423 |
| Final cohort encounters | 99,343 |
| Unique patients | 69,990 |
| Positive rate (`<30` readmission) | 11.39% |

`gender == "Unknown/Invalid"` rows are retained (never imputed) and are
reported as their own group in subgroup analysis (v0.3).

## Split (patient-level, seed 20260717)

| Split | Patients | Allowed uses |
|---|---|---|
| train | 41,994 | fitting, feature selection, grouped-CV tuning |
| calibration | 6,999 | probability calibration only (v0.2) |
| validation | 10,498 | model/threshold/abstention selection; v0.1 baseline evaluation |
| final test | 10,499 | one-shot final evaluation (v1.0); **never touched before** |

- Split lists: `data/splits/v1/*.txt`; combined SHA-256
  `da093bcccc4176ea6f27866c3c817ded85d4a15e21fd8b3950ef65e3d89ecfdc`.
- Non-overlap is asserted at creation time and by a CI test
  (`tests/test_splits.py::test_committed_split_lists_have_no_overlap`).

## Feature audit (all 50 columns)

Rules (docs/DATA_AND_MODEL_PLAN.md): a feature must be knowable at or before
discharge; features encoding the outcome are excluded; every decision is
recorded here. All v1 encodings are fixed rules, not fitted to data.

### Identifiers and target

| Column | Decision | Reason |
|---|---|---|
| `encounter_id` | Key only, never a feature | Identifier; ordering may proxy calendar time |
| `patient_nbr` | Key only, never a feature | De-identified patient key; used solely for splitting |
| `readmitted` | Target only | Outcome |

### Excluded features

| Column | Decision | Reason |
|---|---|---|
| `weight` | Excluded | ~97% missing; too sparse to be informative |
| `examide` | Excluded | Constant ("No") for every row |
| `citoglipton` | Excluded | Constant ("No") for every row |

### Kept features — demographics and administrative

| Column | Decision | Notes |
|---|---|---|
| `race` | Keep | `?` → explicit `Unknown` category; never imputed (protected attribute) |
| `gender` | Keep | `Unknown/Invalid` retained as its own category |
| `age` | Keep | 10-year brackets as provided |
| `admission_type_id` | Keep | Known at admission; categorical code |
| `discharge_disposition_id` | **Keep** (non-excluded values; decision confirmed by v0.2 ablation) | Destination is decided at discharge, hence knowable at the prediction point, but may proxy unmeasured severity. v0.2 ablation (`reports/ablation/discharge_disposition.json`): removing it costs 0.028 validation AUROC (0.677 → 0.649) with no leakage mechanism identified (the feature is fixed at discharge, and outcome-defined values 11/13/14/19/20/21 are already excluded from the cohort), so the feature is kept. See the note on IDs 9/12/15 below |

**Note on retained disposition IDs 9, 12, 15** (release-audit measurement,
2026-07-17, official definitions from the archive's `IDS_mapping.csv`):
ID 9 "Admitted as an inpatient to this hospital" (n=21, 30-day readmission
rate 42.9%), ID 12 "Still patient or expected to return for outpatient
services" (n=3, 66.7%), ID 15 "swing-bed transfer" (n=63, 44.4%) show
elevated rates versus the 11.4% cohort baseline. They are retained because
each value is determined at the discharge decision (no post-discharge
information) and planned continuation of care is legitimate predictive
signal. Limits: together they cover only 87 encounters (0.09% of the
cohort), below LightGBM's `min_child_samples=50` in most splits, so their
individual effect on the model is negligible and their rate estimates carry
wide uncertainty.
| `admission_source_id` | Keep | Known at admission |
| `payer_code` | Keep | Known at admission; `?` → `Missing` |
| `medical_specialty` | Keep | Admitting physician specialty; `?` → `Missing` (~49% missing) |
| `time_in_hospital` | Keep | Length of stay; known at discharge |

### Kept features — utilization history (strictly pre-encounter)

| Column | Decision | Notes |
|---|---|---|
| `number_outpatient` | Keep | Outpatient visits in the preceding year |
| `number_emergency` | Keep | Emergency visits in the preceding year |
| `number_inpatient` | Keep | Inpatient visits in the preceding year |

### Kept features — in-stay measurements (available at discharge)

| Column | Decision | Notes |
|---|---|---|
| `num_lab_procedures` | Keep | Count during stay |
| `num_procedures` | Keep | Count during stay |
| `num_medications` | Keep | Count during stay |
| `number_diagnoses` | Keep | Count on the record |
| `max_glu_serum` | Keep | `None` is a legitimate "not measured" category |
| `A1Cresult` | Keep | `None` is a legitimate "not measured" category |
| `diag_1`, `diag_2`, `diag_3` | Keep, grouped | Raw ICD-9 codes mapped to the Strack et al. disease groups (`*_group` features); `?` → `Missing` |

### Kept features — medications (in-stay prescriptions, known at discharge)

`metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`,
`acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`,
`rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`,
`insulin`, `glyburide-metformin`, `glipizide-metformin`,
`glimepiride-pioglitazone`, `metformin-rosiglitazone`,
`metformin-pioglitazone` — all Keep (dose-direction categories
No/Steady/Up/Down); plus `change` and `diabetesMed` (Keep, summary flags).

## Protected attributes and fairness constraints

- `race`, `gender`, `age` are used for **subgroup audit** (v0.3), not for
  performance optimization.
- Missing/unknown categories are reported as groups, never imputed.
- Subgroup results must always pair metrics with group size and confidence
  intervals; groups under 50 patients are labeled "insufficient sample for
  conclusions".

## Reproducibility record

Every run manifest (`reports/runs/<run_id>/manifest.json`) stores: archive
SHA-256, preprocessing version, split SHA-256, config SHA-256, seeds,
package versions, and the git commit. Identical seed and config reproduce
identical split hashes and metrics (verified 2026-07-17; determinism test in
`tests/test_splits.py`).
