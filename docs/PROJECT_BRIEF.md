# Project Brief

## Background

Machine-learning models are increasingly proposed for clinical risk
prediction, but most public portfolio projects stop at a headline accuracy
number. In real medical AI work, the harder and more valuable questions are:
Is the evaluation leak-free? Are the probabilities calibrated? For whom does
the model fail? When should the system refuse to answer? Can an LLM explain
the model's output without inventing medicine? Can anyone reproduce the
result?

**MedRisk Audit Lab** is a public portfolio project that answers those
questions end to end. It is deliberately framed as an **audit platform for a
medical risk model**, not as another readmission classifier.

## Problem

Using the public UCI "Diabetes 130-US Hospitals" dataset (1999–2008), the
project predicts whether an inpatient encounter is followed by a hospital
readmission within 30 days of discharge, and then subjects that model to a
full safety audit: patient-level leak prevention, calibration, threshold and
abstention analysis, subgroup evaluation with uncertainty, constrained LLM
explanation, and quantitative evaluation of the LLM output itself.

## Intended audience

- medical AI recruiters
- machine-learning engineers
- LLM application engineers
- backend engineers

## Portfolio capabilities to demonstrate

1. Leakage-aware ML evaluation: patient-level splits, isolated final test set,
   automated non-overlap tests.
2. Probability quality: calibration, Brier score, threshold analysis,
   confidence intervals via patient-level bootstrap.
3. Safety mechanisms: abstention on uncertain cases, subgroup audit with
   explicit sample-size caveats.
4. Constrained LLM engineering: the LLM explains structured model output under
   a strict contract and is itself evaluated quantitatively, with failures
   published.
5. Security engineering: secret hygiene, CI without paid API calls,
   fork-safe workflows, incident response.
6. Reproducibility: checksummed data acquisition, versioned splits and
   configs, deterministic reruns, run manifests.

## Differentiation from common student projects

A typical student project trains a classifier on this dataset and reports
accuracy. This project treats the classifier as the *subject under audit*:
the deliverable is the audit methodology and its honest results — including
failure cases — not a leaderboard score. Reported performance is never
improved by weakening evaluation validity.

## In scope

- Core ML Mode: preprocessing, training, evaluation, calibration, threshold
  analysis, explainability (SHAP), subgroup evaluation, uncertainty and
  abstention — all without any API key.
- Demo LLM Mode: stored LLM explanations and stored evaluation results
  (successes and failures) — without any API key.
- Live LLM Mode: optional, disabled by default, user-supplied local key.
- A Streamlit demo UI and a CLI over a tested Python package.
- Security tooling, CI, and the nine design documents plus Data Card and
  Model Card.

## Out of scope

- diagnosis
- treatment recommendations
- emergency triage
- real patient care
- identifiable patient information
- real-time or production inference infrastructure
- clinical validation of any kind
- causal claims from SHAP attributions
- public Live LLM access (unless separately approved)

## Data era disclaimer

The dataset covers US hospitals from 1999–2008. Results do not generalize to
current clinical practice and do not transfer to present-day Japanese
healthcare. The project exists for research, education, and portfolio
demonstration only.

## Future extensions (not v1.0 dependencies)

After v1.0, a credentialed MIMIC-IV or the public MIMIC-IV demo subset may be
added as a second audit target. MIMIC is **not** a dependency of any v0.x or
v1.0 deliverable; see `IMPLEMENTATION_ROADMAP.md`.

## Success definition

All mandatory criteria in `ACCEPTANCE_CRITERIA.md` pass with objective
evidence.
