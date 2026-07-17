# MedRisk Audit Lab — Claude Code Instructions

## Project purpose

This repository is a public GitHub portfolio project demonstrating reproducible,
safety-aware clinical machine learning and constrained LLM explanation evaluation.

It is not a medical device and must not be presented as suitable for diagnosis,
treatment decisions, emergency triage, or real patient care.

## Authorized workspace

The working root is:

`/Users/watanabe/Desktop/MedRiskAuditLab`

Within this directory and its descendants, Claude Code may autonomously:

- create, read, edit, move, rename, and delete project files
- create virtual environments
- install project dependencies
- run formatting, linting, type checking, tests, builds, and local applications
- initialize and manage the local Git repository
- create branches
- run `git add` and `git commit`
- update source code, tests, documentation, and local configuration

Do not modify, move, delete, or create files outside the working root.

The following operations always require explicit human approval:

- `git push`
- creating or changing remote repositories
- opening, merging, or closing pull requests
- deployment
- invoking paid APIs
- enabling public Live LLM access
- publishing datasets, model artifacts, or generated reports
- destructive Git history rewriting
- system-level configuration changes
- `sudo`

## Development priorities

Use this order when making trade-offs:

1. Safety
2. Prevention of secret and personal-data exposure
3. Evaluation validity and prevention of data leakage
4. Reproducibility
5. Automated verification
6. Clear documentation
7. Maintainability
8. Model performance
9. Visual polish

Never improve reported model performance by weakening evaluation validity.

## Mandatory work process

Before a non-trivial change:

1. Read the relevant documents in `docs/`
2. Inspect the current implementation and tests
3. State the intended change
4. Identify security, privacy, leakage, and reproducibility risks
5. Define verification steps

After implementation:

1. Run the relevant checks
2. Inspect `git diff`
3. Update documentation in the same change
4. Evaluate applicable acceptance criteria as Yes or No
5. Report failures and unverified items honestly

## Secrets

Never read, print, edit, copy, summarize, expose, or commit:

- `.env`
- `.env.*` except `.env.example`
- API keys
- access tokens
- private keys
- credential files
- deployment secrets

Never place realistic-looking credentials in source code, tests, fixtures,
documentation, screenshots, logs, prompts, or commit messages.

If a possible secret is found:

1. stop the affected operation
2. do not display the full value
3. report only the file and approximate location
4. recommend immediate revocation and rotation
5. check Git history and secret-scanning results

## Medical and personal data

Do not use real patient information.

Only use:

- authorized public datasets
- synthetic examples
- appropriately de-identified examples permitted by the dataset license

Do not log raw medical inputs by default.

## Machine-learning evaluation

Prevent leakage between training, calibration, validation, and final test data.

When patient identifiers exist, split by patient rather than row.

Never use the final test set for:

- feature selection
- hyperparameter tuning
- threshold selection
- probability calibration
- prompt tuning
- model selection

Report at least:

- AUROC
- AUPRC
- recall
- specificity
- precision
- F1
- Brier score
- confusion matrix
- calibration results

Add confidence intervals or uncertainty estimates where technically valid.

## LLM constraints

The LLM is an explanation component, not the clinical predictor.

The LLM must receive structured model outputs and must not independently invent:

- diagnoses
- treatment recommendations
- emergency assessments
- probabilities
- features not present in the input

Live LLM Mode must be optional and disabled by default.

Core ML Mode and Demo LLM Mode must work without an API key.

Normal tests and CI must not call paid external APIs.

## Completion report

End each task with:

1. Summary
2. Files changed
3. Commands executed
4. Verification results
5. Security and privacy checks
6. Known limitations
7. Recommended next task
