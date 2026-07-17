# MedRisk Audit Lab

> A reproducible evaluation platform for clinical risk prediction and safety-aware LLM explanations

## Status

v0.1 — data pipeline and baseline model (design documents approved; see
`docs/IMPLEMENTATION_ROADMAP.md` for what each version contains)

## Quickstart (no API key required)

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # install pinned dependencies
uv run medrisk pipeline # download (SHA-256 verified) → preprocess → split → train baseline
uv run pytest           # run the test suite (includes split non-overlap checks)
uv run streamlit run apps/streamlit_app/app.py  # demo UI (Demo LLM Mode, no key)
```

Further pipeline stages (all key-free): `train-main`, `ablation`, `explain`,
`audit`, `llm-cases`, `llm-generate`, `llm-eval` — see
`docs/ARCHITECTURE.md`. macOS note: if LightGBM fails to load libomp, run
`scripts/fix_lightgbm_macos.sh` (or `brew install libomp`).

The pipeline downloads the public UCI "Diabetes 130-US Hospitals" dataset
(CC BY 4.0, 1999–2008), builds the cohort, creates patient-level
train/calibration/validation/final-test splits, trains a logistic-regression
baseline, and writes validation metrics plus a reproducibility manifest to
`reports/runs/`. See `docs/DATA_CARD.md` for the dataset audit.

Note on data era: the dataset covers US hospitals from 1999–2008; results do
not generalize to current clinical practice or to present-day Japanese
healthcare.

## Purpose

This public GitHub portfolio project demonstrates how to design, evaluate, and
document a clinical machine-learning system without treating model output as a
medical diagnosis.

The project will focus on:

- reproducible medical risk prediction
- data-leakage prevention
- probability calibration
- threshold analysis
- subgroup evaluation
- explainability
- uncertainty and abstention
- constrained LLM explanations
- quantitative LLM failure evaluation
- API, UI, testing, CI, and security

## Operating modes

### Core ML Mode

Works without an API key.

### Demo LLM Mode

Shows stored, non-sensitive demonstration outputs without an API key.

### Live LLM Mode

Disabled by default. Users must configure their own API key locally.

## Safety boundary

This project is for research, education, and portfolio demonstration only.

It must not be used for:

- diagnosis
- treatment decisions
- emergency triage
- real patient care
- processing identifiable patient information

## Start here

1. Read `docs/PROJECT_BRIEF.md`
2. Run Claude Code from the repository root
3. Paste `prompts/01_design_only.md`
4. Review and approve the design before implementation
