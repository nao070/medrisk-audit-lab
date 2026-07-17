# MedRisk Audit Lab

> A reproducible evaluation platform for clinical risk prediction and safety-aware LLM explanations

## Status

Design phase

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
