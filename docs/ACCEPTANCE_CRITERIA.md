# Acceptance Criteria

All criteria must be objectively answerable with Yes or No.

## Repository and secrets

- [ ] `.env` does not appear in `git ls-files`
- [ ] only `.env.example` is committed
- [ ] secret scanning passes
- [ ] no realistic credential appears in documentation, tests, fixtures, or history
- [ ] normal CI does not invoke a paid external API

## API-key-free operation

- [ ] Core ML Mode works without an API key
- [ ] Demo LLM Mode works without an API key
- [ ] Live LLM Mode is disabled safely when no key is configured
- [ ] missing credentials do not crash unrelated features
- [ ] logs do not reveal credentials

## Reproducibility

- [ ] training and evaluation run through a documented command sequence
- [ ] dataset splitting is deterministic or versioned
- [ ] machine-readable metrics are saved
- [ ] dependencies are declared and reproducible

## ML evaluation

- [ ] AUROC is saved
- [ ] AUPRC is saved
- [ ] Brier score is saved
- [ ] calibration results are saved
- [ ] confusion-matrix values are saved
- [ ] at least 10 misclassified cases are analyzed
- [ ] final test data is not used for model selection or tuning

## LLM evaluation

- [ ] at least 100 versioned evaluation cases exist
- [ ] numeric alteration is measured
- [ ] fabricated features are measured
- [ ] unsupported assertions are measured
- [ ] abstention compliance is measured
- [ ] failed outputs are published alongside successful outputs

## Documentation and portfolio

- [ ] README states purpose, prohibited uses, and current status near the top
- [ ] Data Card exists
- [ ] Model Card exists
- [ ] security documentation exists
- [ ] Japanese and English README versions exist by v1.0
- [ ] architecture and evaluation results are visible without running the project

## Privacy and medical safety

- [ ] no identifiable patient information is used
- [ ] public demos do not accept real patient data
- [ ] the project does not claim diagnostic or treatment capability
- [ ] SHAP output is not described as causal evidence
- [ ] public Live LLM access does not expose the owner's unrestricted API key
