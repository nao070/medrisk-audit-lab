# MedRisk Audit Lab — Design-only instruction for Claude Code

## Objective

Design a public GitHub portfolio project named **MedRisk Audit Lab**.

The project must demonstrate reproducible, safety-aware clinical machine learning,
constrained LLM explanation, quantitative LLM evaluation, software engineering,
security, and clear portfolio presentation.

The primary audience is recruiters and engineers working in medical AI,
machine learning, LLM applications, or backend engineering.

## Critical rule

Do not implement the application in this session.

Do not create source code, notebooks, dependencies, datasets, or application
directories.

First inspect the current repository, Git state, `CLAUDE.md`, and existing
documents. Then produce a design proposal.

Ask only questions that materially change scope, dataset selection, evaluation
validity, security, cost, or public-release strategy. Ask one question at a time.

## Required operating modes

### Core ML Mode

Must work without an API key:

- preprocessing
- training
- evaluation
- calibration
- threshold analysis
- explainability
- subgroup evaluation
- uncertainty analysis
- abstention

### Demo LLM Mode

Must work without an API key:

- stored synthetic or non-sensitive examples
- stored LLM outputs
- stored evaluation results
- successful and failed examples

### Live LLM Mode

Optional and disabled by default:

- user supplies their own key locally
- keys are loaded only from local environment configuration
- keys are never logged
- inputs and outputs are not stored by default
- missing keys never break Core ML Mode or Demo LLM Mode
- public deployment must keep this mode disabled unless separately approved

## Compare alternatives

Before choosing the design, compare 2–3 credible options for:

- prediction task
- public medical dataset
- UI approach
- LLM provider abstraction
- experiment tracking
- deployment strategy
- repository structure

For each option compare:

- portfolio value
- medical-AI validity
- implementation difficulty
- reproducibility
- cost
- security
- M1 Pro 32 GB feasibility
- Kaggle feasibility
- differentiation from common student projects

Recommend one option and explain why.

## Medical safety constraints

The project is for research, education, and portfolio demonstration.

It must not:

- diagnose
- recommend treatment
- perform emergency triage
- process identifiable patient information
- present SHAP as causal evidence
- present unverified LLM output as medical truth

The LLM is an explanation component, not the predictor.

## Security requirements

The design must cover:

- `.env.example`
- `.gitignore`
- local secret configuration
- GitHub secret scanning and push protection
- a local secret scanner
- pre-commit checks
- GitHub Actions
- no secrets in forked pull-request workflows
- normal tests without live paid API calls
- manual-only live API integration tests
- incident response and credential rotation
- logging that excludes secrets and raw medical inputs

## Documents to complete after approval

- `docs/PROJECT_BRIEF.md`
- `docs/PRODUCT_REQUIREMENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/SECURITY_DESIGN.md`
- `docs/DATA_AND_MODEL_PLAN.md`
- `docs/LLM_EVALUATION_PLAN.md`
- `docs/PORTFOLIO_PRESENTATION_PLAN.md`
- `docs/IMPLEMENTATION_ROADMAP.md`
- `docs/ACCEPTANCE_CRITERIA.md`

Do not edit them yet. First present what each document should contain.

## Required response format

1. Current repository state
2. Material unknowns
3. Two or three design alternatives
4. Comparison table
5. Recommended design
6. Project scope and exclusions
7. System architecture
8. Security design
9. Data and model plan
10. LLM evaluation plan
11. GitHub portfolio presentation
12. Document plan
13. Versioned implementation roadmap
14. Verification strategy
15. Objective Yes/No acceptance criteria
16. Decisions requiring human approval

End with only the remaining material question.

Do not implement or create files in this session.
