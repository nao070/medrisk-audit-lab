# GitHub Publication Checklist (pre-push)

Every box must be checked by a human before `git push` to a public remote.
Items marked (owner) are actions only the repository owner can perform.

## Secrets and data

- [ ] `git ls-files | grep -E '^\.env'` returns exactly `.env.example`
- [ ] gitleaks full-history scan is clean (`.tools/gitleaks git --redact`)
- [ ] gitleaks working-tree scan is clean (`.tools/gitleaks dir --redact .`)
- [ ] No file under `data/` except `data/splits/` and `data/llm_eval/` is tracked
- [ ] `.env.example` contains only variable names and obviously fake values

## Quality gates

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy` passes
- [ ] `uv run pytest` passes (includes split non-overlap, final-test guard,
      stored-LLM-asset validation, and UI smoke tests)
- [ ] `uv run pip-audit --skip-editable` reports no known vulnerabilities

## Documentation

- [ ] README.md (Japanese) and README_EN.md agree on disclaimers, status,
      and quickstart
- [ ] MODEL_CARD, DATA_CARD, ETHICS_AND_LIMITATIONS, SECURITY.md are final
- [ ] `docs/ACCEPTANCE_CRITERIA.md` evaluation is complete with evidence
- [ ] License file decision made (currently ABSENT — decide before publishing)

## GitHub settings after creating the remote (owner)

- [ ] Create the remote as public/private per decision (approval-gated)
- [ ] Enable secret scanning AND push protection
- [ ] Enable Dependabot alerts and security updates
- [ ] Protect `main` (require CI green)
- [ ] Verify the first CI run is green and declares no secrets
- [ ] Confirm `live-llm-manual.yml` shows no stored `ANTHROPIC_API_KEY`
      secret unless deliberately added later

## Deliberate omissions (do NOT enable without a new approval)

- [ ] No public deployment (Streamlit Cloud / HF Spaces)
- [ ] No public Live LLM access
- [ ] No paid-API calls from CI
