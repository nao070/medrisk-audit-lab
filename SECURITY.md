# Security Policy

## Supported stage

This repository is initially a portfolio and research project, not a production
medical system.

## Secret handling

Never commit:

- `.env`
- API keys
- tokens
- private keys
- credential files
- deployment secrets

Only `.env.example` may be committed.

## If a secret is exposed

1. Revoke the exposed credential immediately
2. Issue a new credential
3. Remove the credential from the working tree and Git history
4. Run secret scanning
5. Review logs and provider usage
6. Document the incident without reproducing the secret

## Data handling

Do not submit real patient data or personally identifiable medical information.

## Verification in this repository

- Normal CI declares no API-key secrets, uses only `push`/`pull_request`
  triggers (never `pull_request_target`), and makes no paid API calls;
  fork PRs therefore cannot exfiltrate anything.
- gitleaks runs in pre-commit, in CI, and over the full git history
  (`.gitleaks.toml`; untracked vendor dirs `.venv/`/`.tools/` are excluded
  from directory scans, tracked content is always scanned).
- Live LLM Mode is opt-in only (`MEDRISK_LLM_MODE=live` plus a key in the
  local environment), keys are read in exactly one module
  (`src/medrisk/config.py`), and the only workflow referencing a key secret
  is manual-only (`.github/workflows/live-llm-manual.yml`).
- Logs and reports contain aggregates, hashes, and verdicts — never raw
  dataset rows and never key material.

Full design: `docs/SECURITY_DESIGN.md`.

## Reporting

Report security issues privately to the repository owner before public disclosure.
