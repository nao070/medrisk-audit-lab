# Security Design

## Threat model

In scope:

1. Secret exposure: an API key committed, logged, printed in CI, or embedded
   in reports/screenshots.
2. Fork-PR exfiltration: a malicious pull request attempting to read repository
   secrets via CI.
3. Data-handling failure: raw dataset rows or realistic medical inputs leaking
   into logs, Git history, or published reports.
4. Dependency compromise: a vulnerable or malicious package version.
5. Cost abuse: CI or a public demo triggering paid LLM calls.

Out of scope: production-grade multi-tenant security, DoS resistance, and
clinical-environment threats — the system is a local portfolio project and is
documented as such.

## Assets to protect

- API keys and tokens
- local credentials
- dataset access credentials
- user inputs
- generated logs and reports
- Git history

## Attack surface

- Git repository and its history (public once pushed)
- GitHub Actions workflows (especially around fork PRs)
- Local `.env` and shell environment
- The Streamlit app's input surface
- Third-party dependencies

## Secret management

- All secrets live only in the local `.env` (gitignored) or the local shell
  environment; `.env.example` contains variable **names** and obviously fake
  placeholder values only.
- Code reads keys exclusively through a single config module; no other module
  touches the environment, which keeps "never log keys" auditable.
- Keys never appear in: logs, exceptions, run manifests, reports, committed
  fixtures, screenshots, prompts, or commit messages.
- GitHub Actions secrets are used only by the manual live-integration
  workflow (below); no normal workflow declares them.

## Privacy and logging

- The dataset is public and de-identified; nevertheless raw feature rows are
  not logged by default and are never committed.
- Live LLM inputs/outputs are not persisted by default (see
  `PRODUCT_REQUIREMENTS.md`, Live LLM behavior).
- Published artifacts are aggregates: metrics, plots, curated de-identified
  example cases, and LLM evaluation records built from them.

## GitHub protections

Enabled at remote-creation time (requires human approval to create the
remote):

- Secret scanning and push protection
- Branch protection on `main` requiring green CI
- Dependabot security alerts and version updates

## CI workflow security

- Normal CI triggers: `push` and `pull_request` only. The `pull_request_target`
  trigger is not used anywhere.
- Normal CI declares **no** secrets; therefore fork PRs cannot exfiltrate any.
- Normal CI makes **no** paid API calls: LLM tests run against StubClient and
  ReplayClient stored outputs.
- Live LLM integration tests exist in a separate workflow triggered only by
  `workflow_dispatch` (manual), restricted to the repository owner.

## Local tooling

- pre-commit hooks: gitleaks (secret scan), ruff, standard hygiene hooks
  (end-of-file, trailing whitespace, large-file guard blocking accidental
  dataset commits).
- The same gitleaks scan runs in CI so the guarantee does not depend on local
  hook installation.

## Dependency security

- Dependencies pinned via a lock file; Dependabot monitors advisories.
- `pip-audit` runs in CI and fails the build on known critical
  vulnerabilities.
- New dependencies require a stated justification in the PR description
  (keeps the dependency surface small).

## Incident response

If a secret is suspected to be exposed:

1. Stop the affected operation immediately.
2. Do not display or copy the full value anywhere.
3. Revoke and rotate the credential at the provider first.
4. Inspect `git log -p`, GitHub secret-scanning alerts, and CI logs to bound
   the exposure window.
5. If it reached Git history: rewrite history only with explicit human
   approval, then force-push (also approval-gated) and invalidate caches.
6. Record the incident and the fix in `SECURITY.md`'s incident log section.

See also the repository-level policy in `../SECURITY.md`.

## Verification

Each release verifies, with commands recorded in `ACCEPTANCE_CRITERIA.md`:

- `git ls-files` contains no `.env`, no data files, no key-like strings
  (gitleaks clean over history).
- CI logs contain no secret material and no paid-API calls in normal runs.
- Fork-PR simulation: a PR from a fork runs CI successfully with zero secrets
  available.
- Grep-based check that only the config module reads LLM key variables.
