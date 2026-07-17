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

## Reporting

Report security issues privately to the repository owner before public disclosure.
