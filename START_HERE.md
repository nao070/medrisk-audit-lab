# Start Here

## 1. Extract

Place the contents of this package in:

`/Users/watanabe/Desktop/MedRiskAuditLab`

## 2. Open Terminal

```bash
cd "/Users/watanabe/Desktop/MedRiskAuditLab"
```

## 3. Initialize Git if needed

```bash
git init
git add .
git commit -m "chore: add project design scaffold"
```

## 4. Start Claude Code without confirmation prompts

```bash
claude --dangerously-skip-permissions
```

The repository instructions still prohibit work outside the project,
secret access, pushes, deployment, and paid API calls. This is guidance plus
Claude Code configuration, not a perfect operating-system security boundary.

## 5. Paste the design prompt

Open and paste:

`prompts/01_design_only.md`

## 6. Review before implementation

Approve the design before using:

- `prompts/02_write_approved_specs.md`
- `prompts/03_implement_v0_1.md`

## Important

Do not put a real API key in this repository.

For Live LLM Mode later:

```bash
cp .env.example .env
```

Then edit `.env` locally. `.env` must remain untracked.
