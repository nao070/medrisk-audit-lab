# Start Here

Current state: **v1.0 — all milestones implemented** (data pipeline,
audited LightGBM model, one-shot final-test evaluation, constrained LLM
explanation layer with quantitative evaluation, Streamlit demo UI).
Japanese overview: [README.md](README.md) — English: [README_EN.md](README_EN.md).

## Run it (no API key required)

```bash
uv sync                                          # pinned dependencies
uv run medrisk pipeline                          # download (SHA-256 verified) → preprocess → split → baseline
uv run pytest                                    # 76 tests incl. split non-overlap
uv run streamlit run apps/streamlit_app/app.py   # demo UI (Demo LLM Mode)
```

macOS: if LightGBM cannot load libomp, run `scripts/fix_lightgbm_macos.sh`
(or `brew install libomp`).

## Where to look

- Results: `reports/final/FINAL_REPORT.md` (final-test metrics with CIs)
- Dataset audit: `docs/DATA_CARD.md` — model: `docs/MODEL_CARD.md`
- Design documents: `docs/` (nine documents, written before implementation)
- Development history: staged prompts in `prompts/` (design → specs →
  implementation), executed with Claude Code under `CLAUDE.md` constraints
- Before publishing to GitHub: `docs/RELEASE_CHECKLIST.md`

## Live LLM Mode (optional, off by default)

For local experiments only:

```bash
cp .env.example .env   # then edit .env locally; .env must remain untracked
```

Set `MEDRISK_LLM_MODE=live` and a local `ANTHROPIC_API_KEY`. Never put a
real API key in tracked files. A missing key never breaks any other
feature. Live API calls are paid actions — run them deliberately.
