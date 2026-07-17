# Write the approved design documents

The proposed design is approved.

Do not implement the application yet.

Write the approved design into:

- `docs/PROJECT_BRIEF.md`
- `docs/PRODUCT_REQUIREMENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/SECURITY_DESIGN.md`
- `docs/DATA_AND_MODEL_PLAN.md`
- `docs/LLM_EVALUATION_PLAN.md`
- `docs/PORTFOLIO_PRESENTATION_PLAN.md`
- `docs/IMPLEMENTATION_ROADMAP.md`
- `docs/ACCEPTANCE_CRITERIA.md`

Preserve the safety and secret-handling requirements in `CLAUDE.md`.

After writing:

1. scan for TBD, TODO, placeholders, and vague requirements
2. resolve contradictions between documents
3. convert subjective completion language into objective Yes/No criteria
4. identify overengineering
5. verify that API-key-free operation remains the default
6. verify that no real or realistic-looking secret has been added
7. inspect `git diff`

Do not create source code, notebooks, datasets, or application directories.

Report:

1. files changed
2. major design decisions
3. unresolved decisions
4. acceptance-criteria quality
5. security review
6. items requiring human approval before implementation
