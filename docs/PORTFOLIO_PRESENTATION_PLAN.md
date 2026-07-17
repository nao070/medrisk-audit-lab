# Portfolio Presentation Plan

## Three-minute reviewer path

1. What the project is — README title block: "an audit platform for a
   clinical risk model", one paragraph, disclaimer immediately visible.
2. Why the problem matters — leakage, calibration, fairness, and LLM
   hallucination as the real problems of medical ML.
3. What was built — architecture diagram and the three modes.
4. How it was evaluated — headline results table with CIs, calibration plot,
   subgroup table, LLM validator rates per condition.
5. What failed — misclassification analysis and published LLM failure cases.
6. How safety and security were handled — link to `SECURITY_DESIGN.md`,
   key-free defaults, fork-safe CI.
7. How to reproduce it — three-command quickstart (setup, pipeline, app),
   all key-free.

## README opening section

Order, top of file: project name and one-line value proposition →
non-medical-device disclaimer and data-era limitation (1999–2008 US data; not
generalizable to current practice or to Japanese healthcare) → status badges
(CI, license) → demo GIF → three-command quickstart → results summary →
"Why this is not just another readmission classifier" section.

## Visual assets

- demo GIF (Streamlit walkthrough, Demo LLM Mode)
- screenshots of each Streamlit page
- architecture diagram (mermaid, same as `ARCHITECTURE.md`)
- model-comparison table with bootstrap CIs
- calibration plot (before/after calibration)
- subgroup results table with group sizes and CIs
- abstention coverage-vs-performance curve
- LLM evaluation table per condition, plus one annotated failure example

All assets are generated from committed reports so they can be regenerated;
no asset may show any key, token, or local path containing a username.

## Japanese and English documentation

- `README.md` (English) is canonical; `README.ja.md` (Japanese) mirrors the
  opening section, quickstart, results summary, and disclaimers by v1.0.
- Design documents remain English-only; the Japanese README links to them.
- Both READMEs carry the same prohibited-uses list.

## Interview talking points

Documented in the repository wiki-style section at v1.0 (short, factual, tied
to evidence in `reports/`):

1. Why patient-level splitting matters and how the non-overlap test proves it.
2. Why the final test set was touched exactly once, and how the run manifest
   shows it.
3. What calibration adds beyond AUROC for clinical risk scores.
4. Why the system abstains and what the coverage/performance trade-off was.
5. What the subgroup audit can and cannot conclude given sample sizes.
6. How the LLM is constrained, how its failures were measured, and one
   concrete failure it produced.
7. How the repo guarantees no key and no paid call in CI, including for
   fork PRs.
