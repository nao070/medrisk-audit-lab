# Implement only v0.1

The design documents are approved.

Implement only the v0.1 scope defined in
`docs/IMPLEMENTATION_ROADMAP.md`.

Do not implement later-version functionality early.

Before implementation:

1. read all relevant design documents
2. inspect the current Git status
3. list the v0.1 work items and acceptance criteria
4. identify data-leakage, security, privacy, cost, and reproducibility risks
5. define the implementation and verification order

During implementation:

- use public or synthetic data only as approved by the design
- never read or create `.env`
- never invoke a paid API
- keep Core ML Mode independent of all LLM providers
- add tests with each implemented behavior
- update documentation with behavior changes
- make small coherent commits

After implementation run all applicable checks, including:

- formatting
- linting
- type checking
- unit tests
- secret scanning
- reproducibility command
- `git diff`

Evaluate every applicable v0.1 acceptance criterion as Yes or No.
Do not mark the milestone complete while a mandatory criterion is No.

Report:

1. summary
2. files changed
3. commands executed
4. test and verification results
5. security and privacy checks
6. acceptance-criteria results
7. known limitations
8. recommended next task
