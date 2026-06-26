# RowHammer-OpenEnv specification bundle

This bundle contains the normative engineering specification and supporting machine-readable artifacts for a simulation-only reinforcement-learning environment built with OpenEnv and Ramulator 2.1.

## Contents

- `SPECIFICATION.md` — complete architecture, policy API, disturbance model, task system, implementation plan, and test suite.
- `schemas/action.schema.json` — logical `CallToolAction` and tool argument schemas.
- `schemas/program.schema.json` — transaction and direct-command AST schemas.
- `schemas/observation.schema.json` — public observation schema.
- `schemas/task.schema.json` — administrator task-instance schema.
- `schemas/profile.schema.json` — signed empirical-profile manifest schema.
- `schemas/errors.schema.json` — stable public error schema.
- `SOURCE_MANIFEST.template.yaml` — immutable source/data provenance template.
- `repository-tree.txt` — condensed proposed repository layout.
- `test-matrix.csv` — CI/admission/release test summary.
- `examples/` — validated task, action, and observation examples.
- `VALIDATION_REPORT.md` — local schema and cross-artifact validation results.

## Required pins

- Ramulator 2.1: `278f1effc3838099a6ffe0ad5f9f572fea80c948`
- OpenEnv v0.3.1: `7449c5dfe375c4c6e6f0827826925a46efd9249f`
- Primary DDR4 data: ReadDisturbanceVTS25 `5d734309457cc8a4ea3b1ec36b93932925548bac`

The specification requires real Ramulator execution, empirically calibrated profiles, genuine mitigations, and a fail-closed hardened sandbox. It prohibits mocks and simplified fallbacks in production, training, evaluation, integration testing, and release qualification.
