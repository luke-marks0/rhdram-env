# Local validation report

Generated: 2026-06-26T06:05:18.102083+00:00

- PASS: `schemas/action.schema.json` parses and is a valid Draft 2020-12 schema.
- PASS: `schemas/errors.schema.json` parses and is a valid Draft 2020-12 schema.
- PASS: `schemas/observation.schema.json` parses and is a valid Draft 2020-12 schema.
- PASS: `schemas/profile.schema.json` parses and is a valid Draft 2020-12 schema.
- PASS: `schemas/program.schema.json` parses and is a valid Draft 2020-12 schema.
- PASS: `schemas/task.schema.json` parses and is a valid Draft 2020-12 schema.
- PASS: `examples/command_action.json` validates against `action.schema.json`.
- PASS: `examples/initial_observation.json` validates against `observation.schema.json`.
- PASS: `examples/task.yaml` validates against `task.schema.json`.
- PASS: `SOURCE_MANIFEST.template.yaml` parses and contains immutable-source policy plus seven source entries.
- PASS: `test-matrix.csv` parses with 43 test rows.
- PASS: `SPECIFICATION.md` has balanced code fences, required no-mock block, implementation plan, and test-suite section (3109 lines).
- PASS: embedded normative task and initial-observation examples validate against their schemas.
- PASS: required Ramulator, OpenEnv, and VTS25 pins are present.

This report covers local syntax, schema, and cross-artifact consistency. Runtime simulator, calibration, mitigation, and sandbox conformance tests are specified but require the future repository implementation.
