# Validation report

## P20 Release Re-Qualification

The release gate is `scripts/verify_release.py`. It now checks:

- all admitted phase gates: P0-P9 and P11-P19;
- the full `tests/` suite with zero required skips;
- deterministic replay for seeds 20, 21, and 22 under `SOURCE_MANIFEST.yaml`;
- no-mock symbol scanning over `cpp/`, `rowhammer_env/`, `profile_builder/`,
  `sdk/`, `scripts/`, and `tests/`;
- source-pin provenance for admitted sources and tracked-file hygiene;
- release bundle artifacts: `SOURCE_MANIFEST.yaml`, this validation report,
  `spec/SBOM.md`, the README, and the admitted DDR4 model card.

`SOURCE_MANIFEST.yaml` is marked `phase: P20`; deferred HBM2 source fields may
remain pending because that source is not admitted.

- `action` schema is valid Draft 2020-12.
- `observation` schema is valid Draft 2020-12.
- `task` schema is valid Draft 2020-12.
- `examples/action.issue.json` validates against `schemas/action.schema.json`.
- `examples/observation.reset.json` validates against `schemas/observation.schema.json`.
- `examples/task.known_target.json` validates against `schemas/task.schema.json`.
- Markdown code fences are balanced in the main documents.

## Line counts

- `IMPLEMENTATION_PLAN.md`: 64 lines
- `README.md`: 16 lines
- `REPOSITORY_TREE.md`: 64 lines
- `SOURCE_MANIFEST.template.yaml`: 40 lines
- `SPEC.md`: 281 lines
- `TEST_PLAN.md`: 88 lines
- `examples/action.issue.json`: 36 lines
- `examples/observation.reset.json`: 54 lines
- `examples/task.known_target.json`: 30 lines
- `schemas/action.schema.json`: 157 lines
- `schemas/observation.schema.json`: 105 lines
- `schemas/task.schema.json`: 169 lines
