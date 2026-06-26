# RowHammer-OpenEnv concise specification bundle

This is a compact v2 replacement for the earlier verbose bundle. It keeps the engineering decisions and safety constraints, but removes oversized prose and giant schemas.

Files:

- `SPEC.md` — concise normative specification.
- `IMPLEMENTATION_PLAN.md` — phase order, serial gates, parallel workstreams, and acceptance criteria.
- `TEST_PLAN.md` — release-oriented test suite.
- `REPOSITORY_TREE.md` — proposed repository skeleton.
- `SOURCE_MANIFEST.template.yaml` — source/provenance manifest template.
- `schemas/*.schema.json` — compact structural schemas for the policy action, observation, and task config contracts.
- `examples/*.json` — small examples validated against the schemas.
- `VALIDATION_REPORT.md` — local validation summary.

The schemas intentionally validate only the wire shape. DRAM legality, timing, disturbance statistics, sandbox policy, hidden-state non-leakage, and reward correctness are enforced by the simulator, sandbox, and tests rather than encoded in thousands of schema lines.
