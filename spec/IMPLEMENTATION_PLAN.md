# Implementation plan with serial gates and parallel work

## Planning model

Build a thin, real vertical slice first, then broaden. The critical path is:

```text
P0 provenance/threat model
  -> P1 real Ramulator/OpenEnv bootstrap
  -> P2 simulator worker + command semantics
  -> P3 first empirical profile
  -> P4 disturbance engine integration
  -> P5 OpenEnv tools/tasks/reward
  -> P6 sandboxed script path, if scripts are in release scope
  -> P10 release qualification
```

Preparatory work for later phases can start early, but a feature cannot be advertised until its serial gate passes with the real implementation.

## Phase table

| Phase | Serial or parallel? | Goal | Hard prerequisites | Parallelizable work during phase | Admission gate |
|---|---|---|---|---|---|
| **P0. Provenance, threat model, no-mock policy** | Serial gate; prep can parallelize | Freeze source pins, licenses, safety invariants, host targets, and acceptance rules. | None. | Repo skeleton, CI stubs, source manifest template, sandbox runtime evaluation, ADR drafts. | Approved manifest format, denylist, no-mock CI policy, and threat model. |
| **P1. Real vertical bootstrap** | Mostly serial | `reset -> step -> Ramulator 2.1 External/frontend path -> completion` with real simulated memory request. | P0 pins. | OpenEnv contract tests, build containers, baseline DDR4 config, worker IDL draft. | One real read/write request completes through Ramulator; no fake memory or canned result. |
| **P2. Worker, protocol, memory, commands** | Serial for public protocol; implementation can split | Separate simulator worker, typed RPC, fresh episode lifecycle, sparse memory overlay, direct command execution, event trace. | P1. | Fuzz tests, command AST parser, address projection tests, event-log tooling. | Functional reads/writes and accepted command sequences are deterministic and legally checked by Ramulator. |
| **P3. Empirical profile pipeline** | Parallel after P0; serial admission | Ingest VTS25 DDR4 data, canonicalize, fit, validate, package, and sign first profile. | P0; P2 event schema for final integration. | Data download verifier, parsers, fitting notebooks converted to tests, model-card template. | Profile has source hashes, fitted parameters, held-out validation, supported-domain metadata, and signature. |
| **P4. Disturbance engine** | Serial integration | Consume issued events, accumulate exposure, apply fitted latent thresholds, update flip overlay, handle refresh/decay/writes. | P2 + admitted P3 profile. | C++ unit tests, refresh tracker, deterministic RNG tests, row/cell sampling tests. | A real simulated flip occurs under a known test pattern; counterfactual controls do not flip. |
| **P5. OpenEnv tools, task compiler, reward** | Serial for API/reward; task variants parallelize | Implement tool interface, initial/step observations, task configs, target predicates, budgets, stable errors, direct-path lockout. | P2 for non-disturbance tasks; P4 for disturbance tasks. | Known-target tasks, hidden-target tasks, examples, docs, baseline policies. | Reward comes only from trusted simulator state; hidden-state non-leakage tests pass. |
| **P6. Script sandbox and SDK** | Parallel after API shape; serial before script rollouts | Run policy-authored Python in hardened sandbox with `rh_sdk` broker to same tools. | P0 runtime decision; P5 tool contract. | SDK docs, broker tests, seccomp/AppArmor/gVisor/Kata/Firecracker evaluation, resource-limit tests. | Script path is trace-equivalent to direct tools and blocks host memory, network, mounts, and devices. |
| **P7. Mitigations** | Parallel branches, admitted one by one | Add `oracle` first, then validated mitigations that exist in Ramulator or are real ports. | P2 command trace; P4 disturbance semantics; P5 config plumbing. | Paper-to-code mapping, differential traces, per-mitigation config tests. | Each mitigation changes issued events/protection state as expected; unavailable mitigations fail closed. |
| **P8. Advanced task families** | Parallel after P5/P4 | Unknown adjacency, hidden target, any-flip, target-cell, pattern, mitigation-aware curricula. | P5; P4 for flips; P7 for mitigation tasks. | Difficulty calibration, baseline policies, leak tests, task manifests. | Tasks have reproducible seeds, calibrated difficulty bands, and no hidden-state leaks. |
| **P9. Additional profiles/standards** | Parallel after profile interface | HBM2, DDR3/DDR3L, temperature/RowPress/temporal-variation extensions. | P3 package interface; P4 disturbance hooks. | HBM2 ingestion, source/legal review, separate model cards. | Each profile is admitted independently; no parameter pooling without documented validation. |
| **P10. Performance, security, release qualification** | Final serial gate | Optimize only after a semantic oracle exists; run full release test suite and publish artifacts. | All release-scope features admitted. | Trace compression, batching, dashboards, documentation. | Full CI/release matrix passes with no required skips and no mocks/fallbacks. |

## Workstreams that can run concurrently

| Lane | Starts | Owns | Depends on |
|---|---|---|---|
| A. Build/provenance/release | P0 | pins, containers, CI, SBOM, source manifests | None |
| B. Ramulator worker | P1 | C++ worker, External frontend integration, event trace, RPC | A |
| C. Memory/disturbance | P2 | sparse memory, exposure accounting, flip overlay, refresh integration | B, D for profile admission |
| D. Data/profile pipeline | P0 | artifact ingestion, fitting, validation, signed profiles, model cards | A; B/C event schema for integration |
| E. OpenEnv/tasks/reward | P1/P2 | tools, observations, tasks, rewards, baselines | B; C for disturbance tasks |
| F. Sandbox/SDK/security | P0/P5 | hardened runner, broker, `rh_sdk`, isolation tests | A, E |
| G. Mitigations | P2/P4 | oracle and admitted mitigation ports | B, C, E |
| H. Evaluation/ops | P5 | curriculum, held-out evaluation, metrics, release reports | E plus relevant profile/mitigation lanes |

## Minimal milestones

1. **Integration milestone:** P0-P2. Real OpenEnv-to-Ramulator loop with functional memory, no disturbance tasks advertised.
2. **First disturbance milestone:** P0-P5 for one DDR4 known-target task, direct tools only.
3. **Script milestone:** P6 admitted; script and direct-tool traces are equivalent.
4. **Benchmark milestone:** P8 plus selected P7/P9 features admitted.
5. **Release milestone:** P10 with all release-scope tests passing.

## Scheduling guidance

- Do P0 before merging executable feature work; it prevents later rework around provenance and safety.
- Start the data/profile lane during P1 because it is source-heavy and mostly independent until integration.
- Do not build a broad task suite before P4 proves one real flip; otherwise teams will encode assumptions that the disturbance engine may violate.
- Keep schemas compact and stable. Add semantic tests rather than schema complexity.
- Admit mitigations one at a time; this prevents ambiguous failures when disturbance, refresh, and mitigation logic interact.
- Optimize last. Any performance optimization must have differential tests against the pre-optimization real path.
