# P29 — Release readiness (de-phase restructure + SPEC §12 acceptance)

> **Standalone execution spec.** This is the self-contained work order for P29 of
> `spec/IMPLEMENTATION_PLAN_V3.md`. It is meant to be executed **on its own branch,
> after the `training` branch (P21–P28) is merged to `main`**, by a fresh agent. It
> reconciles the (older, V2-era) `spec/PRE_RELEASE_RESTRUCTURE.md` with the **current**
> tree, which has grown a `rowhammer_env/llm/` training package, several already-created
> component dirs, a secret-mapping engine, and the Tier 2 discovery families/tests since
> that doc was written.
>
> **Read first, in order:** `spec/SPEC.md` (esp. §2 non-negotiables, §9 reward, §12
> acceptance), `spec/IMPLEMENTATION_PLAN_V3.md` §0–§1 and the **P29** section,
> `spec/PRE_RELEASE_RESTRUCTURE.md` (the "PR" restructure this P29 finally executes),
> and `docs/adr-0004-secret-address-mapping.md` (the P24 mapper). This document assumes
> all three.
>
> **This phase changes no behavior.** It is a rename/restructure plus new *release-gate*
> tests and docs. Every test that passed before must pass after, unchanged in meaning.
> If a rename would force a logic change, that change is out of scope — stop and flag it.

---

## 0. Preconditions (do not start until all true)

1. **The `training` branch (P21–P28) is merged to `main`.** P29 builds on the collapsed
   env + discovery families + curriculum/shaping. Start P29 from a clean `main`.
2. **Solo branch, atomic change set.** The restructure renames the gate/build scripts and
   the core env modules everything imports; run it alone and land it so imports never
   dangle. This repo has an **auto-commit hook** (edits get committed without an explicit
   `git commit`) and a **live co-editing caveat** — do **not** run `git stash`/`git stash
   pop` here (it has surfaced unrelated old stashes before). Work forward-only.
3. **Baseline captured.** Run `python3 -m unittest discover -s tests` and record the exact
   pass/skip/fail set. On the dev host the expected baseline is **263 tests, 1 error**:
   `test_phase19` errors only because `fastapi`/`uvicorn`/`websockets` aren't installed on
   this host (documented pre-existing host gap, not a product defect). **Everything else
   is green.** The post-P29 suite must match this meaning (same assertions, re-homed).
4. **Worker builds.** `build/phase2/ramulator_worker` exists (rebuildable via
   `scripts/build_phase2.py`). Many acceptance tests are worker-gated.

---

## 1. Current-tree ground truth (verified 2026-07-12)

What exists now, so the restructure folds into it rather than assuming a greenfield:

- **Env chain (to collapse):** `rowhammer_env/phase1_env.py` (vestigial bootstrap),
  `phase2_env.py` (base `RowHammerEnv` + `Phase2Action/Observation/State` + `_issue`,
  expansion, timing digest), `phase4_env.py` (`RowHammerDisturbanceEnv`), `phase5_env.py`
  (`RowHammerTaskEnv` — tasks/reward/budgets/candidates/`_issue_acts_ceiling`).
- **Component dirs that ALREADY EXIST** (partially populated — fold into these, don't
  recreate): `rowhammer_env/server/` (`app.py`), `tools/` (`addressing.py`), `tasks/`
  (`compiler.py`, `disclosure.py`), `rewards/` (`predicates.py`), `observability/`
  (`metrics.py`), and `llm/` (the training/eval package: `grpo_env.py`, `rollout.py`,
  `multiturn_rollout.py`, `policies.py`, `curriculum.py`, `shaping.py`, `tools.py`,
  `training.py`, `wandb_logging.py`).
- **Root modules not in the PR-doc rename table** (added in P21–P28 — assign homes, §2.1):
  `geometry.py`, `standards.py`, `mappers.py` (secret Row→Bank XOR mapper, P24),
  `mitigations.py`, plus `client.py`, `disturbance.py`, `openenv_source.py`,
  `profiles.py`, `script_sandbox.py`, `worker_protocol.py` (these last six are in the
  PR-doc table).
- **cpp** is already component-named (`cpp/simulator_service/`, `cpp/ramulator_extensions/`);
  only `cpp/simulator_service/ramulator_worker.cpp` still contains "phase" strings.
- **Scripts:** `verify_phase0.py … verify_phase20.py` (24 gates), `verify_release.py`
  (already exists — a single verifier with a `FORBIDDEN_SYMBOLS`/`EXECUTABLE_ROOTS`
  no-mock scan + `PHASE_GATE_NUMBERS`), `build_phase1.py`, `build_phase2.py`,
  `fetch_phase1_sources.py`, `fetch_phase3_sources.py`, `rl_run.py`, `train_grpo.py`,
  `train_phase19_policy.py`.
- **Tests:** flat `tests/`. Phase-named: `test_phase{0,1_env,2_env,4_disturbance,8_9_release,
  11..20}.py`. Already component-named (P13/P21–P28): `test_activation_budget`,
  `test_issue_expansion`, `test_discovery_families`, `test_discovery_geometry`,
  `test_hidden_adjacency`, `test_secret_mapping`, `test_probe_signal`,
  `test_reference_policy`, `test_curriculum`, `test_probe_shaping`,
  `test_multiturn_rollout`, `test_profile_fit`, `test_profile_package`,
  `test_task_script_mitigation`.
- **docs:** `phase0.md`, `phase3.md` (to fold/rename); `api.md`, `operations.md`,
  `threat_model.md`, `mitigations-roadmap.md`, `adr-0001..0004` (keep).
- **Counts to drive to zero:** `find … -iname '*phase*'` → **87 files**;
  `grep -rniI "phase" …` over the release surface → **578 hits** across **83 files**.
- **`SOURCE_MANIFEST.yaml`:** top-level `phase: P20` (→ `stage: release` or remove);
  `fetch: scripts/fetch_phase3_sources.py` pointer (→ `fetch_sources.py`).

---

## 2. Task A — De-phase restructure (the PR change set)

Execute `spec/PRE_RELEASE_RESTRUCTURE.md` §2–§5 against the tree above. Below are the
reconciled specifics; where this doc and the PR-doc agree, the PR-doc is normative.

### 2.1 Module files → component layout

Verbatim from PR-doc §3.1 (still correct):

| Old | New |
|---|---|
| `rowhammer_env/phase1_env.py` | **delete** (move its "a real RD/WR completes through Ramulator" assertion into `tests/ramulator/test_bootstrap.py`; drop `Phase1*`/`RowHammerBootstrapEnv` from `__init__`) |
| `rowhammer_env/phase2_env.py` | `rowhammer_env/server/environment.py` + `server/types.py` |
| `rowhammer_env/phase4_env.py` | folded into `server/environment.py` (disturbance wiring) |
| `rowhammer_env/phase5_env.py` | folded into `server/environment.py` (tasks/reward/budgets) |
| `rowhammer_env/worker_protocol.py` | `rowhammer_env/server/worker_protocol.py` |
| `rowhammer_env/openenv_source.py` | `rowhammer_env/server/openenv_types.py` |
| `rowhammer_env/disturbance.py` | `rowhammer_env/disturbance/engine.py` (new pkg) |
| `rowhammer_env/script_sandbox.py` | `rowhammer_env/sandbox/broker.py` (new pkg) |
| `rowhammer_env/profiles.py` | `rowhammer_env/profiles/loader.py` (new pkg) |

**Reconciliation additions (not in the PR-doc — decide within the component principle,
recommended homes):**

| Old | Recommended new | Rationale |
|---|---|---|
| `rowhammer_env/geometry.py` | `rowhammer_env/dram/geometry.py` | DRAM geometry model |
| `rowhammer_env/standards.py` | `rowhammer_env/dram/standards.py` | standard registry |
| `rowhammer_env/mappers.py` | `rowhammer_env/dram/mappers.py` | secret Row→Bank XOR mapper (P24) |
| `rowhammer_env/mitigations.py` | `rowhammer_env/mitigations/__init__.py` (or `engine.py`) | mitigation ports |
| `rowhammer_env/client.py` | `rowhammer_env/server/client.py` | policy-side client (PR-doc §2 lists it under `server/`) |
| `rowhammer_env/llm/` | keep as `rowhammer_env/llm/` | already component-named; only fix its `phaseN_env`/`Phase2*` imports and purge "Phase N" docstrings |

> A new `rowhammer_env/dram/` package (geometry+standards+mappers) is the cleanest home
> for the addressing/geometry domain; if you prefer folding them under `server/` or
> `tools/`, that's acceptable as long as the result is component-grouped and phase-free.
> Keep the `__init__.py` re-exports stable so importers change path, not names.

### 2.2 Public symbol renames (`rowhammer_env/__init__.py` + every importer)

| Old | New |
|---|---|
| `Phase2Action` | `RowHammerAction` |
| `Phase2Observation` | `RowHammerObservation` |
| `Phase2State` | `RowHammerState` |
| `RowHammerEnv` (phase2 base) + `RowHammerDisturbanceEnv` (phase4) + `RowHammerTaskEnv` (phase5) | **collapse to one** public `RowHammerEnv` in `server/environment.py` |
| `Phase1Action/Observation/State`, `RowHammerBootstrapEnv` | **delete** |

The collapsed `RowHammerEnv` is the single OpenEnv `Environment` subclass exported.
Prefer **composition** over the phase2→4→5 inheritance chain; a thin internal base, if
kept, is named by role (`_CommandEnv`), never by phase. **Importers to update** (current
list — re-grep to be sure): `rowhammer_env/{client,openenv_source,script_sandbox}.py`,
`rowhammer_env/server/app.py`, `rowhammer_env/llm/{rollout,multiturn_rollout}.py`,
`configs/training/grpo_qwen8b.yaml` (path/string only), and every migrated test. The env
carries several load-bearing internals the tests reach into — preserve them by name on the
collapsed class: `_issue`, `_issue_acts_ceiling`, `expand_commands`/`IssueExpansionError`
(module-level in the ex-`phase2_env`), `_TimingDigest`, `_compiled`, `_decode`,
`_from_worker`, `_error`, disclosure/leakage projection hooks. Grep the tests for
`env\._` and `phase2_env\.` before renaming so nothing silently loses an attribute.

### 2.3 Tests → component dirs

PR-doc §4.1 target dirs: `tests/{contract,ramulator,memory,disturbance,profiles,tasks,
sandbox,security,mitigations,release}/`. Migrate the phase gates' assertions and rename
the phase-named test files; **home the already-named ones too** (they carry no "phase" in
the filename but their bodies import `phase2_env`/`Phase2*`, which must be fixed).
Recommended map (agent judgment allowed; the invariant is *every assertion keeps a home
and the suite's meaning/count is preserved*):

| Current test | Recommended home |
|---|---|
| `test_phase0.py` | `tests/release/test_no_mock.py` (denylist + no-mock symbols + capabilities) |
| `test_phase1_env.py` | `tests/ramulator/test_bootstrap.py` (+ the deleted-env assertion) |
| `test_phase2_env.py` | `tests/contract/test_commands.py` |
| `test_phase4_disturbance.py` | `tests/disturbance/test_flip.py` |
| `test_phase8_9_release.py` | `tests/release/test_release.py` |
| `test_phase11.py` | `tests/ramulator/test_issued_events.py` |
| `test_phase12.py` | `tests/contract/test_disclosure.py` |
| `test_phase13.py` | `tests/tasks/test_compiler.py` |
| `test_phase14.py` | `tests/disturbance/test_fidelity.py` |
| `test_phase15.py` | `tests/disturbance/test_standards.py` |
| `test_phase16.py` | `tests/mitigations/test_mitigations.py` |
| `test_phase17.py` | `tests/contract/test_serving.py` |
| `test_phase18.py` | `tests/sandbox/test_broker.py` |
| `test_phase19.py` | `tests/contract/test_serving_rollout.py` (keep the host-gate skip/error behavior) |
| `test_phase20.py` | `tests/release/test_qualification.py` |
| `test_activation_budget.py` | `tests/contract/test_activation_budget.py` |
| `test_issue_expansion.py` | `tests/contract/test_issue_expansion.py` |
| `test_discovery_geometry.py` | `tests/contract/test_geometry_disclosure.py` |
| `test_probe_signal.py` | `tests/contract/test_probe_signal.py` |
| `test_multiturn_rollout.py` | `tests/contract/test_multiturn_rollout.py` |
| `test_discovery_families.py` | `tests/tasks/test_discovery_families.py` |
| `test_hidden_adjacency.py` | `tests/tasks/test_hidden_adjacency.py` |
| `test_secret_mapping.py` | `tests/tasks/test_secret_mapping.py` |
| `test_reference_policy.py` | `tests/tasks/test_reference_policy.py` |
| `test_curriculum.py` | `tests/tasks/test_curriculum.py` |
| `test_probe_shaping.py` | `tests/rewards/test_probe_shaping.py` |
| `test_profile_fit.py`, `test_profile_package.py` | `tests/profiles/` |
| `test_task_script_mitigation.py` | `tests/mitigations/test_task_script_mitigation.py` |

Notes:
- Add `tests/<component>/__init__.py` only if the current flat layout relies on package
  imports — it does **not** today (`test_multiturn_rollout` does a bare
  `from test_reference_policy import _FakeDiscoveryEnv`, which resolves because `discover`
  puts each start-dir on `sys.path`). After nesting, **fix such cross-imports** (share the
  fake env via a `tests/tasks/_fakes.py` helper or move the fixture). Verify
  `discover -s tests` still finds every test after nesting (it recurses).
- `test_reference_policy`/`test_curriculum`/`test_probe_shaping` are worker-gated
  integration + host unit; keep both layers.

### 2.4 Scripts: gates → tests, build/fetch consolidation

- **Delete** `scripts/verify_phase0.py … verify_phase20.py` after their assertions live in
  `tests/` (2.3). Keep **one** `scripts/verify_release.py` that runs the whole `tests/`
  suite + provenance/no-mock checks (it already does — update it per §3/§4).
- `scripts/build_phase1.py` + `build_phase2.py` → one `scripts/build.py` building
  `libramulator.so`, the smoke binary, and the worker. Update output paths
  `build/phase1|phase2` → `build/ramulator|worker` **everywhere** (env defaults in
  `server/environment.py`, `build.py`, and ~48 test/code references to `build/phase2`).
- `scripts/fetch_phase1_sources.py` + `fetch_phase3_sources.py` → one
  `scripts/fetch_sources.py`; update `SOURCE_MANIFEST.yaml`'s `fetch:` pointer.
- `scripts/train_phase19_policy.py` → `scripts/train_reference_policy.py` (or fold into an
  examples/ script); `train_grpo.py`/`rl_run.py` keep their names (already phase-free) but
  purge any "Phase N" strings/paths inside.

### 2.5 Config + cpp + manifest string purge

- `configs/ramulator/p1_external_ddr4.py` → `configs/dram/ddr4.py`; regenerate the built
  YAML under the new build path; update every importer (the worker-config builder, tests).
- `cpp/simulator_service/ramulator_worker.cpp`: `"<op> is not admitted in Phase 2"` →
  `"<op> is not an admitted command"`; purge any other "Phase" strings. **Rebuild the
  worker** and re-run the worker-gated tests (this is the one C++ touch).
- `SOURCE_MANIFEST.yaml`: `phase: P20` → `stage: release` (or remove); update
  `verify_release.py`'s manifest check accordingly.
- Purge "phase" from `profile_builder/{manifest,paths,package/*}.py` comments/keys and any
  remaining product docstrings/messages (incl. `rowhammer_env/llm/*` "Phase N" mentions).

### 2.6 Archive the planning docs (the only place "phase" may remain)

Move `spec/IMPLEMENTATION_PLAN.md`, `spec/IMPLEMENTATION_PLAN_V2.md`,
`spec/IMPLEMENTATION_PLAN_V3.md`, `spec/TIER2_DISCOVERY_PLAN.md`,
`spec/PRE_RELEASE_RESTRUCTURE.md`, this file, and the `spec/*P29*`/history-style docs into
`spec/history/`. Add a one-line `README.md` pointer to `spec/history/` for build history.
`docs/phase0.md` → fold into `docs/threat_model.md` + new `docs/provenance.md`;
`docs/phase3.md` → `docs/profile_modeling.md`.

---

## 3. Task B — Determinism / replay incl. mapper selection (SPEC §12.7)

New `tests/release/test_determinism.py` (worker-gated). For a spread of families
(known-target, `bounded_sweep`, `hidden_adjacency`) and seeds:

1. Run the same `(task, seed)` through a fresh `RowHammerEnv` **twice** with an identical
   deterministic policy (the reference/`CIHammer` fixture); assert **byte-identical**
   trajectories: same accepted/rejected counts, same `cycle`, same `public_counters`, same
   `new_public_flips`/`public_flips`, same final reward.
2. **Mapper-selection determinism:** for the discovery families, assert the per-episode
   secret mapper chosen from `(task_id, seed)` is **identical across replays of the same
   seed** and **differs across seeds** (sample enough seeds to show variation). Compare via
   the env's decode (`env._decode`, as `test_reference_policy` does) — same seed ⇒ same
   `DECODE` for a fixed linear address; the recovered same-bank set must be identical run
   to run. This is the "same seed ⇒ same mapper ⇒ same trace/flips" clause.

Keep it host-runnable where the worker is built; skip cleanly otherwise.

---

## 4. Task C — No-mock scan extension + capability discovery (SPEC §12.8, §2)

- The existing scan in `scripts/verify_release.py` (`FORBIDDEN_SYMBOLS`,
  `EXECUTABLE_ROOTS = ("cpp","rowhammer_env","profile_builder","sdk","scripts","tests")`)
  already covers the new modules by root. **Update it for the rename:** drop the deleted
  `verify_phase*` exemptions, retarget `SYMBOL_SCAN_EXEMPTIONS` to the surviving files, and
  remove `PHASE_GATE_NUMBERS`/the per-phase-gate runner (it now runs `tests/`).
- Add a **capability-discovery assertion** (in `tests/release/` and/or `verify_release.py`):
  the discovery families (`bounded_sweep`, `hidden_adjacency`) and the timing digest are
  **advertised in `dram.info`/reset metadata** at their disclosure level, and every
  advertised capability is backed by a real implementation (nothing advertised that fails
  closed). Conversely, an unavailable feature is **absent** from capability discovery and
  returns a stable error code — assert one such fail-closed path (e.g. `physical`-form
  addressing under a secret mapper → `ADDRESS_NOT_DISCLOSED`, per P24 task 4).
- Add a **de-phase guard** to `verify_release.py`: the PR-doc §6 grep returns no hits over
  the release surface (fail the release verifier if "phase" reappears in shipped files).

---

## 5. Task D — Docs / provenance (SPEC §10, §12.5)

- **`docs/api.md`:** document the full tool/observation/reward contract **including** the
  `timing_digest` shape (`acts_delta`/`cycles_delta`/`first_clk`/`last_clk`/`per_addr_hits`,
  coordinate-free) and the numeric-candidate discovery families
  (`bounded_sweep`/`hidden_adjacency`), the `logical_only`+`full_trace` disclosure, and the
  compact `HAMMER`/`repeat` forms.
- **P24 mapper ADR:** `docs/adr-0004-secret-address-mapping.md` already exists — ensure it
  is current (RoBaRaCoChRowXOR, seedable `xor_offset`, per-episode secret) and linked from
  `docs/api.md`.
- **Discovery quickstart:** new `docs/discovery_quickstart.md` — launch the server → attach
  an LLM/reference policy → run the curriculum (`configs/training/grpo_curriculum.yaml`),
  with the P26 host reference-gate (`tests/tasks/test_curriculum.py`) as the pre-flight.
- Update `spec/VALIDATION_REPORT.md` (Tier 2 discovery results, reference-policy success
  windows, determinism), the profile **model cards**, and the **SBOM**/`SOURCE_MANIFEST.yaml`
  for the v2.1.0 pin.

---

## 6. Task E — SPEC §12 acceptance mapping

Tie the acceptance clauses to concrete, named tests (mostly existing; add a
`tests/release/test_acceptance_spec12.py` index that asserts each is present/green):

| SPEC §12 clause | Backing test |
|---|---|
| §12.1 DDR4 profile admitted | `tests/profiles/` (ex-`test_profile_package`) |
| §12.2 real flip via issued events | Tier 2b reference policy (`tests/tasks/test_reference_policy.py`) |
| §12.3 fails when budgets/mitigations should prevent it | the `victim ± row_stride` control + timing-blind control (`test_secret_mapping` / `test_reference_policy` differential) |
| §12.4 API/script/reward/projection contract | leakage/projection tests (`test_probe_signal`, `test_secret_mapping`, `test_disclosure`) |
| §12.5 sandbox blocks host escape | `tests/sandbox/` (ex-`test_phase18`) |
| §12.6 statistical validation on held-out units | `tests/profiles/` (ex-`test_profile_fit`) |
| §12.7 deterministic replay incl. mapper | **Task B** (`tests/release/test_determinism.py`) |
| §12.8 no mock/fallback imports | **Task C** (`verify_release.py` scan) |

---

## 7. Definition of done (the P29 gate)

All must hold, on `main` after merge:

1. **Zero phase references in the release surface.** Both return nothing (except under
   `spec/history/`, `third_party/`, `.git/`, `build/`, `__pycache__`):
   ```sh
   grep -rniI "phase" rowhammer_env cpp sdk configs scripts tests docs \
     README.md SOURCE_MANIFEST.yaml CMakeLists.txt 2>/dev/null
   find rowhammer_env cpp sdk configs scripts tests docs -iname '*phase*'
   ```
2. **Public API smoke:** `from rowhammer_env import RowHammerEnv, RowHammerAction,
   RowHammerObservation, RowHammerState` works; none of `Phase1*`, `Phase2*`,
   `RowHammerBootstrapEnv`, `RowHammerDisturbanceEnv`, `RowHammerTaskEnv` resolve.
3. **Behavior parity:** `python3 -m unittest discover -s tests` matches the §0.3 baseline
   in count and meaning (same host-gap skips; no new failures). `scripts/verify_release.py`
   is green. An end-to-end hammer rollout still earns reward only on a real flip; the
   negative + oracle + timing-blind controls still fail as before.
4. **Build parity:** `scripts/build.py` produces a working worker at `build/worker/…`; the
   env launches it with no phase-named paths. Worker-gated tests pass against the rebuilt,
   de-phased worker.
5. **Determinism** (Task B), **no-mock scan + capability discovery** (Task C), **docs**
   (Task D), and **§12 mapping** (Task E) are all green/present.
6. **Discovery families appear in capability discovery** and are backed by real
   implementations (no advertised-but-unavailable capability).

---

## 8. Execution discipline

- **Atomic + forward-only.** Land the rename+fixups as one coherent change set; never leave
  imports dangling between a rename and its fixups. No `git stash` (see §0.2).
- **Behavior-preserving.** No feature logic changes. If a rename seems to require one, stop
  and flag it — it belongs in a feature phase, not P29.
- **Suite-green invariant.** Re-run `discover -s tests` after each cohesive step (env
  collapse; test migration; script/config rename; docs). Rebuild the worker after the cpp
  string edit before trusting worker-gated results.
- **No new gate scripts.** Acceptance is `unittest discover` + the single `verify_release.py`
  (SPEC/PR-doc discipline). Do not add `verify_phaseN.py`-style gates.
- **Report** the final baseline diff (test count before/after, the two zero-hit greps, the
  API smoke, the determinism + no-mock results) and anything in the tree that contradicted
  this plan (it was verified 2026-07-12 but the tree moves).
