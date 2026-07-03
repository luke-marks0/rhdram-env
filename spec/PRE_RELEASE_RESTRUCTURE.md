# Pre-Release Restructure & De-Phasing (handle: **PR**)

> A standalone, behavior-preserving cleanup phase. It runs **after every feature
> phase P11–P19 in `spec/IMPLEMENTATION_PLAN_V2.md` has been admitted green**, and
> **immediately before P20** (release re-qualification). Its only job is to turn
> the phase-built codebase into a **component-structured release** with **zero
> "phase" references in any shipped product file, path, symbol, config, build
> target, or doc.**
>
> This phase changes **no behavior**. No feature logic is added or altered. Every
> test that passed before must pass after, unchanged in meaning. If a rename would
> require a logic change, that change belongs in a feature phase, not here.

---

## 0. Why this phase exists

The `phaseN_*` naming is an artifact of the phase-gated *build process*, not the
intended architecture. `spec/REPOSITORY_TREE.md` already specifies a **component**
layout. This phase realizes it and retires the process scaffolding so a release
artifact ships nothing named after a build phase.

Separation of concerns after this phase:
- **Product code** (`rowhammer_env/`, `cpp/`, `sdk/`, `configs/`): component-named,
  phase-free. This is what ships.
- **Process history** (`spec/IMPLEMENTATION_PLAN*.md`): the *only* place the word
  "phase" survives, because it describes how the project was built. Archive it (see
  §5) so it is excluded from the release surface.

---

## 1. Preconditions (do not start until all are true)

1. All feature phases **P11–P19 admitted green** (their gates pass with real
   implementations).
2. Run on a **single dedicated branch with no other agents mid-phase.** This phase
   renames the gate/build scripts other agents depend on; running it concurrently
   with feature work will break them.
3. Land it as **one atomic change set** so imports never dangle between rename and
   fixup.
4. Baseline captured: full suite green *before* starting (record the exact set of
   passing tests to diff against afterward).

---

## 2. Target component layout (product)

```text
rowhammer_env/
  server/
    environment.py     # the single public env (collapses the phase2->4->5 chain)
    types.py           # RowHammerAction / RowHammerObservation / RowHammerState
    app.py             # OpenEnv HTTPEnvServer wiring            (from P17)
    client.py          # policy-side client                     (from P17)
    worker_protocol.py # stdio RPC to the C++ worker            (was rowhammer_env/worker_protocol.py)
    openenv_types.py   # OpenEnv base-class loader/import        (was openenv_source.py)
  tools/               # dram.info/read/write/issue, episode.finish, script.run
    addressing.py      # physical/logical/handle projection      (from P12)
  tasks/               # task compiler + disclosure projection    (from P13/P12)
  rewards/             # trusted success predicates                (from P13)
  disturbance/         # exposure + flip engine                    (was disturbance.py)
  sandbox/             # OS-isolated script broker                 (from P18, was script_sandbox.py)
  profiles/
    loader.py          # signed-profile verify/admit               (was profiles.py)
  security/            # denylist / info-flow checks
  observability/       # metrics + trace summaries                 (from P19)
cpp/
  simulator_service/   # worker (unchanged location; de-phase strings only)
  ramulator_extensions/# issued-event plugin                       (from P11)
  disturbance/         # optional C++ engine                       (if chosen in P11)
configs/
  dram/ddr4.py         # was configs/ramulator/p1_external_ddr4.py
  tasks/               # task YAMLs
  mitigations/
tests/                 # component dirs (see §4): contract/ ramulator/ memory/
                       # disturbance/ profiles/ tasks/ sandbox/ security/
                       # mitigations/ release/
scripts/
  fetch_sources.py     # was fetch_phase1_sources.py + fetch_phase3_sources.py
  build.py             # was build_phase1.py + build_phase2.py
  verify_release.py    # single release verifier (see §4)
build/                 # gitignored output: build/ramulator, build/worker
                       #   (was build/phase1, build/phase2)
```

---

## 3. Product renames (files + public symbols)

### 3.1 Module files
| Old | New |
|---|---|
| `rowhammer_env/phase1_env.py` | **retire** (see §3.3) |
| `rowhammer_env/phase2_env.py` | `rowhammer_env/server/environment.py` + `server/types.py` |
| `rowhammer_env/phase4_env.py` | folded into `server/environment.py` (disturbance wiring) |
| `rowhammer_env/phase5_env.py` | folded into `server/environment.py` (tasks/reward/budgets) |
| `rowhammer_env/worker_protocol.py` | `rowhammer_env/server/worker_protocol.py` |
| `rowhammer_env/openenv_source.py` | `rowhammer_env/server/openenv_types.py` |
| `rowhammer_env/disturbance.py` | `rowhammer_env/disturbance/engine.py` |
| `rowhammer_env/script_sandbox.py` | `rowhammer_env/sandbox/broker.py` |
| `rowhammer_env/profiles.py` | `rowhammer_env/profiles/loader.py` |

### 3.2 Public symbols (update `rowhammer_env/__init__.py` and all importers)
| Old | New |
|---|---|
| `Phase2Action` | `RowHammerAction` |
| `Phase2Observation` | `RowHammerObservation` |
| `Phase2State` | `RowHammerState` |
| `RowHammerEnv` (phase2 base) / `RowHammerDisturbanceEnv` (phase4) / `RowHammerTaskEnv` (phase5) | **collapse to one** public `RowHammerEnv` in `server/environment.py`, composed from the `tools`/`tasks`/`rewards`/`disturbance`/`sandbox` components |

The collapsed `RowHammerEnv` is the single OpenEnv `Environment` subclass exported.
Prefer **composition** (hold `tools`, `tasks`, `rewards`, `disturbance` objects) over
the old inheritance chain; if a thin internal base is still useful, name it by role
(e.g. `_CommandEnv`), never by phase.

### 3.3 Retire vestigial code
`phase1_env.py` (`RowHammerBootstrapEnv`, `Phase1Action/Observation/State`) exists
only to prove the P1 Ramulator bootstrap. It is not product. **Delete it** and move
its "a real read/write completes through Ramulator" assertion into
`tests/ramulator/test_bootstrap.py`. Remove all four `Phase1*`/`RowHammerBootstrapEnv`
names from `__init__.py` and `__all__`.

### 3.4 In-code "phase" strings (product)
Purge phase wording from messages/comments/docstrings. Known sites (verify with the
§6 grep — some are already removed by earlier phases):
- `disturbance/engine.py`: docstring "for Phase 4"; the "Phase 4 admits DDR4 only"
  error (already removed in P15) — confirm gone.
- `server/environment.py` (ex-phase2): "Phase 2 admits logical addresses only"
  (removed in P12), "Phase 2 worker is not built" → "simulator worker is not built".
- `cpp/simulator_service/ramulator_worker.cpp`: `"<op> is not admitted in Phase 2"`
  → `"<op> is not an admitted command"`.
- `server/openenv_types.py` (ex-openenv_source): "required for Phase 1" → "required".
- `profile_builder/{manifest,paths,package/build}.py`, `worker_protocol.py`: replace
  any "phase" comments/keys with component terms.

---

## 4. De-phase the process scaffolding (gates, build, config, docs)

The per-phase gates were a *development admission* mechanism. For release, convert
them into the permanent **component test suite** plus **one** release verifier.

### 4.1 Gates → tests
- Migrate the assertions in `scripts/verify_phase0.py … verify_phase9.py` into
  `tests/<component>/` per the spec tree (`contract/`, `ramulator/`, `memory/`,
  `disturbance/`, `profiles/`, `tasks/`, `sandbox/`, `security/`, `mitigations/`,
  `release/`). Nothing is lost — every phase assertion gets a component home.
- Rename `tests/test_phase*.py` into those component dirs (e.g.
  `test_phase4_disturbance.py` → `tests/disturbance/test_flip.py`,
  `test_phase2_env.py` → `tests/contract/test_commands.py`,
  `test_phase8_9_release.py` → `tests/release/test_release.py`).
- Keep a single **`scripts/verify_release.py`** that runs the whole `tests/` suite +
  provenance checks. Delete `scripts/verify_phase0.py … verify_phase9.py`.

### 4.2 Build/fetch/config
- `scripts/build_phase1.py` + `build_phase2.py` → one `scripts/build.py` (builds
  `libramulator.so`, the smoke binary, and the worker). Update output paths
  `build/phase1|phase2` → `build/ramulator|worker` everywhere they are referenced
  (env defaults in `server/environment.py`, `build.py`, tests).
- `scripts/fetch_phase1_sources.py` + `fetch_phase3_sources.py` → one
  `scripts/fetch_sources.py`.
- `configs/ramulator/p1_external_ddr4.py` → `configs/dram/ddr4.py`; regenerate the
  built YAML under the new build path.

### 4.3 Docs
- `docs/phase0.md` → fold its admission/error-code/denylist content into
  `docs/threat_model.md` and a new `docs/provenance.md`.
- `docs/phase3.md` → `docs/profile_modeling.md`.
- Ensure `docs/api.md` and `docs/operations.md` (from P17/P19) exist. No `docs/phaseN.md`.

### 4.4 `SOURCE_MANIFEST.yaml` / manifest fields
- Rename the `fetch:` pointer and any `phase:` key. The top-level `phase: P10` field
  (build-stage marker) → `stage: release` (or remove). Update `verify_release.py`'s
  check accordingly.

---

## 5. Archive the planning docs (the one place "phase" may remain)

Move `spec/IMPLEMENTATION_PLAN.md`, `spec/IMPLEMENTATION_PLAN_V2.md`, and this file
into `spec/history/` (or a top-level `PLANNING/` excluded from the release build).
These describe process, not product; archiving them keeps the shipped artifact
phase-free while preserving provenance. Add a one-line note in `README.md` pointing
to `spec/history/` for build history.

---

## 6. Definition of done (the phase gate for PR)

1. **Zero phase references in the release surface.** This command returns **no
   hits** (the only allowed matches are under `spec/history/`, `third_party/`,
   `.git/`, `build/`, and `__pycache__`):
   ```sh
   grep -rniI "phase" \
     rowhammer_env cpp sdk configs scripts tests docs \
     README.md SOURCE_MANIFEST.yaml CMakeLists.txt 2>/dev/null
   ```
   Also: `find rowhammer_env cpp sdk configs scripts tests docs -iname '*phase*'`
   returns nothing.
2. **Public API smoke:** `from rowhammer_env import RowHammerEnv, RowHammerAction,
   RowHammerObservation, RowHammerState` works; none of `Phase1*`, `Phase2*`,
   `RowHammerBootstrapEnv`, `RowHammerDisturbanceEnv`, `RowHammerTaskEnv` resolve.
3. **Behavior parity:** the full migrated `tests/` suite passes with the same count
   and meaning as the pre-PR baseline (§1.4); `scripts/verify_release.py` is green;
   an end-to-end hammer rollout still earns reward only on a real flip (the negative
   and oracle controls still hold).
4. **Build parity:** `scripts/build.py` produces a working worker at the new path;
   the env launches it with no phase-named paths.

Only when all four hold does PR admit. Then proceed to **P20** (which now runs the
component `tests/` suite + `verify_release.py`, not `verify_phaseN.py`).

---

## 7. Fit into the parallelization plan

PR is **serial and solo.** It touches nearly every file and renames the gate scripts
the other instances use, so **no other phase may run concurrently with it.** Sequence:

```
… all feature instances converge (P11–P19 admitted)
  -> PR  (one instance, alone, atomic change set)
  -> P20 release re-qualification
```

In instance terms: after the last feature phase lands on any instance, **one
instance runs PR to completion, then P20.** Do not fan out during PR.
