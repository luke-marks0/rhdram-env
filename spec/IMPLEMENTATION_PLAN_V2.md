# Implementation Plan v2 — Road to LLM-in-the-loop

> This plan continues the project after the original `spec/IMPLEMENTATION_PLAN.md`
> phases P0–P10 were admitted. It is self-contained: an implementing agent can
> read this file plus `spec/SPEC.md` and proceed without other context. It
> supersedes and makes redundant the ad-hoc "post-P10 gaps" project memory.
>
> **Objective of this plan:** bring the repository from a *lean vertical slice*
> to a state where a **language-model policy can be trained and evaluated in the
> environment** — connected over the real OpenEnv HTTP transport, acting through a
> meaningful address/command space, and rewarded from an integrity-preserving,
> issued-event-driven disturbance model.
>
> All original non-negotiables still hold (`spec/SPEC.md` §2): **no mocks, no
> fake fallbacks; unavailable features are absent from capability discovery and
> fail closed with a stable error code.** Every phase below admits only real
> implementations behind a `scripts/verify_phaseN.py` gate, matching the existing
> phase-gated admission convention.

---

## 0. Ground truth as of 2026-07-03 (replaces the stale memory)

### 0.1 Source pins (current)
- **Ramulator** is now pinned to the **`v2.1.0` release tag**, commit
  `38c51d40a976c6b07fbc09de869a7e08dc187d29` (previously the stripped
  `278f1eff…`). Pin is recorded in `SOURCE_MANIFEST.yaml`,
  `scripts/fetch_phase1_sources.py`, and enforced by `scripts/verify_phase1.py`.
  License MIT. Vendored (gitignored) at `third_party/ramulator2/`.
- **OpenEnv** pinned to `v0.3.1` (`7449c5df…`), vendored at `third_party/openenv/`.
- **DDR4 VTS25** profile source pinned (`5d734309…`); one admitted profile
  `ddr4_vts25_v1`. HBM2 source is `deferred_pending_license_and_hash`.

### 0.2 What the v2.1.0 pin changed (IMPORTANT — corrects prior belief)
The old `278f1eff` tree shipped **no** RowHammer mitigations. `v2.1.0` ships a
near-complete suite as **real, reusable code**. Verified present:
- **Controller-level:** `blockhammer_controller.cpp`, `prac_controller.cpp`
  (`src/ramulator/controller/impl/`).
- **Plugins** (registered names in `src/ramulator/controller/plugin/impl/`):
  `PARA`, `OracleRH`, `TWiCeIdeal`, `Graphene`, `Hydra`, `RRS`, `AQUA`,
  `RFMManager`, `IdealTRR`, `SamsungTRR`, `HynixTRR`, plus `CommandCounter`,
  `CmdTraceRecorder`, `BinTraceRecorder`, `LiveTraceStreamer`.
- **Standards:** adds `DDR4_VRR`, `DDR5_RFM`, `DDR5_RFM_VRR`, `DDR5_VRR`,
  `GDDR7`, `LPDDR6` on top of DDR3/4/5, GDDR6, HBM1–4, LPDDR5.

Consequence: the mitigation lane (SPEC §6) is now **"wire up + validate existing
Ramulator code"**, not "port from scratch."

### 0.3 Verified defect inventory (the issues this plan closes)
| ID | Defect | Primary files | Fixed in |
|---|---|---|---|
| A | Disturbance is driven by **frontend RD completions, not issued ACT commands**; ignores row-buffer locality. Worker never surfaces ACT/PRE/REF/RFM. Violates SPEC §4/§5, test D3. | `rowhammer_env/disturbance.py:76`, `cpp/simulator_service/ramulator_worker.cpp:190` | **P11** |
| I | Address forms **`logical`-only**; disclosure matrix partly fake (hidden/unknown-adjacency candidates handed as logical addrs). | `rowhammer_env/phase2_env.py:134`, `rowhammer_env/phase5_env.py:136` | **P12** |
| J | Task compiler thin; single hardcoded `target_row=10`, `row_bytes=8192`, one family per env, difficulty band literal `"smoke"`. | `rowhammer_env/phase5_env.py:23`, `rowhammer_env/disturbance.py:34` | **P13** |
| B | Uses ~one field of a rich profile: single hardcoded stratum, always bit 0 / dir `0->1`; no multiplicity, RowPress, temperature, hierarchical variance. | `rowhammer_env/disturbance.py:37,124,134` | **P14** |
| C | Blast radius hardcoded ±1 logical row, not physical-row / profile-driven. | `rowhammer_env/disturbance.py:95` | **P14** |
| E | Refresh/decay absent; `oracle` is a counter reset, not a target-row-refresh model. | `rowhammer_env/disturbance.py:70,105` | **P14** |
| F | DDR4 hard-gate + hardcoded geometry; no HBM/other-standard path; profile schema DDR4-shaped. | `rowhammer_env/disturbance.py:46`, `profiles/ddr4_vts25_v1/profile.json` | **P15** |
| L | Mitigations only `none`/`oracle` (Python); Ramulator's real ones unused. | `rowhammer_env/disturbance.py:56`, worker config | **P16** |
| G | No OpenEnv HTTP serving; env only driven in-process. `HTTPEnvServer` exists but unused. | `rowhammer_env/openenv_source.py` (loads only `types`/`interfaces`) | **P17** |
| K | `script.run` sandbox is an in-process AST DSL, not OS-level isolation. Test-plan S2–S7 unmet. | `rowhammer_env/script_sandbox.py` | **P18** |
| H | No real NN/LLM baseline; "policy" is a hardcoded hammer loop + 3-arm bandit. | `scripts/rl_run.py` | **P19** |
| D | Disturbance engine is Python, not the C++ `cpp/disturbance/` the SPEC §11 tree implies. Architectural; only a real defect *combined with A*. | `cpp/` (only `simulator_service/`) | Resolved by **P11** design decision (§P11.4) |

### 0.4 Framework facts already verified (use these; do not re-derive)
- **Issued commands are available in Ramulator** via the controller plugin hook
  `on_issue(const Request& req)`, invoked immediately after
  `m_device.issue_command(...)` in
  `src/ramulator/controller/impl/generic_ddr_controller.cpp:59-64` — i.e. **true
  post-schedule** commands. `Request` (`src/ramulator/base/request.h`) carries
  `command` (ACT/PRE/RD/WR/REF/…), `addr_vec` (decoded channel/rank/bankgroup/
  bank/row/column), `type_id`, `arrive`, `depart`. `CmdTraceRecorder`,
  `BinTraceRecorder`, and `LiveTraceStreamer` are worked examples of consuming it.
- **Row-buffer state** is tracked by the controller (`s_row_hits` in
  `controller_base.cpp`). Under `row_policy: Open`, repeated same-row accesses are
  row hits → **one** ACT; this is exactly the locality the current engine ignores.
- **Address decoding**: mapper `RoBaRaCoCh` (`addr_mapper/impl/ro_ba_ra_co_ch.cpp`)
  fills `req.addr_vec`; level order follows `DRAMSpec.level_names`. Physical
  coordinates in observations/actions must be derived from this decode, not from
  logical-address arithmetic.
- **Worker API is byte-identical** between the old pin and `v2.1.0`
  (`receive_external_requests`, `Factory::create_frontend/create_memory_system`,
  `"External"` frontend registration), so the existing worker should recompile
  unchanged — but this is **not yet rebuilt** (see P11 task 1 / P20).
- **OpenEnv serving pattern to copy**: `third_party/openenv/envs/echo_env/`
  (`server/echo_environment.py` = `Environment` subclass, `server/app.py` builds
  the FastAPI app via `HTTPEnvServer(env=<factory>, action_cls, observation_cls)`
  + `register_routes(app)`, run with uvicorn; `client.py` = policy-side client;
  `openenv.yaml` declares `app: server.app:app`).

---

## 1. Planning model

Critical path to "LLM can be trained in the env":

```
P11 issued-event stream (correctness foundation)
  -> P12 address forms + disclosure
  -> P13 task compiler + families
  -> P17 HTTP serving  ---------------\
  -> P19 LLM baseline + eval  <--------  (the "LLM-testable" milestone)

parallel fidelity/breadth (do not block the milestone):
  P14 disturbance fidelity   (after P11)
  P15 standards/profiles     (after P14)
  P16 mitigations via Ramulator (after P11, interplay with P14)
  P18 OS-level sandbox       (after P17 tool contract)

P20 release re-qualification (final serial gate)
```

**Minimum set for the LLM-testable milestone:** P11 + P12 + P13 + P17 + P19,
with P14 admitting at least the profile-driven ACT model so rewards are
meaningful. P15/P16/P18 raise fidelity, breadth, and safety but are not required
to *first* put an LLM in the loop.

Each phase: **Goal → Prerequisites → Tasks (file-level) → Admission gate →
Parallelization**. A feature may not be advertised in `dram.info`/capability
discovery until its gate passes with a real implementation.

---

## P11 — Real issued-event stream (fixes A; resolves D)

**Goal.** The disturbance model consumes **actual post-schedule DRAM commands**
(ACT/PRE/RD/WR/REF/RFM) with decoded coordinates and row-buffer state, not
frontend read/write completions. Row-buffer locality is respected: N reads to an
open row produce ~1 ACT, not N hammers.

**Prerequisites.** v2.1.0 pin (done). Rebuilt worker.

**Design decision (resolves D).** Keep the fitted statistical flip model in
Python (where the profile lives) but feed it a **real C++ issued-event stream**.
Implement a small Ramulator controller plugin that captures `on_issue` events and
have the worker emit them in its per-request JSON `events[]`. This satisfies SPEC
§4/§5 intent without a full C++ rewrite. (A full C++ port into `cpp/disturbance/`
remains a valid alternative but is out of scope here; document the choice in an
ADR under `docs/`.)

**Tasks.**
1. **Rebuild** the worker against v2.1.0: `python3 -B scripts/build_phase1.py &&
   python3 -B scripts/build_phase2.py`. Fix any drift (expected none per §0.4).
2. **Issued-event capture plugin.** Add a controller plugin (model on
   `cmd_trace_recorder.cpp`) that records, per issued command: `clk`, command
   name, full `addr_vec` (channel/rank/bankgroup/bank/row/column), `type_id`,
   and whether it was a row hit. Register it and add it to the DDR4 config used by
   the worker (`configs/ramulator/p1_external_ddr4.py` / built YAML). Prefer an
   in-memory sink drained by the worker over a file, to keep per-episode isolation.
   Location: `cpp/ramulator_extensions/issued_event_recorder.*` (new dir per
   SPEC §11 tree).
3. **Worker emits issued events.** Modify
   `cpp/simulator_service/ramulator_worker.cpp` so each `READ/WRITE/ISSUE`
   response's `events[]` contains the **issued commands** captured since the last
   request (ACT/PRE/RD/WR/REF/RFM), each with decoded coords + `row_hit`, instead
   of the single synthetic frontend `RD/WR` event it emits today
   (`ok_json`, lines ~190-201).
4. **Disturbance consumes ACT events.** Rework `DisturbanceEngine.consume()`
   (`rowhammer_env/disturbance.py:70`) to accumulate exposure from **ACT** events
   on aggressor rows (keyed by decoded bank+row), not `RD`. WR still triggers
   restore. REF/RFM handling stubbed here, completed in P14.
5. **Determinism.** Ensure the same seed + command sequence yields identical
   issued-event streams and flips (replay test).

**Admission gate — `scripts/verify_phase11.py` + `tests/`:**
- **D3**: rejected/illegal commands produce **no** issued events and no exposure
  change.
- **Row-buffer locality**: 50 000 reads to a single open row → ~1 ACT → **no
  flip**; alternating double-sided reads → many ACTs → flip. Prove the two differ.
- **A-regression**: a fixed pattern that flipped before still flips, now driven by
  ACT counts; counterfactual (single-row) control does not.
- Deterministic replay for a fixed seed + manifest.

**Parallelization.** Serial foundation; blocks P14. Its **schema** (issued-event
JSON shape) should be frozen early so P12/P17 can build against it.

---

## P12 — Address forms + disclosure projection (fixes I)

**Goal.** Support `physical` and `opaque_handle` address forms in addition to
`logical`, and implement the real disclosure matrix
(`mapping/adjacency/victim/profile/feedback`, SPEC §7-8) with **no hidden-state
leakage**.

**Prerequisites.** P11 (decoded `addr_vec` available from the worker).

**Tasks.**
1. **Address projection module.** New `rowhammer_env/tools/addressing.py`:
   bidirectional map between `{kind:physical, channel,rank,bankgroup,bank,row,
   column}` and the linear address the worker/External frontend consumes, using
   the DDR4 geometry from the Ramulator config (org `count` + `RoBaRaCoCh` bit
   layout). Replace `phase2_env._addr_value` (`phase2_env.py:134`) which currently
   hard-rejects non-logical.
2. **Opaque handles.** Implement a per-episode handle table
   (`row:target:0`, `cell:...`) resolved server-side only; handle names must be
   non-invertible (random ids, not encodings of coordinates).
3. **Disclosure projection.** New `rowhammer_env/tasks/disclosure.py` enforcing
   each level; `dram.info` advertises only permitted address forms per task.
   Error `ADDRESS_NOT_DISCLOSED` for forms the task hides.
4. **Leakage guard.** Ensure hidden physical info is unrecoverable from error
   messages, handle names, trace ids, or timing fields (SPEC §8).

**Admission gate — `scripts/verify_phase12.py`:**
- Round-trip physical↔logical decode matches Ramulator's own `addr_vec` for a
  sample of addresses (differential test against the worker).
- Disclosure levels: for each `mapping/adjacency/victim` setting, disallowed forms
  fail closed; permitted forms work.
- **Non-leakage tests**: fuzz error/observation fields for any hidden coordinate.

**Parallelization.** Design (schema, projection math) can run **concurrently with
P11**; integration/tests need P11's decoded coords. Serial before P13.

---

## P13 — Task compiler + families (fixes J)

**Goal.** A real task compiler that samples targets, seeds, budgets, and
difficulty, and implements the SPEC §7 families (known-target, hidden-target,
unknown-adjacency, any-flip, target-row, target-cell, pattern, mitigation-aware,
low-disclosure, profile-generalization).

**Prerequisites.** P12 (disclosure + address forms). P11.

**Tasks.**
1. **Task config loading.** Read `configs/tasks/*.yaml` in the SPEC §10 shape;
   validate against `spec/schemas/task.schema.json`. Remove hardcoded
   `target_row=10` / single-family behavior from `phase5_env.py:23` and the
   `row_bytes=8192` default in `disturbance.py:34` (derive geometry from config).
2. **Target sampling.** Per-seed sampling of bank/row/cell targets and aggressor
   sets; deterministic from `(task_id, seed, manifest)`.
3. **Difficulty bands.** Calibrate `easy/medium/hard` bands from
   profile thresholds + budgets (replace literal `"smoke"` in
   `phase5_env.py:104`). Record expected success rate of a reference policy per
   band.
4. **Family predicates.** Move success predicates out of the monolithic
   `_trusted_success` (`phase5_env.py:73`) into `rowhammer_env/rewards/` with one
   tested predicate per family, reading **only** trusted simulator state.
5. **Task manifests + examples** under `configs/tasks/` and `spec/examples/`.

**Admission gate — `scripts/verify_phase13.py`:**
- Each family instantiates, runs, and terminates with reproducible seeds.
- Difficulty bands hit their calibrated success-rate windows for a reference
  policy.
- Hidden-target / unknown-adjacency families expose **no** derivable target
  (uses P12 leakage guard).

**Parallelization.** Serial after P12. Family authoring (task 4/5) can be split
across agents once the compiler API (task 1-3) is fixed.

---

## P14 — Disturbance fidelity (fixes B, C, E)

**Goal.** Replace the toy model with a **profile-driven** one: correct stratum
selection, multiplicity, direction bias, RowPress dwell, hierarchical variance,
refresh/decay/RFM, and profile-driven blast radius.

**Prerequisites.** P11 (ACT/REF/RFM event stream). Admitted `ddr4_vts25_v1`.

**Tasks.**
1. **Stratum selection.** Choose `single|double × all_ones|all_zeros` from the
   **actual** aggressor pattern (single- vs double-sided, inferred from issued ACT
   neighborhood) and the **written data pattern** of the victim region — not the
   constructor default (`disturbance.py:37`).
2. **Threshold sampling.** Use the hierarchical model in the profile
   (`sampling_model`: `mu + module_offset(σ_between) + row_eps(σ_within)`), not
   just `hcfirst_lognormal.mu/sigma`.
3. **Direction & multiplicity.** Flip direction from `direction.dominant_flip_
   direction`/`biased`/`bias_strength`; number of flipped bits per event from
   `multiplicity`. Remove hardcoded bit 0 / `0->1` (`disturbance.py:107,124`).
4. **RowPress.** Model open-row dwell time (from ACT→PRE spacing / open duration
   in the issued stream) using `rowpress.rowhammer_to_rowpress_hc_reduction` and
   `operating_point_hc`; only for profiles with `rowpress.supported` (test D7).
5. **Blast radius.** Drive victim set from profile/standard (±1, ±2/half-double)
   over **physical** rows via decoded `addr_vec`, replacing the ±1 logical loop
   (`disturbance.py:95`).
6. **Refresh/decay + oracle.** Consume REF/RFM issued events: model restoration of
   affected rows (test D8). Replace the counter-reset `oracle`
   (`disturbance.py:105`) with a real target-row-refresh model; **validate it
   against Ramulator's `OracleRH` plugin** (`tRH` param) as the reference (differential
   trace).
7. **Temperature/domain.** Reject out-of-domain temperatures unless the profile
   supports extrapolation (test D9); use `temperature` tables where present.

**Admission gate — `scripts/verify_phase14.py` (disturbance/ tests D4–D10):**
- D5 known-flip fixture and D4 no-flip control across fixed seeds.
- D6 single vs double-sided exposure differs per profile.
- D7 RowPress affects only RowPress-supporting profiles.
- D8 refresh/RFM restoration behaves as specified; Python `oracle` matches
  `OracleRH` on a differential trace.
- D9 domain bounds enforced. D10 flips persist until overwritten/restored.
- Statistical validity: sampled thresholds reproduce profile quantiles on
  held-out chips (reuse `profile_builder/validate/`).

**Parallelization.** Runs **concurrently with P12 and P13** once P11 lands (it
only needs the event stream, not addressing/tasks). Sub-tasks 1-3 / 4-5 / 6-7 can
be split across agents behind a shared `DisturbanceEngine` interface.

---

## P15 — Standards & profile generalization (fixes F)

**Goal.** Remove the DDR4 hard-gate; make engine + profile schema parameterized by
standard; add a second profile/standard (HBM2 target, pending license) without
parameter pooling.

**Prerequisites.** P14 (generic, profile-driven engine). P13 (task plumbing).

**Tasks.**
1. **De-hardcode.** Remove `"Phase 4 admits DDR4 only"` (`disturbance.py:46`);
   derive geometry (`row_bytes`, bank/row counts) from the Ramulator config +
   `DRAMSpec`, not constants.
2. **Profile schema v2.** Extend the schema to carry standard-specific dimensions
   HBM needs — pseudo-channel, stack/die layer, subarray, on-die-ECC interaction,
   and RFM/refresh semantics — while remaining backward-compatible with the DDR4
   profile. Version it (`schema_version: 2`).
3. **Standard adapters.** Small per-standard hooks for: blast topology, on-die ECC
   masking (HBM), RFM/VRR refresh semantics (DDR5_RFM/VRR, HBM). Select the
   Ramulator `dram.impl` + controller per profile standard.
4. **HBM2 ingestion.** Wire `profile_builder/ingest/` for the HBM2 artifact once
   its license + hashes are resolved in `SOURCE_MANIFEST.yaml`
   (`hbm2_read_disturbance`, currently `deferred_pending_license_and_hash`). Keep
   it `deferred`/fail-closed until admitted.

**Admission gate — `scripts/verify_phase15.py`:**
- DDR4 path unchanged (regression).
- A second standard config (e.g. DDR5 or HBM2 skeleton) instantiates the generic
  engine with **no** DDR4 constants leaking.
- Each profile admitted independently; **no parameter pooling** across
  chips/standards without a documented held-out validation.

**Parallelization.** After P14. Schema design (task 2) can start during P14.
Independent of P16/P17.

---

## P16 — Mitigations via Ramulator (fixes L)

**Goal.** Advertise the SPEC §6 mitigation menu by **wiring Ramulator's real
implementations** (now present at v2.1.0), with capability discovery and
fail-closed behavior for anything not yet validated.

**Prerequisites.** P11 (issued-event stream, so mitigation effects are observable
in the trace). P14 (disturbance semantics to interplay with). P13 config plumbing.

**Tasks.**
1. **Config surface.** Map task `mitigation.{name,params}` to Ramulator controller
   config: `OracleRH` (`tRH`), `PARA`, `TWiCeIdeal`, `Graphene`, `Hydra`, `RRS`,
   `AQUA`, `RFMManager`, plus controller variants `BlockHammer`
   (`blockhammer_controller`) and `PRAC` (`prac_controller`). Generate/extend the
   worker YAML per episode from the task's mitigation.
2. **Capability discovery.** `dram.info` lists only mitigations that pass
   conformance; unimplemented/unvalidated ones return `UNAVAILABLE_CAPABILITY`
   (never silently act as `none`). Remove the Python `{none,oracle}` allowlist in
   `disturbance.py:56` in favor of the validated set.
3. **Admit one at a time** (SPEC §6, P7 discipline). Order: `oracle` (as
   `OracleRH`) → `PARA` → `Graphene`/`TWiCe` → `BlockHammer` → `PRAC` → rest.
4. **Per-mitigation conformance.** For each, a differential trace showing it
   changes issued events / protection state as expected (e.g. inserts
   preventive refreshes / defers ACTs), and that a task that succeeds under `none`
   fails under the mitigation at matched budget.

**Admission gate — `scripts/verify_phase16.py` (mitigations/ tests):**
- Each admitted mitigation changes the issued-command trace as specified.
- Unavailable mitigations fail closed.
- Mitigation-aware task family (from P13) reaches expected success/failure bands.

**Parallelization.** After P11+P14. **Each mitigation is an independent
sub-agent task** behind the P16.1 config surface — highly parallel. Concurrent
with P15 and P17.

---

## P17 — OpenEnv HTTP serving + client (fixes G)

**Goal.** Serve the environment over the real OpenEnv HTTP transport so external
RL/LLM harnesses and parallel rollout workers can attach; provide a policy-side
client.

**Prerequisites.** Stable `Phase2Action`/`Phase2Observation` schema (freeze after
P12 lands the new address fields). Can otherwise start immediately.

**Tasks.**
1. **Server app.** New `rowhammer_env/server/app.py` mirroring
   `third_party/openenv/envs/echo_env/server/app.py`: build
   `HTTPEnvServer(env=RowHammerTaskEnv, action_cls=Phase2Action,
   observation_cls=Phase2Observation)`, `register_routes(FastAPI())`, run via
   uvicorn. Factory must create a fresh env (fresh worker + episode) per session.
2. **Serialization.** Ensure `Phase2Action`/`Phase2Observation` (pydantic) round-
   trip through the server's JSON (they already subclass OpenEnv `Action`/
   `Observation`); add tests for base64 data fields and error objects.
3. **Client.** `rowhammer_env/client.py` (model on `echo_env/client.py`) exposing
   `reset/step/state` to a policy; plus a thin `sdk/rh_sdk/` client so scripts and
   external policies share one surface.
4. **Concurrency & isolation.** Verify per-session worker isolation under the
   server's concurrency model; document capacity. No cross-episode state leak.
5. **Deps.** Add `fastapi`/`uvicorn` to the project env; note that
   `rowhammer_env/openenv_source.py`'s bespoke module-injection may need to import
   the full `openenv` server package — validate and, if needed, switch to a normal
   import of the vendored package.

**Admission gate — `scripts/verify_phase17.py` (contract/ tests):**
- End-to-end `reset → step(hammer) → reward` over HTTP equals the in-process
  result (trace-equivalence).
- Two concurrent sessions do not interfere; reward still comes only from trusted
  state.
- Malformed requests map to stable error codes over the wire.

**Parallelization.** **Highly parallelizable** — wraps the existing `Environment`
interface. Can run concurrently with P12–P16; only needs the action/obs schema
frozen. Strong candidate for a dedicated agent early.

---

## P18 — OS-level script sandbox (fixes K)

**Goal.** Replace the in-process AST DSL with real OS-level isolation for
`script.run`, so untrusted (LLM-authored) Python cannot touch host memory,
filesystem, devices, or network — while remaining trace-equivalent to direct
tools.

**Prerequisites.** P17 tool contract (broker calls the same tools).

**Tasks.**
1. **Runtime selection.** Evaluate and pick an isolation mechanism (subprocess +
   `seccomp-bpf` + namespaces, or `nsjail`/`gVisor`; `Kata`/`Firecracker` if VM-
   level is required). Record an ADR in `docs/`. Enforce no-network, no-mounts,
   no-devices, CPU/mem/time limits.
2. **Broker over IPC.** `rh_sdk` inside the sandbox calls back to the env's tools
   over a restricted RPC (stdin/stdout or unix socket), not in-process. Replace
   `rowhammer_env/script_sandbox.py`'s in-process execution.
3. **Resource limits & errors.** `SCRIPT_TIMEOUT`, `SANDBOX_VIOLATION`,
   CPU/trace-volume budgets wired to the episode budget.

**Admission gate — `scripts/verify_phase18.py` (sandbox/security tests S2–S7):**
- Escape attempts blocked: filesystem, `/proc/pagemap` & device denylist
  (SPEC §2), network, fork-bomb/resource exhaustion, host mounts.
- Trace-equivalence: a script and the equivalent direct-tool sequence produce
  identical issued-event traces and reward.

**Parallelization.** Runtime evaluation + harness can run **concurrently with
P12–P16**; final wiring needs P17. Independent of the disturbance lane.

---

## P19 — LLM baseline + evaluation harness (fixes H; the milestone gate)

**Goal.** Demonstrate a **language-model policy acting in the environment** over
HTTP, plus a minimal train/eval loop and metrics. This is the "LLM-testable"
milestone the plan targets.

**Prerequisites.** P17 (serving), P13 (tasks), P11+P14 (real, meaningful rewards).
P12 (address space). P16 optional (for mitigation-aware eval).

**Tasks.**
1. **Tool-calling policy adapter.** Expose the tool surface
   (`dram.info/read/write/issue`, `script.run`, `episode.finish`) as an LLM
   tool/function schema; an adapter that turns model tool-calls into HTTP `step`s.
   Support at least one real model backend (configurable; e.g. an OpenAI-/
   Anthropic-compatible endpoint or a local HF model) — behind an interface so CI
   can run a scripted stand-in **without** claiming it is the LLM (no-mock: the
   stand-in is a test fixture, not an advertised policy).
2. **Rollout loop.** Batched episode rollouts over the HTTP env with a curriculum
   drawn from P13 difficulty bands; collect trajectories (obs, action, reward).
3. **Eval harness.** Held-out task/seed/profile splits (SPEC §7 family 10);
   success-rate, budget-efficiency, and generalization metrics under
   `rowhammer_env/observability/` + `docs/operations.md`.
4. **Reference training example.** One runnable script (RL or reward-model / GRPO-
   style, per the OpenEnv tutorials in `third_party/openenv/tutorial/`) that
   updates a policy from environment reward end-to-end, even if small. Replaces the
   bandit demo in `scripts/rl_run.py`.
5. **Docs.** `docs/api.md` (tool/obs/reward contract for policy authors) and a
   quickstart: launch server → attach LLM → run curriculum.

**Admission gate — `scripts/verify_phase19.py`:**
- An LLM (or CI fixture policy) completes episodes over HTTP and earns reward
  **only** on real trusted-state success; a "claim success without flipping"
  control earns 0.
- Eval harness produces reproducible metrics on a held-out split.
- Training example runs and shows a non-trivial policy update signal.

**Parallelization.** Adapter + eval harness (tasks 1-3) can be built against a
stub env early, then pointed at the real server once P17 lands. Serial for the
final milestone sign-off.

---

## P20 — Release re-qualification (final serial gate)

**Goal.** Prove the whole system still meets SPEC §12 acceptance on the v2.1.0
pin, with no skips and no mocks.

**Prerequisites.** All admitted release-scope phases above.

**Tasks.**
1. **Rebuild everything** against v2.1.0; run the full `verify_phase{0..19}` +
   `verify_release.py` + `unittest discover` matrix.
2. **Determinism/replay** across the pinned manifest + seeds (SPEC §12.7).
3. **No-mock scan** over all executable roots (extend `verify_phase0`'s symbol
   scan to new files).
4. **Docs/provenance**: update `spec/VALIDATION_REPORT.md`, model cards, SBOM;
   confirm `SOURCE_MANIFEST.yaml` has no `pending` for admitted sources.
5. **Retire the stale memory**: confirm this document is the single source of
   truth for post-P10 work.

**Admission gate — `scripts/verify_release.py` (extended):** full matrix green,
no required skips, no mock/fallback imports, deterministic replay.

---

## 2. Parallelization guide (for coding agents)

The **critical serial spine** is `P11 → P12 → P13 → P17 → P19` (with P14 admitting
before P19 so rewards are meaningful). Everything else fans out. A phase is
"parallelizable" if a separate coding agent can work it concurrently without
racing the spine on the same files.

| Phase | Parallelizable? | Can run concurrently with | Hard blockers | Notes for an agent |
|---|---|---|---|---|
| **P11** | No (foundation) | — | v2.1.0 pin (done) | Freeze the issued-event JSON schema first so others build against it. Touches `cpp/` + `disturbance.py`. |
| **P12** | Partly | **P11** (design/schema only) | P11 for integration | Address math + disclosure are self-contained; integration needs P11 coords. |
| **P13** | Partly | P14, P17 | **P12** | Family authoring splits across agents once compiler API is fixed. |
| **P14** | **Yes** | **P12, P13, P16, P17** | **P11** | Pure Python + profile; only needs the event stream. Split as 1-3 / 4-5 / 6-7. Best parallel candidate on the fidelity side. |
| **P15** | Yes | P16, P17 | P14 | Schema design can begin during P14. |
| **P16** | **Yes (highly)** | **P14, P15, P17** | P11 (+P14 for interplay) | **Each mitigation is an independent sub-task** behind the P16.1 config surface — ideal fan-out. |
| **P17** | **Yes (highly)** | **P12–P16, P18** | Action/Obs schema frozen (post-P12) | Wraps existing `Environment`; independent lane. **Recommended dedicated agent, started early.** |
| **P18** | **Yes** | **P12–P16** | P17 for final wiring | Runtime eval + escape-test harness is fully independent infra. |
| **P19** | Partly | build against stub early | **P17, P13, P14** | Adapter/eval scaffolding parallelizable; final sign-off serial. |
| **P20** | No (final gate) | — | all | Serial release gate. |

**Recommended concurrent staffing** (given N agents):
- **Agent 1 (spine):** P11 → P12 → P13, then help P19.
- **Agent 2 (serving/safety):** P17 immediately (against current schema, refit
  after P12) → P18. Independent lane, unblocks P19 early.
- **Agent 3 (fidelity):** P14 as soon as P11's event schema is frozen → P15.
- **Agent 4 (mitigations):** P16, one mitigation per work item, after P11/P14
  interfaces exist.
- Converge on **P19** (LLM-in-the-loop milestone), then **P20** re-qualification.

**Fastest route to "an LLM in the environment":** land **P11 → P12 → P13** on the
spine and **P17** in parallel, admit **P14** for real rewards, then **P19**.
P15/P16/P18 harden and broaden afterward and are not on the milestone's critical
path.
