# Concise specification: RowHammer-OpenEnv RL environment

## 1. Purpose

Build an OpenEnv-compatible reinforcement-learning environment where language-model policies interact only with simulated DRAM and attempt to satisfy rowhammer/read-disturbance target conditions. The environment is for hardware-security research and DRAM-standard hardening. It must never touch host physical memory or provide an unsandboxed path for policy code.

The policy may either call tools directly or submit a restricted script that generates tool calls. The simulator owner, not the policy, computes reward and success from trusted simulated memory state.

## 2. Non-negotiable constraints

> **No mock implementations, no fake fallbacks.** Production code, CI admission, training, evaluation, and release qualification must never use a mock DRAM, placeholder flip generator, synthetic stand-in calibration data, no-op mitigation, permissive sandbox fallback, local unsandboxed script runner, canned reward, or compatibility shim that pretends a missing feature exists. An unavailable feature must be absent from capability discovery and fail closed with a stable error code.

Other constraints:

1. Simulation-only: deny host physical-address APIs, `/proc/pagemap`, `/dev/mem`, `/dev/kmem`, `/dev/kvm`, huge-page discovery, cache-control attack utilities, RDMA, PCIe/GPU memory handles, and all host device access.
2. All policy-visible actions go through OpenEnv `step()` or the script broker that itself calls `step()`-equivalent environment tools. Do not expose a direct control path that bypasses reward, budgets, or termination.
3. Use Ramulator 2.1 for DRAM timing, state transitions, scheduling, refresh, row-buffer behavior, address mapping, request completion, and built-in controller functionality wherever available.
4. Add read-disturbance behavior as a real Ramulator-integrated extension driven by actual issued commands/events, not requested commands that might have been rejected or delayed.
5. Use empirical profiles fitted from public real-chip data. Profiles may be generative only when fitted to real measurements, source-traced, statistically validated, and labeled as sampled rather than exact chip replay.

## 3. External sources to pin in `SOURCE_MANIFEST.yaml`

The build must pin exact commits, artifact hashes, licenses, and retrieval dates. Initial sources:

| Source | Required use | URL |
|---|---|---|
| Ramulator 2.1 branch/repository | Base DRAM simulator and controller framework | `https://github.com/CMU-SAFARI/ramulator2/tree/v2.1` |
| Ramulator 2.1 paper | Design reference for simulator capabilities and validation expectations | `https://arxiv.org/abs/2606.13844` |
| OpenEnv | Environment protocol and Gym-style `reset/step/state` control plane | `https://github.com/huggingface/OpenEnv` |
| OpenEnv release to evaluate first | Suggested initial pin if still current at implementation start | `https://github.com/huggingface/OpenEnv/releases/tag/v0.3.1` |
| DDR4 VTS 2025 read-disturbance artifact | First production DDR4 disturbance profile; includes RowHammer/RowPress characterization from 96 COTS DDR4 chips | `https://github.com/CMU-SAFARI/ReadDisturbanceVTS25` |
| VTS 2025 paper | Scientific reference for the DDR4 artifact | `https://arxiv.org/abs/2503.16749` |
| HBM2 read-disturbance artifact | Optional HBM2 profile after DDR4 is admitted | `https://github.com/CMU-SAFARI/HBM-Read-Disturbance` |
| HBM2 paper | Scientific reference for the HBM2 artifact | `https://arxiv.org/abs/2310.14665` |

If a URL, branch, or release changes, update only the manifest and the associated provenance tests; do not silently substitute data.

## 4. Architecture

```text
RL trainer
  -> OpenEnv client
  -> RowHammerEnv.reset/step/state
  -> trusted Python environment server
  -> simulator worker RPC
  -> Ramulator 2.1 + disturbance extension
  -> trusted reward/termination module

Optional script path:
RL policy -> script.run(code) -> strict sandbox -> rh_sdk broker -> same tool interface -> same simulator worker
```

Components:

- **OpenEnv server:** owns episode lifecycle, budgets, observations, reward, and hidden-state projection.
- **Simulator worker:** separate process per episode or per isolated worker lease; owns Ramulator instance, memory overlay, disturbance state, and event trace.
- **Disturbance extension:** C++/Ramulator-adjacent module consuming issued DRAM events and refresh events.
- **Profile loader:** verifies signed profile packages and exposes only admitted profiles.
- **Task compiler:** samples topology disclosure, target condition, budgets, mitigations, and hidden variables from task configs.
- **Script sandbox:** executes policy-authored Python with no network, no devices, no host mounts, fixed resource limits, and access only to the brokered API.

## 5. Simulation and disturbance fidelity

The environment should claim: **cycle-level DRAM-system simulation with empirically calibrated statistical read-disturbance behavior.** It must not claim transistor-level fidelity or exact prediction for arbitrary commercial modules.

Use Ramulator 2.1 for:

- timing constraints and command legality;
- controller scheduling and request queues;
- row-buffer state;
- address mapping;
- refresh behavior where implemented;
- controller plugins and mitigation hooks where implemented.

Implement the following extension points:

1. **Functional memory overlay.** Sparse simulated memory initialized from the task seed; reads return current simulated bytes; writes restore written cells and update disturbance-relevant state.
2. **Issued-event stream.** Every accepted ACT/PRE/RD/WR/REF/RFM-equivalent event is logged with cycle, decoded coordinates, row-buffer state, and source action id.
3. **Exposure accounting.** For each potential victim region, accumulate exposure from neighboring row activations, single-sided/double-sided patterns, open-row dwell time, refresh age, retention age, and temperature.
4. **Latent vulnerability state.** Sample persistent row/cell/channel vulnerability variables from admitted empirical profiles at episode start. Do not resample independent bit-flip probabilities on each access.
5. **Flip transition.** A bit flips when accumulated exposure crosses the sampled condition for that cell/region. Direction and multiplicity come from the profile. Flips persist until overwritten or refreshed if the model says refresh restores the relevant state.
6. **Noise and temporal variation.** Include fitted temporal variability and noise terms only from validated profile components. Out-of-domain temperatures or timings must be rejected unless the profile explicitly supports extrapolation.
7. **Refresh/decay.** Use Ramulator refresh events as ground truth. The extension tracks which modeled rows are restored or protected by refresh/RFM/mitigation events.

## 6. Mitigations

Mitigations are configured per task:

```yaml
mitigation:
  name: none | oracle | para | twice | graphene | blockhammer | hydra | rrs | aqua | prac | custom
  params: {}
```

Rules:

- Use a Ramulator 2.1 implementation directly if it exists and passes conformance tests.
- If a mitigation exists only in another Ramulator branch/version or paper artifact, port it as real code with paper-to-code tests and differential traces before advertising it.
- `none` means no read-disturbance mitigation beyond baseline refresh.
- `oracle` is the first mitigation to implement because it gives a correctness reference for target-row refresh behavior.
- A mitigation that is not implemented or not validated must return `UNAVAILABLE_CAPABILITY`; it must not act as `none`.

## 7. Task families

Each task config chooses a profile, topology disclosure, target condition, budgets, and success predicate.

Core families:

1. **Known adjacency, known target row:** policy receives physical bank/row coordinates for target and candidate aggressor rows.
2. **Known adjacency, hidden target row:** policy receives a target handle; exact coordinates remain hidden.
3. **Unknown adjacency:** policy must infer effective neighboring rows through permitted simulated reads/writes/commands.
4. **Any flip:** success if any simulated cell flips under budget.
5. **Target row flip:** success if any cell flips in a target row or row set.
6. **Target cell flip:** success if a specified bit or byte satisfies the target condition.
7. **Pattern target:** success if a target mask/value appears after disturbance.
8. **Mitigation-aware:** same objectives with mitigation enabled.
9. **Low-disclosure search:** logical addresses only; physical mapping and adjacency hidden.
10. **Profile generalization:** train and evaluate on different profile/chip/module splits.

Task disclosure levels:

```yaml
mapping: physical | logical_only | opaque_handles
adjacency: exact | candidate_set | hidden
victim: exact | row_handle | cell_handle | hidden_until_finish
profile: public_profile_id | family_only | hidden
feedback: full_trace | summarized_counts | reward_only
```

## 8. Policy-facing interface

The API is a small set of tools. All coordinates refer to simulated DRAM only. Tool schemas live in `schemas/action.schema.json`; semantic rules below are normative.

### Tool summary

| Tool | Purpose | Returns |
|---|---|---|
| `dram.info` | Query disclosed topology, profile, mitigation, budgets, and allowed commands. | Static capability object. |
| `dram.read` | Read simulated memory by logical address, physical coordinate, or opaque handle depending on task disclosure. | Base64 bytes and optional metadata allowed by task. |
| `dram.write` | Write simulated memory; updates functional state and restoration state. | Accepted byte count and cycle delta. |
| `dram.issue` | Submit DRAM-level commands or waits. Commands are validated and executed by Ramulator. | Issued/rejected counts, cycle delta, summarized events. |
| `script.run` | Execute restricted Python using `rh_sdk`, which calls the same tools through a broker. | Script stdout tail, tool-call summary, error if any. |
| `episode.finish` | Declare completion. | Final success, reward, public summary. |

### Address forms

```json
{"kind":"logical","addr":4096}
{"kind":"physical","channel":0,"rank":0,"bankgroup":0,"bank":1,"row":1234,"column":64}
{"kind":"handle","id":"row:target:0"}
```

The task disclosure decides which forms are accepted. Hidden physical information must never be recoverable from error messages, timing artifacts outside the simulated model, logs, trace ids, or handle names.

### Command form

```json
{
  "op": "ACT | PRE | RD | WR | REF | RFM | WAIT",
  "addr": {"kind":"physical", "channel":0, "rank":0, "bankgroup":0, "bank":1, "row":1234, "column":0},
  "cycles": 0,
  "data_b64": "optional-for-WR",
  "tag": "policy-chosen short label"
}
```

Compact forms (expanded server-side into the primitives above before execution,
so budget and disturbance accounting are on the true expanded event count):

- `{"op": <RD|WR|WAIT>, ..., "repeat": N}` (alias `count`) issues that primitive `N` times.
- `{"op": "HAMMER", "rows": [addrA, addrB, ...], "pairs": N}` (alias `count`) issues
  `N` sweeps of one read to each listed row — the canonical double-sided hammer for
  two rows. Equivalent at the worker to writing every alternating read by hand.

Rules:

- `WAIT` uses `cycles` and no address.
- `WR` requires data; `RD` may return data only if the task permits command-read feedback.
- Illegal timing or address requests are rejected with stable errors and do not update disturbance state.
- Disturbance accounting uses actual issued events after controller scheduling, not command list order alone.
- Each action has a budget cost: tool call, simulated cycles, ACT count, bytes read/written, script CPU time, and trace volume.

### Initial observation

`reset()` returns:

- task id, episode id, public seed id, and task family;
- objective in natural language plus structured target summary;
- disclosed topology and accepted address forms;
- profile id/family and mitigation config at the permitted disclosure level;
- allowed tools, command set, limits, and stable error codes;
- reward schedule and termination conditions;
- any target handles or physical coordinates allowed by the task.

### Step observation

Each `step(action)` returns:

```json
{
  "ok": true,
  "observation": {
    "cycle": 12040,
    "budget_remaining": {"tool_calls": 190, "acts": 98000, "cycles": 4980000},
    "last_action": {"accepted": 32, "rejected": 0, "cycle_delta": 1040},
    "public_counters": {"acts": 320, "reads": 64, "writes": 2, "refreshes": 1},
    "feedback": {"new_public_flips": 0, "trace_tail": []}
  },
  "reward": 0.0,
  "terminated": false,
  "truncated": false,
  "error": null
}
```

Feedback fields are task-dependent. Hidden target state is never included unless the task explicitly discloses it.

### Stable error codes

`BAD_SCHEMA`, `UNSUPPORTED_TOOL`, `UNAVAILABLE_CAPABILITY`, `BUDGET_EXCEEDED`, `ADDRESS_NOT_DISCLOSED`, `ILLEGAL_COMMAND`, `QUEUE_FULL`, `SANDBOX_VIOLATION`, `SCRIPT_TIMEOUT`, `PROFILE_REJECTED`, `INTERNAL_SIMULATOR_ERROR`.

## 9. Reward and termination

Default sparse reward:

- `1.0` on target success;
- `0.0` otherwise;
- optional small negative cost for budget exhaustion only in training tasks, not benchmark scoring.

Termination:

- success predicate true;
- `episode.finish` called;
- budget exhausted;
- unrecoverable simulator or sandbox error;
- maximum simulated cycles reached.

The reward module reads trusted simulator state directly. It must not derive success from policy-provided logs, stdout, or submitted claims.

## 10. Configuration files

Minimal task config:

```yaml
id: ddr4_known_target_anybit_v1
profile: ddr4_vts25_v1
standard: DDR4
mitigation: {name: none, params: {}}
disclosure:
  mapping: physical
  adjacency: exact
  victim: exact
  profile: public_profile_id
  feedback: summarized_counts
objective:
  type: target_row_flip
  target: {bank: sampled, row: sampled}
budgets:
  tool_calls: 200
  acts: 100000
  cycles: 5000000
  script_ms: 0
reward: sparse_success
```

Profile packages must include: source manifest, canonical data hashes, fitted parameters, supported standards, supported temperatures/timings, held-out validation report, model card, and signature.

## 11. Repository skeleton

Use the tree in `REPOSITORY_TREE.md`. Keep ownership boundaries stable:

- `cpp/` owns Ramulator integration, worker, memory overlay, disturbance engine, mitigation ports.
- `rowhammer_env/` owns OpenEnv server, tasks, rewards, profiles, sandbox broker, and observability.
- `profile_builder/` owns data ingestion, fitting, validation, and model cards.
- `schemas/` owns compact wire-shape schemas only.
- `tests/` owns unit, integration, statistical, security, and release gates.

## 12. Acceptance definition

A release is acceptable only when:

1. At least one DDR4 profile fitted from VTS25 data is admitted.
2. A known-target task can produce a real simulated flip through issued Ramulator events.
3. The same task fails when budgets or mitigations should prevent success.
4. The OpenEnv API, script path, reward, and hidden-state projection pass contract tests.
5. The sandbox blocks all host-memory and filesystem escape attempts.
6. Statistical validation passes on held-out source units.
7. Replays are deterministic for the same pinned source manifest and seed.
8. No production/test/training path imports or depends on mocks or fallback implementations.
