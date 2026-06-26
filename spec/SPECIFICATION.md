# RowHammer-OpenEnv
## Engineering Specification for a Simulation-Only Reinforcement-Learning Environment

**Document status:** Normative implementation specification  
**Intended audience:** DRAM-simulator engineers, ML/RL infrastructure engineers, security engineers, test engineers, and research scientists  
**Specification version:** `rh-openenv-spec/1.0`  
**Policy API version:** `rh-openenv/v1`  
**Date:** 2026-06-26  

---

## 1. Purpose

RowHammer-OpenEnv is a reinforcement-learning environment in which a language-model policy develops memory-access programs that cause specified disturbance effects in **simulated** DRAM. The environment exists to support defensive hardware-security research: training and evaluating models that can discover access-pattern weaknesses, compare mitigations, and expose shortcomings in future DRAM-security designs before deployment.

The environment shall combine:

1. **Ramulator 2.1** for cycle-level DRAM protocol, controller, scheduling, address-mapping, row-buffer, timing, and refresh behavior;
2. a separately testable **empirical read-disturbance engine** calibrated against public measurements from real DRAM chips;
3. **OpenEnv** for the Gym-style episode boundary and language-model tool interface; and
4. a strict, fail-closed execution sandbox for policy-authored scripts.

An episode succeeds only when the trusted evaluator confirms that the configured target condition has been reached in the simulated memory state. Model text, claimed success, stdout, or simulator crashes never constitute success.

## 2. Normative language

The terms **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** are normative.

An implementation is conformant only when every REQUIRED behavior is implemented through the real production path and all mandatory tests pass without skips.

## 3. Executive design decisions

The implementation shall use the following decisions unless this specification is formally revised:

- Pin Ramulator 2.1 to commit `278f1effc3838099a6ffe0ad5f9f572fea80c948`.
- Pin OpenEnv to release `v0.3.1`, commit `7449c5dfe375c4c6e6f0827826925a46efd9249f`, and lock all transitive package hashes.
- Route every policy tool invocation used in training or evaluation through OpenEnv's `step(CallToolAction(...))` control path. The direct `/mcp` endpoint shall be disabled or unreachable for rollout identities because that path does not perform normal reward, step-count, and termination accounting.
- Run the trusted OpenEnv server, trusted Ramulator worker, and untrusted policy-script runner in separate processes. The script runner shall additionally be placed in an approved hardened isolation runtime.
- Use the public VTS 2025 DDR4 read-disturbance data as the first production calibration source. Add HBM2 only as a separate profile; do not pool measurements across standards.
- Treat Ramulator as the source of truth for legal DRAM commands, command issue timing, controller arbitration, refresh issuance, address mapping, and mitigation actions.
- Treat the empirical disturbance engine as the source of truth for simulated charge-loss state, retention state, latent cell susceptibility, and bit-flip realization.
- Use deterministic, counter-based pseudorandomness so that an episode can be replayed exactly from its build fingerprint, configuration, and seed.
- Fail closed whenever a simulator feature, empirical profile, mitigation, data license, profile signature, or isolation requirement is unavailable.

> [!IMPORTANT]
> ## NON-NEGOTIABLE: NO MOCKS, NO SIMPLIFIED FALLBACKS
>
> Production code, tests, training, evaluation, and CI **must never** substitute a mock DRAM, fake bit-flip generator, synthetic placeholder calibration dataset, no-op mitigation, permissive local executor, or “best effort” sandbox. If Ramulator, a provenance-checked empirical profile, a required mitigation implementation, or the required isolation runtime is unavailable, startup, build, or test execution must fail closed with a clear error.
>
> Early milestones may support fewer features, but every advertised feature must execute through the real Ramulator path and a profile calibrated from real measurements. No feature flag may silently downgrade fidelity or isolation. A generative statistical profile fitted to real measurements is permitted; it is not a mock, but it must be explicitly labeled as generated, carry a model card, remain inside its validated domain, and pass the calibration gates in this specification.

## 4. Scope

### 4.1 In scope

The environment shall support:

- DDR4 first, followed by separately validated DDR3/DDR3L and HBM2 profiles;
- transaction-level access through the real memory controller;
- direct issue of synthetic DRAM commands through a validator/sequencer connected to the real Ramulator timing/state machinery;
- policy-authored Python programs that call the same tool API through a capability-scoped SDK;
- RowHammer and RowPress exposure;
- ordinary retention decay, temperature-conditioned behavior, refresh recovery, write recovery, stochastic chip/row/cell variability, and temporal measurement variability;
- mitigations that are genuinely implemented and verified on the selected Ramulator 2.1 pin;
- tasks ranging from a first flip anywhere to a specific simulated cell, transition, syndrome, count, row, or persistent target condition;
- disclosed, partially disclosed, or hidden physical adjacency and logical-to-physical mapping;
- known, candidate-set, partially known, or hidden target locations;
- deterministic curricula and stochastic held-out evaluation;
- strict simulation-only containment and information-flow controls.

### 4.2 Explicitly out of scope

The project shall not provide or expose:

- access to host physical memory, real DIMMs, memory-controller registers, firmware, SPD, `/dev/mem`, `/proc/pagemap`, MSRs, huge-page address discovery, cache-flush instructions, DMA/RDMA, FPGA boards, or physical address translation;
- utilities that export a discovered pattern into a native physical-memory attack program;
- host-device passthrough or co-location with privileged memory-testing hardware;
- claims of transistor-level or circuit-level accuracy;
- claims that a generated simulated cell is an exact prediction for a particular commercial DIMM unless the profile contains and replays that exact measured identity;
- unsupported extrapolation beyond the profile's measured or explicitly validated domain;
- a toy ECC implementation. ECC tasks may be enabled only through a validated, protocol-correct ECC model or a verified integration such as EINSim.

### 4.3 Fidelity claim

The product shall describe itself as:

> A cycle-level DRAM-system simulation with empirically calibrated statistical read-disturbance behavior.

It shall not describe itself as an electrical, transistor-level, or universally predictive DRAM model. Every profile release shall publish quantitative goodness-of-fit and held-out validation results.

## 5. Pinned upstream dependencies

### 5.1 Ramulator 2.1

Production builds shall use:

- Repository: https://github.com/CMU-SAFARI/ramulator2
- Branch documentation: https://github.com/CMU-SAFARI/ramulator2/tree/v2.1
- Required commit: https://github.com/CMU-SAFARI/ramulator2/commit/278f1effc3838099a6ffe0ad5f9f572fea80c948
- Ramulator 2.1 paper: https://arxiv.org/abs/2606.13844

The build shall verify the full Git object ID, submodule state, patch-series digest, compiler identity, CMake version, and generated-source digest before linking.

Ramulator 2.1 provides the cycle-level memory model, Python-generated DRAM definitions, an `External` request frontend, controller plugins, request completion callbacks, and timing/state machinery. The integration shall use these implementations rather than recreating equivalent scheduler, refresh, row-buffer, timing, or address-mapping logic outside Ramulator.

### 5.2 Mitigation source reference

The public Ramulator 2.1 pin above does not contain the RowHammer-mitigation plugin set present in the older/main Ramulator 2.0 line. The source reference for audited forward ports shall initially be:

- Source tree: https://github.com/CMU-SAFARI/ramulator2/tree/be93be78055d922aa1d4d33e15bcc8f2b0c61a9d/src/dram_controller/impl/plugin
- Source commit: https://github.com/CMU-SAFARI/ramulator2/commit/be93be78055d922aa1d4d33e15bcc8f2b0c61a9d

This source pin is a reference implementation, not the runtime simulator. Each mitigation shall be forward-ported into the pinned 2.1 architecture, code-reviewed algorithm by algorithm, and admitted only after differential and paper-conformance tests pass. A moving `main` branch shall never be consumed in a production build.

### 5.3 OpenEnv

Production builds shall use:

- Repository: https://github.com/huggingface/OpenEnv
- Release: https://github.com/huggingface/OpenEnv/releases/tag/v0.3.1
- Tag tree: https://github.com/huggingface/OpenEnv/tree/v0.3.1
- Required commit: https://github.com/huggingface/OpenEnv/commit/7449c5dfe375c4c6e6f0827826925a46efd9249f

The environment shall subclass the version-pinned OpenEnv environment interface, preferably `MCPEnvironment` where its behavior is compatible with this specification. OpenEnv APIs are treated as versioned external contracts; compatibility shims shall be covered by contract tests and shall not alter episode semantics.

### 5.4 Compiler and runtime baseline

The initial reproducible toolchain shall pin:

- a C++20 compiler;
- CMake 3.14 or newer, with an exact version in the build image;
- Python 3.10 or newer, with an exact patch version;
- OpenEnv and Python dependency hashes;
- the isolation-runtime and container-image digests;
- the host-kernel compatibility range used for release qualification.

The exact production versions shall be recorded in `build/manifest.json` and exposed through `describe_environment` as non-secret build metadata.

## 6. Public empirical data and required use

### 6.1 Primary DDR4 calibration source

**Revisiting DRAM Read Disturbance: Identifying Inconsistencies Between Experimental Characterization and Device-Level Studies (VTS 2025)**

- Paper: https://arxiv.org/abs/2503.16749
- Repository: https://github.com/CMU-SAFARI/ReadDisturbanceVTS25
- Pinned commit: https://github.com/CMU-SAFARI/ReadDisturbanceVTS25/commit/5d734309457cc8a4ea3b1ec36b93932925548bac
- Immutable data directory: https://github.com/CMU-SAFARI/ReadDisturbanceVTS25/tree/5d734309457cc8a4ea3b1ec36b93932925548bac/data

This dataset is the REQUIRED initial DDR4 source. It reports characterization of 96 commodity DDR4 chips and includes raw experiment CSVs. The ingestion pipeline shall recognize, at minimum, the following per-module file families when present:

- `*_retention.csv`: temperature, initialized pattern, wait time, row, and number of bit flips;
- `*_rd_ber.csv`: victim row, data pattern, hammer count, aggressor type, bit-flip count, and iteration;
- `*_rd_hcf.csv`: first-flip/hammer-count characterization using the same major conditioning fields;
- `*_rd_rp.csv`: RowPress characterization;
- `*_ds_ber_sweep.csv`: temperature/victim-row/hammer-count/iteration sweeps.

The parser shall derive the exact schema from the pinned files, reject missing or renamed required fields, preserve anonymized module identity, and produce a machine-readable provenance report. The repository did not present an obvious data license during specification preparation; legal review is REQUIRED before redistribution or publication of derived profiles. Absence of license clearance shall fail the redistribution pipeline, not silently relabel the data.

### 6.2 RowPress protocol and mitigation source

**RowPress: Amplifying Read Disturbance in Modern DRAM Chips**

- Paper: https://arxiv.org/abs/2306.17061
- Repository: https://github.com/CMU-SAFARI/RowPress
- Pinned commit tree: https://github.com/CMU-SAFARI/RowPress/tree/5b6f1502594e9ea7a1557b98b5544c096a30eec5

Use this artifact for experimental protocol, access-pattern semantics, terminology, and mitigation-reference behavior. Use the VTS 2025 `*_rd_rp.csv` measurements as the first raw DDR4 calibration data unless a separately licensed raw RowPress dataset is added to the source manifest. Do not claim that this repository itself contains all raw measurements unless the pinned artifact is audited and found to do so.

### 6.3 HBM2 profile source

**Read Disturbance in High Bandwidth Memory: A Detailed Experimental Study on HBM2 DRAM Chips**

- Paper: https://arxiv.org/abs/2310.14665
- Repository: https://github.com/CMU-SAFARI/HBM-Read-Disturbance
- Pinned repository tree: https://github.com/CMU-SAFARI/HBM-Read-Disturbance/tree/bc9d600e03efbe38c740ba015398510ca9bd1a60
- Archived artifact: https://zenodo.org/records/10257930
- DOI: https://doi.org/10.5281/zenodo.10257930

The archived artifact is approximately 6 GB, has published MD5 `0601628cb43d6e1e7c9d6dd6ad2dfd26`, and is licensed CC BY 4.0. It shall be downloaded only by the offline data-ingestion job, verified, and stored in the controlled raw-data cache. HBM2 observations shall form a separate profile family and shall not be pooled into DDR4 parameter estimates.

### 6.4 Historical and cross-validation sources

These sources may be used for historical regression, aggregate validation, or separate profiles; they shall not be blended into DDR4 production calibration without an explicit scientific justification and held-out evidence.

- Original RowHammer data/code: https://github.com/CMU-SAFARI/rowhammer and https://github.com/CMU-SAFARI/rowhammer/tree/master/data
- DRAM Voltage Study retention data: https://github.com/CMU-SAFARI/DRAM-Voltage-Study/blob/master/characterization_results/retention_time_profile.csv
- Revisiting RowHammer (2020): https://arxiv.org/abs/2005.13121
- Variable Read Disturbance (2025): https://arxiv.org/abs/2502.13075
- Optional ECC simulation integration: https://github.com/CMU-SAFARI/EINSim

Every source shall enter through a signed `SOURCE_MANIFEST` entry containing immutable revision or DOI, per-file checksums, license status, parser version, citation, permitted use, and split policy. Runtime workers shall never download research data.

## 7. System qualities and acceptance priorities

The project shall prioritize, in order:

1. simulation-only containment;
2. semantic correctness and reproducibility;
3. empirical validity and transparent uncertainty;
4. non-leakage of hidden task state;
5. episode throughput;
6. convenience.

An optimization or convenience feature that weakens a higher-priority property shall not be accepted.

---

## 8. Architecture

### 8.1 Trust domains

The deployment shall contain four distinct trust domains:

1. **Trainer/orchestrator client.** Chooses an authorized task family and seed, calls `reset`, sends policy actions to `step`, and records rewards. It does not inspect privileged evaluator state.
2. **Trusted environment service.** Owns the OpenEnv episode state machine, validates all actions, enforces budgets and disclosure rules, calculates reward, redacts observations, and supervises child processes.
3. **Trusted simulator service.** Owns Ramulator, the disturbance model, the simulated backing store, hidden physical mapping, mitigation state, and privileged target verifier.
4. **Untrusted policy-script runner.** Executes policy-supplied Python source in a hardened isolation domain. It sees only the capability-scoped SDK and public episode information.

The untrusted runner shall never be linked into the simulator process, mmap simulator state, share a writable volume with it, inherit its file descriptors, or access its control socket directly.

### 8.2 Logical deployment

```text
+----------------------+           +----------------------------------+
| RL trainer / policy  |  OpenEnv  | Trusted environment service      |
| generation loop      +---------->+ - reset / step / state           |
+----------------------+           | - schema and disclosure checks   |
                                   | - reward and termination         |
                                   | - budget accounting              |
                                   +----------+-----------------------+
                                              |
                        typed, authenticated  | internal RPC
                                              v
                                   +----------------------------------+
                                   | Trusted Ramulator worker         |
                                   | - External request frontend      |
                                   | - direct-command sequencer       |
                                   | - real refresh/controller logic  |
                                   | - mitigation plugins             |
                                   | - disturbance engine             |
                                   | - sparse simulated memory        |
                                   | - privileged verifier            |
                                   +----------------------------------+
                                              ^
                                              |
                                   capability | broker RPC only
                                              |
                                   +----------+-----------------------+
                                   | Untrusted Python runner          |
                                   | - fixed read-only image          |
                                   | - rh_sdk                          |
                                   | - no host/network/device access  |
                                   +----------------------------------+
```

### 8.3 Process lifecycle

A fresh trusted simulator worker shall be created for every episode. Process reuse is forbidden for the first conformant release. A future immutable-template/fork optimization may be admitted only after a written proof of state isolation and differential tests demonstrate bit-for-bit equivalence to fresh-process execution.

On reset, the environment service shall:

1. validate the trainer's signed task request;
2. resolve a task template, profile, Ramulator config, mitigation config, and all seeds;
3. verify all profile and binary signatures/checksums;
4. start a fresh simulator worker with a one-episode capability token;
5. instantiate Ramulator and the backing memory;
6. initialize hidden mapping, latent chip state, target, and memory pattern;
7. perform a reachability/preflight check where required by the task manifest;
8. start a fresh untrusted runner only when `run_script` is first used, or at reset if the sandbox policy requires pre-attestation;
9. return the public initial observation.

On terminal success, failure, truncation, crash, or close, the service shall revoke all episode capabilities, terminate the untrusted runner, terminate the simulator worker, delete ephemeral storage, and emit a signed episode record. Cleanup failure is an infrastructure error and shall quarantine the worker node.

### 8.4 Internal RPC

The trusted environment service and simulator worker shall communicate over a versioned, length-delimited, authenticated local protocol. Protobuf, Cap'n Proto, or an equivalently schema-driven format is acceptable. Ad-hoc pickle, Python object serialization, shell command construction, or unbounded JSON streams are forbidden.

Every internal request shall contain:

- protocol version;
- episode ID;
- monotonic request sequence number;
- unique request ID;
- deadline;
- operation-specific payload;
- integrity/authentication tag where the transport does not already provide equivalent protection.

The simulator shall reject cross-episode tokens, duplicate non-idempotent sequence numbers, messages above configured limits, unknown fields when strict mode is enabled, and requests after termination.

## 9. Ramulator integration

### 9.1 Source of truth

Ramulator shall exclusively determine:

- request queue admission and backpressure;
- transaction-to-command scheduling;
- legal command prerequisites;
- command timing constraints;
- bank/rank/channel state transitions;
- row-buffer state;
- refresh command scheduling and issue timing;
- address mapping configured for the episode;
- mitigation-triggered commands, delays, throttling, or remapping;
- request completion cycle and memory-system statistics.

No Python-side component may approximate these behaviors for production execution.

### 9.2 Transaction path

The simulator worker shall embed Ramulator through the pinned C++ library and `External` frontend. A submitted transaction shall be admitted only through Ramulator's request-ingress method. When Ramulator reports a full queue, the worker shall either retry on later simulated cycles according to the action's explicit execution policy or return a structured partial-completion result. It shall not bypass the queue.

The worker shall tick the real memory system one DRAM cycle at a time or through a verified cycle-skipping optimization that is semantically identical. Request completion callbacks shall drive timed data-plane operations and response completion.

The worker shall reject transaction sizes, alignments, or address ranges that the selected Ramulator configuration does not support. Splitting a large transaction into bursts is permitted only when the selected tool mode specifies splitting and the split exactly matches the memory-system transaction semantics.

### 9.3 Direct-command path

The direct-command interface is a synthetic simulator feature. It shall not expose a physical-memory API.

A `DirectCommandFrontend` shall submit requested commands to the selected Ramulator controller through a narrow extension point that reuses the controller's real prerequisite, timing, state-update, refresh-arbitration, and mitigation paths. The extension must not directly mutate bank state to “make a command work.”

Two timing modes are permitted:

- `strict`: issue at the policy-requested simulated cycle; return `DRAM_TIMING_VIOLATION` or `MISSING_PREREQUISITE` when illegal;
- `earliest_legal`: hold the command until it becomes legal, while allowing mandatory refresh and mitigation arbitration. This mode is available only when the task manifest grants it.

The sequencer shall not invent or auto-insert ACT, PRE, refresh, or other prerequisite commands. It may delay a command in `earliest_legal` mode; it must report the requested and actual issue cycles.

The valid command vocabulary and address fields shall be generated from the selected Ramulator DRAM definition. Hard-coding a DDR4-only command enum in the public API is forbidden.

### 9.4 Controller observation hook

A C++ `ReadDisturbancePlugin` shall attach to each controller and receive every issued command after Ramulator has accepted and applied the scheduling decision. At minimum, the hook shall receive:

- actual issue cycle;
- channel/controller identity;
- command identifier and command class;
- complete resolved DRAM coordinate available at that command level;
- request/source identity where applicable;
- mitigation/refresh/policy source annotation;
- open-row state immediately before and after issue, where needed;
- data-pattern/write-mask metadata for data commands.

The hook shall be implemented against Ramulator 2.1's real plugin lifecycle. Any additional extension point required for pre/post state shall be a minimal reviewed patch and shall have an upstream-diff test.

### 9.5 Configuration generation

Human-authored task/config files shall be compiled into a canonical Ramulator configuration through version-pinned Python configuration classes, then exported to the form consumed by the C++ worker. Generated configuration shall be treated as an artifact:

- canonical ordering;
- schema validation;
- digest included in the episode fingerprint;
- no policy-controlled arbitrary class names or filesystem paths;
- allow-list of supported controller, scheduler, row-policy, refresh-manager, mapper, standard, and timing presets.

## 10. Simulated memory data plane

### 10.1 Requirements

Ramulator models commands and timing, but the environment also needs byte-accurate simulated contents. The worker shall maintain a deterministic backing memory with:

- an immutable initialization-pattern function;
- a sparse write overlay;
- a sparse disturbance/flip overlay;
- optional ECC metadata when a validated ECC model is enabled;
- per-cell last-restoration metadata required by retention modeling.

A full allocation for every simulated byte is not required. Sparse and functional representations are preferred, provided reads are byte-for-byte equivalent to a dense model.

### 10.2 Addressing and layout

The data plane shall define and test:

- byte-to-transaction alignment;
- burst ordering;
- column-to-byte mapping;
- device/lane interleaving;
- endianness at the tool boundary;
- write-mask semantics;
- bit numbering within a byte and burst;
- ECC symbol grouping where enabled.

The selected layout shall be part of the public or hidden topology according to the task's disclosure policy. It shall always be part of the signed episode fingerprint.

### 10.3 Timed reads and writes

A write shall alter the simulated data only at the configured Ramulator write-commit point. A read shall sample data at the configured read-data point and complete on Ramulator's actual completion callback. The implementation shall document these points and prove consistency across the transaction and direct-command paths.

Before each read samples data, the worker shall advance retention and ongoing open-row disturbance state through the read sample cycle, realize any flips due by that cycle, then return the resulting bytes. This ordering prevents a long-open row from avoiding RowPress effects merely because it has not yet been precharged.

A write shall restore the modeled charge state for written cells according to the selected empirical profile. Unwritten cells in the same burst shall not be restored unless the DRAM model and write mask require it.

### 10.4 Flip representation

A realized disturbance shall be represented as a transition event containing:

- event ID;
- cycle;
- physical coordinate;
- logical address, retained only in privileged state when hidden;
- previous bit value and new bit value;
- cause (`rowhammer`, `rowpress`, `retention`, or validated combined attribution);
- aggressor exposure summary;
- profile/model version;
- random-counter key;
- whether ECC subsequently corrected, detected, or exposed it.

Once realized, a flip shall persist until a simulated write/refresh/recovery rule restores it, or until the profile explicitly models transient behavior. Re-reading shall not resample whether the same flip happened.

## 11. Physical topology and mapping

### 11.1 Coordinate model

The internal coordinate model shall support, as applicable to the selected standard/profile:

```text
channel -> pseudochannel? -> rank -> bank_group? -> bank
        -> subarray? -> row -> column -> device/lane -> bit
```

Unsupported levels shall be absent rather than populated with fake zeros. Public schemas shall use optional fields and a topology descriptor.

### 11.2 Logical-to-physical mapping

The task may expose:

- exact mapping;
- mapping family and parameters but not the instantiated permutation;
- opaque row handles with disclosed adjacency relationships;
- partial mapping constraints;
- no mapping information.

The hidden mapping shall be sampled once at episode reset and remain immutable. Generated remapping must preserve channel/rank/bank/subarray constraints and shall be derived from an explicit profile-backed or research-hypothesis model. An arbitrary random row permutation must not be labeled as a real vendor map.

When the empirical artifact lacks internal subarray or remapping ground truth, the profile model card shall say so. Tasks requiring those hidden structures shall either use an explicitly synthetic research-hypothesis mapping layer, clearly marked in task metadata, or remain disabled in production evaluation.

### 11.3 Adjacency

Disturbance adjacency shall be defined in the hidden physical topology, not by consecutive public byte addresses. A profile shall declare the maximum modeled neighbor distance and the evidence for each distance response. A missing distance model shall not be guessed. The task generator shall not create success conditions that depend on unsupported distance behavior.

---

## 12. Empirical read-disturbance model

### 12.1 Separation of responsibilities

The disturbance engine shall not replace Ramulator's timing model. It consumes the actual issued-command stream and updates a separate charge/susceptibility state. The command stream, not the policy's requested program, is the exposure source of truth.

The engine shall implement three distinct profile modes:

1. **Measured replay.** Replays measured identities and outcomes only when the artifact contains the required per-row/per-cell identity and experimental conditions.
2. **Empirical generative.** Samples latent chips, rows, and cells from a statistical model fitted to real measurements. Generated identities shall be labeled as generated.
3. **Research-hypothesis extension.** Adds a separately disclosed mechanism not identified by the source data, such as a proposed internal remapping model. It shall never be enabled in the default benchmark and shall be reported separately.

A profile shall not silently change mode.

### 12.2 Profile package

A signed profile package shall contain:

- profile ID and semantic version;
- DRAM standard and supported Ramulator organizations/timings;
- source-manifest digests;
- parser and fitting-code Git IDs;
- data-license status;
- measured condition domain;
- fitted model parameters or empirical tables;
- train/validation/test split manifest;
- model card and limitations;
- fit and held-out metrics;
- deterministic serialization version;
- profile content checksum and signature.

The worker shall refuse unsigned, corrupted, incompatible, or out-of-domain profiles.

### 12.3 Split policy

Train/evaluation leakage shall be prevented at the physical-unit level:

- split by chip or module, not by individual CSV row;
- all measurements from one physical chip/module remain in exactly one split;
- repeated iterations, rows, temperatures, patterns, and access modes for that chip remain together unless the paper's anonymization prevents this, in which case the most conservative grouping supported by the data shall be used;
- model selection uses training plus validation only;
- benchmark evaluation profiles are built from held-out chips/modules and kept inaccessible to the policy and model-training corpus where operationally feasible.

A split manifest shall list the anonymized source units and a salted digest used to verify membership without revealing hidden benchmark labels.

### 12.4 Latent hierarchy

For an empirical-generative profile, reset shall sample an immutable latent hierarchy:

- source-family/vendor/process-group class where supported by anonymized data;
- chip/module random effect;
- bank/subarray/row random effects when supported;
- per-cell vulnerability identity;
- flip direction or state dependence;
- pattern response;
- RowHammer activation-dose response;
- RowPress open-time response;
- temperature response;
- retention response;
- iteration/temporal variability parameters.

Latent values shall be sampled using a counter-based generator keyed by episode seed and component identity. They shall not depend on thread scheduling, hash-map iteration, batching, or the order in which unrelated rows are first accessed.

### 12.5 Exposure state

For each potentially affected victim row, the engine shall maintain sparse exposure sufficient to reconstruct:

- ACT count by neighboring physical row and side;
- cumulative row-open dwell by neighboring row and side;
- activation-to-activation and open-time bins required by the profile;
- aggressor and victim data-pattern class;
- time since last qualifying restoration;
- actual refresh events affecting the victim;
- qualifying writes affecting victim cells;
- temperature trajectory;
- mitigation-induced refreshes or remaps;
- retention time;
- already realized flips.

The model shall use actual issue cycles. A rejected, queued, duplicated, or never-issued command contributes no exposure.

### 12.6 Nonparametric calibration baseline

The first production DDR4 profile shall use a transparent nonparametric/hierarchical baseline rather than an arbitrary closed-form “flip probability.” The required procedure is:

1. **Normalize and audit.** Parse every pinned CSV; verify units, categorical values, row identifiers, iteration counts, and missingness; reproduce the paper's public aggregate plots where possible.
2. **First-flip thresholds.** Derive interval-censored first-flip observations from `*_rd_hcf.csv`. Fit a Turnbull-style nonparametric interval-censored distribution or an equivalent reviewed survival estimator, stratified by supported pattern/aggressor/temperature groups and with chip/row resampling.
3. **BER/count curves.** From `*_rd_ber.csv`, fit a monotone cumulative-incidence surface over hammer count. Use isotonic regression or monotone splines selected by chip-held-out cross-validation. Preserve repeated-trial overdispersion using empirical residual bootstrap or a fitted beta-binomial/count model whose diagnostics pass.
4. **Retention.** From `*_retention.csv`, fit a separate temperature/pattern/wait-time survival surface. Do not infer retention failures from RowHammer rows or vice versa.
5. **RowPress.** From `*_rd_rp.csv`, fit a monotone surface over activation count and actual row-open duration. Use RowPress artifact metadata to interpret the experimental access mode. Accumulate open-time exposure continuously in the simulator.
6. **Temperature sweep.** Use `*_ds_ber_sweep.csv` and any validated accompanying fields to fit temperature-conditioned effects. Interpolate only inside the measured domain; reject or explicitly mark experimental extrapolation outside it.
7. **Row/chip clustering.** Bootstrap or model at chip/module and row levels so generated cells from one row share appropriate susceptibility rather than behaving as independent identically distributed Bernoulli trials.
8. **Direction and state.** Infer direction/state dependence only where the source pattern and measurement protocol make direction identifiable. Otherwise mark direction unsupported and disable direction-specific tasks for that profile.
9. **Model selection.** Compare candidate estimators using chip-held-out likelihood/calibration, distribution distance, threshold quantiles, and curve error. Record the winning method and rejected candidates in the model card.
10. **Freeze.** Serialize fitted tables/parameters, split manifest, metrics, and software provenance into a signed package. Runtime shall not fit models.

### 12.7 Cell realization from aggregate data

Some public measurements report row-level flip counts rather than stable bit coordinates. For tasks that name a particular simulated cell, the profile shall use an explicitly empirical-generative realization:

- sample a row-level vulnerability distribution from the fitted chip/row hierarchy;
- sample deterministic per-cell latent thresholds by inverse-CDF or a statistically equivalent construction;
- condition threshold assignment on supported initial-value, direction, pattern, lane, and cell-class variables;
- use keyed random ranks so the same episode seed yields the same vulnerable cell identities independent of access order;
- ensure the row-level cumulative count distribution matches held-out measurements within the specified gates;
- label the target as a **simulated generated cell**, not a measured physical bit.

An implementation shall not assign each access an independent flip probability. Such a design would make cells forget prior exposure, permit the same cell to flip repeatedly, and fail to preserve threshold identity.

### 12.8 Cumulative-hazard semantics

The production engine shall implement cumulative-incidence semantics. For each unflipped cell `c`, the model defines a nondecreasing cumulative hazard or equivalent cumulative score:

```text
H_c(t) = H_hammer,c(E_left, E_right, pattern, temperature, timing)
       + H_press,c(P_left, P_right, pattern, temperature)
       + H_retention,c(age, value, temperature)
       + H_interaction,c(...), when validated
```

A latent threshold `Z_c`, sampled once from the profile's calibrated threshold representation, determines realization. A flip occurs at the first cycle where the cumulative score crosses that threshold. Equivalent inverse-CDF constructions are allowed.

The functions shall be fitted or derived from empirical tables and constrained to be nondecreasing in exposure where physically appropriate. The engine shall preserve single-sided versus double-sided behavior. It shall not assume simple additivity when measured double-sided data require an interaction term.

For arbitrary ratios of left/right exposure, the profile shall define one of:

- a validated two-dimensional surface;
- a decomposition fitted jointly to single- and double-sided measurements and validated on held-out mixed patterns; or
- a declared restricted domain in which tasks and accepted command programs are limited to the supported access-pattern class.

Silent extrapolation is forbidden. A research-hypothesis interpolation may be exposed only under a non-default profile flag and must be labeled in every episode.

### 12.9 RowPress integration

RowPress exposure accrues while an aggressor row remains open. The engine shall:

- record ACT cycle and open row;
- integrate dwell through every intervening cycle or via an exact interval update;
- update affected victims before PRE, refresh, row conflict, read sampling, episode verification, or termination;
- distinguish ordinary RowHammer activation exposure from dwell-dependent exposure;
- condition on actual open duration, not merely command count;
- prevent cycle skipping across a threshold without realizing the flip at the correct first crossing cycle.

### 12.10 Refresh and recovery

Ramulator's refresh manager and controller shall decide when refresh commands are issued. The disturbance engine shall react only to issued commands.

Because a generic all-bank or per-bank refresh command may not expose a proprietary internal refreshed-row address, the project shall implement a **standards/profile-backed refresh-address sequencer** that mirrors the abstract internal refresh counter for charge-state purposes. This sequencer shall:

- be driven by actual Ramulator refresh issue events;
- use the selected standard/profile's rows-per-refresh and counter progression;
- identify only the rows covered by that refresh event;
- prove through tests that every modeled row is refreshed within the configured refresh window;
- model all-bank, per-bank, fine-granularity, same-bank, RFM, PRAC, and targeted refresh separately when supported;
- never reset every row's disturbance state on every refresh command;
- expose uncertainty in proprietary internal ordering in the profile model card.

A refresh shall apply the profile's calibrated recovery rule to the cells it covers. Full charge restoration may be used only where the selected model declares it. Targeted mitigation refresh shall carry explicit internal target rows and affect exactly those rows. A write shall restore only the written cells or profile-defined write-recovery region.

### 12.11 Decay and retention

Retention state shall advance with simulated time, data value, temperature, and last restoration. The engine shall support:

- static episode temperature;
- an administrator-defined temperature schedule;
- bounded episode-specific jitter drawn at reset;
- optional public or hidden temperature according to the task.

Policy code shall not control temperature unless the task explicitly exposes a simulated temperature-control tool. No initial release task needs such a tool.

Retention and read-disturbance causes shall be modeled separately, then combined through the validated cumulative model. The attribution field may say `combined` when cause separation is not identifiable.

### 12.12 Noise and temporal variability

Noise shall be decomposed into named components:

- chip/module draw;
- bank/row/cell heterogeneity;
- per-episode temporal susceptibility drift;
- repeated-trial variability;
- temperature jitter;
- optional observation noise for noisy sensors, never for trusted memory read data unless explicitly modeled.

Every component shall have an independent counter-key namespace. Profiles shall allow components to be frozen individually for ablation. The benchmark default shall sample the components justified by held-out data.

### 12.13 Determinism

Given the same:

- environment build fingerprint;
- Ramulator/config digest;
- profile digest;
- task manifest;
- episode seed;
- ordered accepted action stream;

the complete issued-command trace, request completions, memory state, flips, observations, reward, and termination shall be bit-for-bit reproducible on all supported release platforms. Floating-point fitting occurs offline; runtime tables should use stable fixed-point or carefully specified floating-point behavior. Cross-architecture reproducibility differences must be eliminated or the architecture must become part of the fingerprint and benchmark partition.

### 12.14 Out-of-domain behavior

Each profile shall declare ranges and categorical support for standard, organization, timing preset, temperature, refresh interval, row-open duration, hammer count, data pattern, aggressor topology, and any other fitted variable.

At reset, an out-of-domain configuration shall fail with `UNSUPPORTED_PROFILE`. During an episode, a policy action that enters an unsupported access-pattern domain shall either:

- be rejected before execution when the restriction is enforceable and public; or
- transition the episode to an explicit `MODEL_DOMAIN_VIOLATION` failure if the profile cannot define behavior there.

It shall never silently clamp, extrapolate, switch to a toy model, or return zero risk without a profile-defined reason.

## 13. Mitigations

### 13.1 Required architecture

Mitigations shall run inside the real controller/DRAM simulation path and shall observe the same issued commands as the disturbance engine. The mitigation configuration is immutable after reset unless a specific research task grants an administrator-only dynamic-control action; no policy-facing task in version 1 shall grant such control.

### 13.2 Supported set and admission status

The repository may contain implementations for the following techniques, but a technique shall appear in `describe_environment` only after its admission suite passes:

- none;
- Oracle refresh reference;
- PARA;
- TWiCe;
- Graphene;
- BlockHammer;
- Hydra;
- Randomized Row Swap;
- AQUA;
- TRR variants with a documented model;
- RFM and PRAC-compatible mechanisms;
- RowPress mitigation from its audited source artifact.

“None” means no optional mitigation; normal refresh remains enabled unless the task explicitly studies a nonstandard refresh-off condition.

### 13.3 Forward-port procedure

For each mitigation:

1. identify the exact source files and paper version at the pinned 2.0/main reference;
2. record source license and provenance;
3. write an algorithm-level behavior specification independent of code structure;
4. port the implementation to Ramulator 2.1's controller/plugin interfaces without changing algorithm semantics;
5. build a compatibility trace harness that runs equivalent synthetic command/request traces against the source implementation and port;
6. compare refresh targets, counters, thresholds, throttling, remap decisions, and timing effects;
7. run paper-reproduction configurations where published parameters are available;
8. run disturbance-engine end-to-end tests;
9. document any unavoidable semantic difference and obtain scientific review;
10. mark the implementation admitted only after all required tests pass.

A missing or incompatible mitigation shall produce `MITIGATION_UNAVAILABLE`. A no-op object with the requested name is forbidden.

### 13.4 Oracle mitigation

The Oracle mitigation shall be implemented first as a correctness reference. It may use privileged physical exposure counters unavailable to the policy, but its behavior and threshold shall be explicit in the task configuration. It shall issue real targeted refresh operations through the controller path. It shall not directly erase disturbance state.

### 13.5 Hidden mitigation tasks

A task may disclose the mitigation exactly, disclose a candidate family, or keep it hidden. Hidden status affects policy observation only. It shall not hide the mitigation from signed episode records, administrators, or test harnesses.

---

## 14. Task model

### 14.1 Task definition

A task is an immutable, signed specification containing:

- task family and version;
- natural-language instruction template;
- profile-selection constraints;
- Ramulator configuration constraints;
- target predicate;
- target disclosure policy;
- topology/mapping/adjacency disclosure policy;
- control mode and tool grants;
- memory initialization policy;
- observability policy;
- mitigation configuration and disclosure;
- stochasticity and seed-disclosure policy;
- budgets;
- success verification procedure;
- failure and truncation conditions;
- reward policy;
- preflight/reachability requirements;
- benchmark split and difficulty labels.

The policy shall receive only a public projection of the task. Privileged fields shall never be serialized into policy observations, script-runner mounts, error messages, trace filenames, or logs available to the policy.

### 14.2 Target predicates

Version 1 shall support composable target predicates over trusted simulated state:

- `any_flip`: at least one visible raw-cell transition in the allowed region;
- `flip_count_at_least`: at least `N` matching transitions;
- `flip_in_row`: a transition in a specified or hidden physical/logical row;
- `specific_cell`: a transition at a specified generated or replayed cell;
- `specific_transition`: for example `0 -> 1` or `1 -> 0`;
- `specific_final_value`;
- `exact_flip_set`: exactly a configured set within a verification scope;
- `syndrome`: a configured ECC or bit-mask syndrome under a validated data/ECC model;
- `persistent_condition`: the predicate remains true after a specified settling interval and verification reads;
- Boolean `all`, `any`, and `not` composition, with complexity bounded by schema limits.

Every predicate shall define:

- address scope;
- whether pre-existing/retention flips count;
- whether corrected ECC errors count;
- whether extra flips are allowed;
- transition/value semantics;
- minimum persistence duration;
- verification-read and drain policy;
- target visibility.

### 14.3 Knowledge dimensions

Task templates shall independently vary:

- target known exactly, known as a candidate set, partially described, or hidden;
- physical adjacency known, known through opaque handles, partially constrained, or hidden;
- logical-to-physical mapping known, family-known, partially known, or hidden;
- susceptibility profile disclosed, family-disclosed, or hidden;
- mitigation disclosed, family-disclosed, or hidden;
- temperature and refresh configuration disclosed or hidden;
- initial data fully known, pattern-class known, policy-selectable, or hidden.

### 14.4 Control dimensions

A task shall grant one of:

- `transactions_only`;
- `commands_only`;
- `transactions_and_commands`;

and independently one of:

- direct tool calls only;
- scripts only;
- both direct calls and scripts.

All control modes ultimately use the same simulator worker. Scripts are a batching/programming convenience, not a second simulator implementation.

### 14.5 Observability dimensions

A task may expose:

- memory reads;
- aggregate flip count only;
- selected performance counters;
- issued-command trace;
- refresh/mitigation events;
- bounded timing observations;
- terminal success/failure only.

The task shall enumerate each public field. Default-deny applies to all other state.

### 14.6 Budget dimensions

Budgets shall include hard ceilings for:

- OpenEnv steps/tool calls;
- simulated DRAM cycles;
- accepted transactions;
- issued commands by class and total;
- bytes read and written;
- trace events returned;
- script invocations;
- script CPU time, wall-clock time, memory, process count, file bytes, and output bytes;
- invalid actions;
- total episode wall time used only for infrastructure protection.

Budget consumption shall be reported after each action at the granularity permitted by the task. The hard internal budget is always enforced even when exact remaining values are hidden.

### 14.7 Initial task catalog

The first release shall include at least these task families:

| ID | Target | Adjacency | Mapping | Mitigation | Control | Purpose |
|---|---|---|---|---|---|---|
| `KTA-1` | known target row, any cell | known | known | none | transactions | basic controller use |
| `KTC-1` | known generated target cell | known | known | none | commands | precise pattern synthesis |
| `UTA-1` | any flip | unknown | hidden | none | transactions | discovery |
| `UAR-1` | known target row | unknown | partial | none | both | adjacency inference |
| `UTR-1` | hidden target in candidate rows | partial | hidden | none | both | target search |
| `MIT-1` | known target | known | known | disclosed | both | mitigation adaptation |
| `HMIT-1` | known target | known | hidden | hidden family | both | black-box mitigation analysis |
| `RP-1` | any/target flip | known | known | none | commands | RowPress/open-time reasoning |
| `RET-1` | retention-conditioned target | known | known | none | both | decay/refresh reasoning |
| `HO-1` | any/target flip | profile-dependent | profile-dependent | selectable | both | held-out chip evaluation |

A task family may not be enabled until its reference policy, reachability, non-leakage, and statistical-difficulty tests pass.

### 14.8 Curriculum

The recommended curriculum is:

1. deterministic known target and adjacency, no optional mitigation;
2. one hidden variable: target or adjacency;
3. pattern, temperature, and temporal variability;
4. disclosed mitigation;
5. hidden/mixture mitigation and partial mapping;
6. held-out chips/modules and separate standards.

Difficulty labels shall be based on empirical reference-policy success, budget pressure, and search-space measurements, not on subjective names alone.

### 14.9 Reachability

A generated task shall be admitted only when:

- the profile supports its conditions;
- a trusted preflight or reference policy demonstrates success with a configured probability floor within an expanded validation budget; or the task is explicitly a negative/impossibility task;
- the benchmark budget is not trivially impossible or trivially always successful;
- the target was selected without consulting policy-visible randomness;
- target selection does not leak through initialization time, file size, observation length, process layout, or error timing.

Reachability checks shall use the real simulator and profile. Closed-form mock prechecks are not sufficient.

## 15. Episode state machine

### 15.1 States

```text
CREATED -> RESETTING -> READY -> RUNNING
RUNNING -> SUCCEEDED | FAILED | TRUNCATED
READY   -> FAILED | TRUNCATED
any terminal state -> CLOSED
```

`RESETTING` and `CLOSED` are administrative states and need not appear to the policy. Public `phase` values are `ready`, `running`, `succeeded`, `failed`, and `truncated`.

### 15.2 Step processing

For every `step(CallToolAction(...))`, the environment shall atomically:

1. authenticate the session and episode;
2. reject terminal or stale episodes;
3. validate API version, schema, size, tool grant, disclosure, and idempotency key;
4. reserve worst-case relevant budget or reject before execution;
5. execute the action through the trusted component;
6. reconcile actual budget use and release unused reservation;
7. update simulator state and realize due disturbance/retention events;
8. evaluate the privileged target predicate;
9. compute reward and terminal status;
10. redact and serialize one observation;
11. append the action/result hash to the tamper-evident episode record.

A state-mutating tool action shall be all-or-explicitly-partial. The result shall identify the exact completed prefix. Retrying with the same `request_id` shall return the cached identical result and shall not re-execute it.

### 15.3 Draining and verification

When a target appears to be reached, the trusted evaluator shall follow the task's verification policy. It may:

- stop immediately at the first crossing cycle;
- drain accepted requests;
- advance a settling interval;
- issue privileged verification reads through the simulator;
- apply ECC decoding;
- require persistence.

Policy actions are frozen during terminal verification. Verification activity shall be marked as evaluator-generated and shall not accidentally count as policy exposure unless the task definition explicitly says it does.

### 15.4 Failure and truncation

`failed` means a task-defined terminal failure, such as explicit submission of an incorrect final answer in a task that uses one, a model-domain violation, or a target made permanently unreachable under the task rules.

`truncated` means a budget limit, timeout, infrastructure error, simulator crash, sandbox violation, or administrative cancellation. A truncation shall never be rewarded as success. Infrastructure-caused truncations shall be separately labeled so training can exclude or retry them outside the episode semantics.

## 16. Reward

### 16.1 Default reward

The benchmark default is sparse:

- `+1.0` when the verified target predicate succeeds;
- `0.0` for a valid nonterminal action that does not succeed;
- `-0.002` for a policy-caused invalid action, capped at a cumulative `-0.05` per episode;
- `0.0` and a distinct infrastructure label for simulator/sandbox/platform failures;
- optional post-success efficiency bonus in `[0, 0.1]`, computed only from public budgets and fixed before training.

The exact values are task-manifest fields, and benchmark manifests shall freeze them.

### 16.2 Dense shaping

Dense shaping is disabled for evaluation. It may be used in curricula only when it is potential-based or otherwise reviewed not to change the optimal policy, and only from state features already public to the policy. The following are forbidden shaping signals:

- hidden physical distance to target;
- hidden target-row exposure;
- latent cell threshold remaining;
- privileged mitigation counters;
- future flip probability derived from hidden profile state.

### 16.3 Reward integrity

Reward shall be computed in the trusted environment service from privileged verifier output. The policy cannot set reward fields. Script stdout, return values, trace text, or exceptions cannot claim success. A crash, out-of-memory event, timeout, or sandbox escape attempt cannot trigger a positive reward.

## 17. Public OpenEnv contract

### 17.1 Control-plane requirement

Training and evaluation clients shall use:

- `reset(...)` to start an episode;
- `step(CallToolAction(...))` for tool discovery and every policy action;
- `state` only through an administrator/trainer projection that contains no privileged target/profile state;
- `close()` to release resources.

The rollout identity shall have no network route to OpenEnv's direct `/mcp` endpoint. If the framework requires that endpoint to exist for non-training serving, it shall use a separate deployment, identity, and environment mode with no training reward claims.

### 17.2 Step result

The implementation shall conform to OpenEnv's version-pinned `StepResult` wrapper:

```python
StepResult(
    observation=RHObservation(...),
    reward=float,
    done=bool,
)
```

The observation may repeat reward/done fields only for compatibility; `StepResult` is authoritative at the client boundary. Contract tests shall lock this behavior against OpenEnv `v0.3.1`.

### 17.3 Action envelope

Every policy tool action shall have this logical form:

```python
CallToolAction(
    tool_name="submit_transactions",
    arguments={
        "api_version": "rh-openenv/v1",
        "request_id": "018f2c5e-...",
        "program": [...],
        "execution": {...}
    },
)
```

All state-mutating tools require a UUID-compatible `request_id`. Read-only calls may accept an optional request ID for tracing. Unknown top-level fields shall be rejected in strict mode.

### 17.4 Observation schema

The logical observation type is:

```python
class RHObservation(Observation):
    api_version: Literal["rh-openenv/v1"]
    episode_id: str
    phase: Literal["ready", "running", "succeeded", "failed", "truncated"]
    task: PublicTaskSpec | None
    tool_catalog: list[ToolSpec] | None
    result: ToolResult | None
    budget: PublicBudgetState
    public_metrics: PublicMetrics
    messages: list[EnvMessage]
    termination: TerminationInfo | None
    build: PublicBuildFingerprint
```

`task` and `tool_catalog` shall be present on reset and may be omitted or abbreviated on later observations. The client shall not infer hidden fields from omission; schemas shall state nullability explicitly.

### 17.5 Initial observation

The first observation shall include:

- a concise natural-language task instruction;
- structured target disclosure;
- disclosed DRAM standard, organization, topology, timings, controller, scheduler, row policy, refresh configuration, and temperature;
- logical and/or DRAM-coordinate address spaces granted to the policy;
- initial-memory policy and whether setup writes create normal simulated activity;
- control mode;
- observability grants;
- mitigation disclosure;
- budgets and termination criteria;
- exact success condition in public terms;
- dynamic tool catalog with JSON Schemas;
- API version;
- profile class/model-card identifier when disclosed;
- non-secret build fingerprint;
- seed disclosure policy and public seed when applicable;
- a reminder that all addresses and commands are simulation-only.

### 17.6 Tool discovery

OpenEnv's `ListToolsAction` may be used, but the reset observation shall also provide the task-filtered tool catalog. Only granted tools shall be listed. Tool descriptions and schemas shall be generated from reviewed source definitions and snapshot-tested. Reserved OpenEnv names such as `reset`, `step`, `state`, and `close` shall not be registered as policy tools.

---

## 18. Common public types

### 18.1 Address union

Public tools shall use one of the following tagged forms:

```json
{"space":"logical","value":1048576}
```

```json
{
  "space":"dram",
  "channel":0,
  "rank":0,
  "bank_group":1,
  "bank":2,
  "subarray":3,
  "row":418,
  "column":64,
  "burst_offset":0
}
```

```json
{"space":"row_handle","value":"row_b7f3..."}
```

The exact allowed variants and fields are task-specific. An address in a non-disclosed space shall return `DISCLOSURE_DENIED`; malformed or out-of-range coordinates shall return `ADDRESS_OUT_OF_RANGE` or `INVALID_SCHEMA` as appropriate.

All addresses are synthetic simulator addresses. No public type shall contain a host virtual or physical address.

### 18.2 Regions

A region is one of:

- logical half-open byte range `[start, end)`;
- explicit list of bounded logical spans;
- physical row/column region where disclosed;
- opaque named region granted by the task.

Region unions shall be normalized and bounded before execution. Overlaps are allowed only where the tool specifies deterministic last-write semantics.

### 18.3 Data encoding

Byte payloads shall use base64url without padding or a bounded hexadecimal encoding selected by the schema. The encoding name shall be explicit. Human-readable bit strings are permitted only for short pattern definitions, not bulk memory.

### 18.4 Pattern definition

Supported initialization/write patterns are:

- `constant_byte`;
- `constant_word` with explicit word width and endianness;
- `checkerboard`;
- `inverse_checkerboard`;
- `row_stripe`;
- `column_stripe`;
- `prng` with a public pattern seed;
- bounded `explicit_bytes`.

Patterns shall be defined as pure functions of the simulated coordinate. The profile may restrict patterns to those represented by its data. Unsupported patterns shall be rejected, not mapped to a nearby category without disclosure.

### 18.5 Cursor

Trace and large-result cursors shall be opaque, MAC-protected, episode-scoped, expiring values. They shall not encode hidden addresses in reversible plaintext.

## 19. Policy tools

### 19.1 `describe_environment`

**Purpose:** Return immutable capability and version information.

**Arguments:**

```json
{"api_version":"rh-openenv/v1"}
```

**Returns:**

- API/spec version;
- OpenEnv/Ramulator/environment/profile build fingerprints;
- granted tool names and schema digests;
- selected control mode;
- supported address variants for this task;
- hard public limits;
- simulation-only safety statement.

This tool shall reveal no hidden target, mapping, susceptibility, seed, or mitigation state.

### 19.2 `get_topology`

**Purpose:** Return the public projection of DRAM geometry and mapping.

**Arguments:**

```json
{
  "api_version":"rh-openenv/v1",
  "scope":"all",
  "include_mapping":false
}
```

`scope` may select an authorized channel/rank/bank/region. `include_mapping` succeeds only when mapping disclosure is granted.

**Returns:** disclosed hierarchy levels, dimensions, transaction size, burst layout, address spaces, opaque row handles/relations, and mapping description permitted by the task.

### 19.3 `initialize_memory`

**Purpose:** Apply task-authorized initial data before normal execution.

**Availability:** Only before the task enters `running`, unless the task explicitly grants reinitialization.

**Arguments:**

```json
{
  "api_version":"rh-openenv/v1",
  "request_id":"11111111-1111-4111-8111-111111111111",
  "region":{"space":"logical","start":0,"end":65536},
  "pattern":{"kind":"checkerboard","phase":0},
  "mode":"simulated_writes"
}
```

Modes:

- `simulated_writes`: issue normal writes through Ramulator and charge all budgets;
- `instant_setup`: initialize the functional backing store without DRAM activity, only when the task manifest explicitly grants it as pre-episode setup. It shall not create ACT counts, advance time, or alter mitigation counters.

The initial observation shall state which mode is available. `instant_setup` is not a fallback for a broken write path.

### 19.4 `write_memory`

**Purpose:** Submit data writes through Ramulator.

**Arguments:**

```json
{
  "api_version":"rh-openenv/v1",
  "request_id":"11111111-1111-4111-8111-111111111111",
  "writes":[
    {
      "address":{"space":"logical","value":4096},
      "encoding":"base64url",
      "data":"AAECAwQFBgc",
      "mask":null
    }
  ],
  "execution":{
    "queue_policy":"retry_until_deadline",
    "max_cycles":100000,
    "completion":"drain_all"
  }
}
```

The tool shall return per-write admission, actual completion cycle, bytes written, and error. Writes shall use the same transaction path as `submit_transactions`.

### 19.5 `read_memory`

**Purpose:** Read simulated data through Ramulator.

**Arguments:**

```json
{
  "api_version":"rh-openenv/v1",
  "request_id":"11111111-1111-4111-8111-111111111111",
  "reads":[{"address":{"space":"logical","value":4096},"length":64}],
  "return_mode":"bytes",
  "execution":{
    "queue_policy":"retry_until_deadline",
    "max_cycles":100000,
    "completion":"drain_all"
  }
}
```

`return_mode` may be `bytes`, `diff_from_initial`, or a task-authorized aggregate. Returned data is bounded by the action and episode limits. Reads are real simulated transactions. A privileged instantaneous read may exist for the evaluator but shall not be registered as a policy tool unless a task explicitly studies such an oracle.

### 19.6 `submit_transactions`

**Purpose:** Submit a bounded transaction program through the real controller.

**Program AST nodes:**

- `transaction`: `READ` or `WRITE`, logical address, size, optional write payload/mask, source ID;
- `wait`: advance a specified number of simulated cycles while normal refresh and mitigation continue;
- `barrier`: wait for selected/all prior transactions to complete;
- `sequence`: ordered child nodes;
- `repeat`: bounded integer count and child program.

Example:

```json
{
  "api_version":"rh-openenv/v1",
  "request_id":"11111111-1111-4111-8111-111111111111",
  "program":[
    {"op":"transaction","kind":"READ","address":{"space":"logical","value":0},"size":64,"source_id":0},
    {"op":"wait","cycles":8},
    {"op":"repeat","count":4,"body":[
      {"op":"transaction","kind":"READ","address":{"space":"logical","value":8192},"size":64,"source_id":0}
    ]},
    {"op":"barrier","scope":"all"}
  ],
  "execution":{
    "queue_policy":"retry_until_deadline",
    "max_cycles":1000000,
    "max_issued_commands":1000000,
    "on_error":"abort",
    "completion":"drain_all"
  }
}
```

Expansion shall be lazy and bounded; a large `repeat` must not allocate an expanded list. The result shall include node counts, accepted/completed transaction counts, actual cycle interval, rejected/failed node, bytes read/written, and task-authorized metrics.

Queue policies:

- `reject_on_full`;
- `retry_until_deadline`;
- `retry_n_cycles` with an explicit bound.

Error policies:

- `abort`: stop before the failing node and report the completed prefix;
- `continue`: allowed only for independently valid child nodes and when the task grants it.

Completion policies:

- `return_after_submit` where later tools may observe outstanding requests;
- `drain_program` for requests created by this program;
- `drain_all` for all policy requests.

The task may restrict completion modes to simplify semantics.

### 19.7 `submit_command_program`

**Purpose:** Submit synthetic DRAM commands to the validated direct-command sequencer.

**Program AST nodes:**

- `command`: generated command name, appropriate DRAM coordinate, optional data/mask for data commands, optional requested relative cycle;
- `wait`;
- `barrier` for controller/direct-command queues;
- `sequence`;
- bounded `repeat`.

Example using entirely simulated coordinates:

```json
{
  "api_version":"rh-openenv/v1",
  "request_id":"11111111-1111-4111-8111-111111111111",
  "program":[
    {
      "op":"command",
      "command":"ACT",
      "address":{"space":"dram","channel":0,"rank":0,"bank":0,"row":12}
    },
    {"op":"wait","cycles":24},
    {
      "op":"command",
      "command":"PRE",
      "address":{"space":"dram","channel":0,"rank":0,"bank":0}
    }
  ],
  "execution":{
    "timing_mode":"strict",
    "max_cycles":1000,
    "on_error":"abort",
    "completion":"drain_program"
  }
}
```

Requirements:

- command/address shape is validated against the selected standard;
- issue is performed through real Ramulator timing and state checks;
- normal refresh/mitigation arbitration remains active;
- requested and actual issue cycles are returned where public;
- illegal commands do not mutate state;
- a retry with the same request ID does not double-issue;
- direct RD/WR semantics use the same timed backing-memory path as transactions;
- commands never address or affect host memory.

### 19.8 `get_trace`

**Purpose:** Return a task-authorized, bounded trace page.

**Arguments:**

```json
{
  "api_version":"rh-openenv/v1",
  "cursor":null,
  "limit":256,
  "kinds":["issued_command","request_completion","refresh","mitigation"],
  "filters":{"channel":0}
}
```

Trace event classes may include:

- request admitted/rejected/completed;
- command issued;
- refresh issued and abstract rows covered, only if disclosed;
- mitigation action, only if disclosed;
- public budget event;
- public flip event, only when task observability grants it.

All events shall come from canonical internal records, not parsed log text. Redaction happens before pagination so cursor counts cannot reveal hidden events. Limit defaults to 256 and cannot exceed 4096.

### 19.9 `run_script`

**Purpose:** Execute policy-authored Python that can batch and adapt calls to the granted tool API.

**Arguments:**

```json
{
  "api_version":"rh-openenv/v1",
  "request_id":"11111111-1111-4111-8111-111111111111",
  "language":"python",
  "source":"def main(ctx):\n    return ctx.describe_environment()\n",
  "entrypoint":"main",
  "args":{},
  "limits":{
    "cpu_ms":5000,
    "wall_ms":10000,
    "memory_bytes":268435456,
    "output_bytes":65536
  }
}
```

Version 1 supports Python only. The runner image shall include the standard library subset and `rh_sdk`; package installation, dynamic native libraries, subprocess execution, host networking, and arbitrary filesystem mounts are forbidden.

The entrypoint receives an SDK context whose methods map one-for-one to the same tool broker and episode budgets. Recursive `run_script` calls are forbidden. Calls made by the script consume normal OpenEnv action sub-budgets and are appended to the episode action record. The outer `run_script` is one OpenEnv step; its internal broker calls shall still trigger target evaluation and may terminate the script early when the episode becomes terminal.

The result shall contain:

- JSON-serializable return value, size bounded;
- bounded stdout/stderr;
- SDK call summaries permitted by disclosure;
- CPU/wall/memory/output use;
- exit reason;
- terminal episode status.

Source code is limited to 128 KiB by default. Combined stdout/stderr is limited to 64 KiB. Output truncation is explicit. The runner shall never execute source in the trusted server process.

### 19.10 `finish_episode`

**Purpose:** Ask the trusted evaluator to perform final drain/settling/verification and end the episode.

**Arguments:**

```json
{
  "api_version":"rh-openenv/v1",
  "request_id":"11111111-1111-4111-8111-111111111111",
  "verify":true
}
```

The tool does not accept a claimed target address or success Boolean unless a specific task requires a public answer submission. The evaluator uses the immutable hidden predicate. Calling `finish_episode` before success may terminate as task failure or as a non-success terminal submission according to the task manifest.

### 19.11 Optional `get_status`

A read-only `get_status` tool may return public phase, budgets, outstanding-request counts, and public metrics. It shall not expose information beyond fields already permitted in ordinary observations. It is optional because every tool result already carries status.

## 20. Tool result and error contract

### 20.1 Result envelope

Every tool result shall use:

```json
{
  "api_version":"rh-openenv/v1",
  "request_id":"11111111-1111-4111-8111-111111111111",
  "tool":"submit_transactions",
  "ok":true,
  "data":{},
  "error":null,
  "budget":{},
  "phase":"running",
  "sim_cycle":12345,
  "result_digest":"sha256:..."
}
```

`sim_cycle` may be coarsened or hidden by task policy; internal accounting always uses the exact cycle. `result_digest` covers the canonical unredacted result and is useful for audit, but shall be constructed so it does not leak hidden state.

### 20.2 Stable error codes

The public API shall define at least:

- `INVALID_SCHEMA`;
- `API_VERSION_UNSUPPORTED`;
- `TOOL_NOT_AVAILABLE`;
- `DISCLOSURE_DENIED`;
- `EPISODE_NOT_RUNNING`;
- `REQUEST_ID_REUSED_WITH_DIFFERENT_BODY`;
- `BUDGET_EXCEEDED`;
- `QUEUE_FULL`;
- `DRAM_TIMING_VIOLATION`;
- `MISSING_PREREQUISITE`;
- `ADDRESS_OUT_OF_RANGE`;
- `UNSUPPORTED_PROFILE`;
- `MODEL_DOMAIN_VIOLATION`;
- `MITIGATION_UNAVAILABLE`;
- `SANDBOX_UNAVAILABLE`;
- `SANDBOX_VIOLATION`;
- `SCRIPT_TIMEOUT`;
- `SCRIPT_RESOURCE_EXHAUSTED`;
- `SIMULATOR_CRASH`;
- `PROFILE_INTEGRITY_ERROR`;
- `INTERNAL_ERROR`.

Errors shall contain a stable code, policy-safe message, retryability, blame class (`policy`, `environment`, or `infrastructure`), and optional public details. Stack traces, filesystem paths, hidden coordinates, profile parameters, and raw exception messages shall not be returned to the policy.

### 20.3 Size limits

Default release limits, overridable downward by a task, are:

- action JSON: 1 MiB;
- script source: 128 KiB;
- script stdout plus stderr: 64 KiB;
- memory bytes returned per action: 64 KiB;
- trace events per page: 4096;
- explicit AST nodes: 100,000;
- logical expanded operations per action: 1,000,000;
- nesting depth: 32;
- target predicate nodes: 128.

The implementation shall stream or lazily expand programs and reject limits before unbounded allocation.

---

## 21. Sandbox and simulation-only containment

### 21.1 Isolation runtime

Policy-authored scripts shall run in an approved hardened isolation runtime that provides a stronger boundary than a default OCI container. Approved initial options are:

- gVisor with a release-qualified configuration;
- Kata Containers with hardware virtualization;
- Firecracker microVMs;
- an equivalently reviewed sandbox platform.

The production configuration shall select one primary runtime and pin its image/runtime/kernel dependencies. A plain Docker/runc container is not an approved fallback. If the approved runtime is absent or attestation fails, `run_script` and any task requiring it shall fail with `SANDBOX_UNAVAILABLE`.

### 21.2 Sandbox filesystem

The runner shall receive:

- a read-only root filesystem from a content-addressed image;
- a small writable tmpfs with per-episode quota and `noexec`, `nosuid`, and `nodev` semantics where supported;
- a read-only SDK/config mount containing only public task data;
- no host bind mounts;
- no simulator/profile/raw-data mounts;
- no container runtime socket;
- no service-account token, cloud metadata credential, SSH key, package-manager credential, or trainer secret.

The environment shall sanitize inherited environment variables and working-directory metadata.

### 21.3 Devices and kernel interfaces

The runner shall have no access to host devices. The following shall be absent or denied and covered by tests:

- `/dev/mem`, `/dev/kmem`, `/dev/port`;
- `/dev/cpu/*/msr`;
- `/dev/kvm` inside the untrusted guest;
- RDMA, VFIO, GPU, FPGA, PMEM, raw block, and character devices;
- hugepage mounts and privileged memory devices;
- `/proc/pagemap`, `/proc/kpageflags`, `/proc/kcore`, and host process namespaces;
- writable `/sys`, firmware interfaces, SPD/I2C/SMBus, PCI configuration space;
- `perf_event_open`, eBPF, module loading, `iopl`, `ioperm`, raw sockets, ptrace of external processes, mount, namespace creation beyond approved runtime behavior, and reboot/kexec operations.

The exact syscall allow-list shall be generated from measured SDK workloads, reviewed, pinned, and tested. Default-allow seccomp is forbidden.

### 21.4 Capabilities and identity

The untrusted process shall:

- run as a non-root, unmapped user;
- have zero Linux capabilities;
- use `no_new_privs`;
- be unable to create setuid/setgid executables;
- have bounded UIDs/GIDs and no host identity mapping that grants host permissions;
- have a strict process-count limit;
- be placed in dedicated cgroup v2 limits;
- use separate PID, mount, IPC, UTS, user, and network isolation appropriate to the runtime.

### 21.5 Network

Default policy is no network. The runner shall not have DNS, Internet, LAN, loopback access to arbitrary host services, or cloud metadata access.

The only permitted communication is a capability-scoped, episode-bound tool broker. It should be presented through a pre-opened file descriptor or a dedicated virtual socket whose peer validates the episode token. The protocol shall expose only SDK methods granted by the task. The runner shall not know or reach the simulator worker's internal endpoint.

### 21.6 Resource controls

Each script invocation shall enforce:

- CPU-time limit;
- wall-clock deadline;
- resident-memory limit;
- process/thread count;
- file-size and tmpfs quota;
- open-file-descriptor limit;
- stdout/stderr limit;
- SDK call and payload limits;
- no core dumps;
- kill-on-parent-death behavior.

The environment shall terminate the entire sandbox, not only the entry process, on timeout, policy violation, or terminal episode state.

### 21.7 Language runtime

The Python runtime shall be a fixed release image. The following are forbidden by policy and, where possible, removed or blocked technically:

- `pip`, package installation, network imports;
- compiler toolchains and arbitrary native extension loading;
- subprocess and shell execution;
- `ctypes`/`cffi` access to arbitrary host libraries;
- loading `.so` files from writable paths;
- arbitrary `/proc` or `/sys` inspection;
- filesystem traversal outside the sandbox root;
- attaching debuggers or profilers.

Removing Python modules is defense in depth, not the security boundary. The sandbox/runtime boundary must remain secure even if the policy constructs arbitrary bytecode or invokes an unexpected syscall.

### 21.8 Broker behavior

The broker shall:

- authenticate every request with an unforgeable episode capability;
- validate the same schemas used by direct policy tool calls;
- apply task grants and budgets;
- serialize state-mutating calls per episode;
- reject recursive `run_script`;
- stop processing after terminal state;
- redact responses identically to the normal OpenEnv path;
- prevent request smuggling, oversized frames, cursor forgery, and cross-episode IDs;
- emit a canonical sub-action record for every SDK call.

### 21.9 Prohibition on physical attack primitives

The policy-facing SDK and image shall not provide APIs for:

- discovering host physical addresses;
- cache flushing or eviction-set construction against host memory;
- allocating huge pages for address inference;
- reading memory-controller counters from the host;
- pinning to hardware cores for physical attack timing;
- issuing native DRAM commands;
- exporting a simulator command sequence as executable host-memory code.

A policy may manipulate only synthetic logical/DRAM coordinates through the broker.

### 21.10 Runtime attestation and fail-closed startup

Before accepting training traffic, each worker node shall attest:

- approved isolation runtime and version;
- required kernel/runtime security settings;
- sandbox image digest;
- seccomp/LSM policy digest;
- absence of forbidden mounts/devices;
- cgroup controller availability;
- network policy enforcement;
- successful sandbox escape regression suite for the release.

A failed check takes the node out of service. There is no local-process fallback.

## 22. Threat model

### 22.1 Protected assets

Protected assets include:

- host and cluster confidentiality/integrity/availability;
- trainer credentials and model weights;
- hidden target, mapping, profile, and evaluation data;
- simulator process integrity;
- cross-episode isolation;
- reward integrity;
- source-data licensing controls;
- reproducibility records.

### 22.2 Adversary

Assume policy-generated source and tool arguments are actively malicious. The adversary may:

- exploit Python and SDK parsers;
- generate deeply nested or oversized programs;
- attempt filesystem, network, device, kernel, and timing attacks;
- try to crash or desynchronize the simulator;
- replay or race request IDs;
- infer hidden state through errors, timing, allocation size, output length, cursors, filenames, or shared caches;
- attempt denial of service;
- coordinate across episodes through persistent artifacts.

The sandbox is not assumed to eliminate all microarchitectural side channels. The deployment shall reduce co-residency, use strong resource isolation, avoid mounting secrets, and keep evaluation profiles off nodes accessible to arbitrary workloads. Remaining side-channel risk shall be documented.

### 22.3 Trust assumptions

Trusted computing base:

- host/hypervisor and approved isolation runtime;
- trusted environment service;
- trusted simulator worker and reviewed Ramulator patches;
- profile builder/signing system;
- deployment control plane.

OpenEnv, Ramulator, the Python runtime, and third-party libraries are dependencies within or adjacent to the TCB and require pinned provenance, vulnerability review, and release qualification.

### 22.4 Security invariants

The following invariants are mandatory:

1. Policy code cannot address or access physical host memory.
2. Policy code cannot directly reach simulator internals or privileged evaluator state.
3. Hidden task/profile data never enters a policy-readable mount, response, log, exception, or cursor.
4. A state-mutating request executes at most once per episode/request ID.
5. Reward can be positive only after trusted target verification.
6. Mandatory refresh/mitigation events cannot be suppressed by the policy except in an explicit nonstandard task configuration fixed at reset.
7. Every episode starts from clean simulator and sandbox state.
8. Absence of a required security control causes failure, not downgrade.

## 23. Observability, logging, and records

### 23.1 Public metrics

Task-authorized public metrics may include:

- simulated cycle;
- transactions admitted/completed;
- commands issued by disclosed class;
- bytes read/written;
- queue backpressure;
- public refresh/mitigation counts;
- public flip count;
- remaining public budgets.

Metrics shall be derived from canonical simulator records. They shall be deterministic and schema-versioned.

### 23.2 Privileged episode record

The environment shall write a tamper-evident episode record containing:

- episode/task IDs and split;
- all build/config/profile/sandbox fingerprints;
- public and privileged seed IDs under access control;
- canonical accepted action stream;
- issued command and request-completion trace;
- flip events and target verification;
- mitigation/refresh events;
- budget use;
- reward and termination;
- infrastructure incidents;
- hashes of script source and bounded outputs;
- cleanup result.

Sensitive evaluation data shall be encrypted at rest with access auditing and retention limits.

### 23.3 Policy-visible logs

Policy-visible messages shall use a small controlled vocabulary and structured fields. They shall not include C++ exceptions, backtraces, internal class names that vary with hidden configuration, raw file paths, raw profile IDs when hidden, or event counts that the task does not disclose.

### 23.4 Timing channels

The service shall decouple policy-observed wall-clock latency from hidden simulator complexity where practical. At minimum:

- the public response schema/length shall not vary with undisclosed target coordinates;
- hidden events shall be redacted before pagination and serialization;
- errors for hidden address validity shall be uniform within the public address grant;
- target selection shall not change mounted-file size or runner image;
- benchmark evaluation may batch or delay responses to reduce coarse timing leakage, with the policy-facing simulated cycle remaining authoritative.

## 24. Configuration and reproducibility

### 24.1 Configuration layers

Configuration shall be layered and immutable after reset:

1. release defaults;
2. signed deployment policy;
3. signed task template;
4. sampled task instance;
5. trainer-authorized seed selection.

Policy actions cannot edit configuration files or environment variables.

### 24.2 Episode fingerprint

Every episode shall expose a public fingerprint and store a privileged fingerprint. The privileged fingerprint shall hash:

- environment source commit;
- Ramulator source commit and patch series;
- OpenEnv source/package hashes;
- compiler/linker/build flags;
- container and sandbox image digests;
- Ramulator canonical config;
- profile package;
- task instance;
- mapping and latent-state seed material;
- reward policy;
- schema versions.

The public fingerprint may omit or commit to hidden components using salted commitments.

### 24.3 Replay

An administrator-only replay tool shall reconstruct an episode from the privileged record and compare:

- every accepted action;
- every issued command and cycle;
- request completion;
- every flip event;
- every public observation digest;
- reward and termination.

Replay mismatch is a release-blocking defect.

### 24.4 Data build

Raw data shall be fetched and processed only by offline jobs. The required pipeline is:

```text
SOURCE_MANIFEST -> download -> checksum/license gate -> immutable raw cache
 -> parser/audit -> canonical parquet/arrow tables -> split
 -> fit -> validate -> model card -> sign profile package
```

The runtime image shall contain only admitted profile packages, not raw research data or fitting notebooks.

---

## 25. Repository skeleton

The repository shall use the following structure. Names may change only through an architecture decision record that preserves the separation of trust domains and test ownership.

```text
rowhammer-openenv/
├── README.md
├── LICENSE
├── SECURITY.md
├── THREAT_MODEL.md
├── CONTRIBUTING.md
├── CODEOWNERS
├── CITATION.cff
├── CHANGELOG.md
├── pyproject.toml
├── uv.lock
├── openenv.yaml
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/
│   ├── Toolchains.cmake
│   ├── ReproducibleBuild.cmake
│   └── Dependencies.cmake
├── third_party/
│   ├── ramulator2/                 # submodule: 278f1eff...
│   ├── ramulator2_source_ref/      # source-only ref: be93be78...
│   ├── patches/
│   │   ├── series
│   │   ├── 0001-controller-event-hook.patch
│   │   └── ...
│   ├── LICENSES/
│   └── README.md
├── cpp/
│   ├── protocol/
│   │   ├── simulator.proto
│   │   ├── broker.proto
│   │   ├── include/rh/protocol/
│   │   └── src/
│   ├── simulator_service/
│   │   ├── include/rh/sim/
│   │   ├── src/main.cpp
│   │   ├── src/service.cpp
│   │   ├── src/episode.cpp
│   │   └── src/request_id_cache.cpp
│   ├── memory/
│   │   ├── include/rh/memory/
│   │   ├── src/functional_memory.cpp
│   │   ├── src/pattern.cpp
│   │   ├── src/sparse_overlay.cpp
│   │   └── src/ecc_adapter.cpp
│   ├── disturbance/
│   │   ├── include/rh/disturbance/
│   │   ├── src/engine.cpp
│   │   ├── src/profile.cpp
│   │   ├── src/hazard.cpp
│   │   ├── src/retention.cpp
│   │   ├── src/rowpress.cpp
│   │   ├── src/counter_rng.cpp
│   │   └── src/flip_log.cpp
│   ├── ramulator_extensions/
│   │   ├── read_disturbance_plugin.cpp
│   │   ├── direct_command_frontend.cpp
│   │   ├── refresh_address_tracker.cpp
│   │   ├── event_annotation.cpp
│   │   └── mitigations/
│   │       ├── oracle.cpp
│   │       ├── para.cpp
│   │       ├── graphene.cpp
│   │       └── ...
│   └── tests/
│       ├── unit/
│       ├── integration/
│       ├── fuzz/
│       └── fixtures/
├── rowhammer_env/
│   ├── __init__.py
│   ├── version.py
│   ├── models.py
│   ├── client.py
│   ├── server/
│   │   ├── app.py
│   │   ├── environment.py
│   │   ├── openenv_adapter.py
│   │   ├── tool_registry.py
│   │   ├── observation.py
│   │   └── dependencies.py
│   ├── orchestrator/
│   │   ├── lifecycle.py
│   │   ├── simulator_process.py
│   │   ├── sandbox_process.py
│   │   ├── capabilities.py
│   │   ├── budgets.py
│   │   └── replay.py
│   ├── tasks/
│   │   ├── schema.py
│   │   ├── compiler.py
│   │   ├── generator.py
│   │   ├── disclosure.py
│   │   ├── predicates.py
│   │   ├── reachability.py
│   │   └── catalog/
│   ├── rewards/
│   │   ├── sparse.py
│   │   ├── potential.py
│   │   └── integrity.py
│   ├── tools/
│   │   ├── describe.py
│   │   ├── topology.py
│   │   ├── memory.py
│   │   ├── transactions.py
│   │   ├── commands.py
│   │   ├── trace.py
│   │   ├── scripts.py
│   │   └── finish.py
│   ├── simulator/
│   │   ├── client.py
│   │   ├── protocol.py
│   │   └── canonical.py
│   ├── profiles/
│   │   ├── manifest.py
│   │   ├── loader.py
│   │   ├── signing.py
│   │   └── model_card.py
│   ├── sandbox/
│   │   ├── interface.py
│   │   ├── gvisor.py
│   │   ├── kata.py
│   │   ├── broker.py
│   │   ├── attestation.py
│   │   └── policies/
│   ├── observability/
│   │   ├── records.py
│   │   ├── redaction.py
│   │   ├── tracing.py
│   │   └── metrics.py
│   └── security/
│       ├── request_ids.py
│       ├── size_limits.py
│       ├── information_flow.py
│       └── audit.py
├── sdk/
│   └── rh_sdk/
│       ├── __init__.py
│       ├── context.py
│       ├── types.py
│       ├── errors.py
│       └── _transport.py
├── schemas/
│   ├── action.schema.json
│   ├── observation.schema.json
│   ├── task.schema.json
│   ├── profile.schema.json
│   ├── program.schema.json
│   ├── errors.schema.json
│   └── internal/
├── data/
│   ├── manifests/
│   │   ├── SOURCE_MANIFEST.yaml
│   │   └── licenses.yaml
│   ├── raw/.gitignore
│   ├── canonical/.gitignore
│   ├── profiles/.gitignore
│   └── README.md
├── profile_builder/
│   ├── ingest/
│   │   ├── vts25.py
│   │   ├── hbm2_dsn24.py
│   │   └── retention_voltage.py
│   ├── audit/
│   ├── split.py
│   ├── fit/
│   │   ├── hcf.py
│   │   ├── ber.py
│   │   ├── rowpress.py
│   │   ├── retention.py
│   │   └── hierarchy.py
│   ├── validate/
│   ├── package.py
│   └── model_card.py
├── configs/
│   ├── dram/
│   ├── controllers/
│   ├── mitigations/
│   ├── profiles/
│   ├── tasks/
│   └── sandbox/
├── scripts/
│   ├── fetch_data.py
│   ├── verify_data.py
│   ├── build_profiles.py
│   ├── validate_profiles.py
│   ├── export_ramulator_config.py
│   ├── replay_episode.py
│   ├── verify_release.py
│   └── reproduce_release.py
├── reference_policies/
│   ├── known_target_transactions.py
│   ├── known_target_commands.py
│   ├── unknown_adjacency.py
│   ├── rowpress.py
│   └── random_baseline.py
├── tests/
│   ├── unit/
│   ├── contracts/
│   ├── ramulator/
│   ├── memory/
│   ├── disturbance/
│   ├── refresh/
│   ├── calibration/
│   ├── mitigations/
│   ├── tasks/
│   ├── rewards/
│   ├── e2e/
│   ├── sandbox/
│   ├── security/
│   ├── information_flow/
│   ├── fuzz/
│   ├── statistical/
│   ├── concurrency/
│   ├── performance/
│   ├── data/
│   └── release/
├── deployments/
│   ├── docker/
│   ├── kubernetes/
│   │   ├── base/
│   │   └── overlays/
│   └── sandbox/
├── docs/
│   ├── specification.md
│   ├── api.md
│   ├── architecture.md
│   ├── disturbance-model.md
│   ├── calibration.md
│   ├── task-catalog.md
│   ├── sandbox.md
│   ├── testing.md
│   ├── operations.md
│   └── adr/
└── .github/
    ├── workflows/
    │   ├── pr-real-simulator.yml
    │   ├── nightly.yml
    │   ├── weekly-calibration.yml
    │   └── release.yml
    └── dependabot.yml
```

### 25.1 Ownership boundaries

- `cpp/simulator_service`, `cpp/ramulator_extensions`, and `cpp/disturbance` require DRAM-simulator and security code owner approval.
- `profile_builder` requires research-science approval for model changes and data-governance approval for source changes.
- `rowhammer_env/sandbox` and deployment policies require security approval.
- public schemas require API owner approval and compatibility tests.
- mitigation ports require both algorithm and simulator approvals.

### 25.2 Generated files

Generated Ramulator code, Protobuf bindings, JSON Schema snapshots, and profile artifacts shall be reproducibly generated and checked for drift. Generated files shall carry headers naming the generator version and input digest. Hand-editing generated outputs is forbidden.

## 26. Detailed implementation plan

The order below minimizes rework by establishing the real end-to-end path, semantics, and security boundary before adding statistical richness or task breadth. No phase uses a mock implementation. A phase may expose only a narrow real feature subset.

### Phase 0 — Scope, provenance, and threat-model freeze

**Work**

- Approve this specification and create architecture decision records for trust boundaries, supported standard/profile scope, direct-command semantics, data licensing, and approved isolation runtime.
- Pin Ramulator, OpenEnv, compiler, build images, mitigation source reference, and initial datasets.
- Build the source manifest and complete legal review for data/code redistribution.
- Define the no-mock CI guard and forbidden fallback list.
- Define release-supported host/kernel/architecture targets.

**Acceptance criteria**

- Every external dependency has an immutable identifier and license record.
- Threat model and security invariants are signed off.
- CI fails on a floating Git ref, unverified download, missing license status, or mock/fallback symbol.
- The VTS25 source can be fetched into a controlled cache and its file inventory/checksums recorded, even if redistribution remains restricted.

### Phase 1 — Reproducible real-simulator bootstrap

**Work**

- Add pinned Ramulator as a submodule and build it unmodified.
- Run all upstream Ramulator 2.1 smoke, latency-throughput, device-timing, and controller-scheduling suites.
- Add pinned OpenEnv `v0.3.1` and a minimal environment subclass.
- Export one canonical DDR4 configuration.
- Execute one real write and one real read through OpenEnv `reset -> step(CallToolAction) -> Ramulator External frontend -> completion callback`.
- Produce build manifests and signed container images.

**Acceptance criteria**

- The end-to-end path contains no fake memory system or canned result.
- Upstream tests pass unchanged.
- A queue-full condition and retry are observed in a real integration test.
- Two clean builds produce identical binaries or documented/reviewed reproducibility exceptions.
- OpenEnv sync and async contract tests pass against the exact pin.

### Phase 2 — Trusted simulator worker and protocol

**Work**

- Move Ramulator into a dedicated C++ worker process.
- Implement typed authenticated RPC, request sequencing, deadlines, and crash containment.
- Implement fresh-worker-per-episode lifecycle and canonical episode config.
- Add command/request event recording and exact replay of the minimal transaction episode.
- Add request-ID idempotency in the trusted service.

**Acceptance criteria**

- Simulator crashes cannot crash the OpenEnv server.
- Cross-episode tokens and stale sequence numbers are rejected.
- Duplicate request IDs return identical cached results with no second Ramulator admission.
- Fresh reset produces no state from the previous episode.
- Episode replay reproduces the issued-command trace exactly.

### Phase 3 — Functional memory and direct-command semantics

**Work**

- Implement deterministic initialization patterns, sparse write overlay, timed read/write commit, masks, and layout.
- Implement the direct-command frontend/sequencer through real timing/prerequisite/state paths.
- Implement transaction and command AST interpreters with lazy repeat expansion and budgets.
- Keep the disturbance engine disabled as an unavailable feature, not a zero-flip placeholder.

**Acceptance criteria**

- Transaction and direct-command reads/writes agree byte-for-byte when they produce equivalent legal command sequences.
- Illegal direct commands do not mutate state.
- Refresh and controller arbitration continue during waits and command programs.
- All alignment, burst, lane, mask, and commit-cycle tests pass.
- The environment refuses disturbance tasks with `UNSUPPORTED_PROFILE` until Phase 4.

### Phase 4 — Data pipeline and first real DDR4 disturbance profile

**Work**

- Implement VTS25 fetch, checksum, inventory, parser, canonical-table conversion, and chip/module split.
- Reproduce key aggregate source results.
- Fit first-flip, BER/count, and retention models using the required nonparametric/hierarchical baseline.
- Implement profile packaging/signing/loading.
- Implement the C++ disturbance engine, latent cell thresholds, actual-command exposure, flip overlay, and counter-based RNG.
- Implement all-bank refresh-address tracking for the selected DDR4 config.
- Validate on held-out chips/modules.

**Acceptance criteria**

- Every runtime profile parameter is traceable to a source file, fitting step, or documented constant.
- No synthetic placeholder dataset enters the build.
- Same seed/actions produce identical flips.
- Held-out calibration gates in Section 27 pass for supported conditions.
- Refresh affects only the modeled rows covered by actual issued refreshes.
- A real reference command/transaction program causes a verified simulated flip in an admitted task.

### Phase 5 — OpenEnv tools, task compiler, and trusted reward

**Work**

- Implement all common types, schemas, initial observation, tool discovery, memory/transaction/command/trace/finish tools, disclosure projection, budgets, and stable errors.
- Implement target predicates and sparse reward.
- Implement known-target/known-adjacency tasks first.
- Add reachability preflight and reference policies using the real simulator.
- Disable the direct `/mcp` path for rollout identities.

**Acceptance criteria**

- JSON Schema and OpenEnv contract suites pass.
- Success is triggered only by trusted memory state.
- Hidden target state is absent from all public serialized structures.
- Reference policies succeed at the task-manifest floor; random baselines remain below the maximum allowed floor/ceiling.
- Direct MCP calls cannot be made by a training identity.

### Phase 6 — Strict script runner and SDK

**Work**

- Build the fixed Python runner image and `rh_sdk`.
- Implement hardened runtime launcher, capability broker, resource controls, output handling, and attestation.
- Route SDK calls through the same validation/budget/simulator path.
- Add the complete sandbox and information-flow attack suite.

**Acceptance criteria**

- All prohibited device, filesystem, syscall, network, credential, and cross-episode tests pass under the actual production runtime.
- Removing or misconfiguring the approved runtime causes fail-closed startup.
- Script and equivalent direct tool calls produce identical simulator traces.
- A terminal target reached inside a script stops further SDK execution and returns the correct reward.
- No plain-container or local-subprocess fallback exists.

### Phase 7 — RowPress, temperature, and temporal variability

**Work**

- Fit and integrate VTS25 RowPress and temperature-sweep data with audited RowPress protocol semantics.
- Implement continuous open-time integration and first-crossing behavior.
- Add named temporal/noise components and ablation controls.
- Add RowPress and retention task families.

**Acceptance criteria**

- Open-row exposure is applied before reads, PRE, refresh, verification, and termination.
- RowPress/temperature/variability held-out gates pass.
- Out-of-domain dwell or temperature behavior fails explicitly.
- Frozen-noise ablations are deterministic and independent.

### Phase 8 — Hidden mapping, adjacency, and search tasks

**Work**

- Implement opaque row handles, partial topology disclosure, hidden target generation, mapping commitments, and non-leaking trace redaction.
- Add unknown-target and unknown-adjacency tasks.
- Add profile-backed mapping or clearly labeled research-hypothesis mapping packages.
- Calibrate task difficulty with real reference and random/search baselines.

**Acceptance criteria**

- Hidden canary suite finds no leakage through observations, errors, cursors, lengths, timing classes, files, or logs.
- Task generation preserves bank/subarray constraints and stated adjacency.
- Reference success and random baselines meet manifest ranges.
- Generated mapping assumptions are visible in model cards and benchmark metadata.

### Phase 9 — Separate HBM2 and DDR3/DDR3L profiles

**Work**

- Ingest and verify the Zenodo HBM2 artifact and pinned repository.
- Fit an HBM2-specific profile against Ramulator HBM2 organization/timing.
- Add a separate DDR3/DDR3L historical/retention profile where source data and licensing permit.
- Extend coordinate, refresh, and command schemas only as required by the actual standards.

**Acceptance criteria**

- No parameter pooling across standards unless a documented hierarchical study proves it and benchmark owners approve it.
- Each standard passes independent calibration, command-timing, refresh, and end-to-end suites.
- A task cannot select a profile/standard mismatch.

### Phase 10 — Mitigations, one at a time

**Work**

- Implement Oracle first.
- Forward-port and admit each mitigation separately following Section 13.3.
- Add source/port differential harnesses, paper-parameter tests, performance counters, and mitigation-aware tasks.
- Add hidden-mitigation task variants only after non-leakage review.

**Acceptance criteria per mitigation**

- Source pin and paper mapping documented.
- Differential trace suite passes for refresh targets, counters, decisions, and timing effects within declared equivalence.
- End-to-end outcomes change for the expected reason, not because the mitigation directly edits flip state.
- Unsupported standard/config pairs fail with `MITIGATION_UNAVAILABLE`.
- The feature is absent from tool discovery until admitted.

### Phase 11 — Curriculum and held-out evaluation

**Work**

- Freeze task mixtures, profile splits, reward policies, and benchmark seeds/commitments.
- Add curriculum schedules and held-out evaluation harness.
- Measure difficulty, variance, and confidence intervals.
- Add anti-overfitting rotation of private held-out profile packages.

**Acceptance criteria**

- Training profiles share no physical source unit with held-out evaluation profiles.
- Evaluation is reproducible from privileged records.
- Score reports separate task family, profile, standard, mitigation, and infrastructure failures.
- Dense shaping is absent from benchmark evaluation.

### Phase 12 — Performance engineering

**Work**

- Profile real workloads.
- Add sparse state compression, batched RPC, trace compression, safe cycle skipping, and immutable-template startup only where valuable.
- For every optimization, build a semantic differential test against the pre-optimization implementation.

**Acceptance criteria**

- Issued commands, completions, flips, observations, and reward remain bit-for-bit equivalent.
- No optimization changes RNG consumption.
- Performance SLOs are measured on a named reference platform and included in release notes.
- A slower real path is retained as a differential oracle where practical, not replaced by a mock.

### Phase 13 — Production hardening and release

**Work**

- Run security red-team, fuzz, statistical, reproducibility, and long-duration concurrency campaigns.
- Produce signed SBOMs, SLSA-style provenance, vulnerability reports, profile model cards, source/license manifests, deployment manifests, and operations documentation.
- Freeze API/spec compatibility policy and incident response.

**Acceptance criteria**

- All release-blocking suites pass with zero required skips.
- No critical/high unmitigated vulnerability is accepted without documented risk approval.
- Release images and profiles are signed and reproducible.
- Disaster cleanup, node quarantine, key rotation, and record replay drills pass.

---

## 27. Test suite specification

### 27.1 Test principles

All conformance tests shall execute real production components. Unit tests may isolate a pure function, but they shall not replace Ramulator, a profile, a mitigation, or the sandbox in any test that claims integration, calibration, end-to-end, security, or release coverage.

Tests shall be deterministic unless explicitly statistical. Statistical tests shall use preregistered sample sizes, random seeds, metrics, confidence intervals, and failure thresholds. Required tests shall not be skipped because a dependency is unavailable; the job shall fail and report the missing prerequisite.

Each test shall declare:

- owner;
- trust domain(s);
- requirement IDs covered;
- fixtures and their provenance;
- expected runtime tier;
- deterministic/statistical classification;
- failure triage guidance;
- whether it blocks pull requests, nightly builds, profile admission, or releases.

### 27.2 Dependency and provenance tests

**Required tests**

- `PROV-001`: Ramulator submodule full SHA equals `278f1effc3838099a6ffe0ad5f9f572fea80c948`.
- `PROV-002`: OpenEnv package/source resolves to `v0.3.1` and commit `7449c5dfe375c4c6e6f0827826925a46efd9249f`.
- `PROV-003`: mitigation source reference equals the approved immutable SHA.
- `PROV-004`: every source-manifest file checksum matches.
- `PROV-005`: every profile signature and source-manifest digest verifies.
- `PROV-006`: compiler, linker, CMake, Python, container, and sandbox-runtime versions match release policy.
- `PROV-007`: no dependency URL uses an unpinned branch for production input.
- `PROV-008`: SBOM contains every linked/runtime dependency and license classification.
- `PROV-009`: two clean builds compare reproducibly according to the release policy.
- `PROV-010`: generated sources are in sync and identify their generator/input digests.

**No-mock guard**

A static and runtime guard shall fail on:

- classes/modules named or registered as mock/fake/stub/dummy for simulator, disturbance, mitigation, or sandbox production interfaces;
- a no-op mitigation registered under a real mitigation name;
- a random bit-flip function not loaded from an admitted profile;
- a local subprocess runner selected when hardened sandbox mode is required;
- a “continue without profile” or “disable validation on error” branch;
- test-only dependency injection compiled into release images.

The guard is supplemental; code review and runtime path assertions remain required.

### 27.3 Upstream Ramulator tests

Run the pinned upstream suites unchanged:

- smoke;
- latency-throughput;
- device timings;
- controller scheduling.

Add a patch-drift job that runs the suites on unpatched Ramulator and patched Ramulator. Any changed upstream result requires a documented intentional difference and review.

### 27.4 OpenEnv contract tests

**Reset/step/state**

- initial reset observation validates against `observation.schema.json`;
- synchronous and asynchronous clients produce equivalent results;
- `StepResult.observation`, `reward`, and `done` match the pinned OpenEnv contract;
- state step count increments exactly once per outer policy action;
- terminal state rejects further actions consistently;
- reset produces a new episode ID and clean resources.

**MCP/tool behavior**

- `ListToolsAction` returns only granted tools;
- every tool schema matches the checked-in snapshot and JSON Schema validator;
- reserved names cannot be registered;
- malformed arguments return stable typed errors;
- `CallToolAction` flows through reward/termination logic;
- the direct `/mcp` route is unreachable for rollout credentials and network policy;
- production-serving mode, if shipped, is deployed separately and cannot claim training reward semantics.

### 27.5 Public schema and serialization tests

- round-trip every public model through JSON and the OpenEnv transport;
- reject unknown fields in strict mode;
- reject integers outside safe ranges, NaN/Infinity, duplicate keys, invalid UTF-8, oversized strings, and excessive nesting;
- verify canonical serialization and result digests;
- property-test address/region normalization;
- snapshot every stable error code and policy-safe message class;
- ensure cursors are episode-scoped, authenticated, expiring, and non-malleable;
- ensure request IDs reused with a different body are rejected;
- ensure duplicate request IDs with identical bodies return the identical cached result.

### 27.6 Simulator integration tests

**Transactions**

- queue admission and full-queue retry follow actual Ramulator return values;
- request completion cycle matches the Ramulator callback;
- transaction size/alignment restrictions match selected config;
- multiple source IDs preserve ordering rules;
- row-hit/row-miss behavior matches command traces;
- waits continue refresh and controller progress;
- outstanding requests behave correctly for each completion policy;
- partial action result names the exact completed prefix.

**Direct commands**

- legal command sequences issue on expected cycles;
- each timing constraint has boundary tests at `n-1`, `n`, and `n+1` cycles;
- missing ACT/PRE and wrong bank/rank state return stable errors;
- `strict` never delays a command silently;
- `earliest_legal` reports requested/actual issue cycles and permits mandatory arbitration;
- illegal commands do not mutate row state, counters, exposure, memory, or budgets beyond the documented invalid-action cost;
- command vocabulary and address shape are generated correctly for each supported standard;
- direct data commands and transaction-generated commands share the same data-plane commit semantics.

**Lifecycle**

- worker crash, hang, malformed reply, and premature EOF are contained and classified;
- worker is killed after terminal state and cannot accept stale RPC;
- 10,000 reset/close cycles show no descriptor, process, shared-memory, or ephemeral-file leak;
- prior-episode canaries never appear in a later episode.

### 27.7 Functional memory tests

- every initialization pattern matches a dense reference for bounded regions;
- sparse overlay reads equal dense-model reads under randomized writes;
- overlapping writes obey deterministic last-commit order;
- masks affect only selected bits/bytes;
- byte/word endianness and lane interleaving match documentation;
- reads sample after due disturbance and before/at the configured completion point;
- writes restore only intended cells according to profile rules;
- flips persist across reads and unrelated writes;
- a restoring write removes the correct flip overlay;
- a retry or duplicate request never applies a write twice;
- memory and flip state serialize/replay exactly.

### 27.8 Disturbance engine deterministic tests

Use small admitted profile fixtures derived from real source rows and packaged through the real profile builder. They may be reduced in size but not fabricated.

- `DIST-DET-001`: no issued aggressor exposure produces no RowHammer/RowPress flip; retention may occur only when configured by the real profile.
- `DIST-DET-002`: rejected/unissued commands produce zero exposure.
- `DIST-DET-003`: ACT/PRE issue cycles create the exact exposure intervals.
- `DIST-DET-004`: a threshold crossing realizes exactly once at the first crossing cycle.
- `DIST-DET-005`: the same seed/action sequence produces identical latent cells and flips.
- `DIST-DET-006`: unrelated access order does not change a row's keyed latent state.
- `DIST-DET-007`: duplicate request replay does not double-count exposure.
- `DIST-DET-008`: open-time exposure is integrated before a read while the row remains open.
- `DIST-DET-009`: refresh/write recovery updates only intended cells/rows.
- `DIST-DET-010`: realized flips persist and are visible in subsequent reads.
- `DIST-DET-011`: single- and double-sided exposure select the correct fitted response.
- `DIST-DET-012`: unsupported patterns/temperatures/dwell conditions fail explicitly.
- `DIST-DET-013`: cycle skipping, if enabled, produces the same first-crossing cycle as single-cycle execution.
- `DIST-DET-014`: cross-thread and batch execution are bit-for-bit deterministic.

### 27.9 Disturbance physical-consistency properties

Within each profile's declared domain, test:

- expected cumulative incidence is nondecreasing with qualifying exposure;
- expected retention failure is nondecreasing with wait time and profile-supported temperature relation;
- refresh/recovery does not increase accumulated damage unless the profile explicitly models an observed anomaly;
- cells beyond the declared neighbor distance receive no unsupported coupling;
- a row cannot flip a second time in the same direction without being restored;
- initial data state and direction constraints are obeyed;
- RowPress risk responds to actual open time, not only activation count;
- ordinary reads/writes create only the exposures implied by their actual command trace;
- mitigation-generated activity is attributed and modeled consistently.

These are profile-conditional properties; a documented empirical exception may override a generic monotonicity assumption only through a reviewed model-card rule and dedicated regression fixture.

### 27.10 Refresh tracker tests

For every supported standard/configuration:

- enumerate a reduced geometry and prove each row is covered within `tREFW` under normal all-bank refresh;
- verify counter wraparound;
- verify rows-per-refresh grouping;
- verify rank/bank scope for all-bank and per-bank commands;
- verify fine-granularity modes where supported;
- verify that one refresh does not reset unrelated rows;
- verify targeted mitigation refresh affects exactly named rows;
- verify RFM/PRAC semantics against the selected Ramulator model;
- verify refresh continues during transaction and direct-command waits;
- verify refresh arbitration cannot be disabled by a policy action;
- compare a long-run refresh coverage trace to a separately reviewed standards-level oracle.

### 27.11 Data ingestion tests

For each source:

- download only from the manifest URL/DOI and verify checksum;
- reject changed file inventory or unrecognized schema;
- validate row counts, nulls, categorical domains, units, monotonic fields, and duplicate experiment keys;
- preserve source module/chip/row/iteration identifiers;
- reproduce selected tables/figures or aggregate statistics from the paper/artifact;
- verify no source unit spans train/validation/test;
- ensure license status permits the requested processing/output action;
- record parser warnings and fail on unexplained loss of rows;
- make canonical output deterministic and hash-stable.

### 27.12 Calibration and held-out statistical tests

The profile builder shall produce metrics at source-family, chip/module, row, pattern, aggressor type, temperature, and exposure strata where sample size permits.

**Required metrics**

- first-flip threshold quantiles and relative error;
- empirical CDF distance: Kolmogorov-Smirnov and/or Cramér-von Mises;
- Wasserstein distance in log-exposure space;
- mean/variance and quantiles of bit-flip counts;
- BER curve integrated absolute/relative error;
- calibration curve and expected calibration error;
- predictive interval coverage;
- zero-inflation and overdispersion diagnostics;
- pattern/aggressor ranking agreement;
- direction proportion error where identifiable;
- RowPress two-dimensional surface error;
- retention survival error;
- chip/row intraclass-correlation or equivalent clustering diagnostic;
- temporal-variation distribution distance.

**Initial admission gates**

The first baseline study shall compute uncertainty and may tighten these gates. Relaxation requires a specification/model-card revision and scientific approval; CI shall never silently relax them.

- key first-flip quantile relative error: `<= 15%` on held-out source units for supported strata with adequate samples;
- KS distance for primary threshold/count distributions: `<= 0.10`;
- direction proportion absolute error: `<= 5 percentage points` where direction is supported;
- nominal 90% predictive interval empirical coverage: `85%–95%`;
- BER/count curve normalized integrated error: `<= 15%` for primary benchmark strata;
- rank correlation of condition susceptibility: `>= 0.8` when at least five comparable conditions exist;
- no statistically significant train/evaluation source-unit leakage according to split audit.

A profile that fails a stratum shall either improve the model or remove that stratum from its declared domain. It shall not average away a failing subgroup.

**Statistical procedure**

- use source-unit bootstrap, not independent-row bootstrap when rows share a chip/module;
- publish confidence intervals;
- correct for multiple comparisons or define a hierarchical gate;
- predefine random seeds and sample counts;
- store plots/tables and machine-readable results as profile-admission artifacts;
- compare generated and measured distributions using held-out chips/modules only.

### 27.13 Mitigation tests

For every candidate mitigation:

**Algorithm unit tests**

- counters, thresholds, tables, reset/decay, random choices, and address scope;
- parameter validation and boundary values;
- deterministic RNG under fixed seed.

**Source differential tests**

Feed equivalent request/command traces into the pinned source implementation and the 2.1 port. Compare:

- mitigation-trigger cycles;
- selected victim/refresh/remap rows;
- counter evolution;
- throttling/stall decisions;
- issued mitigation commands;
- performance impact attributable to the algorithm.

Exact equality is required unless the porting report documents an interface-induced timing convention and defines a reviewed equivalence tolerance.

**End-to-end tests**

- mitigation acts through real refresh/throttle/remap commands;
- disturbance state is never directly edited by the mitigation;
- below-threshold workloads are not spuriously mitigated beyond algorithm semantics;
- above-threshold workloads trigger expected actions;
- mitigation-on and mitigation-off traces differ as expected;
- unsupported standard/config combinations fail closed;
- hidden mitigation does not leak through tool catalog, errors, or observation shape.

**Paper conformance**

Where the paper/source publishes configurations and results, reproduce qualitative trends and numeric values within an approved tolerance. Record deviations.

### 27.14 Task-generation tests

- every generated task validates against schema;
- target lies in the admitted address/profile domain;
- target disclosure matches policy exactly;
- known adjacency relations are true in hidden topology;
- hidden target/mapping fields are absent from public projection;
- target predicate is satisfiable under preflight success floor, unless labeled negative;
- reference policy success falls within manifest range;
- random/search baseline falls within manifest range;
- task seeds produce stable instances;
- different seeds provide adequate diversity without split leakage;
- task text and structured fields are semantically consistent;
- budgets are internally consistent and cannot overflow counters;
- target selection does not alter observation length or setup-resource footprint in a target-dependent way.

### 27.15 Reward and verifier tests

- exact predicate truth tables for every primitive/combinator;
- transition, count, scope, persistence, and ECC semantics;
- extra flips allowed/disallowed behavior;
- evaluator drain/settling/read ordering;
- stdout or return value claiming success has no effect;
- spoofed trace/result fields have no effect;
- simulator crash, timeout, sandbox violation, and infrastructure cancellation yield no positive reward;
- invalid-action penalties cap correctly;
- efficiency bonus cannot exceed bound or turn failure into positive reward;
- dense shaping, when enabled, uses public state only and is disabled in evaluation;
- reward and terminal state replay exactly.

### 27.16 End-to-end policy tests

Run against real Ramulator, admitted profiles, and the production OpenEnv path:

- known-target transaction reference policy succeeds;
- known-target direct-command policy succeeds;
- equivalent script SDK policy produces equivalent trace/outcome;
- unknown-adjacency reference policy performs probing and succeeds at expected rate;
- random baseline does not exceed the task's maximum allowed rate;
- disclosed mitigation reference adapts and produces expected actions;
- hidden mitigation tasks remain solvable at intended difficulty;
- held-out profile tasks use only evaluation source units;
- terminal success during a script interrupts subsequent calls;
- replay reproduces every public observation digest.

### 27.17 Sandbox security tests

All tests run inside the exact release runtime and policy image.

**Filesystem/device tests**

Attempt and verify denial of:

- `/dev/mem`, `/dev/kmem`, `/dev/port`, `/dev/cpu/*/msr`, `/dev/kvm`;
- host block/character devices, RDMA, GPU, FPGA, PMEM;
- host `/proc`, `/sys`, container runtime sockets, kube service tokens, host mounts;
- traversal/symlink/hardlink tricks out of writable tmpfs;
- setuid/setgid execution and file capabilities;
- core dumps and secret environment variables.

**Kernel/syscall tests**

Attempt and verify denial or safe virtualization of:

- ptrace outside sandbox;
- mount/pivot_root and unauthorized namespace creation;
- module loading, eBPF, `perf_event_open`, raw sockets;
- `iopl`, `ioperm`, reboot/kexec;
- arbitrary device ioctls;
- userfaultfd or other high-risk interfaces according to release policy;
- escape regressions relevant to the pinned runtime.

**Network tests**

- no DNS, Internet, LAN, cloud metadata, host-loopback, or arbitrary Unix socket access;
- only the authenticated broker is reachable;
- broker token cannot be reused across episodes;
- malformed/oversized broker frames are rejected without broker crash.

**Resource/DoS tests**

- fork/thread bombs;
- memory bombs;
- file/output bombs;
- infinite loops and sleep;
- decompression/regex/parser bombs;
- excessive SDK calls and nested data;
- orphaned grandchildren;
- repeated sandbox startup/teardown.

The complete sandbox must terminate within policy limits and leave no process, mount, file, token, or network endpoint.

**Fail-closed tests**

- remove the approved runtime binary;
- alter its configuration/policy digest;
- remove required cgroup controller;
- weaken network policy;
- introduce forbidden device/mount;
- fail attestation.

Each must prevent task admission. No default-container/local-exec fallback may run.

### 27.18 Information-flow tests

Create unique hidden canaries in target coordinates, mapping tables, profile IDs, seeds, mitigation names, file paths, and privileged trace records. Search for canaries or derived encodings in:

- initial and step observations;
- tool catalogs and schemas;
- errors and exception classes;
- stdout/stderr and SDK exceptions;
- trace pages and cursors;
- response lengths and JSON key presence;
- log files available to policy;
- sandbox filesystem and environment;
- process arguments and `/proc` view;
- metrics endpoints;
- episode IDs and filenames.

Run differential experiments varying one hidden field while keeping the public task fixed. Test that public serialization and policy-visible resource footprint remain identical except for behavior legitimately caused by policy interactions. Coarse wall-time distributions shall be monitored for leakage.

### 27.19 Fuzzing

**Python/property fuzzing**

Use Hypothesis or equivalent for:

- public action/observation schemas;
- address/region/pattern normalization;
- AST interpreters and budget accounting;
- target predicates;
- cursor and request-ID logic;
- redaction.

**Native fuzzing**

Use libFuzzer/AFL++ or equivalent for:

- internal RPC decoders;
- profile package parser;
- generated Ramulator configuration parser boundary;
- direct-command AST decoder;
- memory payload/mask decoder;
- broker protocol.

Sanitizer lanes shall include ASan, UBSan, and where feasible MSan/TSan. Fuzz corpora shall include minimized regressions and real schema-derived seeds.

### 27.20 Concurrency and isolation tests

- run many episodes concurrently across channels/profiles/tasks;
- verify no cross-episode request, trace, RNG, memory, budget, or capability state;
- stress reset/terminal races and client disconnects;
- kill services at each lifecycle transition and verify cleanup;
- ensure per-episode serialization with parallel episodes;
- verify deterministic results are independent of global scheduling;
- enforce global and tenant quotas fairly;
- verify one malicious script cannot starve simulator control traffic.

### 27.21 Performance tests

Performance gates shall be established after a correct baseline on a named reference host. Measure:

- reset cold-start and warm infrastructure overhead, while still using a fresh episode worker;
- simulated cycles per wall second for representative transaction and command workloads;
- policy tool-call latency distribution;
- script sandbox startup and SDK-call overhead;
- memory use as a function of touched rows/cells and trace size;
- concurrent episode throughput;
- profile-loading and signature-verification cost;
- replay throughput.

Every optimization must pass a semantic differential suite comparing command traces, completions, flips, observations, rewards, and RNG event keys. Performance tests never authorize a lower-fidelity fallback.

### 27.22 Long-duration and fault-injection tests

- multi-hour and multi-day episode farms;
- worker SIGKILL at every operation boundary;
- disk full, tmpfs full, log sink failure, network partition to internal broker, and clock anomalies;
- corrupted profile/package/record;
- partial cleanup and node restart;
- trainer disconnect/reconnect and duplicate last action;
- cgroup OOM and sandbox runtime crash;
- profile-signing key rotation and revocation.

Faults shall produce classified truncation, preserve reward integrity, and leave the node safe or quarantined.

### 27.23 CI and release matrix

**Per pull request**

- format/lint/type checks;
- dependency/provenance checks;
- no-mock guard;
- real Ramulator upstream smoke/device-timing subset;
- C++/Python unit tests;
- schema/OpenEnv contracts;
- deterministic simulator/memory/disturbance subset;
- sandbox policy static checks;
- sanitizer smoke.

**Nightly**

- full Ramulator suites;
- all integration/e2e tests with real profiles;
- sandbox attack suite in production runtime;
- information-flow differential suite;
- mitigation differential suites;
- statistical smoke gates;
- concurrency/fault injection;
- fuzzing budget.

**Weekly/profile admission**

- full raw-data audit;
- refit from clean cache;
- held-out calibration gates and model-card generation;
- large statistical tests;
- HBM2 artifact verification when enabled;
- long-duration stress;
- reproducible build and profile package comparison.

**Release**

- all required lanes with zero skips;
- security red-team signoff;
- SBOM/provenance/signature verification;
- exact replay corpus;
- deployment attestation;
- documentation/source/license review;
- signed artifacts and rollback test.

### 27.24 Test evidence

Every release shall publish or archive:

- machine-readable test results;
- exact code/profile/image identifiers;
- calibration tables and plots;
- failed/retried infrastructure jobs;
- fuzz/sanitizer summaries;
- sandbox attestation report;
- non-leakage report;
- mitigation port reports;
- reproducibility and replay results;
- list of unsupported/disabled features.

---

## 28. Nonfunctional requirements

### 28.1 Correctness over throughput

The release shall ship a measured performance baseline, but no fixed throughput target is imposed before Phase 12. Any production timeout must be large enough to distinguish a slow valid simulation from a hung worker on the release reference platform. Infrastructure timeouts shall not alter simulated-time behavior.

### 28.2 Availability and fault isolation

- A failed episode worker shall not affect other episodes.
- A malicious runner shall not exhaust node-wide resources beyond configured tenant/node quotas.
- The environment service shall use bounded queues and backpressure.
- Profile or simulator corruption shall quarantine the artifact/node.
- Administrative health checks shall exercise a real minimal Ramulator episode, not only process liveness.

### 28.3 Compatibility

`rh-openenv/v1` follows semantic compatibility:

- adding optional response fields is allowed only when strict clients ignore them by version policy;
- adding a tool or enum value requires catalog/schema-version handling;
- changing tool semantics, defaults, required fields, reward semantics, target semantics, bit numbering, timing modes, or error classification requires a new API or task version;
- profile revisions that change outcomes require a new profile version and benchmark partition;
- OpenEnv upgrades require a full contract requalification and shall not occur through an unconstrained dependency update.

### 28.4 Documentation

Release documentation shall include:

- architecture and trust boundaries;
- API schemas and examples;
- task catalog and disclosure matrix;
- disturbance methodology and limits;
- profile model cards and calibration results;
- mitigation port reports;
- sandbox threat model and deployment requirements;
- operations, incident response, replay, and cleanup procedures;
- source/data/license manifest;
- exact reproduction commands.

### 28.5 Accessibility to language models

Tool names and descriptions shall be concise, unambiguous, and stable. Each schema field shall include a description and units. Errors shall state the violated public rule and a policy-safe corrective hint when appropriate. The initial observation shall avoid requiring the model to infer command vocabulary or timing units from prose alone.

## 29. Deployment requirements

### 29.1 Service identities

Use separate least-privilege identities for:

- trainer control plane;
- OpenEnv service;
- simulator worker launcher;
- sandbox launcher/broker;
- profile reader;
- episode-record writer;
- metrics reader;
- release/profile signer.

The policy runner receives no cluster identity.

### 29.2 Kubernetes or equivalent

Where Kubernetes is used:

- enforce Restricted Pod Security or stricter controls for trusted services;
- use a dedicated runtime class for untrusted runners;
- deny hostPID, hostIPC, hostNetwork, privileged mode, hostPath, added capabilities, and device plugins;
- use default-deny network policy;
- use read-only root filesystems and explicit ephemeral volumes;
- set requests/limits and cgroup v2 enforcement;
- schedule evaluation profiles on isolated nodes when required;
- prevent the runner from accessing the Kubernetes API or metadata services;
- admit workloads only through signed images and policy checks.

The actual deployment shall be validated, not assumed secure because YAML contains desired fields.

### 29.3 Secrets

Profile signing keys, record encryption keys, trainer credentials, and private evaluation manifests shall never be mounted into untrusted runners or simulator processes that do not require them. Verification keys may be mounted read-only. Key IDs and rotation status belong in privileged records.

### 29.4 Updates

No automatic dependency or base-image update shall reach production without:

- provenance verification;
- vulnerability review;
- full required CI tier for the affected trust domain;
- reproducibility check;
- rollback artifact.

## 30. Example task instance

The following example is illustrative of the canonical task representation. Hidden fields are shown because this is an administrator artifact; the policy receives only the configured public projection.

```yaml
schema_version: rh-task/v1
task_id: KTC-1
task_instance_id: 9b715f84-7fa2-4d72-b389-5e8f4d2e92af
split: train
profile:
  id: vts25-ddr4-generative
  version: 1.0.0
  selection:
    source_unit_pool: train
    seed_ref: secret://episode/profile
ramulator:
  standard: DDR4
  organization: DDR4_8Gb_x8
  timing: DDR4_2400R
  ranks: 1
  controller: GenericDDR
  scheduler: FRFCFS
  row_policy: Open
  refresh_manager: AllBank
  address_mapper: RoBaRaCoCh
  clock_ratio: 1
mitigation:
  id: none
  disclosure: exact
target:
  predicate:
    kind: persistent_condition
    cycles: 1000
    verification_reads: 2
    child:
      kind: all
      children:
        - kind: specific_cell
          coordinate_ref: secret://episode/target-cell
        - kind: specific_transition
          from: 1
          to: 0
  disclosure:
    target: exact_generated_cell
    adjacency: exact
    mapping: exact
control:
  modes: [commands]
  script_access: both
  command_timing_modes: [strict]
initial_memory:
  policy: fixed
  pattern: {kind: constant_byte, value: 255}
  setup_mode: instant_setup
observability:
  memory_reads: true
  command_trace: true
  refresh_events: true
  mitigation_events: false
  flip_events: terminal_only
  counters: [sim_cycle, commands_issued, reads, writes]
budgets:
  openenv_steps: 128
  simulated_cycles: 10000000
  issued_commands: 5000000
  transactions: 100000
  bytes_read: 1048576
  bytes_written: 1048576
  scripts: 8
  script_cpu_ms_total: 30000
  invalid_actions: 32
reward:
  success: 1.0
  valid_nonterminal: 0.0
  invalid_action: -0.002
  invalid_penalty_floor: -0.05
verification:
  drain: all
  settlement_cycles: 1000
  privileged_reads: 2
  extra_flips_allowed: true
preflight:
  required: true
  policy: reference_policies/known_target_commands.py
  minimum_success_probability: 0.95
  validation_budget_multiplier: 2.0
```

## 31. Example initial observation

```json
{
  "api_version":"rh-openenv/v1",
  "episode_id":"67ece91a-7f8a-47fb-8eeb-c001e0ca855a",
  "phase":"ready",
  "task":{
    "task_family":"KTC-1",
    "instruction":"Cause a persistent 1-to-0 transition at the disclosed simulated target cell. All coordinates refer only to the simulated DDR4 device.",
    "target":{
      "kind":"specific_generated_cell",
      "address":{"space":"dram","channel":0,"rank":0,"bank_group":0,"bank":1,"row":1024,"column":128},
      "bit":3,
      "transition":"1_to_0",
      "persistence_cycles":1000
    },
    "dram":{
      "standard":"DDR4",
      "organization":"DDR4_8Gb_x8",
      "timing":"DDR4_2400R",
      "transaction_bytes":64,
      "cycle_unit":"dram_cycle"
    },
    "control":{"mode":"commands_only","timing_modes":["strict"],"scripts":true},
    "mitigation":{"disclosure":"exact","id":"none"},
    "initial_memory":{"pattern":{"kind":"constant_byte","value":255},"setup_complete":true},
    "success":{"verification":"drain_then_two_reads","extra_flips_allowed":true}
  },
  "tool_catalog":[
    {"name":"describe_environment","input_schema":{"$ref":"schemas/action.schema.json#/$defs/DescribeArgs"}},
    {"name":"submit_command_program","input_schema":{"$ref":"schemas/action.schema.json#/$defs/SubmitCommandArgs"}},
    {"name":"read_memory","input_schema":{"$ref":"schemas/action.schema.json#/$defs/ReadMemoryArgs"}},
    {"name":"get_trace","input_schema":{"$ref":"schemas/action.schema.json#/$defs/GetTraceArgs"}},
    {"name":"run_script","input_schema":{"$ref":"schemas/action.schema.json#/$defs/RunScriptArgs"}},
    {"name":"finish_episode","input_schema":{"$ref":"schemas/action.schema.json#/$defs/FinishArgs"}}
  ],
  "result":null,
  "budget":{
    "steps":{"used":0,"limit":128,"remaining":128},
    "simulated_cycles":{"used":0,"limit":10000000,"remaining":10000000},
    "issued_commands":{"used":0,"limit":5000000,"remaining":5000000}
  },
  "public_metrics":{"sim_cycle":0},
  "messages":[{"level":"info","code":"SIMULATION_ONLY","text":"No tool can access host or physical memory."}],
  "termination":null,
  "build":{
    "environment":"sha256:...",
    "ramulator_commit":"278f1effc3838099a6ffe0ad5f9f572fea80c948",
    "openenv_commit":"7449c5dfe375c4c6e6f0827826925a46efd9249f",
    "profile":"vts25-ddr4-generative@1.0.0"
  }
}
```

## 32. Profile model-card minimum contents

Every admitted profile's model card shall state:

- intended use and prohibited claims;
- DRAM standard/organization/timing compatibility;
- exact source URLs, revisions, checksums, and citations;
- license status and redistribution constraints;
- source-unit counts and split method;
- experimental conditions and measured domain;
- parser assumptions and excluded records;
- fitted estimator details;
- generated latent hierarchy and RNG construction;
- cell-identity status: measured replay or generated;
- refresh/recovery assumptions;
- spatial/adjacency assumptions;
- unsupported conditions and out-of-domain policy;
- held-out metrics with confidence intervals and subgroup failures;
- known limitations and scientific uncertainties;
- profile version/change history;
- signing identity and package digest.

## 33. Definition of done for version 1

Version 1 is complete only when all of the following are true:

- the pinned real Ramulator/OpenEnv path is reproducibly built and tested;
- the DDR4 profile is built from pinned real VTS25 data and passes held-out gates;
- transaction and direct-command interfaces use actual Ramulator timing/state;
- memory, refresh, RowHammer, RowPress, retention, noise, and verification semantics are implemented and validated for the declared domain;
- at least the initial known/unknown target and adjacency task families are admitted;
- the untrusted Python runner passes the production-runtime sandbox suite and has no fallback;
- Oracle and every advertised mitigation are genuinely implemented and admitted; unimplemented mitigations are absent from capability discovery;
- reward and hidden-state non-leakage tests pass;
- all required CI/release tests pass with zero skips;
- signed code images, profiles, SBOM, provenance, model cards, mitigation reports, and operations documentation are available;
- the release explicitly lists every unsupported feature and fidelity limitation.

## 34. References

1. Ramulator 2.1 source and user guide: https://github.com/CMU-SAFARI/ramulator2/tree/v2.1
2. Required Ramulator commit: https://github.com/CMU-SAFARI/ramulator2/commit/278f1effc3838099a6ffe0ad5f9f572fea80c948
3. Ramulator 2.1 paper: https://arxiv.org/abs/2606.13844
4. Ramulator 2.0/main mitigation source reference: https://github.com/CMU-SAFARI/ramulator2/tree/be93be78055d922aa1d4d33e15bcc8f2b0c61a9d/src/dram_controller/impl/plugin
5. OpenEnv v0.3.1: https://github.com/huggingface/OpenEnv/releases/tag/v0.3.1
6. Required OpenEnv commit: https://github.com/huggingface/OpenEnv/commit/7449c5dfe375c4c6e6f0827826925a46efd9249f
7. OpenEnv MCP tutorial/control-path behavior: https://github.com/huggingface/OpenEnv/blob/main/docs/source/tutorials/mcp-environment.md
8. VTS25 DDR4 paper: https://arxiv.org/abs/2503.16749
9. VTS25 immutable data: https://github.com/CMU-SAFARI/ReadDisturbanceVTS25/tree/5d734309457cc8a4ea3b1ec36b93932925548bac/data
10. RowPress paper: https://arxiv.org/abs/2306.17061
11. RowPress artifact: https://github.com/CMU-SAFARI/RowPress/tree/5b6f1502594e9ea7a1557b98b5544c096a30eec5
12. HBM2 paper: https://arxiv.org/abs/2310.14665
13. HBM2 artifact repository: https://github.com/CMU-SAFARI/HBM-Read-Disturbance/tree/bc9d600e03efbe38c740ba015398510ca9bd1a60
14. HBM2 archived data: https://zenodo.org/records/10257930
15. Original RowHammer artifact: https://github.com/CMU-SAFARI/rowhammer
16. Revisiting RowHammer: https://arxiv.org/abs/2005.13121
17. Variable Read Disturbance: https://arxiv.org/abs/2502.13075
18. DRAM Voltage Study retention profile: https://github.com/CMU-SAFARI/DRAM-Voltage-Study/blob/master/characterization_results/retention_time_profile.csv
19. EINSim: https://github.com/CMU-SAFARI/EINSim
20. gVisor security model: https://gvisor.dev/docs/architecture_guide/security/
21. Kubernetes Pod Security Standards: https://kubernetes.io/docs/concepts/security/pod-security-standards/
22. Docker seccomp documentation: https://docs.docker.com/engine/security/seccomp/

---

**End of normative specification.**
