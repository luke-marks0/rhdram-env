# SPEC.md

Architectural source of truth for **rhdram-env** — an RL system that trains
language-model policies to discover RowHammer / read-disturbance bit-flip
patterns against a simulation-only DRAM environment.

Read this before touching env, reward, disclosure, disturbance, sandbox, or
training code. Code implements this document; where they disagree, one of the two
is wrong — see [Drift](#drift) for known cases, and don't silently "fix" code to
match a fuzzy reading of the spec or vice-versa.

**Relationship to `spec/`.** `spec/SPEC.md` and the rest of `spec/` are the
original *concise normative design bundle* (requirements + wire schemas +
provenance rules). This document is the *current architecture as built*, tagged
and pointed at real files. Section references like "SPEC §5" scattered through the
code point at `spec/SPEC.md`; the tags here (`@spec:<area>-<name>`) are this
document's stable handles. When the design bundle and the code disagree, that is a
Drift entry, not a license to improvise.

**Tag convention.** Every specified unit of behavior carries a `@spec:<area>-<name>`
tag and a **Defined in:** pointer. Test and implementation code may reference tags
by name. A tag is a contract; anything not yet built is marked *Not yet implemented*
or lives in [Open / unsettled](#open--unsettled) instead of being tagged.

---

## System overview

A policy (an LLM, or a deterministic reference/fixture policy) drives a
simulation-only DRAM environment through a small tool API. Each episode compiles a
*task* — an objective plus a topology-disclosure level plus a resource budget —
against the real geometry a Ramulator 2.1 worker reports, samples a hidden victim
(and, for discovery families, a per-episode secret address mapping), and runs the
policy's DRAM commands through the real simulator. A profile-driven disturbance
engine folds the worker's *actual issued events* into persistent simulated bit
flips. Reward is sparse and trusted: `1.0` only when trusted simulator state
satisfies the task's success predicate, `0.0` otherwise — never derivable from
policy claims, logs, or stdout. The environment never touches host physical memory.

The core loop: `reset(seed, task)` → disclosed observation → policy emits a tool
call → `step(action)` runs it on the worker, updates disturbance state, charges
budget, and returns a disclosure-projected observation → repeat until a success
predicate fires, `episode.finish`, budget exhaustion, or error.

---

## Architecture

```
RL trainer / policy
  → OpenEnv client (HTTP POST or WS)         rowhammer_env/client.py, server/app.py
  → RowHammerTaskEnv.reset/step              rowhammer_env/phase5_env.py
      ├ TaskSpec.from_config / compile       rowhammer_env/tasks/compiler.py
      ├ Disclosure + AddressResolver         rowhammer_env/tasks/disclosure.py
      ├ per-episode (secret) address mapper  rowhammer_env/mappers.py
      ├ trusted success predicate            rowhammer_env/rewards/predicates.py
      └ RowHammerDisturbanceEnv.step         rowhammer_env/phase4_env.py
          ├ RowHammerEnv (tools, budgets)    rowhammer_env/phase2_env.py
          │   → WorkerClient (stdio RPC)      rowhammer_env/worker_protocol.py
          │       → ramulator_worker (C++)    cpp/simulator_service/ramulator_worker.cpp
          │           = Ramulator 2.1 + IssuedEventRecorder plugin
          │             cpp/ramulator_extensions/*, third_party/ramulator2/
          └ DisturbanceEngine.consume/apply  rowhammer_env/disturbance.py
              ├ StandardModel (blast/refresh) rowhammer_env/standards.py
              └ signed profile package        rowhammer_env/profiles.py, profile_builder/

Optional script path (policy submits Python):
  step(script.run) → RestrictedScriptBroker → unshare+bwrap child → rh_sdk broker
  → same env.step tool surface               rowhammer_env/script_sandbox.py, sdk/rh_sdk/

Training harness (torch/trl only in scripts/):
  grpo_env (prompt/parse/reward) · multiturn_rollout (loop + completion mask)
  · curriculum · shaping · policies (reference/fixture)   rowhammer_env/llm/*
```

**Environment class hierarchy** (each layer adds one concern; `RowHammerTaskEnv`
is what the server serves):

- `RowHammerEnv` (phase2) — tool dispatch, worker RPC, logical addressing, command
  expansion, the bare (untask'd) budget-free env.
- `RowHammerDisturbanceEnv` (phase4) — adds geometry fetch, the address mapper,
  the disclosure/resolver, and the disturbance engine wired to issued events.
- `RowHammerTaskEnv` (phase5) — adds the task compiler, per-family disclosure,
  budgets, handles, secret mappers, and the trusted per-family reward.

("phase" filenames are historical build-order names, not runtime phases.)

---

## Specified behavior

### RL formulation

#### `@spec:rl-action-space` — tool set and action envelope
The policy's entire influence on the world is a sequence of tool calls. An action
is `{tool: str, args: dict}`. The admitted tool set is exactly:
`dram.info`, `dram.read`, `dram.write`, `dram.issue`, `script.run`, `episode.finish`.
A missing/empty `tool` fails closed with `BAD_SCHEMA` (never OpenEnv's transport
`VALIDATION_ERROR`). Unknown tools return `UNSUPPORTED_TOOL`.
Defined in: `rowhammer_env/phase5_env.py` (`ALLOWED_TOOLS`, `step`),
`rowhammer_env/phase2_env.py` (`Phase2Action`, `RowHammerEnv.step`),
`spec/schemas/action.schema.json`, `rowhammer_env/llm/tools.py` (`TOOL_SCHEMAS`).

#### `@spec:rl-observation-reset` — initial observation
`reset(seed, episode_id=None, task=None, budgets=None)` returns a `Phase2Observation`
whose `metadata`/`info` carry: `task_id`, `task_family`, `difficulty` (band + seed +
expected success window), `objective` (+ structured `target`/`candidates` at the
disclosed level), `disclosure` (public axes), `address_forms`, `allowed_tools`,
`budget_remaining`, `profile`, `mitigation`, `geometry` (public block), and a public
`disturbance` summary. Hidden target state appears **only** at the disclosure level
that permits it.
Defined in: `rowhammer_env/phase5_env.py` (`reset`, `_task_metadata`,
`_objective_and_target`, `_public_disturbance`), `rowhammer_env/phase4_env.py`
(`reset`, `_disturbance_metadata`).

#### `@spec:rl-observation-step` — step observation
`step(action)` returns `Phase2Observation` with `cycle`, `last_action`
(`accepted`/`rejected`/`cycle_delta`), `public_counters` (`acts`/`reads`/`writes`/
`refreshes`), `feedback` (disclosure-projected: `new_public_flips`, optional
`trace_tail`, `timing_digest`, `public_flips`), `reward`, `done`, `error`, and
`budget_remaining` in metadata. `metadata` is mirrored into a serialized `info`
field because OpenEnv drops `metadata` from the wire payload.
Defined in: `rowhammer_env/phase2_env.py` (`Phase2Observation`, `_from_worker`),
`rowhammer_env/phase5_env.py` (`step`).

#### `@spec:rl-reward` — sparse, trusted reward
Default reward is sparse: `1.0` on the task's success predicate, else `0.0`. The
reward is computed **only** from trusted simulator state (the disturbance engine's
committed flips / decoded flipped-row keys), never from policy-provided logs,
stdout, or claims. `episode.finish` returns `1.0` iff the predicate already holds.
A **target row is a full physical coordinate**: every target-row family decides
success by comparing the compiled task's decoded
`(channel, rank, bankgroup, bank, row)` victim key against the engine's
`flipped_row_keys`, never a row *index* — the index alone names one physical row
per bank, and under a secret mapper a flip's linear address does not encode its
row at all. The `target_cell` and `pattern_target` predicates likewise do not
reconstruct a byte from the engine's flip metadata or data-pattern stratum. Once a
committed flip reaches the target cell, they obtain the worker's actual stored byte
through the server-internal `READ … STORED` form, apply the disturbance overlay,
and compare that resulting byte with the compiled bit/mask/value condition. The
stored-byte read is side-effect-free, so checking reward cannot create unbudgeted
DRAM events or alter cycles and public counters.
Defined in: `rowhammer_env/rewards/predicates.py` (`success_for`, `PREDICATES`),
`rowhammer_env/phase5_env.py` (`_trusted_success`, `_trusted_read_byte`, `step`,
`_script`), `cpp/simulator_service/ramulator_worker.cpp` (`read`).

#### `@spec:rl-budgets` — resource budgets and enforcement
Every episode carries a budget dict (a subset of `tool_calls`, `acts`, `cycles`).
Each `step` charges one `tool_call`, the observed `cycle_delta`, and
the true `acts` delta from `public_counters`. A single `dram.issue` may not spend
more activations than the remaining `acts` budget: the server expands compact forms
and issues one primitive at a time, checking the ceiling **before** each primitive
leaves for the worker — so **no flip is ever credited that the budget could not pay
for**. A `step` with no `tool_calls` left is likewise refused before dispatch (it
reaches neither the worker nor `_charge`), so the budget is spent by actions that
ran, never by an action the budget could not afford. Exhaustion sets `error =
BUDGET_EXCEEDED` and `done`. Families without an `acts` budget hammer freely.
The `script.run` path carries no separate budget: its per-call `timeout_ms` arg is
a wall-clock safety deadline (not a reward-shaping cost), and its brokered inner
`rh.*` calls charge the same `tool_calls`/`acts`/`cycles` budgets as any other tool
call. See `DECISION.md` for why a cumulative `script_ms` budget was removed rather
than implemented.
Defined in: `rowhammer_env/phase5_env.py` (`_charge`, `_issue_acts_ceiling`),
`rowhammer_env/phase2_env.py` (`_issue`, `MAX_ISSUE_ACTIVATIONS`).

#### `@spec:rl-episode-termination` — termination
An episode ends when: the success predicate becomes true; `episode.finish` is
called; a budget is exhausted; an unrecoverable simulator/sandbox error occurs; or
the worker's per-request cycle deadline is hit. Any of these sets `done=True`.
Termination is enforced by the env, not assumed of the driver: once an episode has
ended, every further `step` is **refused before dispatch** — no worker call, no
disturbance, no budget charged — with `done` and a stable error code, and the worker
is torn down. Only the conditions above latch; the codes that reject a single
*action* (`BAD_SCHEMA`, `UNSUPPORTED_TOOL`, `ADDRESS_NOT_DISCLOSED`,
`ILLEGAL_COMMAND`, `SANDBOX_VIOLATION`, `SCRIPT_TIMEOUT`, `QUEUE_FULL`,
`PROFILE_REJECTED`, `UNAVAILABLE_CAPABILITY`) leave the episode running **and report
`done=False`**, so a driver that stops on `done` does not abandon an episode over a
typo — an episode that cannot start at all is ended by `reset` itself, whatever code
that failure carries.
Defined in: `rowhammer_env/phase2_env.py` (`_error`, `_terminal_error`,
`TERMINAL_ERROR_CODES`, `reset`, `_issue`),
`rowhammer_env/phase5_env.py` (`step`, `reset`, `_refuse`, `_charge`),
`rowhammer_env/phase4_env.py` (`reset`),
`cpp/simulator_service/ramulator_worker.cpp` (`complete_request` deadline).

### Tools (policy-facing interface)

#### `@spec:tool-dram-info`
Returns disclosed `address_forms`, `commands` (`RD`/`WR`/`WAIT`), admitted
`mitigations`, and the public `geometry` block. Static per episode; discloses no
hidden state.
Defined in: `rowhammer_env/phase2_env.py` (`step` `dram.info` branch, `_geometry_block`).

#### `@spec:tool-dram-read` / `@spec:tool-dram-write`
Read/write simulated memory at a disclosed address (logical / physical / handle per
task). Reads return base64 bytes with any disturbance flips applied to the returned
copy; writes update functional state and record the row's data pattern (for stratum
selection) and restore overwritten flipped cells. `note_write` takes the write's
*decoded* row key, not its linear address: which row an address belongs to is a
property of the active mapper, which only the worker knows.
Defined in: `rowhammer_env/phase2_env.py` (`step`), `rowhammer_env/phase4_env.py`
(`_from_worker`), `rowhammer_env/disturbance.py` (`apply`, `note_write`, `restore`).

#### `@spec:tool-dram-issue` — command list + compact expansion
`args.commands` is a non-empty list of primitive or compact commands. Primitives:
`RD`, `WR`, `WAIT`. Compact forms are expanded **server-side into the exact
primitive sequence** before execution, so budget and disturbance accounting run on
the true expanded event count:
- `{op: RD|WR|WAIT, repeat|count: N}` — issue the primitive N times; an absent
  count defaults to 1.
- `{op: HAMMER, rows: [...], pairs|count|repeat: N}` — N sweeps of one RD to each
  listed row (the canonical double-sided hammer for two rows). Bit-for-bit
  equivalent at the worker to writing every alternating RD by hand. The sweep
  count is **required** — HAMMER with no `pairs`/`count`/`repeat` fails
  `BAD_SCHEMA` rather than silently expanding to nothing, so a policy cannot
  spend a turn on a hammer that never happened. An explicit `pairs: 0` is a legal
  no-op.
Expansion beyond `MAX_ISSUE_ACTIVATIONS` (2,000,000) fails `ILLEGAL_COMMAND`.
`ACT`/`PRE`/`REF`/`RFM` are **controller-generated, not policy-issuable** — the
worker rejects them with `ILLEGAL_COMMAND`. They appear only in the issued-event
stream, where they drive disturbance accounting.
Defined in: `rowhammer_env/phase2_env.py` (`expand_commands`, `_issue`),
`cpp/simulator_service/ramulator_worker.cpp` (`issue`).

#### `@spec:tool-script-run` — brokered restricted-Python path
`args = {language: "python-rh-sdk", code, timeout_ms}`. Runs policy Python in an
OS-isolated child; its `rh.info/read/write/issue/finish` calls are brokered back
through the *same* `env.step` tool surface (never a bypass). Only logical addresses
are available to scripts. Success is re-checked from trusted state after the script.
Defined in: `rowhammer_env/phase5_env.py` (`_script`),
`rowhammer_env/script_sandbox.py` (`RestrictedScriptBroker`), `sdk/rh_sdk/`.

#### `@spec:tool-episode-finish`
Declares completion; returns final trusted success/reward and ends the episode.
Defined in: `rowhammer_env/phase5_env.py` (`step`), `rowhammer_env/phase2_env.py` (`step`).

#### `@spec:action-address-forms` — address resolution + disclosure gate
Three forms: `{kind:logical, addr}`, `{kind:physical, channel,rank,bankgroup,bank,
row,column}`, `{kind:handle, id}`. A bare int is shorthand for a logical address
(needed for `HAMMER.rows`). The task's `mapping`/`victim` disclosure decides which
forms are accepted; a hidden form fails closed with `ADDRESS_NOT_DISCLOSED`.
`physical` decodes through the geometry-derived mapper; `handle` resolves through
the per-episode table. Physical coordinates are never recoverable from error text,
timing outside the model, logs, trace ids, or handle names.
Defined in: `rowhammer_env/tasks/disclosure.py` (`AddressResolver`, `Disclosure`),
`rowhammer_env/tools/addressing.py` (`AddressMapper`).

#### `@spec:error-codes` — stable error set
The stable, policy-visible error codes are exactly:
`BAD_SCHEMA`, `UNSUPPORTED_TOOL`, `UNAVAILABLE_CAPABILITY`, `BUDGET_EXCEEDED`,
`ADDRESS_NOT_DISCLOSED`, `ILLEGAL_COMMAND`, `QUEUE_FULL`, `SANDBOX_VIOLATION`,
`SCRIPT_TIMEOUT`, `PROFILE_REJECTED`, `INTERNAL_SIMULATOR_ERROR`.
Transport-layer errors must be mapped into this set, never surfaced raw.
Defined in: `rowhammer_env/phase2_env.py` (`_error`),
`rowhammer_env/tools/addressing.py` (`AddressError`),
`rowhammer_env/script_sandbox.py` (`ScriptError` subclasses),
`cpp/simulator_service/ramulator_worker.cpp` (`error_json`).

### Simulator backend

#### `@spec:env-worker-protocol` — Ramulator worker RPC
A per-episode `ramulator_worker` subprocess speaks newline-delimited requests over
stdio: `INFO`, `READ id addr len [STORED]`, `WRITE id addr hex`, `ISSUE id op ...`,
`DECODE id linear`, `ENCODE id c0 c1 … cN`, `QUIT`. Each response is one JSON line
(`ok`, `cycle`, `last_action`, `public_counters`, `request`, `events`, optional
`data_hex`/`addr_vec`/`linear`/`geometry`). `DECODE` and `ENCODE` are the two
directions of the same map — coordinates in DRAMSpec level order, one per level,
channel first and column last — and both are side-effect-free (no tick, no drain, no
counter change). `READ … STORED` returns raw functional-memory bytes without
submitting a frontend request, ticking, draining events, or changing counters; it
is used only by trusted reward evaluation, which applies the disturbance overlay
server-side. `READ … STORED`, `DECODE`, and `ENCODE` are **server-internal only** and
are never in the policy tool surface. A fresh worker = a fresh episode (sparse
memory overlay reset, empty issued-event buffer).
Defined in: `rowhammer_env/worker_protocol.py` (`WorkerClient`, `WorkerRequest`),
`cpp/simulator_service/ramulator_worker.cpp`.

#### `@spec:env-issued-event-stream` — real post-schedule events
Disturbance and counters are driven by the *actual issued* DRAM events Ramulator
emits after scheduling (via the `IssuedEventRecorder` plugin), not by the requested
command order. Each event carries `op`, `clk`, decoded coordinates, `type_id`,
`row_hit`. Rejected/undelivered commands produce no events and cannot change state.
Defined in: `cpp/ramulator_extensions/issued_event_recorder.{h,cpp}`,
`rowhammer_env/phase4_env.py` (`_from_worker`), `rowhammer_env/disturbance.py` (`consume`).

#### `@spec:env-geometry` — geometry derivation
`Geometry` is built from the worker's `INFO` (standard, `tx_bytes`, `prefetch`,
level names/sizes, `command_names`). It derives the linear **row stride** (bytes
between physically adjacent rows) from the real RoBaRaCoCh layout, for any standard,
and the **row span** (bytes one row occupies inside its own bank — the
prefetch-adjusted Column field). These are not the same number and are not
interchangeable: the stride steps over every bank/bankgroup/rank at a row index, so
it measures "the next row along" while the span measures "still inside this row"
(131072 vs 8192 on the admitted DDR4 geometry). It also exposes a `public_block()` (row stride + row/bank/bankgroup counts + standard)
that is identical across every episode of a profile and leaks nothing about the
target. `command_names` is the standard's whole `DRAMSpec` command vocabulary — the
closed set of `op` names the issued-event stream can contain. It is **not** part of
`public_block()`: it is an internal integrity input, not policy-facing. A worker
`INFO` without it is rejected at `_fetch_geometry`, so the vocabulary check below can
never be silently skipped.
Defined in: `rowhammer_env/geometry.py`, `cpp/simulator_service/ramulator_worker.cpp`
(`info`), `rowhammer_env/phase4_env.py` (`_fetch_geometry`).

#### `@spec:env-address-mapper` — physical↔linear projection
`AddressMapper` reproduces Ramulator's RoBaRaCoCh decode/encode from the reported
geometry (single-channel only; fails closed otherwise). Its round-trip matches the
worker's own `addr_vec`. Used for `physical`-form addressing and for exact-victim
disclosure. It reproduces **only** the public RoBaRaCoCh mapper.
Defined in: `rowhammer_env/tools/addressing.py` (`AddressMapper`).

#### `@spec:env-secret-mapper` — per-episode secret row→bank mapping
Discovery families run the authored `RoBaRaCoChRowXOR` worker mapper (RoBaRaCoCh
decode plus a Row→Bank XOR with a seedable `xor_offset`), chosen deterministically
from `(task_id, seed)` and **never disclosed** (not in any observation, error,
handle, digest, or the public geometry). Under it, `victim ± row_stride` lands in a
*different* bank, so adjacency is not computable from a numeric address and must be
found via the timing channel. The Python projection is disabled for secret mappers
(`physical` disclosure fails closed); the compiler builds candidate sets via the
worker `DECODE` op instead, and the disturbance model anchors victim flips via the
worker `ENCODE` op (`@spec:sim-exposure-flip`), so a flip is readable at the address
the task disclosed rather than at an arithmetic guess.
Defined in: `rowhammer_env/mappers.py`, `cpp/ramulator_extensions/row_xor_addr_mapper.cpp`,
`rowhammer_env/phase5_env.py` (`_active_mapper`), `docs/adr-0004-secret-address-mapping.md`.

### Disturbance model (what is simulated)

Claim level: **cycle-level DRAM-system simulation with empirically calibrated
statistical read-disturbance**. Not transistor-level; not exact replay of any
specific commercial module.

#### `@spec:sim-disturbance-engine` — event → flip overlay
`DisturbanceEngine.consume(events, request)` folds one worker response into state.
Every issued command is dispatched by `classify_command` into exactly one class:

| class | commands | effect |
|---|---|---|
| `hammer` | `ACT` | drives exposure and the oracle counter |
| `close` | `PRE*`, `RDA`, `WRA` | settles the RowPress dwell of every open row in scope |
| `refresh` | `REF*`, `RFM*` | decays exposure over the refresh window |
| `inert` | `RD`, `WR`, `VRR` | no direct disturbance effect |

`RDA`/`WRA` are closes because Ramulator runs `PREpb::action` as their own action —
no `PRE` event marks the close. `VRR` is inert because its victim-row refresh is
modelled from the ACT counter (`@spec:mitigation-oracle`) and it is targeted rather
than part of the JEDEC auto-refresh window, so folding it into `refresh` would
wrongly advance the window counter.

Because dispatch is by command *name*, the name set must be closed: the engine
validates the worker-published `command_names` (`@spec:env-geometry`) at
construction and **fails closed** on any command it cannot classify, rather than
silently dropping it. A `WR` request also restores the cells it overwrote.
`apply(addr, data)` overlays committed flips onto returned bytes. All keyed by
*decoded* physical `(channel,rank,bankgroup,bank,row)`.
Defined in: `rowhammer_env/disturbance.py` (`DisturbanceEngine`, `classify_command`,
`unclassified_commands`, `AUTO_PRECHARGE_OPS`, `INERT_OPS`).

#### `@spec:sim-latent-vulnerability` — per-row latent state, sampled once
Each victim row's thresholds (double-sided `hcfirst`, single-sided `hcfirst`),
flip direction, and first-flip bit are sampled **once at episode start** from the
profile's hierarchical lognormal (`log N = mu + module_offset(σ_between) +
row_eps(σ_within)`) and never resampled per access. Stratum (single/double ×
data pattern) is inferred from the issued ACT neighbourhood and the victim region's
written pattern.
Defined in: `rowhammer_env/disturbance.py` (`Victim`, `_victim`, `_sample_threshold`,
`_module_offset`, `_sample_first_bit`).

#### `@spec:sim-exposure-flip` — exposure accumulation and flip transition
An ACT exposes the standard's blast neighbourhood (±1, optional ±2 half-double).
Double-sided exposure `= min(left,right)*2 + bonus` vs the double threshold;
single-sided `= max(left,right) + bonus` vs the single threshold. Crossing the
threshold flips ≥1 cell; multiplicity grows with exposure/threshold per the profile
`multiplicity` curve, capped at `MAX_FLIPPED_BITS_PER_ROW` (64). Flips persist until
overwritten or refresh-restored. Flip positions are deterministic from the seed;
cell 0 is always column 0 / `first_bit`.
A flip's recorded linear address is **the victim row's own column-0 base plus an
offset inside `[0, row_span)`**, so every recorded address decodes back to the
victim's `(channel, rank, bankgroup, bank, row)` key. The anchor is obtained by
encoding that decoded key (with column 0) through the **active mapper**, via the
engine's `row_encoder` — the worker `ENCODE` op in the live env. It is never derived
arithmetically from the activating access: `aggressor ± d * row_stride` holds only
for the public RoBaRaCoCh mapper and lands in a *different bank* under
`@spec:env-secret-mapper`, which would file a real flip at an address that reads
clean and leave a phantom one where nothing flipped. The anchor is therefore
independent of the intra-row offset and of the address the aggressor access carried.
A victim row outside the device's row range is skipped rather than encoded.
The `[0, row_span)` offset assumes the mapper keeps Column as one contiguous field
just above the transaction offset, which both admitted mappers (`RoBaRaCoCh` and
`RoBaRaCoChRowXOR`) do; a mapper that splits the Column field would need each cell's
address encoded individually, not just the row's base.
Defined in: `rowhammer_env/disturbance.py` (`_hammer`, `_row_addr`, `_maybe_flip`,
`_flip`, `_multiplicity_bits`, `_flip_positions`), `rowhammer_env/phase4_env.py`
(`_encode`), `cpp/simulator_service/ramulator_worker.cpp` (`encode`),
`cpp/ramulator_extensions/issued_event_recorder.cpp` (`AddrInverter`).

#### `@spec:sim-rowpress` — open-row dwell
A row held open longer than `ROWPRESS_DWELL_NOMINAL` accrues extra effective
hammers, scaling to the profile's `rowhammer_to_rowpress_hc_reduction` at
`ROWPRESS_DWELL_SATURATION`. Applied only for profiles that characterise RowPress;
ordinary back-to-back traffic accrues no bonus.

A row's dwell is the span from its `ACT` to the command that actually closes it —
any command in the `close` class of `@spec:sim-disturbance-engine` (the per-bank
`PREpb`; the rank-scoped `PREab` that precedes every auto-refresh, whose decoded
bankgroup/bank of -1 means "every bank in this rank"; an auto-precharge `RDA`/`WRA`)
— or a conflicting `ACT` in the same bank. Dwell is therefore bounded by the refresh
interval: an idle `WAIT` between two activations does not extend the preceding one's
dwell, and cannot buy the saturated bonus.
Defined in: `rowhammer_env/disturbance.py` (`_close_bank`, `_settle_dwell`,
`_rowpress_factor`).

#### `@spec:sim-refresh-decay` — refresh window decay
Ramulator auto-refresh events are ground truth. At each JEDEC refresh-window
boundary (`refresh_commands_per_window`, 8192) the rank's accumulated exposure is
cleared, so a hammer too slow/spread is refreshed away and must re-accumulate; a
fast burst crossing threshold within one window still flips. Already-flipped cells
are **not** corrected by refresh.
Defined in: `rowhammer_env/disturbance.py` (`_refresh`).

#### `@spec:sim-known-target` — fixed calibrated reference row
The engine keeps one "known target" row with a fixed calibrated `hcfirst` and a
deterministic first flip (bit `first_bit`, direction from pattern), so the
deterministic fixtures stay reproducible. Every other row is fully profile-sampled.
The known row's full decoded key is pinned — channel/rank/bankgroup/bank as well as
row (all zero under the public mapper; the scrambled bank under a secret mapper, and
whatever rank the compiler's decode reports) — so the fixed threshold lands on the
same victim the success predicate reads.
Defined in: `rowhammer_env/disturbance.py` (`known_threshold`, `_known_target_key`,
`known_target_row`/`_channel`/`_rank`/`_bank`/`_bankgroup`/`known_first_bit`).

#### `@spec:sim-standard-model` — standard-generic parameters
`StandardModel.from_geometry` derives the blast neighbourhood, refresh-window
length, and Ramulator `dram.impl`/controller per standard from source-traceable
`StandardFacts` + the real geometry. Fails closed if the geometry's levels
contradict the standard's dimensions (e.g. HBM PseudoChannel). No statistical
parameter is pooled across standards.
Defined in: `rowhammer_env/standards.py`, `profile_builder/standards.py`.

#### `@spec:sim-profile-loading` — admitted, signed profiles only
Only profiles in `ADMITTED_PROFILES` load; the package is verified
(`verify_package`) and its id/validation checked. A profile with failed validation,
out-of-domain temperature, or advertised extrapolation is rejected
(`PROFILE_REJECTED` / `ValueError`). `ddr4_vts25_v1` is admitted; HBM2 is deferred.
Defined in: `rowhammer_env/profiles.py`, `rowhammer_env/disturbance.py` (`__init__`
domain/temperature/validation guards), `profile_builder/package/build.py`.

#### `@spec:sim-not-modeled` — explicit non-goals
The environment does **not** model: transistor-level physics; exact prediction for
arbitrary real modules; host physical memory, TLB/cache, or OS page mapping;
policy-issuable ACT/PRE/REF/RFM (controller-generated only); multi-channel address
projection; temperatures/timings outside the profile's admitted domain. These are
contract-level absences, not TODOs.
Defined in: this document; enforced across `disturbance.py`, `addressing.py`,
`ramulator_worker.cpp`.

### Tasks, disclosure, and difficulty

#### `@spec:task-config` — task config parse
A task config is either the `spec/SPEC.md §10` YAML shape or the `{"family": ...}`
shorthand. `TaskSpec.from_config` parses the geometry-independent parts: family,
disclosure, objective, mitigation, budgets, difficulty band, target hints. A config
with no family infers one from objective type + disclosure. Unknown families/bands/
mitigations fail closed (`TaskConfigError`).
Defined in: `rowhammer_env/tasks/compiler.py` (`TaskSpec`, `_derive_family`,
`_canonical_family`), `configs/tasks/*.yaml`, `spec/schemas/task.schema.json`.

#### `@spec:task-families` — the family set
The canonical families (objective + default disclosure + target kind) are the
`FAMILIES` table: `known_target_anybit` (alias `known_target`), `target_row`,
`target_cell`, `pattern_target`, `any_flip`, `hidden_target` (deprecated),
`hidden_adjacency`, `unknown_adjacency`, `bounded_sweep`, `mitigation_aware`,
`low_disclosure`, `profile_generalization`. Each maps to exactly one success
predicate. Discovery families (`bounded_sweep`, `hidden_adjacency`) set
`secret_mapping=True`.
Defined in: `rowhammer_env/tasks/compiler.py` (`FAMILIES`, `FamilyDef`),
`rowhammer_env/rewards/predicates.py` (`PREDICATES`).

#### `@spec:task-compiler` — per-episode compilation
`TaskSpec.compile(seed, geometry, decode)` samples the concrete target row/bit,
resolves budgets and difficulty, pins disclosure, builds the candidate window (for
discovery families, via the worker `DECODE` op against the true mapping), and
records the victim's decoded `(channel, rank, bankgroup, bank, row)` key
(`CompiledTask.target_row_key`) — from the worker `DECODE` op under a secret mapper,
and from the equivalent public projection (`AddressMapper`) otherwise, so no
coordinate of it is ever assumed. Which of the two applies is a property of the
**family**, not of the caller: a `secret_mapping` family is exactly the family the
env runs under a secret mapper, so compiling one without the worker `DECODE` op
fails closed with `TaskConfigError`. So does a geometry the public projection cannot
represent. Neither may yield a victim key no flip can match. Produces an immutable
`CompiledTask` that carries everything reward and disclosure read. Deterministic per
`(task_id, seed, manifest)`.
Defined in: `rowhammer_env/tasks/compiler.py` (`TaskSpec.compile`, `CompiledTask`),
`rowhammer_env/phase5_env.py` (`_disturbance_overrides`).

#### `@spec:task-difficulty-bands` — calibrated budgets
`easy`/`medium`/`hard` map to calibrated activation budgets (`BAND_ACTS`) and
expected reference-policy success windows (`BAND_WINDOW`); for discovery families
they also set the candidate-window size (`BAND_CANDIDATES`). Bands are contracts the
difficulty-calibration gates check against; the concrete numbers are calibration,
not spec, and may be retuned.
Defined in: `rowhammer_env/tasks/compiler.py` (`BAND_ACTS`, `BAND_WINDOW`,
`BAND_CANDIDATES`).

#### `@spec:task-candidates` — discovery candidate windows
For `bounded_sweep` (Tier 2a, opaque handles) and `hidden_adjacency` (Tier 2b,
numeric addresses), the compiler builds an N-candidate window against the *secret*
mapping: exactly two true aggressors (the victim's same-bank rows ±1) plus
same-bank-far and different-bank decoys, shuffled so list position leaks no role.
Tier 2b additionally camouflages proximity (adjacent-shell different-bank decoys),
so no disclosed address reveals the aggressor — only the timing channel + trusted
flip can.
Defined in: `rowhammer_env/tasks/compiler.py` (`_build_candidates`,
`_build_numeric_candidates`, `_bank_slot_addrs`, `Candidate`).

#### `@spec:disclosure-levels` — the five disclosure axes
`Disclosure` narrows what the policy sees along five independent axes:
`mapping` (physical | logical_only | opaque_handles) → accepted address forms;
`adjacency` (exact | candidate_set | hidden); `victim` (exact | logical_addr |
row_handle | cell_handle | hidden_until_finish); `profile` (public_profile_id |
family_only | hidden); `feedback` (full_trace | summarized_counts | reward_only).
Defined in: `rowhammer_env/tasks/disclosure.py` (`Disclosure`).

#### `@spec:disclosure-leakage-guard` — the single enforcement point
`Disclosure.project_feedback` / `project_trace` are the one place hidden physical
state is stripped from public output: trace coordinates are removed unless mapping
is physical; the trace is dropped entirely under `summarized_counts`/`reward_only`;
per-flip coordinates survive only when victim is exact *and* mapping is physical.
The disturbance engine always consumes *raw* decoded events; only policy-facing
feedback is projected. `timing_digest` is coordinate-free by construction and keyed
by policy-supplied tokens, so it is safe under `logical_only`.
Defined in: `rowhammer_env/tasks/disclosure.py` (`project_feedback`, `project_trace`,
`expose_*`), `rowhammer_env/phase4_env.py` (`_from_worker`),
`rowhammer_env/phase2_env.py` (`_TimingDigest`, `_digest_addr_key`, `_trace_disclosed`).

#### `@spec:disclosure-handles` — opaque handle table
Per-episode handle ids are `h_<sha256(seed:role)[:16]>` — deterministic for replay,
non-invertible, encoding no coordinate/address/threshold. Resolved server-side only;
an unknown handle fails `ADDRESS_NOT_DISCLOSED`.
Defined in: `rowhammer_env/tasks/disclosure.py` (`HandleTable`),
`rowhammer_env/phase5_env.py` (`_register_handles`).

#### `@spec:timing-channel` — bank-conflict timing digest
`dram.issue` returns a `timing_digest` (under `full_trace`) aggregating real issued
events across the expanded primitives: `acts_delta`, `cycles_delta`, first/last clk,
and per-address `{acts, hits, misses}` keyed by the token the policy supplied. The
same-vs-different-bank discriminator is `acts_delta` (whether an alternating access
forced a new ACT), **never** a policy RD's own `row_hit` (always true). This is the
load-bearing signal discovery families are designed around.
Defined in: `rowhammer_env/phase2_env.py` (`_TimingDigest`, `_issue`),
`spec/TIER2_DISCOVERY_PLAN.md`.

### Mitigations

#### `@spec:mitigation-capabilities` — admit one at a time, fail closed
Only mitigations in `ADMITTED` are advertised in `dram.info` and accepted; every
other known SPEC name (`para`, `graphene`, `blockhammer`, `twice`, `prac`, …)
fails closed with `UNAVAILABLE_CAPABILITY` and must **not** silently act as `none`.
Currently admitted: `none` (baseline refresh) and `oracle`.
Defined in: `rowhammer_env/mitigations.py` (`ADMITTED`, `require_admitted_mitigation`,
`public_mitigation_capabilities`, `worker_config_for_mitigation`).

#### `@spec:mitigation-oracle` — faithful OracleRH target-row refresh
The `oracle` mitigation is a port of Ramulator `OracleRH`: per-aggressor ACT
counter, victim-row refresh (exposure reset) at `tRH`, counters cleared on all-bank
refresh. `tRH` defaults to a conservative fraction of the known `hcfirst`;
overridable per task.
Defined in: `rowhammer_env/disturbance.py` (`_oracle_on_act`, `_refresh` oracle
branch), `third_party/ramulator2/.../oracle_rh.cpp`.

### Security / sandbox

#### `@spec:security-simulation-only` — host isolation
Simulation only: no host physical-address APIs, `/proc/pagemap`, `/dev/mem`,
`/dev/kmem`, `/dev/kvm`, huge-page discovery, cache-attack utilities, RDMA, or PCIe/
GPU handles. All policy-visible effects go through `step()` or the script broker
that itself calls `step()` — no path bypasses reward/budget/termination.
Defined in: `docs/threat_model.md`, `docs/adr-0003-os-script-sandbox.md`;
enforced by `rowhammer_env/script_sandbox.py`.

#### `@spec:sandbox-script` — restricted-Python execution
`RestrictedScriptBroker` runs policy Python under `unshare --net --user
--map-root-user` + `bwrap` (isolated net/ipc/pid/uts, read-only `/usr`, tmpfs
`/tmp`, cleared env), with `RLIMIT_CPU/AS/FSIZE/NOFILE/NPROC` caps. A child runner
AST-validates the code (no imports except `from rh_sdk import rh`, no dunder access,
whitelisted builtins/calls only) and brokers `rh.*` calls over JSON-line IPC back
to `env.step`. Violations/timeouts return `SANDBOX_VIOLATION`/`SCRIPT_TIMEOUT`/
`UNAVAILABLE_CAPABILITY`.
Defined in: `rowhammer_env/script_sandbox.py`.

#### `@spec:sandbox-attestation` — fail-closed self-test
The runtime is admitted only after a live differential attestation: host-fs blocked,
`/proc/self/pagemap` PFN zeroed, `/dev/mem` & `/dev/kvm` absent, network blocked,
*and* a positive control (`/usr/lib` readable). If the runtime is missing or any
probe fails, `script.run` is `UNAVAILABLE_CAPABILITY` — never a permissive fallback.
Defined in: `rowhammer_env/script_sandbox.py` (`SandboxRuntime.detect`, `attest`,
`PROBE_RUNNER`).

### Serving and transport

#### `@spec:server-transport` — OpenEnv HTTP/WS server
The env is served through the vendored OpenEnv HTTP transport. `make_env`/
`create_rowhammer_app` build a FastAPI app; each session gets a fresh
`RowHammerTaskEnv` (fresh worker episode). Config via `MAX_CONCURRENT_ENVS`,
`RH_SERVER_MODE` (`simulation`/`production`), `RH_TASK` (default task JSON).
Defined in: `rowhammer_env/server/app.py`, `rowhammer_env/openenv_source.py`.

#### `@spec:server-session-model` — stateful WS vs stateless HTTP
Episodes are **stateful only over the WebSocket `/ws` transport**
(`RowHammerClient`). HTTP `POST /reset` and `/step` are stateless (fresh env per
request) and cannot carry an episode. `RH_SERVER_MODE=production` drops the
stateless routes and exposes only `/ws`, `/health`, `/schema`, `/metadata`, `/mcp`.
An orchestrator may override the task per episode via `reset(task=...)` over WS.
Defined in: `rowhammer_env/server/app.py`, `rowhammer_env/client.py`.

### Training and evaluation harness

All training glue is torch/trl-free and host-testable; the torch/trl binding lives
in `scripts/train_grpo.py` (and `scripts/train_curriculum.py`).

#### `@spec:train-prompt` — disclosed observation → prompt
`build_messages` turns a disclosed reset observation into GRPO chat messages
(fixed `SYSTEM_PROMPT` + a task view of *only disclosed* fields). `reference_hints`
are gated by `HINT_LEVELS` (`full`/`geometry`/`none`); the dataset-build and rollout
prompts for one run must use the same level (the trainer matches prompts to tasks).
Defined in: `rowhammer_env/llm/grpo_env.py` (`build_messages`, `SYSTEM_PROMPT`,
`public_hints`, `set_hint_level`).

#### `@spec:train-completion-parse` — completion → tool calls
`parse_actions` tolerantly parses a completion (fenced/raw JSON, several shapes)
into an ordered `ToolCall` list; `summarize_actions` reports the *expanded*
activation count (the key diagnostic). An unparseable completion yields `[]` →
reward 0.
Defined in: `rowhammer_env/llm/grpo_env.py` (`parse_actions`, `_coerce_action`,
`summarize_actions`, `completion_text`).

#### `@spec:train-reward-eval` — trusted replay reward
Parsed tool calls are replayed through the *same* verified rollout path
(`ScriptedPolicy` → `run_episode`) and scored by the trusted episode reward
(`1.0` only on a real flip). Batched, concurrency-bounded; a broken episode scores
0, never crashes training. Nothing here can fabricate reward from model text.
Defined in: `rowhammer_env/llm/grpo_env.py` (`ScriptedPolicy`, `evaluate_rewards`,
`evaluate_item`), `rowhammer_env/llm/rollout.py`.

#### `@spec:train-multiturn-rollout` — turn-by-turn training loop
`run_training_episode[_local]` / `run_batched_training_episodes[_local]` drive a
turn-by-turn chat loop (assistant completion → parsed tool call → `env.step` →
rendered tool-result turn), trace-equivalent to the eval loop. Records the full
chat transcript alongside the trajectory as a `MultiTurnRollout`.
Defined in: `rowhammer_env/llm/multiturn_rollout.py`, `rowhammer_env/llm/rollout.py`.

#### `@spec:train-completion-mask` — assistant-span token mask
`build_masked_completion` tokenizes a full transcript and marks only assistant-turn
tokens trainable (each turn anchored independently, robust to the Qwen3
`enable_thinking` `<think></think>` quirk); `to_grpo_example` emits
`(prompt_ids, completion_ids, completion_mask, reward)`. `enable_thinking` must
match how the dataset prompt was baked.
Defined in: `rowhammer_env/llm/multiturn_rollout.py` (`build_masked_completion`,
`to_grpo_example`, `_as_token_ids`).

#### `@spec:train-curriculum` — ordered, gated curriculum
`load_curriculum` parses the `curriculum:` config block into ordered
`CurriculumStage`s and enforces non-decreasing difficulty
(`tier0 → tier2a → tier2b`, `easy → medium → hard`). Each stage carries a
`reference_min_success` gate (the P26 reference policy must solve that fraction
before the stage earns training time). This module only defines/orders stages; it
computes no reward.
Defined in: `rowhammer_env/llm/curriculum.py`, `configs/training/grpo_curriculum.yaml`.

#### `@spec:train-reward-shaping` — bounded, outcome-neutral, training-only
An optional auxiliary shaping term rewards each *decisive* bank-conflict probe
(a pairwise, repeated alternation whose real `timing_digest` lands cleanly in one
regime), computed only from disclosed timing, bounded in `[0,1]`, outcome-neutral
(same-bank and different-bank score equally). `validate_shaping_weight` fails closed
unless the weight is `>= 0` and **strictly below** `success_weight`, so shaping can
never rival one real flip. Training-only; eval/benchmark scoring uses the trusted
reward alone.
Defined in: `rowhammer_env/llm/shaping.py`.

#### `@spec:train-reference-policy` — deterministic DRAMA reference solver
`ReferenceProbePolicy` is a hand-written reference (labelled a fixture, never the
LLM policy) that solves the discovery families using only disclosed tools/signals:
classify each candidate by `timing_digest.acts_delta`, drop different-bank ones, then
double-side the same-bank survivors. `use_timing=False` is the differential control
(hammers everything → trips `BUDGET_EXCEEDED`), which is what makes the timing signal
load-bearing. Other policies (`CIHammerFixturePolicy`, `TrainableHammerPolicy`,
`OpenAICompatibleToolPolicy`, `ClaimSuccessFixturePolicy`) are fixtures/adapters.
Defined in: `rowhammer_env/llm/policies.py`.

#### `@spec:eval-metrics` — episode result + aggregation
`EpisodeResult.from_rollout` normalizes one rollout (family, difficulty, split,
success = reward > 0, steps, final cycle, budget); `summarize_episodes` aggregates
success rate / reward / budget efficiency, bucketed by family/difficulty/split.
Defined in: `rowhammer_env/observability/metrics.py`.

### Cross-cutting invariants

These must hold everywhere; a violation is a bug regardless of which component
introduced it.

#### `@spec:invariant-no-mock` — no mocks, fail closed
No production/CI/training/eval path may use a mock DRAM, placeholder flip
generator, synthetic calibration, no-op mitigation, permissive sandbox fallback,
canned reward, or compatibility shim. An unavailable feature is **absent from
capability discovery** and fails closed with a stable error code.
Defined in: enforced across the codebase; gate `scripts/verify_release.py`,
`scripts/verify_phase20.py`.

#### `@spec:invariant-trusted-reward` — reward from trusted state only
Success/reward derives only from committed simulator state (`DisturbanceEngine.flips`
/ `flipped_row_keys`), never from policy logs, stdout, submitted claims, or
`episode.finish` assertions.
Defined in: `rowhammer_env/rewards/predicates.py`, `rowhammer_env/phase5_env.py`.

#### `@spec:invariant-no-leakage` — hidden state never leaks
Hidden physical coordinates/addresses/thresholds/mapper-identity never reach the
policy through observations, feedback, errors, handle names, trace ids, or timing
outside the simulated model. The disclosure projection is the enforcement point.
Defined in: `rowhammer_env/tasks/disclosure.py`, `rowhammer_env/mappers.py`,
`rowhammer_env/phase2_env.py` (`_TimingDigest`).

#### `@spec:invariant-budget-honesty` — no over-budget credit
Activations the `acts` budget cannot pay for are never issued to the worker, so no
flip they would have caused is ever credited; budget counters are monotone
non-increasing within an episode. The same holds on the `tool_calls` axis: an action
with no tool call left to pay for it is refused before dispatch rather than executed
and charged afterwards, and no action at all runs once the episode is over. Both
guards are pre-dispatch checks in the env, so the invariant does not depend on the
driver honouring `done`.
Defined in: `rowhammer_env/phase5_env.py` (`step`, `_refuse`, `_charge`,
`_issue_acts_ceiling`), `rowhammer_env/phase2_env.py` (`_issue`, `_acts_issued`).

#### `@spec:invariant-determinism` — replayable episodes
An episode is deterministic given `(task_id, seed)` and the pinned source manifest/
profile: target sampling, thresholds, handles, secret-mapper choice, and candidate
windows are all seed-derived (sha256 of `task_id:seed:salt`).
Defined in: `rowhammer_env/tasks/compiler.py` (`_rng`), `rowhammer_env/disturbance.py`
(seed-derived RNGs), `rowhammer_env/mappers.py` (`select_secret_mapper`),
`rowhammer_env/tasks/disclosure.py` (`HandleTable`).

---

## Provenance and admitted sources

Not code behavior, but a release contract: builds pin exact commits/hashes/licenses
for Ramulator 2.1, OpenEnv, and the read-disturbance data artifacts, and admit
profiles only after held-out statistical validation and signing. Ramulator is
pinned to v2.1.0 (`38c51d40`). `ddr4_vts25_v1` (from the VTS25 real-chip data) is
admitted; HBM2 is deferred until its license/hashes/validation resolve.
Defined in: `SOURCE_MANIFEST.yaml`, `spec/SBOM.md`, `spec/SHA256SUMS.txt`,
`profile_builder/` (ingest/fit/validate/trust/package).

---

## Open / unsettled

Deliberately **not** tagged as contracts — in flux, aspirational, or advisory:

- **Difficulty / budget numbers.** `BAND_ACTS`, `BAND_WINDOW`, `BAND_CANDIDATES`,
  discovery hammer/probe pair counts, and `tRH` defaults are calibration values,
  expected to be retuned. The *existence* of graded bands is spec; the numbers are not.
- **Reward shaping in use.** `@spec:train-reward-shaping` fixes the envelope (bounded,
  outcome-neutral, sub-success, training-only). Whether/at what weight shaping is
  actually enabled in a given run is an experiment config, not spec.
- **Additional mitigations.** `mitigations.py` names an `ADMISSION_ORDER`
  (`para`, `graphene`, `twice`, `blockhammer`, `prac`, `hydra`, `rrs`, `aqua`, `rfm`)
  and aliases, but only `none`/`oracle` are admitted. The rest are roadmap, currently
  fail-closed. See `docs/mitigations-roadmap.md`.
- **Additional standards / profiles.** The engine is standard-generic and
  `standards.py` lists DDR5/HBM2 adapters, but only DDR4 (`ddr4_vts25_v1`) has an
  admitted profile. HBM2 is explicitly deferred.
- **`hidden_target` family.** Marked deprecated in `FAMILIES` (kept working for
  legacy leakage/regression tests); superseded by `hidden_adjacency`. Do not build
  new tasks on it.
- **P29 release-readiness restructure.** `spec/P29_*.md` and
  `spec/PRE_RELEASE_RESTRUCTURE.md` describe a de-phase-naming restructure that is
  **spec'd but not executed** — the `phaseN_env.py` / `verify_phaseN.py` names are
  still the reality. Treat the phase-rename as future work, not current architecture.
- **Training scale-out.** The batched multi-turn rollout paths (vLLM/HF batch
  generation) exist and are host-tested, but full GRPO training runs are
  instance-side and not covered by the host CI gates.

---

## Drift

Places where code and the design bundle / docs / comments currently disagree.
Flagged, not resolved — a human decides which side is authoritative.

3. **`note_write` has no production caller, so the written-pattern half of the latent
   model is unreachable.** `@spec:tool-dram-write` says writes "record the row's data
   pattern (for stratum selection)" and names `note_write` under **Defined in**, but
   nothing outside tests calls it: `phase4_env._from_worker` calls `consume` and
   `apply` only. `_row_pattern` is therefore always empty, `_victim_pattern` always
   returns `all_zeros`, and every victim gets `direction = "0->1"`. The profile's
   `single|all_ones` / `double|all_ones` strata and its `direction.bias_strength` are
   dead, so `@spec:sim-latent-vulnerability`'s "single/double × data pattern" is in
   practice "single/double × constant". Wiring it in is decided (audit decision C)
   but remains separate work; the stored-byte reward prerequisite that previously
   blocked it is now resolved by `@spec:rl-reward`.
   Files: `rowhammer_env/disturbance.py` (`note_write`) vs `SPEC.md`
   (`@spec:tool-dram-write`, `@spec:sim-latent-vulnerability`).

4. **The oracle port clears its counters per refresh *window*, not per all-bank
   refresh.** `@spec:mitigation-oracle` claims a faithful `OracleRH` port with
   "counters cleared on all-bank refresh"; `oracle_rh.cpp` clears `m_table` on every
   `REFab`, while `disturbance._refresh` clears them below the window-boundary early
   return, i.e. every 8192nd. The port is materially *more* protective than the plugin
   it claims to port. Left as-is on purpose (audit decision D): the one-line hoist
   would make `oracle` close to a no-op at the current `tRH` default and move the
   phase-7 protection fixture, which is a `tRH` calibration decision, and `tRH`
   defaults are still under "Open / unsettled". `mitigation: none` is the default and
   no trained task enables `oracle`.
   Files: `rowhammer_env/disturbance.py` (`_refresh`) vs
   `third_party/ramulator2/.../plugin/impl/oracle_rh.cpp`.

4. **A written-over cell can never flip again.** `@spec:tool-dram-read` says reads
   return bytes "with any disturbance flips applied", but `restore` pops overwritten
   cells from `flips` while leaving `victim.flipped_bits` untouched, and
   `_flip_positions` never re-emits a cell index below it — so one `dram.write`
   permanently prevents that cell from ever flipping again, which is unphysical. On
   the overlay rather than on reward (`flipped_row_keys` is mapper-agnostic, so the
   trusted success predicates are unaffected).
   (The other half of this entry — victim addresses derived arithmetically as
   `aggressor ± d * row_stride`, wrong under a secret mapper — is **resolved**: the
   worker now publishes an `ENCODE` op and the engine anchors every victim at its own
   column-0 address. See `@spec:sim-exposure-flip`.)
   Files: `rowhammer_env/disturbance.py` (`restore`, `_flip`, `_flip_positions`) vs
   `SPEC.md` (`@spec:tool-dram-read`).

---

## Tag index

`@spec:rl-action-space`, `@spec:rl-observation-reset`, `@spec:rl-observation-step`,
`@spec:rl-reward`, `@spec:rl-budgets`, `@spec:rl-episode-termination`,
`@spec:tool-dram-info`, `@spec:tool-dram-read`, `@spec:tool-dram-write`,
`@spec:tool-dram-issue`, `@spec:tool-script-run`, `@spec:tool-episode-finish`,
`@spec:action-address-forms`, `@spec:error-codes`,
`@spec:env-worker-protocol`, `@spec:env-issued-event-stream`, `@spec:env-geometry`,
`@spec:env-address-mapper`, `@spec:env-secret-mapper`,
`@spec:sim-disturbance-engine`, `@spec:sim-latent-vulnerability`,
`@spec:sim-exposure-flip`, `@spec:sim-rowpress`, `@spec:sim-refresh-decay`,
`@spec:sim-known-target`, `@spec:sim-standard-model`, `@spec:sim-profile-loading`,
`@spec:sim-not-modeled`,
`@spec:task-config`, `@spec:task-families`, `@spec:task-compiler`,
`@spec:task-difficulty-bands`, `@spec:task-candidates`,
`@spec:disclosure-levels`, `@spec:disclosure-leakage-guard`,
`@spec:disclosure-handles`, `@spec:timing-channel`,
`@spec:mitigation-capabilities`, `@spec:mitigation-oracle`,
`@spec:security-simulation-only`, `@spec:sandbox-script`, `@spec:sandbox-attestation`,
`@spec:server-transport`, `@spec:server-session-model`,
`@spec:train-prompt`, `@spec:train-completion-parse`, `@spec:train-reward-eval`,
`@spec:train-multiturn-rollout`, `@spec:train-completion-mask`,
`@spec:train-curriculum`, `@spec:train-reward-shaping`, `@spec:train-reference-policy`,
`@spec:eval-metrics`,
`@spec:invariant-no-mock`, `@spec:invariant-trusted-reward`,
`@spec:invariant-no-leakage`, `@spec:invariant-budget-honesty`,
`@spec:invariant-determinism`.
