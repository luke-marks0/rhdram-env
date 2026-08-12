# Audit: `rowhammer_env/phase2_env.py` timing digest

## Scope

Audited `_TimingDigest`, `_digest_addr_key`, and their direct execution,
projection, budget, worker-event, and training-shaping paths against
`@spec:disclosure-leakage-guard`, `@spec:timing-channel`,
`@spec:env-issued-event-stream`, `@spec:rl-budgets`, and the cross-cutting
invariants. None of the findings below is listed in `SPEC.md`'s Drift section.

## Findings

### High — A late invalid primitive leaves committed simulator state but discards the digest and ACT charge

- **Location:** `rowhammer_env/phase2_env.py:355-383` validates/resolves each
  primitive only immediately before sending it, while
  `rowhammer_env/phase2_env.py:304-307` replaces a later validation failure with a
  fresh error observation. The already-issued prefix has been consumed at
  `rowhammer_env/phase4_env.py:177-196`; `rowhammer_env/phase5_env.py:297-303`
  then sees no public counters and does not debit its ACTs.
- **Concrete scenario:** reset an `unknown_adjacency` task with `full_trace` and
  budgets `acts: 10`; issue two commands in one action: first an RD to a valid
  candidate handle, then an RD to an unknown handle. The first RD completes and
  produces one ACT. The returned observation is
  `ADDRESS_NOT_DISCLOSED` with `feedback == {}` and
  `public_counters == {}`, while `env._public_counters["acts"] == 1` and the
  disturbance engine has consumed that ACT. The returned
  `budget_remaining["acts"]` is still 10. A longer valid hammer prefix can likewise
  add exposure or reach a real flip before a bad final address/data field causes
  the action to be reported only as rejected.
- **Why wrong:** `spec/SPEC.md` section 8 requires illegal timing/address requests
  not to update disturbance state, and `@spec:timing-channel` says the digest
  aggregates the real issued events of the expanded issue. Here the issue mutates
  state but exposes neither its events/digest nor its accepted count, and
  `@spec:rl-budgets` does not charge the true ACT delta. Reward remains derived
  from trusted simulator state, so this is not a policy-claim/trusted-reward
  violation; it is a partial-execution and budget-accounting violation.
- **Minimal fix:** use a two-phase `_issue`: first expand and fully validate every
  primitive (address resolution, WAIT cycles, and WR base64/size) into immutable
  prepared requests and leak-safe digest keys; only then dispatch any request. If
  a runtime worker failure can still occur after a committed prefix, return the
  aggregate observation/counters/digest for that prefix together with the stable
  error so charging remains honest.

### Medium — Invalid feedback modes fail open and disclose the timing channel

- **Location:** `rowhammer_env/phase2_env.py:424-433`, mirrored by
  `rowhammer_env/tasks/disclosure.py:88-106`; invalid values enter through
  `Disclosure.from_config()` at `rowhammer_env/tasks/disclosure.py:28-37` without
  enum validation.
- **Concrete scenario:** configure `hidden_adjacency` as normal except misspell
  `feedback: reward_only` as `feedback: reward-only`. Reset accepts the task.
  Because `_trace_disclosed()` treats every value other than the two known hidden
  modes as disclosed, `dram.issue` returns `timing_digest` and the trace projection
  returns events. The load-bearing same-bank discriminator is therefore visible
  even though the requested disclosure mode was not `full_trace`.
- **Why wrong:** `@spec:disclosure-levels` is a closed enum, and
  `@spec:timing-channel` permits the digest only under `full_trace`. Invalid
  configuration must fail closed, not select the most informative path. This is a
  disclosure-control violation even though the digest itself contains no decoded
  coordinates.
- **Minimal fix:** validate every disclosure axis when the task is parsed and
  reject unknown values as `TaskConfigError`/`BAD_SCHEMA`. Independently make both
  trace and digest admission use `feedback == "full_trace"`, so future validation
  regressions remain fail-closed.

### Medium — The claimed bounded digest has an unbounded policy-controlled address map

- **Location:** `_TimingDigest.absorb()` at
  `rowhammer_env/phase2_env.py:68-85` creates one permanent bucket for every
  distinct key; `as_dict()` at lines 87-94 returns the entire map. Only the raw
  trace is capped at `rowhammer_env/phase2_env.py:397-398`.
- **Concrete scenario:** submit
  `{"op":"HAMMER", "rows":[0,1,...,99999], "pairs":1}` to a `full_trace`
  episode without a restrictive ACT budget. This is below
  `MAX_ISSUE_ACTIVATIONS`, but `per_addr_hits` contains 100,000 entries. The
  admitted maximum is two million entries, yielding a very large in-memory map
  and policy response despite the class and P22 contract calling the digest
  bounded.
- **Why wrong:** bounding `trace_tail` does not bound the observation when
  `per_addr_hits` grows linearly with attacker-controlled unique addresses. This
  is an avoidable memory, serialization, transport, and prompt-size denial of
  service on oversized but admitted input.
- **Minimal fix:** define a small, documented maximum number of distinct digest
  address tokens and reject an issue exceeding it before dispatch with a stable
  schema/command error. Do not silently merge excess addresses because downstream
  probe shaping relies on the exact number of address buckets.

### Low — Physical-address buckets are keyed by a derived linear address, not the supplied token

- **Location:** fallback in `_digest_addr_key()` at
  `rowhammer_env/phase2_env.py:443-452`.
- **Concrete scenario:** in the full-disclosure disturbance environment, issue an
  RD to physical token
  `{channel:0, rank:0, bankgroup:0, bank:0, row:200, column:0}`. The returned
  `per_addr_hits` key is `"26214400"`, the resolved linear address. It is not a
  canonical form of the physical token the policy supplied. Mixing the equivalent
  logical and physical forms in one issue also collapses both tokens into that one
  bucket.
- **Why wrong:** both `@spec:disclosure-leakage-guard` and
  `@spec:timing-channel` say buckets are keyed by policy-supplied tokens. The
  derived value is safe under physical disclosure because it is publicly
  computable, so this is not a hidden-address leak, but it is a contract divergence
  and loses token identity.
- **Minimal fix:** assign each admitted form a canonical, namespaced key derived
  only from the supplied token (for example `logical:<n>`, `handle:<id>`, and a
  stable serialization of the supplied physical coordinates). Use that same key
  during the up-front validation/preparation recommended above.

## Checks that passed

- For valid feedback modes, the digest is present only with `full_trace`; logical
  and handle tasks build it from already-projected events, so decoded coordinates
  do not enter any digest field. Handle buckets use the disclosed handle id and do
  not expose its resolved linear address.
- `acts_delta`, cycle totals, event clocks, and `{acts,hits,misses}` are derived
  from real worker replies. The apparent `misses` count on ACT/PRE events matches
  the explicit original design note that `row_hit == false` belongs to those
  controller events, while a policy RD itself reports a hit.
- `_TimingDigest` is local to one `_issue`, so no digest state survives between
  actions or episodes. It uses no randomness; ordering and values follow the
  deterministic worker event stream.
- The activation ceiling stops a budget-truncated issue before the next primitive
  is dispatched, and the digest accurately summarizes the paid prefix on that
  normal truncation path. No component-specific path was found where the digest
  itself fabricates success or credits a flip from policy claims.

## Verification performed

- `python -m unittest tests.test_probe_shaping tests.test_probe_signal tests.test_activation_budget tests.test_issue_expansion`
  — 44 tests passed with the built Ramulator worker.
- End-to-end reproduction confirmed the late-invalid-handle partial execution:
  the returned observation had empty feedback/counters and an unchanged ACT
  budget while the live worker counter had advanced by one.
- End-to-end physical-form reproduction confirmed the derived key
  `"26214400"` for physical row 200.

---

# Audit: secret address mapper

## Scope

Audited `rowhammer_env/mappers.py` and
`cpp/ramulator_extensions/row_xor_addr_mapper.cpp` in full, plus their direct
selection/configuration path in `phase4_env.py` and `phase5_env.py`, Ramulator's
`AddrMapperBase` and stock `RoBaRaCoCh` implementation, the worker `DECODE`/request
boundary, logical-address resolution, task compilation/candidate construction,
feedback projection, and the mapper integration tests. The governing contract is
`@spec:env-secret-mapper`, with `@spec:task-compiler`,
`@spec:disclosure-leakage-guard`, and the cross-cutting invariants. None of the
findings below is listed in `SPEC.md`'s Drift section.

## Findings

### High — A task disclosure override bypasses the secret-mapper guard and publishes decoded coordinates

- **Location:** `rowhammer_env/phase5_env.py:205-211` checks
  `fam.disclosure.mapping`, the immutable family *default*, instead of the active
  `self.spec.disclosure.mapping`. The overridden disclosure is subsequently
  installed at `rowhammer_env/phase5_env.py:218-221`, and
  `rowhammer_env/tasks/disclosure.py:89-95` returns raw coordinates whenever that
  override says `mapping: physical`.
- **Concrete scenario:** construct `RowHammerTaskEnv` with family
  `hidden_adjacency` and disclosure
  `{mapping: physical, adjacency: candidate_set, victim: logical_addr,
  profile: public_profile_id, feedback: full_trace}`, then reset with seed 7 and
  issue one RD to a disclosed candidate. Reset succeeds, advertises both
  `logical` and `physical` forms, and the returned trace contains the active secret
  mapper's decoded values, for example
  `{channel: 0, rank: 0, bankgroup: 2, bank: 3, row: 21248, column: 0}`.
  The same issue exists for an overridden `bounded_sweep` task. If `victim: exact`
  is also selected, reset instead reaches the later `_physical_target` exception,
  rather than rejecting the task at admission with a stable error.
- **Why wrong:** `@spec:env-secret-mapper` requires physical disclosure to fail
  closed under a secret mapper, and `@spec:invariant-no-leakage` forbids the hidden
  coordinates and mapper secret from reaching policy feedback or errors. Publishing
  each candidate's decoded bank/row eliminates the intended timing-only discovery
  problem. The task config is an orchestration input rather than a policy action,
  but it is still an admitted configuration that creates a policy-visible leak;
  the environment must reject an internally inconsistent contract.
- **Minimal fix:** validate the *effective* `TaskSpec.disclosure` when parsing or
  admitting every `secret_mapping` family and reject any mapping other than the
  permitted non-physical form before starting a worker. At minimum, change the
  guard to inspect `self.spec.disclosure.mapping` and catch `TaskConfigError` on
  reset so it becomes a stable `BAD_SCHEMA`; retaining a second defensive check
  before installing the resolver/feedback projection would keep the path fail
  closed if task parsing regresses.

### Medium — Derived mapper YAML publication is non-atomic across concurrent resets

- **Location:** `rowhammer_env/mappers.py:86-97` maps every identical mapper choice
  to the same output path, then opens that shared path with truncation and rewrites
  it directly. `rowhammer_env/phase4_env.py:62-74` immediately launches a worker
  from the returned path. No lock or atomic publication separates writers from
  readers.
- **Concrete scenario:** two concurrent `hidden_adjacency` resets select the same
  `xor_offset` (only six output paths exist). Reset A finishes writing; reset B
  opens the same file and truncates it; before B finishes its write, A's worker
  opens the empty/partial YAML. A then receives an initialization error instead of
  its deterministic episode. A process interruption between truncation and write
  leaves the same bad artifact for a reader already starting.
- **Why wrong:** the server advertises concurrent sessions, so a valid reset must
  not depend on filesystem scheduling. This breaks ordinary reset correctness and
  `@spec:invariant-determinism`: identical `(task_id, seed)` episodes can either
  start or fail based on another episode's timing.
- **Minimal fix:** serialize to a uniquely named temporary file in the same
  directory, flush/close it, then atomically replace the content-addressed target.
  Concurrent writers are safe because they publish identical bytes. Validate the
  loaded base shape before publication and never expose a partially written path.

### Medium — Negative and oversized logical addresses alias valid DRAM coordinates

- **Location:** `rowhammer_env/tasks/disclosure.py:157-176` converts a logical
  address to `int` without a device-range check; worker `parse_u64` at
  `cpp/simulator_service/ramulator_worker.cpp:72-75` accepts `-1` via `strtoull`
  and assigns it to signed `Addr_t`; and
  `cpp/ramulator_extensions/row_xor_addr_mapper.cpp:52-69` consumes only the mapped
  low bits, silently discarding high bits. The worker's `DECODE`, read, and write
  paths do not impose a capacity bound (`ramulator_worker.cpp:174-180,193-220`).
- **Concrete scenario:** on the admitted DDR4 geometry, `DECODE 0` and
  `DECODE 8589934592` (one byte past the 8 GiB mapped address space) both return
  `(channel,rank,bankgroup,bank,row,column) = (0,0,0,0,0,0)`.
  `DECODE -1` and `ISSUE RD -1` are also accepted and decode to the last mapped
  row/column. Functional bytes remain keyed by the original linear value, so two
  addresses can be different memory-overlay cells while Ramulator and disturbance
  accounting treat them as the same physical DRAM location.
- **Why wrong:** the mapper's documented bijection holds only inside the device
  domain, but that domain is never enforced. Oversized policy input therefore
  creates address collisions and splits functional memory from simulator state,
  violating ordinary decode/address correctness and the requirement to fail closed
  on oversized inputs. ACTs are still honestly charged and reward still reads
  trusted disturbance state, so this is not independently a reward or budget
  violation.
- **Minimal fix:** derive the mapped byte capacity from geometry and reject logical
  addresses unless the complete requested byte range lies in `[0, capacity)`.
  Enforce this once in policy-facing resolution and again at the worker boundary
  for `READ`, `WRITE`, `ISSUE`, and internal `DECODE`; parse signedness/range
  explicitly rather than assigning `strtoull` results to signed `Addr_t`.

### Low — Mapper selection returns mutable process-global parameter state

- **Location:** `rowhammer_env/mappers.py:42-44` stores mutable dicts in
  `SECRET_MAPPERS`; `select_secret_mapper` at lines 57-60 returns the selected
  tuple and dict by reference.
- **Concrete scenario:** after
  `impl, params = select_secret_mapper("audit", 7)`, setting
  `params["xor_offset"] = 31` changes the entry held by `SECRET_MAPPERS`; a later
  call with the same or any input selecting that slot returns 31 rather than the
  seed-derived admitted value. An environment exposes the same object as
  `_active_mapper_params`, so accidental trusted-side mutation can contaminate
  later episodes in the process.
- **Why wrong:** a supposedly pure seed-derived selection depends on mutable state
  surviving between episodes, contrary to reset isolation and
  `@spec:invariant-determinism`. No current policy-facing path can mutate this
  private object, so the issue is robustness rather than a direct policy exploit.
- **Minimal fix:** store immutable mapper choices (for example, a frozen value
  object or tuple of parameter pairs) and return a fresh dict for each episode.

### Low — Oversized `xor_offset` values invoke an undefined C++ shift instead of failing closed

- **Location:** `cpp/ramulator_extensions/row_xor_addr_mapper.cpp:41-46` rejects
  only negative offsets; line 69 right-shifts a signed 32-bit `int` by the accepted
  offset. `worker_config_for_mapper` accepts an arbitrary parameter mapping and
  performs no corresponding validation (`rowhammer_env/mappers.py:63-97`).
- **Concrete scenario:** generating a mapper config with `xor_offset: 32` (or
  larger) passes mapper initialization, then decoding any address executes
  `row >> 32` on an `int`. C++ leaves that operation undefined, so the BankGroup
  decode can vary by compiler/build or fail unpredictably rather than rejecting
  the config.
- **Why wrong:** invalid/oversized mapper input must fail closed, and mapper output
  is part of deterministic episode state. The built-in `SECRET_MAPPERS` values are
  all safe, so shipped episodes do not currently reach this path.
- **Minimal fix:** validate `xor_offset` against the Row field's bit width during
  `init` (and validate the admitted parameter set in Python), then use an unsigned
  type for shifts/masks. Also reject a missing or zero-width Bank level if this
  implementation is ever admitted on a geometry other than the current DDR4 one,
  because its all-scattering property otherwise silently disappears.

## Checks that passed

- For the shipped discovery-family defaults, mapper selection is SHA-256-derived
  from `(task_id, seed)`, all admitted offsets preserve the raw Row field, and every
  tested `addr + row_stride` changes decoded bank identity. Selection varies across
  seeds and is stable when process-global state is not mutated.
- The C++ field slicing matches Ramulator's stock `RoBaRaCoCh` placement before the
  Row-to-Bank/BankGroup XOR. Within the admitted device address range the XOR is
  invertible given Row, and candidate construction uses the worker's active decoder
  rather than duplicating the secret function in Python.
- Under the canonical `logical_only` disclosure, decoded coordinates are removed
  from trace feedback, the mapper name/offset is absent from reset/info/step
  observations, `DECODE` is not a policy tool, and physical target projection is
  refused defensively.
- Mapper code has no route from policy claims to reward. Success continues to use
  decoded committed flip keys, and issued ACTs/cycles flow through the normal
  worker counters and budget guards; no mapper-specific trusted-reward or
  budget-honesty defect was found beyond the separately documented Drift issue for
  secret-mapper flip read-back.
- A fresh worker remains the episode-state boundary; no C++ mapper instance state
  was found to survive worker teardown.

## Questions

- Is `RoBaRaCoChRowXOR` intentionally restricted to the admitted DDR4 geometry, or
  is it meant to be standard-generic? On current DDR4, Bank has two bits and all
  configured offsets are valid. If it is meant to be generic, initialization
  should explicitly require a nonzero Bank width and exact power-of-two mapped
  levels rather than silently degrading the all-scattering contract.

## Verification performed

- `python -m unittest tests.test_secret_mapping tests.test_hidden_adjacency` — 32
  tests passed with the built Ramulator worker.
- End-to-end reproduction confirmed the disclosure-override leak: reset advertised
  physical addressing and one candidate RD returned raw decoded bank/row
  coordinates under the secret mapper.
- Direct worker probes confirmed `0` and `8589934592` decode to the same coordinate,
  and that `-1` is accepted by both `DECODE` and `ISSUE RD`.
- A process-local reproduction confirmed that mutating the dict returned by
  `select_secret_mapper` changes the value returned by later selections.

---

# Audit: geometry and public address projection

## Scope

Audited `rowhammer_env/tools/addressing.py` and
`rowhammer_env/geometry.py` in full against `@spec:env-address-mapper`,
`@spec:env-geometry`, and `@spec:action-address-forms`, plus their direct worker,
task-compiler, disclosure-projection, disturbance, and standard-adapter call paths.
The cross-cutting trusted-reward, no-leakage, budget-honesty, and determinism
invariants were checked explicitly. None of the findings below appears in
`SPEC.md`'s Drift section.

The out-of-domain aliasing of negative/oversized logical addresses is not repeated
here: it is already reported as **Medium — Negative and oversized logical
addresses alias valid DRAM coordinates** in the preceding secret-mapper audit.
`AddressMapper.decode()` deliberately exhibits the same truncation for values
outside its mapped bit width (`addressing.py:83-85`), so the domain must be enforced
at the policy/worker boundary described by that finding.

## Findings

### Medium — The hidden-coordinate scrubber fails open for geometry levels outside the DDR coordinate tuple

- **Location:** `rowhammer_env/tools/addressing.py:8-11` hard-codes
  `COORD_KEYS` to `(channel, rank, bankgroup, bank, row, column)`;
  `rowhammer_env/tasks/disclosure.py:89-95` removes only those names from a
  non-physical trace. The worker instead serializes every name published in the
  active `DRAMSpec.level_names` at
  `cpp/simulator_service/ramulator_worker.cpp:321-333`.
- **Concrete scenario:** the repository's real HBM2 worker reports the levels
  `Channel, PseudoChannel, BankGroup, Bank, Row, Column`. An RD whose decoded
  pseudo-channel is 1 emits an ACT containing `"pseudochannel": 1`. Projecting that
  event with `Disclosure(mapping="logical_only", feedback="full_trace")` removes
  channel/bankgroup/bank/row/column but returns
  `{..., "pseudochannel": 1, ...}` to the policy.
- **Why wrong:** `@spec:disclosure-leakage-guard` calls this projection the single
  enforcement point, and `@spec:invariant-no-leakage` forbids *hidden physical
  coordinates*, not only the six DDR-shaped names. The failure mode is a denylist:
  any present or future standard-specific coordinate silently becomes public.
  HBM2's empirical profile is currently deferred, so the shipped DDR4 task path
  does not reach this leak today; the failure is nevertheless concrete in the
  repository's real second-standard geometry and would activate on profile
  admission.
- **Minimal fix:** make hidden-mapping trace projection allowlist only the public,
  coordinate-free event fields (`op`, `clk`, `type_id`, `row_hit`) rather than
  trying to enumerate coordinate names. If coordinate enumeration remains
  necessary, derive it from the episode's validated `Geometry.level_names` and
  reject any unclassified event field instead of passing it through.

### Medium — Inconsistent level-name/size vectors are silently reinterpreted as a different device

- **Location:** `rowhammer_env/geometry.py:27-30` constructs `level_sizes` with
  `zip(names, sizes)` without checking equal lengths or unique names. The resulting
  dictionary then drives `row_stride`, `row_span`, `public_block`, and every
  `AddressMapper` field width/shift. `rowhammer_env/phase4_env.py:164-177` checks
  only that `command_names` is present before accepting the object.
- **Concrete scenario:** start from the admitted DDR4 INFO payload, accidentally
  omit `BankGroup` from `level_names`, but leave the six-element `level_sizes`
  vector unchanged. `Geometry` accepts it, silently pairs `Row` with size 4 and
  `Column` with size 65536, and drops the final size 1024. It reports
  `row_stride = 2,097,152` and `row_span = 524,288`; its mapper encodes `{row: 1}`
  as 2,097,152. The real DDR4 worker decodes that address as row 16, while the
  accepted Python mapper decodes it as row 1.
- **Why wrong:** `@spec:env-geometry` and `@spec:env-address-mapper` require the
  Python geometry/projection to reproduce the worker, and the geometry feeds target
  compilation, flip placement, reward row keys, and public disclosure. A malformed
  or version-skewed INFO response must fail closed; silently building a different
  topology can make a target impossible, attribute disturbance to the wrong row,
  and publish false geometry. Duplicate names produce a related overwrite without
  rejection.
- **Minimal fix:** validate INFO as one atomic schema before deriving anything:
  require equal non-empty vector lengths, case-insensitively unique names, Channel
  first, Column last, exactly one Row, required Bank for the current public block,
  positive exact-power-of-two sizes, and a prefetch no larger than/dividing Column.
  Have `_fetch_geometry` map any validation failure to a terminal stable simulator
  error rather than letting `KeyError`/`ValueError` escape reset.

### Low — Physical coordinates that are not JSON integers are silently coerced

- **Location:** `rowhammer_env/tools/addressing.py:111-117`; `_coord_int()` calls
  `int(value)` instead of checking the value's type. The broad runtime action model
  does not apply the nested JSON schema before this path, so this conversion is the
  effective validator.
- **Concrete scenario:** under physical disclosure,
  `{kind: "physical", row: 1.9}` and `{kind: "physical", row: true}` both execute
  as row 1; `{kind: "physical", row: "2"}` executes as row 2. Each should be
  malformed under `spec/schemas/action.schema.json`, whose coordinate fields are
  JSON integers, but no `BAD_SCHEMA` is returned.
- **Why wrong:** an invalid address is silently redirected to a different physical
  row. This violates the fail-closed malformed-coordinate behavior documented by
  `AddressError` and makes policy behavior depend on Python coercion rules rather
  than the action contract. It does not by itself leak hidden state or bypass
  activation accounting.
- **Minimal fix:** accept only `isinstance(value, int) and not isinstance(value,
  bool)` for each coordinate, then range-check it; reject every other type with
  `AddressError("BAD_SCHEMA", ...)`. Apply the same strict rule to logical
  addresses in `AddressResolver` so the two admitted numeric forms do not diverge.

## Checks that passed

- On the admitted DDR4 geometry and for in-range integral addresses, the computed
  field widths/shifts match Ramulator's stock `RoBaRaCoCh`; encode/decode round trips
  at row, bank, bankgroup, rank, and column boundaries, and the row stride is exactly
  131,072 bytes.
- `row_span` is correctly distinct from `row_stride` (8,192 versus 131,072 bytes on
  admitted DDR4), and the generic stride calculation correctly includes HBM2's
  PseudoChannel level. `public_block()` exposes only the contract's static
  standard/count/stride fields and excludes mapper identity, field ordering,
  shifts, command vocabulary, target state, and per-episode state.
- Multi-channel public projection, a missing Row level, a non-final Column level,
  and non-power-of-two mapped sizes fail closed. Physical forms are gated before
  encoding, so hidden tasks cannot use range-error messages to probe coordinates;
  the accepted error text contains only policy-supplied values and public geometry
  limits.
- These modules are pure after construction and use no randomness or mutable
  episode state. They cannot derive reward from policy claims and issue no worker
  commands, so no component-local trusted-reward, budget-honesty, reset-state, or
  determinism violation was found.

## Verification performed

- `python -m unittest tests.test_phase11.GeometryTests
  tests.test_phase12.AddressMapperTests tests.test_phase12.DisclosureTests
  tests.test_phase12.HandleTableTests tests.test_phase12.AddressResolverTests
  tests.test_phase15.GeometryStrideTests
  tests.test_discovery_geometry.GeometryBlockUnitTests` — 28 focused tests passed.
- A live HBM2 worker probe confirmed that an ACT event contains
  `pseudochannel: 1` and the logical/full-trace disclosure projection leaves it in
  the returned event.
- Direct reproductions confirmed the malformed vector is accepted with the wrong
  stride/decode and that `True`, `1.9`, and `"2"` are accepted as physical row
  coordinates.
- The broader geometry/addressing test selection could not complete because the
  current dirty worktree has an unrelated integration mismatch:
  `DisturbanceEngine.__init__()` requires `row_encoder`, while its existing direct
  constructors and `RowHammerDisturbanceEnv.reset()` do not supply it. The focused
  component tests above are unaffected; this audit did not modify that work.
