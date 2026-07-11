# ADR 0004 — Realistic secret address mapping for discovery families (P24)

Status: Accepted (IMPLEMENTATION_PLAN_V3 P24)
Supersedes the mapper choice in: `spec/IMPLEMENTATION_PLAN_V3.md` §0.3 / P24
task 1 (the `MOP4CLXOR` premise), `spec/TIER2_DISCOVERY_PLAN.md` §4.1.

## Context

The Tier 2b threat model (`spec/TIER2_DISCOVERY_PLAN.md` §4.1) requires that a
policy given a victim's **numeric** address cannot compute which candidate rows
are same-bank physical neighbours: physical adjacency must be reverse-engineered
through the bank-conflict timing channel (the DRAMA technique, Pessl et al.,
USENIX Security 2016), exactly as on real hardware where the controller's
bank-select function is undocumented. Tier 2a (`bounded_sweep`) wants the same
property so its underlying physics match hardware rather than leaning on a public
linear map where `victim ± row_stride` is *always* the adjacent aggressor.

`IMPLEMENTATION_PLAN_V3.md` proposed getting this "for free" by selecting, per
episode, among Ramulator v2.1.0's stock mappers — naming **`MOP4CLXOR`** as one
that "XORs row bits into the bank/bankgroup indices, so linearly-adjacent
addresses scatter across banks."

### The premise was false (verified against the code + the worker)

Reading `addr_mapper/impl/mop4clxor.cpp` and confirming with the new `DECODE` op
over the real worker:

- **`MOP4CLXOR` XORs *column* bits into the bank index**, not row bits — it is
  cache-line/bank interleaving for bank-level parallelism (that is what
  "MOP…CLXOR" means). `addr + row_stride` changes only the Row field, leaves the
  column unchanged, leaves the XOR mask unchanged → **the bank never changes.**
  Measured: `addr` vs `addr + row_stride` land in a different bank **0 / 200**
  times under `MOP4CLXOR` (and 0/200 under `RoBaRaCoCh`).
- No stock Ramulator mapper XORs Row→Bank: `RoBaRaCoCh` (pure low-bit slice),
  `MOP4CLXOR` (column→bank), `ChRaBaRoCo` (different *stride*, would break P21's
  mapper-independent geometry disclosure and still does not scatter banks),
  `RITAddrMapper` (a mitigation decorator), `PassThroughAddrMapper`.

So the stock-mapper plan cannot satisfy P25's acceptance criterion ("a control
that computes `victim ± row_stride` and hammers it **fails**"): under
`MOP4CLXOR` that control **succeeds**.

## Decision

**Author a real, source-cited Row→Bank XOR mapper** —
`RoBaRaCoChRowXOR` (`cpp/ramulator_extensions/row_xor_addr_mapper.cpp`),
registered into the worker binary alongside the `IssuedEventRecorder` plugin — and
select it (with a seedable `xor_offset`) as the per-episode secret for discovery
families.

- It decodes exactly like `RoBaRaCoCh` (Column at the LSB, then Rank..Row, **Row
  the most-significant field**), so the linear row stride is byte-identical
  (131072 for DDR4_8Gb_x8) and the publicly disclosed geometry (P21,
  `Geometry.public_block`) stays honest and mapper-independent.
- It then XORs a slice of the **Row** bits into the Bank (and, with the seedable
  offset, BankGroup) index. **Bit 0 of Row always folds into Bank**, so `addr`
  and `addr + row_stride` land in **different banks** — measured 300/300 for every
  admitted `xor_offset`. Physical adjacency is therefore not computable from the
  numeric address and must be found by timing.
- The map is a bijection (XOR by a function of Row is invertible given Row), so no
  two linear addresses collide — it is a valid address map, not a lossy hack.

### Why this is within SPEC §2 (no fabrication)

This is **address mapping**, a real controller function — not a fabricated
disturbance/physics term. Undocumented Row/high-bit → Bank XOR is exactly the
bank-select scrambling that AMD and Intel memory controllers apply and that the
DRAMA paper reverse-engineers. The disturbance engine is untouched: flips are
keyed on the recorder's **real decoded** `(bankgroup, bank, row)` from issued
events, so the physics are identical regardless of which mapper is active. The
authored mapper is source-cited here and covered by differential tests
(`tests/test_secret_mapping.py`).

### Mechanism

1. **Selection (`rowhammer_env/mappers.py`).** `select_secret_mapper(task_id,
   seed)` deterministically picks `RoBaRaCoChRowXOR` + an `xor_offset` from a small
   all-scattering set. `worker_config_for_mapper` writes a derived worker YAML that
   swaps only the `addr_mapper` node (geometry/timings/plugins untouched). The
   default public mapper reuses the base config verbatim, so **non-discovery
   families are byte-identical to before**. The choice is server-side only.
2. **`DECODE` op (`cpp/simulator_service/ramulator_worker.cpp` + the recorder
   plugin).** A side-effect-free `DECODE <id> <linear>` returns the true decoded
   coordinates under the *active* mapper — it does not tick the simulator, drain
   events, or touch counters. The `IssuedEventRecorder` publishes the decoder as a
   `std::function` over the controller's own `addr_mapper`, so no mapper logic is
   duplicated in Python. `DECODE` is **never** in `ALLOWED_TOOLS` and is
   unreachable from `step` — it is used only by the compiler.
3. **Candidate construction (`tasks/compiler.py`).** The candidate window is built
   against the true mapping via `DECODE`: the *raw* RoBaRaCoCh field positions are
   public geometry, so enumerating the raw bank slots at a fixed row and decoding
   each yields the decoded-bank → linear-address map (a bijection over the bank
   space) for that row, from which the true same-bank neighbours (aggressors),
   same-bank-far decoys, and different-bank decoys are selected.
4. **Decode-correct reward (`rewards/predicates.py`, `disturbance.py`).**
   `bounded_sweep` uses `_target_bankrow_flip`, which reads the trusted **decoded**
   victim key from `DisturbanceEngine.flipped_row_keys` — correct under any mapper
   — instead of `_target_row_flip`'s `addr // row_bytes`, which assumes the public
   linear layout. The engine's fixed known-target threshold is pinned to the
   victim's decoded `(bankgroup, bank)` so a found aggressor flips reliably even
   though the victim sits in a scrambled bank.
5. **Guards (P24 tasks 4/5).** Discovery families are `logical_only`, so the policy
   never receives a physical decoder; `_physical_target` and physical addressing
   fail closed (`ADDRESS_NOT_DISCLOSED` / `TaskConfigError`) under a non-projectable
   mapper. The active mapper id/params (including `xor_offset`) never appear in
   `dram.info`, reset metadata, errors, handle names, or the timing digest.

## Consequences

- P25's Tier 2b family (numeric victim address, secret adjacency) is now buildable
  on a mapper that genuinely defeats arithmetic adjacency; the `victim ± row_stride`
  control fails (measured 0.0 reward).
- The plan's stated P23↔P24 independence is intentionally **narrowed**:
  `bounded_sweep` (Tier 2a) now runs under the secret mapper by default (user
  decision, for realism), so its candidate construction and success predicate are
  decode-based rather than RoBaRaCoCh-arithmetic. Its P23 tests were rewritten to
  verify the split via `DECODE` (`tests/test_discovery_families.py`).
- Under a secret mapper, the disturbance stores each flip at an arithmetic linear
  address that no longer decodes to the victim row, so byte-accurate *read-back* of
  a disturbed victim byte is unsupported for discovery episodes. This is
  acceptable: discovery families disclose the victim as an opaque handle and score
  on `target_row_flip` via the trusted decoded key, never on a policy read.
- Non-discovery families and the public `RoBaRaCoCh` path are unchanged and green.

## Alternatives considered

- **Select among stock mappers with differing strides** (`RoBaRaCoCh` +
  `ChRaBaRoCo`): rejected — `ChRaBaRoCo`'s true stride (8192) differs from the
  disclosed 131072, which makes P21's "geometry identical every episode"
  disclosure a lie, and it still does not scatter banks.
- **De-scope to Tier 2a handle-opacity only** (drop the secret mapper): rejected —
  abandons the "discover like a real attacker with numeric addresses" deliverable
  (Tier 2b), which is the release headline.
