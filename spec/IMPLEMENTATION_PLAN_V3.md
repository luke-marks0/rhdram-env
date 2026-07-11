# Implementation Plan v3 — Train a policy that discovers real RowHammers

> This plan continues the project after `spec/IMPLEMENTATION_PLAN_V2.md` (P0–P20,
> admitted). It turns the design sketch in `spec/TIER2_DISCOVERY_PLAN.md` into a
> concrete, file-level build-out, and carries the system to a **release-ready
> state in which a language-model policy can be trained to discover genuine
> RowHammer adjacency** — not recite a disclosed hint.
>
> **Read first:** `spec/SPEC.md` (non-negotiables, §2/§9), then
> `spec/TIER2_DISCOVERY_PLAN.md` (why the timing side channel exists, what it can
> and cannot reveal — §0.2 as corrected). This document assumes both.
>
> All SPEC §2 non-negotiables still hold: **no mocks, no fabricated reward, no
> fake fallback signal; unavailable features are absent from capability discovery
> and fail closed with a stable error code.**

---

## 0. Ground truth as of 2026-07-11 (verified against the code)

### 0.0 How this plan is organised, and where it runs

**No new per-phase verify/gate scripts.** The P0–P20 work used one
`scripts/verify_phaseN.py` gate per phase; this plan deliberately does **not**
add more. They convolute the tree, and `spec/PRE_RELEASE_RESTRUCTURE.md` already
wants "phase" removed from the product surface. New Tier 2 work lands as ordinary
code plus **component-named tests** under `tests/` (e.g.
`tests/test_discovery_mapping.py`, `tests/test_discovery_families.py`,
`tests/test_probe_signal.py`) — grouped by feature, not by phase number. Each
phase below states its acceptance as a **"Done when"** list; those bullets are the
tests/checks to write, nothing more ceremonious.

**Where the code is authored vs. exercised.** This is being written on a **host
machine with no `torch`/`trl`** (and no GPU). That is fine for almost everything
here:
- **Runnable on the host:** all of P21–P26 and the env/family/mapping/reference-
  policy code is **pure Python plus the compiled C++ worker** — the existing
  `python3 -m unittest` suite runs without `torch`/`trl`. The one build step is
  the worker (P24 adds a `DECODE` op and mapper wiring; rebuild with
  `scripts/build_phase2.py`).
- **Authored on the host, exercised on the compute instance:** the GRPO training
  loop (P27) and curriculum runs (P28) import `trl`/`torch`. Write and unit-test
  their *non-torch* parts (transcript rendering, completion-mask construction,
  reward attribution) on the host with plain asserts; run the actual training on
  the instance. Keep the `trl`/`torch` imports local to the training entrypoints
  so importing a library module for a host-side unit test does not require them.

### 0.1 What already works (do not rebuild)
- **Interactive multi-turn loop exists.** `rowhammer_env/llm/rollout.py:run_episode`
  already drives genuine turn-by-turn interaction: `policy.next_tool(observation,
  transcript)` is called once per real `step()` and loops to `done`/`max_steps`.
  The *single-shot* flow is specific to the GRPO **training** path
  (`llm/grpo_env.py:ScriptedPolicy`/`evaluate_item`, which parses one completion
  into an action list and replays it). P27's lift is training-side, not the loop.
- **Disclosure leakage guard is real and wired.** `tasks/disclosure.py`
  (`project_trace`/`project_feedback`) strips hidden coordinates; applied on the
  step path at `phase4_env.py:149`. `full_trace` is a real, tested feedback level
  (`tests/test_phase12.py:93-98`; default for the bare env, `phase4_env.py:18-24`).
- **Task compiler + 10 families + per-family reward predicates** exist
  (`tasks/compiler.py`, `rewards/`), with calibrated difficulty bands
  (`compiler.py:24-29` `BAND_ACTS`/`BAND_WINDOW`).
- **HTTP serving + async client + CI fixture policies** exist (`server/app.py`,
  `client.py`, `llm/policies.py:CIHammerFixturePolicy`). The fixture hammers
  *disclosed* addresses; it is **not** a discovery policy.
- **Real issued-event stream** with `clk`/`row_hit`/decoded coords
  (`cpp/ramulator_extensions/issued_event_recorder.cpp`), consumed by the Python
  disturbance model keyed on decoded bank+row.

### 0.2 The timing side channel — corrected mechanics (drives P22/P26)
The environment exposes a **real DRAM bank-conflict timing side channel** (the
DRAMA class of signal). Two facts the naïve reading gets wrong — both verified in
code — govern every probe design below:

1. **A policy `RD` event's `row_hit` is always `true`.** A policy can only issue
   `RD`/`WR`/`WAIT` (`ramulator_worker.cpp:194-216`); an `RD` is a frontend
   request the controller schedules, auto-inserting the `ACT` **before** the `RD`.
   The recorder sets `m_open_row` on that `ACT`, so the `RD`'s `row_hit` is
   computed against the row its own `ACT` just opened → always `true`
   (`issued_event_recorder.cpp:70-81`). `row_hit=false` appears **only** on
   `ACT`/`PRE` events. The same-bank/different-bank discriminator is therefore
   *whether an alternating access forces a new `ACT`*, observable via (in
   increasing plumbing order) the cumulative **`public_counters.acts` delta**,
   **`last_action.cycle_delta`**, or the `ACT`/`PRE` events under `full_trace`.
   Requires the admitted **`Open` row policy** (`configs/ramulator/p1_external_ddr4.py:14`).

2. **`_issue` drops intermediate events.** `phase2_env.py:_issue` (232-257) calls
   the worker once per expanded primitive and returns only the **last** primitive's
   observation; the worker `drain()`s its event sink per call
   (`ramulator_worker.cpp:271`). A multi-primitive `dram.issue` (incl. a big
   `HAMMER`) surfaces only the final primitive's `events[]`/`cycle_delta`. So
   probes must be issued one-primitive-per-tool-call, **or** `_issue` must
   aggregate a per-issue timing digest (P22). `public_counters.acts` is cumulative
   and survives across primitives — the most robust probe channel.

### 0.3 The load-bearing architectural decision — a realistic secret address mapping
The timing channel is only *necessary* when address→bank membership is **not
computable** by the policy. With the current fixed, public `RoBaRaCoCh` mapper
(`configs/ramulator/p1_external_ddr4.py:15`) and disclosed geometry, bank
membership is a pure bit-slice computation, and — because a row-stride step keeps
the bank bits fixed — `victim ± row_stride` is *always* the physically adjacent
aggressor. Two consequences:

- **Tier 2a** stays a genuine discovery task today only because candidates are
  handed as **opaque handles** (non-arithmetic), so their bank cannot be computed.
- **Tier 2b** (the redefined real-attacker model: victim's own *numeric* address
  disclosed) is trivially computable under a public mapper — *not* a discovery.
  Per the resolved decision (`TIER2_DISCOVERY_PLAN.md` §4.1), Tier 2b's hidden
  variable is the **address→bank mapping itself**, kept a per-episode secret so
  adjacency must be reverse-engineered by timing (DRAMA), exactly as on real
  hardware where the controller's bank-XOR function is undocumented.

> **CORRECTION (P24 as-built, 2026-07-12 — see `docs/adr-0004-secret-address-
> mapping.md`).** The `MOP4CLXOR` premise below is **wrong** and was not used.
> Verified against the code and the worker `DECODE` op: `MOP4CLXOR` XORs
> **column** bits into the bank index (cache-line/bank interleaving), *not* row
> bits — so `addr + row_stride` stays in the **same bank** under it (measured
> 0/200 different-bank), exactly as under `RoBaRaCoCh`. No stock Ramulator v2.1.0
> mapper XORs Row→Bank, so none makes `victim ± row_stride` non-computable. P24
> instead **authors** a real, source-cited Row→Bank XOR mapper,
> **`RoBaRaCoChRowXOR`** (`cpp/ramulator_extensions/row_xor_addr_mapper.cpp`):
> RoBaRaCoCh decode (so the row stride and disclosed geometry are unchanged) plus
> a Row→Bank/BankGroup XOR with a seedable `xor_offset` — bit 0 of Row always
> folds into Bank, so `addr + row_stride` lands in a **different bank** (measured
> 300/300). This is address mapping (a real, documented controller function;
> DRAMA / Pessl et al.), not fabricated physics, so it is within SPEC §2. Read
> `MOP4CLXOR` as `RoBaRaCoChRowXOR` throughout the rest of this section and P24.

Ramulator v2.1.0 ships a **real, source-traced** mapper for this: **`MOP4CLXOR`**
("Multi-Offset Physical with 4-CL XOR", `addr_mapper/impl/mop4clxor.cpp`) XORs row
bits into the bank/bankgroup indices, so linearly-adjacent addresses scatter
across banks. Also available: `RoBaRaCoCh`, `ChRaBaRoCo`, `RITAddrMapper`,
`PassThroughAddrMapper`. **No fabrication is required** — the per-episode secret
is *which* real mapper (and its seedable parameters) is active; each mapper is
genuine controller behavior. This is P24, the linchpin for numeric-address
discovery, and it keeps the disturbance model consistent for free (flips already
key on the recorder's *real* decoded bank+row, whatever mapper is active).

---

## 1. Planning model

Execution-order dependency spine (▸ = hard blocker):

```
P21 geometry disclosure ─▸┐
P22 probe observability ─▸┼─▸ P23 Tier 2a family (handle-opaque) ─▸ P26 reference policy ─(proves 2a)─┐
                          │                                                                            │
                          └──────────────────────────────────────────────────────────────────────────┤
P24 realistic secret mapping ─▸ P25 Tier 2b family (numeric addr) ──────────▸ P26 reference policy ─(proves 2b)─┤
                                                                                                        │
                                                            P27 multi-turn training + GRPO ◂────────────┘
                                                            P28 curriculum + bounded shaping ◂── P27
                                                            P29 release readiness ◂── all
```

**Phase → `TIER2_DISCOVERY_PLAN.md` sketch mapping** (the sketch numbered by
topic, not execution order): P21≡sketch-P21, P22≡sketch-P22, P23≡sketch-P23,
**P24 new** (the mapping decision the sketch surfaced but did not phase), **P25
new** (Tier 2b family), P26≡sketch-P26 (reference policy — moved *before* training,
as the sketch itself insists), P27≡sketch-P24 (rollout engine), P28≡sketch-P25
(curriculum/shaping), P29 is release readiness.

**Critical discipline (kept):** P26's deterministic reference policy must solve
each family **before** any GPU time is spent training it. If a hand-written policy
using only the disclosed signals can't solve a band within budget, no GRPO run
will — and that is cheap to find out first, on the host, with no `torch`.

Each phase below: **Goal → Prerequisites → Tasks (file-level) → Done when.** Don't
advertise a capability in `dram.info`/capability discovery until it's backed by a
real implementation — but that's a code review point, not a gate script.

---

## P21 — Public geometry disclosure

**Goal.** Expose architecture-level DRAM geometry (row stride, and *sizes*: row/
bank/bankgroup counts + standard) in `dram.info` and reset metadata,
unconditionally. This is public standard information (identical across every
episode of a profile), so it leaks nothing about the hidden target. **Disclose
only the geometry *sizes and row stride*, never the bank-select bit function** —
that function is P24's per-episode secret and must stay hidden.

**Prerequisites.** None (independent). The env already has `self.address_mapper.
geometry`; the worker `INFO` reports level sizes (`ramulator_worker.cpp:146-160`).

**Tasks.**
1. Add a `geometry` block to `RowHammerTaskEnv._task_metadata`
   (`phase5_env.py:196-215`) and the base `dram.info` handler
   (`phase2_env.py:190-201`): `{"row_bytes": int, "row_count": int, "bank_count":
   int, "bankgroup_count": int, "standard": str}`, sourced from
   `Geometry`/`geometry.row_stride` the compiler already uses
   (`compiler.py:258-259`). Do **not** emit the level *order* or bit offsets.

**Done when.**
- The geometry block is present in `dram.info` + reset metadata at every
  disclosure level, and is byte-identical across many seeds/targets **and** across
  `RoBaRaCoCh` vs `MOP4CLXOR` episodes (a test proving it carries no per-episode /
  no mapping-function information).
- No bank-function bits appear anywhere in the observation surface; existing
  non-leakage tests still pass with the new fields fuzzed.

**Parallelization.** Fully independent; concurrent with P22/P24.

---

## P22 — Probe observability: `full_trace` wiring + per-issue timing digest

**Goal.** Make a probe sequence's timing actually reach the policy, and fix the
`_issue` event-loss (0.2.2) so a probe or a large `HAMMER` returns a bounded,
useful signal instead of one primitive's raw events.

**Prerequisites.** None (independent config + small plumbing).

**Tasks.**
1. **`full_trace` on discovery configs.** Set `feedback: full_trace` on the P23/
   P25 discovery task configs (and optionally on `configs/tasks/unknown_adjacency.
   yaml`, which becomes solvable-by-timing once candidates span banks).
2. **Per-issue timing digest (the real fix).** In `phase2_env.py:_issue`, instead
   of returning only `last`, **accumulate across the expanded primitives** and
   attach a bounded digest to the returned feedback: `{"acts_delta": int,
   "cycles_delta": int, "first_clk": int, "last_clk": int, "per_addr_hits": {addr:
   {"acts": n, "hits": n, "misses": n}}}` — computed from each primitive's real
   worker response (a summary of true issued events; nothing fabricated). Keep the
   raw `trace_tail` for small issues; cap it (e.g. last N events) for large ones.
   The digest carries only counts/clks — no coordinates — so the leakage guard is
   preserved by construction.

**Done when.**
- A same-bank alternating probe shows a nonzero `acts_delta`/large `cycles_delta`;
  a different-bank probe shows ~0 — checked through the real `RowHammerTaskEnv.
  step()`, under `mapping: logical_only` + `full_trace`, with all six `COORD_KEYS`
  stripped from any raw events.
- A 25k-activation `HAMMER`'s returned feedback stays within a documented size
  bound, and `public_counters.acts` reconciles with the summed `acts_delta`.

**Parallelization.** Independent; concurrent with P21/P24. Freeze the digest shape
early — P26's reference policy and P23's configs build against it.

---

## P23 — Tier 2a family: bounded addressable candidate window (handle-opaque)

**Goal.** A family that mirrors real memory-templating: `N` **RD-addressable**
candidate handles at a deliberate **mix of same-bank/different-bank, adjacent/
non-adjacent** offsets, only 1–2 of which are true aggressor neighbors of a real
(but not pre-identified) victim, sized so bank-conflict timing (0.2) plus a final
budgeted hammer solves it. Ships on the **current `RoBaRaCoCh` mapper** — handle
opacity (not a secret mapping) keeps bank membership non-computable, so no P24
dependency. Difficulty scales with candidate-window width `N`.

**Prerequisites.** P21, P22.

**Tasks.**
1. **Family in `tasks/compiler.py`.** Add `bounded_sweep` (or generalize
   `unknown_adjacency`): register `N` candidate handles whose *decoded* coords are
   a controlled mix — some same-bank+row-adjacent (true aggressors), some
   same-bank+far, some different-bank (decoys). The existing `unknown_adjacency`
   offsets `{-1,+1,+2}×row_bytes` are **all same-bank** (`phase5_env.py:151-153`;
   a row-stride step keeps bank bits fixed), so different-bank decoys must be built
   from **bank-stride** offsets computed from the decoded geometry, not guessed.
   `N` is a difficulty parameter (`easy: N=4`, `medium: N=16`, `hard: N=64`).
2. **Candidate emission.** Extend `phase5_env.py:_register_handles` /
   `_objective_and_target` to emit the `N`-handle candidate set (currently
   hard-wired to 3). Keep victim as `row_handle`; keep `feedback: full_trace`.
3. **Budgets.** Do **not** guess `tool_calls`/`acts`. Leave placeholder budgets;
   P26's reference policy measures the real probe+hammer call count and back-fills
   the bands.
4. **Configs.** `configs/tasks/bounded_sweep_{easy,medium,hard}.yaml` mirroring the
   `any_flip_{easy,medium,hard}.yaml` band convention.

**Done when.**
- Each band instantiates, compiles, and runs.
- A fixed-seed fixture confirms the intended true-aggressor fraction **and** the
  intended same/different-bank split (decoded against the real mapper).
- Leakage guard confirms no candidate's "real one" identity is derivable from
  anything but the trusted final flip.

**Parallelization.** After P21+P22. Independent of the P24/P25 mapping lane.

---

## P24 — Realistic secret address mapping (the discovery linchpin)

**Goal.** Make address→bank membership a **per-episode secret** for discovery
families, using **real Ramulator mappers** (no fabrication), so a *numeric*
victim/candidate address does not reveal its bank — forcing genuine DRAMA-style
timing discovery. Keep the disturbance model and every non-discovery family
byte-for-byte unchanged.

**Prerequisites.** P11 event stream (done); P13 config plumbing (done). Concurrent
with P21–P23. **Needs a worker rebuild** (`scripts/build_phase2.py`) — the one C++
touch in this plan.

**Tasks.**
1. **Per-episode mapper selection.** The worker YAML is built from
   `configs/ramulator/p1_external_ddr4.py`. Make `addr_mapper` a per-episode
   parameter sourced deterministically from `(task_id, seed)` for discovery
   families: choose among the real registered mappers — **`MOP4CLXOR`** (row↔bank
   XOR: linearly-adjacent addresses scatter across banks — the primary secret),
   `RoBaRaCoCh`, `ChRaBaRoCo` — and record the choice in an ADR under `docs/`.
   Non-discovery families keep `RoBaRaCoCh` (regression-stable).
2. **Worker `DECODE` request.** Add a cheap `DECODE <id> <linear>` op to
   `ramulator_worker.cpp` returning the true `addr_vec` for a linear address under
   the *active* mapper (reuse the mapper the worker already constructed). This lets
   the env/compiler build candidate sets against the true mapping **without
   duplicating mapper logic in Python** (avoids `tools/addressing.py` drift when
   the mapper isn't `RoBaRaCoCh`). The env never exposes `DECODE` to the policy.
3. **Compiler uses real decode.** In `tasks/compiler.py`, define the victim by a
   linear address and its **decoded** bank/row (via `DECODE`), and construct
   aggressor/decoy candidates by searching for linear addresses whose decoded
   coords are `{same bank, row±1}` (true) / `{different bank}` / `{same bank, far
   row}` (decoys). Under `MOP4CLXOR` this is a genuine search, not arithmetic.
4. **Addressing guard.** `tools/addressing.py`'s Python `RoBaRaCoCh` projection is
   correct only for `RoBaRaCoCh` episodes; for other mappers, `physical`-form
   addressing and `_physical_target` must fail closed (`ADDRESS_NOT_DISCLOSED`).
   Discovery families are `logical_only` anyway, so the policy never gets a
   decoder — assert this.
5. **Leakage guard.** The active mapper id/params must never appear in `dram.info`,
   reset metadata, errors, handle names, or the timing digest.

**Done when.**
- Under `MOP4CLXOR`, `DECODE(addr)` and `DECODE(addr+row_bytes)` land in
  **different banks** for a documented fraction of addresses (proving adjacency is
  not computable from linear arithmetic), while the disturbance still flips the
  **decoded** aggressor→victim pair (real physics unchanged).
- Every non-discovery family + the existing `RoBaRaCoCh` tests stay green.
- Mapper identity is unrecoverable from any disclosed field (leakage fuzz test).

**Parallelization.** Independent lane; concurrent with P21–P23. Blocks P25.

---

## P25 — Tier 2b family: known numeric address, secret adjacency

**Goal.** The redefined `hidden_target` (`TIER2_DISCOVERY_PLAN.md` §4.1): disclose
the victim's **own numeric (logical) address** and a set of **numeric** candidate
addresses (real-attacker knowledge), with the address→bank mapping secret (P24) so
which candidates are same-bank+adjacent must be discovered by timing. This is the
"discover a new RowHammer like a real attacker" deliverable.

**Prerequisites.** P24 (secret mapping + `DECODE`), P22 (probe observability), P21.

**Tasks.**
1. **New disclosure level.** Add a victim disclosure that hands the victim's linear
   address as `{"kind":"logical","addr":N}` **without** physical coords (distinct
   from `victim: exact`, which emits coords via `_physical_target` and would leak
   the mapping). Extend `tasks/disclosure.py` (`Disclosure`, `expose_victim`
   semantics) and `phase5_env.py:_objective_and_target` (241-257).
2. **Numeric candidates.** Emit candidates as numeric logical addresses (not
   handles), built by P24 task 3's real-decode search: a mix of true aggressors +
   same-bank-far + different-bank decoys. `mapping: logical_only`, `feedback:
   full_trace`.
3. **Family + configs.** Register `hidden_adjacency` (redefined `hidden_target`) in
   `FAMILIES` (`compiler.py:65-100`) and add
   `configs/tasks/hidden_adjacency_{easy,medium,hard}.yaml`. Retire or alias the
   old opaque `hidden_target` (dropped per §5); keep a deprecation note.
4. **Leakage guard.** No disclosed field (address, digest, error) may reveal a
   candidate's bank/row or the mapper — only the trusted flip confirms an
   aggressor.

**Done when.**
- The family instantiates and runs; a fixed-seed fixture confirms the true-
  aggressor fraction and bank split (via `DECODE`).
- A control policy that computes `victim ± row_stride` and hammers it **fails**
  (proving the secret mapping makes arithmetic insufficient).
- Leakage guard green over the numeric-candidate surface.

**Parallelization.** After P24. Concurrent with P23 (different families).

---

## P26 — Reference probing policy (BEFORE any training)

**Goal.** Prove the disclosed signals are *sufficient* by hand-writing a
deterministic policy that solves Tier 2a (P23) and Tier 2b (P25) using exactly the
tools/signals an LLM policy would have — and nothing else. Pure Python; runs on the
host with no `torch`. Built once; validates each family as it lands (2a right after
P23, 2b after P25).

**Prerequisites.** P23 for the 2a check; P25 for the 2b check; P22 digest.

**Tasks.**
1. **DRAMA timing solver** as a normal module (a `ReferenceProbePolicy` beside
   `llm/policies.py:CIHammerFixturePolicy`, clearly a deterministic reference — the
   P19 no-mock discipline of labelling fixtures as fixtures). For each candidate:
   warm the victim row open, `RD` the candidate, then `RD` the victim in a
   **separate `dram.issue`**, and classify same-bank vs different-bank from the
   **`acts_delta`/`cycles_delta` digest** (or `public_counters.acts` delta) — **not**
   the access's `row_hit`, which is always true (0.2.1). Keep same-bank candidates.
2. **Budgeted confirmation.** Among survivors, spend the real hammer budget
   (compact `HAMMER`) attempting each, checking `feedback.new_public_flips` after
   each, stopping on success.
3. **Back-fill budgets + bands.** Measure the real probe+hammer `tool_calls`/`acts`
   and write them into the P23/P25 configs and `BAND_ACTS`/`BAND_WINDOW`
   (`compiler.py:24-29`) — this is where budget sizing was deliberately deferred.

**Done when.**
- The reference policy solves `easy`/`medium` for both 2a and 2b across many seeds
  within the measured budget, driven through `run_episode` (a `tests/` case, run
  on the host).
- A **differential control** that ignores the `acts`/`cycle_delta` timing (probes/
  hammers candidates in random order) does markedly worse or fails budget —
  proving the signal is load-bearing, not incidental.
- **This is the go/no-go for training.** No GPU time until it passes.

**Parallelization.** Serial checkpoint between the family phases and training.

---

## P27 — Multi-turn training rollout + GRPO integration

**Goal.** Give the GRPO **training** path genuine turn-by-turn trajectories (the
eval path already has them via `run_episode`), and feed them to the trainer with
correct loss masking.

**Prerequisites.** P26 (proof the tasks are solvable), P22, P23/P25. Non-torch
parts authored/tested on the host; full run on the compute instance (0.0).

**Tasks.**
1. **Training rollout module** (`rowhammer_env/llm/multiturn_rollout.py`):
   generalize `run_episode` to (a) render the running transcript (system + task +
   alternating assistant tool-call turns and `tool`-result turns from the real
   `Phase2Observation`), (b) call the model for one short completion per turn (same
   parse path as `grpo_env.parse_actions`/`_coerce_action`), (c) `await client.
   step(...)`, (d) repeat until `episode.finish`, budget exhaustion, or a
   `max_turns` cap. Reuse `client.RowHammerClient`. Keep `trl`/`torch` out of this
   module so it unit-tests on the host.
2. **GRPO integration (completion-mask pattern).** Build the trajectory outside the
   trainer's generation step, then hand `GRPOTrainer` a `(prompt, completion)` pair
   where `completion` is the whole multi-turn transcript and a **token-level
   completion mask** marks only assistant-authored spans as trainable (tool-result
   turns masked out of the loss, as in multi-turn SFT). **First** check the
   training-host TRL version for a native custom-rollout hook; fall back to the
   mask pattern if absent. Isolate `trl` imports in `scripts/train_grpo.py`, not in
   importable library modules.
3. **Reward attribution.** The final trusted episode reward (`0.0`/`1.0` from
   `_trusted_success()`) is the reward for the whole trajectory; GRPO's group-
   relative advantage is per full rollout. No per-turn reward except P28's shaping.
4. **Throughput.** Re-tune `env.concurrency`/`max_concurrent_envs`
   (`configs/training/grpo_qwen8b.yaml` `env:` block) — round-trip latency now
   dominates. Measure on the instance before assuming.

**Done when.**
- The P26 reference policy driven through the new training loop produces the
  **exact same trajectory + reward** as driving it manually via `RowHammerClient`
  (trace-equivalence — a host-side `tests/` case, no `torch`).
- The completion mask provably covers only assistant spans (host-side unit test on
  a fixed transcript).
- On the instance: a tiny end-to-end run shows a non-trivial gradient/advantage
  signal.

**Parallelization.** Serial after P26.

---

## P28 — Curriculum and (narrowly bounded) reward shaping

**Goal.** Make the sparse end-of-trajectory reward learnable in practice.

**Prerequisites.** P27.

**Tasks.**
1. **Curriculum ordering.** Tier 0 (already saturates at `reward=1.1`) → Tier 2a
   `easy` (N=4, generous budget) → 2a medium → 2a hard → Tier 2b `easy` → 2b
   medium/hard, gated at each step by P26's measured reference success window.
2. **Shaping, scoped to SPEC §9.** Any auxiliary term must (a) derive only from
   already-trusted, already-disclosed-at-this-feedback-level state — e.g. a small
   bonus for a probe that yields a *decisive* same/different-bank digest, computed
   from the real `acts`/`cycle_delta` (never a hint about which candidate is
   correct); (b) apply **only** in training configs, never eval/benchmark scoring;
   (c) never push reward above what a real, budget-respecting successful trajectory
   earns. Add it as a separate, clearly-labelled weight beside
   `reward.success_weight`/`format_weight` (`configs/training/grpo_qwen8b.yaml`).
3. **Re-validate after every curriculum change** with the P26 reference policy
   (host-side, no `torch`) before spending instance time.

**Done when.**
- The reference policy hits its calibrated success window on every band.
- Shaping terms provably vanish from eval/benchmark scoring; no shaping path yields
  reward without a real flip (a host-side `tests/` case on the reward module).

**Parallelization.** After P27.

---

## P29 — Release readiness

**Goal.** The system meets SPEC §12 acceptance on the v2.1.0 pin, with the Tier 2
discovery capability admitted, no mocks. Do this as the standalone pre-release
restructure (`spec/PRE_RELEASE_RESTRUCTURE.md`) plus a final review — **not** a new
gate script.

**Prerequisites.** P21–P28.

**Tasks.**
1. **Restructure.** Collapse the `phaseN_*` files into the component layout and
   remove "phase" from the product surface, per `PRE_RELEASE_RESTRUCTURE.md`. The
   Tier 2 tests written under this plan are already component-named, so they fold in
   cleanly.
2. **One test suite.** `python3 -m unittest discover` is the acceptance run (host-
   runnable for everything except the training smoke, which runs on the instance).
   Retire the per-phase `verify_phaseN.py` scripts as part of the restructure rather
   than adding to them.
3. **Determinism/replay** across the pinned manifest + seeds, **including
   per-episode mapper selection** (same seed ⇒ same mapper ⇒ same trace/flips) —
   SPEC §12.7.
4. **No-mock scan** over all executable roots (extend the existing symbol scan to
   the new modules).
5. **Docs/provenance.** `docs/api.md` (tool/obs/reward contract incl. the timing
   digest and the numeric-candidate discovery families), the P24 mapper ADR, a
   discovery quickstart (launch server → attach LLM → run curriculum), and updated
   `spec/VALIDATION_REPORT.md`, model cards, SBOM.
6. **Acceptance mapping (SPEC §12).** §12.2 real flip via issued events — Tier 2b
   reference policy; §12.3 fails when it should — the `victim ± row_stride` control
   (P25); §12.4 API/reward/projection contract — the P22/P24 leakage tests; §12.8
   no mock/fallback — task 4.

**Done when.** `unittest discover` is green on the host (env/family/mapping/
reference-policy) and on the instance (training smoke); replay is deterministic
including mapper selection; no mock/fallback imports; discovery families appear in
capability discovery.

---

## 2. Open decisions still flagged (resolve as they arise)

1. **TRL custom-rollout capability on the training host** (P27.2) — verify what
   multi-turn/custom-generation support the installed TRL exposes before committing
   to the completion-mask shape. Cannot be checked on the host (no `trl`).
2. **Mapper-secret granularity** (P24.1) — per-episode mapper *selection* from a
   small real set is the baseline. A seedable parameterization *within* `MOP4CLXOR`
   (which XOR offset) is the next rung if more difficulty is wanted — still real,
   still source-traced. Decide from P26's measured difficulty.
3. **Second standard for Tier 2b** — this plan lands DDR4. Extending discovery to
   the HBM2/DDR5 paths is post-release and gated on profile admission; not promised.

## 3. Non-goals (unchanged from `TIER2_DISCOVERY_PLAN.md` §5)

- No fabricated row-distance signal (SPEC §2). The mapper secret (P24) is real
  controller behavior, not an invented physics term.
- No reward path that scores success from anything but `_trusted_success()`/real
  `new_public_flips` (SPEC §9).
- The fully-opaque, zero-addressability `hidden_target` is dropped in favour of the
  numeric real-attacker redefinition (§4.1).

## 4. Parallelization staffing guide

| Lane | Phases | Notes |
|---|---|---|
| **A — disclosure/observability** | P21, P22 | Independent; freeze the geometry block + timing-digest shape early. |
| **B — mapping** | P24 | Independent C++/config lane; the ADR + `DECODE` op unblock P25. Needs a worker rebuild. |
| **C — families** | P23 (needs A), P25 (needs A+B) | Family authoring; 2a ships without B. |
| **D — proof→training** | P26 → P27 → P28 | P26 is the host-runnable go/no-go before any GPU spend. |
| **E — release** | P29 | Restructure + one test suite; needs all. |

**Fastest route to a trainable discovery policy:** land A + P23 + **P26 proving
Tier 2a** (all host-runnable, no `torch`), then P27/P28 on Tier 2a *while* B+P25
bring Tier 2b online — converge on P29. Tier 2a is the earliest end-to-end
trainable target; Tier 2b is the real-attacker headline capability.
