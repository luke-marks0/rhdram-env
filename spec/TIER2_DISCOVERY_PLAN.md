# Tier 2 plan — a policy that discovers RowHammer adjacency, not just recites it

> Audience: an implementing engineer or LLM agent picking this up cold. Read
> `spec/SPEC.md` and `spec/IMPLEMENTATION_PLAN_V2.md` first (P0–P20, admitted);
> this document assumes that ground truth and does not repeat it except where
> directly relevant. It also assumes the current GRPO training harness
> (`scripts/train_grpo.py`, `rowhammer_env/llm/grpo_env.py`) and its bugfixes
> (`154300a`, `71affbd` — compact HAMMER + bare-int address shorthand) are in
> place and working: `known_target_anybit`/`target_cell`/`pattern_target` train
> and saturate at `reward=1.1` almost immediately, because those families
> disclose the exact target and the exact activation threshold — solving them is
> closer to "copy this number into a template" than "discover a hammer."
>
> **What this document is for.** Tier 2 means: a policy that has to *find*
> something true about the hidden simulated DRAM state (which row is a real
> aggressor neighbor) using only legitimate, trusted, issued-event signal — not
> a disclosed hint. This plan covers what's realistic, what isn't, and the
> concrete phased engineering (config, environment, and training-harness
> changes) to get there. All SPEC §2 non-negotiables still apply: no mocks, no
> fabricated reward, no fake fallback signal, fail-closed on anything
> unavailable.

---

## 0. The central technical finding (read this before designing anything)

The environment already exposes a **real, non-fabricated timing/row-buffer
side channel** — but it reveals less than "the exact adjacent row," and the
gap matters for what's achievable. This section is the reasoning that the rest
of the plan is built on.

### 0.1 What's actually in an issued event

Every issued command the worker emits (built in `cpp/ramulator_extensions/
issued_event_recorder.cpp:98-105`, serialized by `events_json` in
`ramulator_worker.cpp:247-267`) carries:

```json
{"op": "ACT", "clk": 118432, "type_id": 3, "row_hit": false}
```

(plus decoded coordinates `channel/rank/bankgroup/bank/row/column`, but those
are exactly `COORD_KEYS` in `rowhammer_env/tools/addressing.py:11` and get
stripped by `Disclosure.project_trace` whenever `mapping != "physical"` —
`rowhammer_env/tasks/disclosure.py:76-82`). Critically, **`clk` and `row_hit`
are *not* coordinate keys and survive that projection** even under
`mapping: logical_only`. Today, though, none of the discovery families
actually disclose the trace at all: `hidden_target`/`unknown_adjacency` are
configured `feedback: summarized_counts`, which makes `project_trace` return
`[]` unconditionally (`disclosure.py:76-77`). `feedback: full_trace` is a
real, already-tested feedback level (`phase4_env.py:23`, `tests/
test_phase12.py:93-98`) — turning it on for a discovery family is a **config
change**, not new engine code.

### 0.2 What `row_hit` and `clk` can and cannot tell a policy

A policy cannot issue `ACT`/`PRE` directly — the worker admits only `RD`/`WR`/
`WAIT` (`ramulator_worker.cpp:194-216`); everything else is `ILLEGAL_COMMAND`.
Every policy `RD` is a *frontend memory request* the controller schedules,
auto-inserting the `ACT` (and any `PRE`) needed to open its row under the
admitted **`Open` row policy** (`configs/ramulator/p1_external_ddr4.py:14`).
`row_hit` is computed only for *accessing* commands (RD/WR), per bank
(`flat_bank = bankgroup*num_banks + bank`), against the row the recorder
believes is currently open (`issued_event_recorder.cpp:70-81,120-126`):

```cpp
row_hit = is_accessing && (m_open_row[flat_bank] == this_command's_row)
```

**Subtlety the probe design must respect.** Because the controller issues the
row's `ACT` *before* the `RD`, and the recorder sets `m_open_row` on that `ACT`,
the `RD` event's `row_hit` is **always `true`** — a policy `RD` never observes
`row_hit=false` on its own access, in either bank case. `row_hit=false` appears
only on the accompanying `ACT`/`PRE` events. So the discriminator is **not** the
access's `row_hit` flag; it is *whether an alternating access forces a new
`ACT`*. Warm both rows open (`RD A`, `RD B`), then `RD A` again:
- **Different bank:** B's access did not evict A's open row, so the second
  `RD A` is a true row-buffer hit — **no new `ACT`**, so `public_counters.acts`
  does not move and `last_action.cycle_delta` is small.
- **Same bank:** B evicted A's row, so the second `RD A` must `PRE`+`ACT` first
  — **the `acts` counter increments**, `cycle_delta` is large, and `ACT`/`PRE`
  (`row_hit=false`) events appear in the trace.

The signal is therefore observable three equivalent ways, in increasing order of
plumbing needed: the cumulative **`public_counters.acts` delta** (always present,
needs no trace at all), **`last_action.cycle_delta`**, and — only under
`full_trace` — the `ACT`/`PRE` events themselves. The "count of `row_hit=true`
vs `false`" digest P22 mentions *does* carry the signal, but because its `false`
entries are exactly those `ACT`/`PRE` events — not because the access reads
false. (This all assumes the `Open` row policy above; a closed-row/auto-precharge
config would `ACT` on every access regardless of bank and erase the channel — so
the policy is a load-bearing precondition, not a detail.)

This is the real DRAM bank-conflict timing side channel (the same class of
signal as Pessl et al.'s DRAMA reverse-engineering technique) and it is
genuinely present and genuinely trusted-state-derived here. **What it reveals
is "same bank, yes/no" — a `log2(bank_count)`-bit signal (4 bits for the
16-bank DDR4_8Gb_x8 geometry the worker publishes) — via a handful of cheap
interleaved reads (tens of activations, not thousands).** It does **not**
reveal row distance: two different rows in the same bank produce the
*identical* `row_hit=false`/large-`clk`-gap signature regardless of whether
they're neighbors or 30000 rows apart. There is no other structural signal in
this engine that encodes row distance (no modeled sense-amp/crosstalk term
that varies with physical proximity) — inventing one would be exactly the kind
of non-empirical fabrication SPEC §2 forbids, so it should not be added.

Consequence: **finding the bank a hidden target is in (≈4–16 cheap probes) is
realistic and buildable. Finding the exact adjacent row inside a 65536-row
bank, from timing alone, is not** — the search space (16 bits) is far larger
than what ~200 tool-calls of real-flip testing (~25k activations each) can
brute-force (65536 × 25k ≈ 1.6B activations vs. a ~100k `acts` budget).

### 0.3 The address mapping is public, deterministic, and *not* the secret

`RoBaRaCoCh` (`rowhammer_env/tools/addressing.py`) is a fixed bit-slice map —
for the admitted DDR4 geometry: `column[6:13) → bankgroup[13:15) → bank[15:17)
→ row[17:33)`. The row stride (`row_bytes = 2**17 = 131072`) is **standard
geometry, identical across every episode of a profile** — it is not
episode-specific secret information. Yet nothing in `dram.info` or the reset
metadata discloses it today (`RowHammerEnv.step`'s `dram.info` handler,
`phase2_env.py:188-201`, and `RowHammerTaskEnv._task_metadata`/
`_public_disturbance`, `phase5_env.py:196-226`, expose `address_forms`,
`disclosure`, `disturbance.{family,stratum}`, and — only when
`expose_victim()` — `known_target_row`/`known_threshold`. No row/bank/column
counts or stride anywhere). This is a real, fixable gap (§2, P21 below):
disclosing standard geometry leaks nothing about *which* row is the hidden
target, exactly as a real attacker knows the DDR4 spec's row size without
knowing which physical row their victim data landed in.

Even with the stride disclosed, though, a policy **cannot** compute
`target_addr ± row_bytes`, because it never learns `target_addr` as a number —
`hidden_target`'s victim is an opaque, non-invertible handle
(`HandleTable`, `disclosure.py:100-127`) resolved only via
`{"kind":"handle","id":...}`, which has no offset parameter
(`AddressResolver.to_linear`'s handle branch, `disclosure.py:166-167`). This is
deliberate (SPEC §8 leakage guard) and correct — but it means `hidden_target`
as currently defined asks for something **stricter than the real-world
attacker's actual epistemic position**: a real Rowhammer researcher knows the
numeric (virtual/physical) address of their own allocated memory; what's
hidden from *them* is the mapping function, not their own addresses. This
environment's `unknown_adjacency` family (a handle for the *victim*, plus
**RD-addressable (opaque) candidate handles** for 3 neighbor-row
guesses — `phase5_env.py:151-153`) is much closer to that real threat model
than `hidden_target` (a handle with zero addressable neighbors at all). Note,
though, that those 3 candidates are registered at offsets
`{-1, +1, +2} × row_bytes` — all row-stride offsets, so **all three land in the
victim's own bank** (a row-stride step changes only the Row field, per the bit
layout above). §0.2's bank-conflict timing therefore cannot discriminate among
the *existing* candidates — they are all same-bank. The timing channel only
becomes load-bearing once the candidate window deliberately **spans multiple
banks**, which is exactly what P23 introduces.

### 0.4 What this means for scope

| Target | Tractable? | Why |
|---|---|---|
| **Tier 2a — bounded, addressable candidate window** (extend `unknown_adjacency`'s 3-handle idea to a wider real candidate set; realistic memory-templating framing) | **Yes, buildable now** | Bank-conflict timing narrows candidates cheaply; final confirmation is a real (budgeted) hammer attempt on the narrowed set. Mirrors real methodology. |
| **Tier 2b — victim's own address known, DRAM bank-mapping secret** (redefined per §4.1: the real-attacker-knowledge model — disclose the victim's numeric address + numeric candidates, keep the address→bank function a per-episode secret so adjacency must be reverse-engineered by timing, DRAMA-style) | **Yes — now the committed Tier 2b, via a realistic secret address mapping** | The timing channel reveals exactly the secret it needs to (bank membership, §0.2); physical row adjacency then follows from the public row stride. Requires the realistic-mapping engine change (`IMPLEMENTATION_PLAN_V3.md` P24). The *fully-opaque, zero-addressability* variant (a handle with no numeric address at all) stays out of scope: it asks for stricter-than-real-attacker knowledge and would need a row-distance signal this engine won't fabricate. |

**Tier 2a ships first** (no engine-mapping change needed — candidate opacity via
handles keeps bank membership non-computable). **Tier 2b is now a committed
deliverable too**, redefined to the real-attacker-knowledge model above and built
on a realistic secret address mapping (`IMPLEMENTATION_PLAN_V3.md` P24–P25). The
detailed, file-level, admission-gated build-out of both — through to release and a
trainable discovery policy — lives in `spec/IMPLEMENTATION_PLAN_V3.md`.

---

## 1. Goal

Train (and evaluate) a policy that, given a task where the true aggressor row
is genuinely not disclosed but is drawn from a **bounded, real, addressable**
candidate set, uses multi-turn interaction with the trusted simulator —
cheap timing probes to narrow candidates, then a real budgeted hammer on the
survivors — to cause a real flip. Success is measured exactly as today: the
trusted `episode.finish`/`_trusted_success()` predicate, sparse reward, no
policy-claimed success (SPEC §9).

## 2. Prerequisites

- P11–P20 admitted (issued-event stream, disclosure/addressing, task compiler,
  disturbance fidelity, HTTP serving, LLM baseline) — all in place per
  `spec/IMPLEMENTATION_PLAN_V2.md`.
- The two GRPO harness fixes (`154300a` compact HAMMER expansion, `71affbd`
  bare-int address shorthand) — in place; Tier 0 families saturate at
  `reward=1.1`.
- `full_trace` feedback level — already implemented and tested
  (`disclosure.py`, `tests/test_phase12.py:93-105`); no engine work needed,
  only config wiring (P22).

## 3. Planning model

```
P21 public geometry disclosure  ────┐
P22 full_trace wiring for discovery ─┼──> P23 Tier 2a task family ──> P24 multi-turn
                                     │                                  rollout engine
                                     │                                     │
                                     └─────────────────────────────────────┼──> P25 curriculum
                                                                           │     + shaping
                                                                           └──> P26 reference
                                                                                 probe policy
                                                                                 + admission gate
```

P21/P22 are small, independent config/plumbing changes and can land in either
order or in parallel. P23 depends on both. **P26 (a scripted reference
policy) should be built and proven *before* spending GPU time on P24/P25's RL
training** — exactly the discipline the rest of this repo already follows
(P11's row-buffer-locality proof, P14's `OracleRH` differential trace): if a
hand-written policy using only the same trusted signals can't solve Tier 2a
within budget, no amount of GRPO will either, and that's cheap to find out
first.

---

## P21 — Public geometry disclosure

**Goal.** Expose standard-level DRAM geometry (`row_bytes`/row stride, row
count, bank count, bankgroup count) in `dram.info` and reset metadata,
unconditionally — this is architecture-level public information (same for
every episode of a profile/standard), not hidden episode state, so disclosing
it cannot leak the hidden target.

**Tasks.**
1. Add a `geometry` block to `RowHammerTaskEnv._task_metadata`
   (`phase5_env.py:196-215`) and to the base `dram.info` handler
   (`phase2_env.py:188-201`): `{"row_bytes": int, "row_count": int,
   "bank_count": int, "bankgroup_count": int, "standard": str}`, sourced from
   the same `Geometry`/`geometry.row_stride` the compiler already uses
   (`compiler.py:265`, `TaskSpec.compile`'s `row_bytes = geometry.row_stride`).
2. **Non-leakage test**: for `hidden_target`/`unknown_adjacency`/a low-
   disclosure family, assert the geometry block is present and *identical*
   across many different seeds/targets (proving it carries no per-episode
   information) — extend `tests/test_phase12.py`'s leakage-guard style tests.

**Admission gate.** New geometry fields present in `dram.info`/reset
metadata for every disclosure level; identical across seeds; existing
non-leakage/fuzz tests (SPEC §8) still pass with the new fields included in
the fuzzed observation surface.

---

## P22 — `full_trace` feedback for discovery families

**Goal.** Let the issued-event trace (with `clk`/`row_hit`/`op`, coordinates
stripped per §0.1) actually reach the policy for tasks meant to be solved by
probing.

**Tasks.**
1. Set `feedback: full_trace` on the discovery-oriented task configs (the new
   Tier 2a config from P23; optionally `unknown_adjacency.yaml` too, since its
   existing 3-handle version becomes strictly more interesting with real
   timing feedback instead of blind brute force).
2. Confirm (regression test) that under `mapping: logical_only` +
   `full_trace`, `project_trace` still strips all six `COORD_KEYS` per event
   but preserves `op`/`clk`/`type_id`/`row_hit` — this is already covered by
   `test_phase12.py:93-98` in the abstract; add a task-level integration test
   that exercises it through `RowHammerTaskEnv.step()`'s real `feedback.
   trace_tail`, not just the `Disclosure` unit directly.
3. Make a probe sequence's timing **observable** — and note the trace-volume
   worry is inverted from what it first looks like. `_issue`
   (`phase2_env.py:232-257`) expands the command list, calls the worker **once
   per primitive**, and returns only the **last** primitive's observation; the
   worker `drain()`s its event sink on every call (`ramulator_worker.cpp:271`).
   So a multi-primitive `dram.issue` — including a HAMMER of thousands of RDs —
   already surfaces only the *final* primitive's `events[]` in `trace_tail` (and
   only the final `last_action.cycle_delta`). `full_trace` therefore does **not**
   flood the prompt with thousands of events; the actual defects are the
   opposite: (a) every *intermediate* probe event is silently dropped, so a probe
   sequence issued as one command list cannot convey the timing signal at all,
   and (b) per-primitive `cycle_delta` is lost. Cumulative `public_counters.acts`
   *does* survive across primitives (worker-side counter), which is why it is the
   most robust probe channel. Two ways to make probing work:
   - **No engine change:** issue each probe `RD` as its own one-primitive
     `dram.issue` and read the `acts`/`cycle_delta` deltas between calls. Costs
     one `tool_call` per probe — size the P23 budget accordingly.
   - **Aggregation (new plumbing in `_issue`):** accumulate a bounded **timing
     digest** across the expanded primitives (total ACTs, first/last `clk`,
     per-address hit/miss counts) so a multi-RD probe *or* a 25k-activation
     HAMMER returns one small summary instead of a single primitive's raw
     events. Prefer this if per-probe tool-call budget gets tight.

**Admission gate.** A probing sequence's `trace_tail` shows real `clk`/
`row_hit` values with zero coordinate leakage; a full-size HAMMER's trace
stays within a documented size bound.

---

## P23 — Tier 2a task family: bounded addressable candidate window

**Goal.** A family that mirrors real memory-templating: several **real,
directly addressable** candidate rows, only 1–2 of which are true aggressor
neighbors of a real (but not pre-identified) victim, sized so bank-conflict
timing (§0.2) plus a final budgeted hammer is enough to solve it — and sized
in *bands* so difficulty scales with candidate-window width.

**Tasks.**
1. New family (or a generalized `unknown_adjacency`) in
   `rowhammer_env/tasks/compiler.py`: instead of exactly 3 fixed offsets
   (`-row_bytes, +row_bytes, +2*row_bytes` — `phase5_env.py:152`), register
   `N` candidate handles at a **mix of same-bank/different-bank, adjacent/non-
   adjacent** offsets from the target, where `N` is a difficulty parameter
   (e.g. `easy: N=4`, `medium: N=16`, `hard: N=64`), keeping `victim:
   row_handle` (or upgrade the *candidates* to `logical` addresses directly,
   which is more realistic per §0.3's reframing — recommended: give
   candidates as real logical addresses, not just handles, since a policy
   needs to *combine* candidates arithmetically for the timing test, and an
   opaque handle-to-handle test is less natural than testing real numbers).
2. Budgets: raise `tool_calls` well above the Tier 0 `200` — each bank-conflict
   probe round costs at least one `dram.issue` tool call, and `N` candidates
   at even a handful of probe rounds each adds up quickly. Size from the
   reference policy's actual call count (P26) once it exists, don't guess.
3. `feedback: full_trace` per P22.
4. New `configs/tasks/bounded_sweep_{easy,medium,hard}.yaml` (or extend
   `unknown_adjacency_{easy,medium,hard}.yaml`) mirroring the existing
   `any_flip_{easy,medium,hard}.yaml` difficulty-band convention
   (`compiler.py:22-27`, `BAND_ACTS`/`BAND_WINDOW`).

**Admission gate.** Each band instantiates, compiles, and runs; a fixed-seed
fixture confirms exactly the intended fraction of candidates are true
aggressors; leakage guard confirms no candidate's identity as "the real one"
is derivable from anything but the trusted final flip outcome.

---

## P24 — Multi-turn interactive rollout engine

**Goal.** Give the *GRPO training* path genuine turn-by-turn interaction: the
model sees a real `step()` observation, decides the next tool call, sees the
*next* real observation, and so on, bounded by the task's real budget and a
max-turn cap, before terminating.

**Note what already exists.** The *evaluation* path is already multi-turn and
interactive: `rowhammer_env/llm/rollout.py:run_episode` calls
`policy.next_tool(observation, transcript)` once per real `step()` and loops to
`max_steps`/`done` (used by `verify_phase19.py` with `CIHammerFixturePolicy`).
The single-shot flow is specific to the **GRPO training** path
(`grpo_env.py`'s `ScriptedPolicy`/`evaluate_item`, which parses one completion
into an action list and replays it). So the lift here is **not** building the
interactive loop from zero — it is (i) generalizing `run_episode` for training
rollouts and (ii) feeding those multi-turn transcripts into the trainer
(the completion-mask problem, task 2 below). Concretely:

1. **Trajectory collection loop** (new module, e.g.
   `rowhammer_env/llm/multiturn_rollout.py`): for one task instance, run a
   real interactive episode: render the running transcript (system + user
   task description + alternating `assistant` tool-call turns and `tool`-
   result turns built from the real `Phase2Observation`), call the model for
   one short completion (one tool call, same parse path as
   `grpo_env.parse_actions`/`_coerce_action`), `await client.step(...)`,
   append the real result as the next turn, repeat until the model emits
   `episode.finish`, the task budget (`tool_calls`/`acts`/`cycles`) is
   exhausted, or a `max_turns` cap is hit. Reuse `rowhammer_env.client.
   RowHammerClient` (already async, already used by `rollout.py`) — this loop
   is a generalization of `run_episode`, not a replacement for the
   env/server/reward layer.
2. **Feeding this into GRPO.** TRL's default `GRPOTrainer` calls the policy
   once per prompt to get `completions`, then scores them — it does not
   natively interleave environment calls *during* generation. Two options,
   in order of preference:
   - **Check whether the installed TRL version (on the training host — not
     installed in this dev checkout, verify there first) exposes a custom
     rollout/generation hook** that lets you substitute the multi-turn loop
     above as the thing that produces a "completion" for a given prompt.
   - **If not, build the trajectory *outside* the trainer's generation
     step**: run the full multi-turn loop yourself (using the same model
     weights, via direct HF/vLLM inference calls) to produce a complete
     transcript, then hand `GRPOTrainer` a `(prompt, completion)` pair where
     `completion` is the *entire* multi-turn transcript text and a
     **token-level completion mask** marks only the assistant-authored spans
     as trainable (the interleaved `tool`-turn text — the real observations —
     must be masked out of the loss, the same way multi-turn SFT masks
     non-assistant turns). This is the standard pattern for tool-integrated
     RL and doesn't require a specific TRL feature, just care in how the
     `(prompt, completion, mask)` triple is constructed before it reaches the
     trainer.
3. **Reward attribution.** Exactly as today: the *final* trusted episode
   reward (`0.0`/`1.0` from `_trusted_success()`) is the reward for the whole
   trajectory — GRPO's group-relative advantage is computed per full rollout,
   not per turn. Do not invent a per-turn reward from anything but real
   trusted state (§4/P25 below governs the one place shaping is allowed).
4. **Throughput.** Each rollout is now `O(turns)` real environment round-trips
   instead of one. Expect the `env.concurrency`/`max_concurrent_envs` knobs
   (`configs/training/grpo_qwen8b.yaml`'s `env:` block) to need re-tuning —
   probably *lower* per-GPU generation concurrency in exchange for more
   parallel environment sessions, since round-trip latency, not model
   inference, becomes a larger share of step time. Measure before assuming.

**Admission gate.** A scripted multi-turn policy (see P26) run through this
loop produces the exact same trajectory/reward whether driven by the new loop
or by manually issuing the same calls one at a time via the existing
`RowHammerClient` — i.e., trace-equivalence, the same discipline P18 already
applies to the script sandbox.

---

## P25 — Curriculum and (carefully bounded) reward shaping

**Goal.** Make the sparse end-of-trajectory reward learnable in practice.

**Tasks.**
1. **Curriculum ordering**: Tier 0 (already saturated) → Tier 2a easy
   (`N=4` candidates, generous budget) → medium → hard (`N=64`, tighter
   budget) → *only if P26's reference policy proves it tractable* — a
   fully-hidden variant.
2. **Reward shaping, scoped narrowly.** SPEC §9 allows "optional small
   negative cost for budget exhaustion only in training tasks, not benchmark
   scoring" — stay inside that: any auxiliary shaping term must (a) derive
   only from already-trusted, already-disclosed-at-this-feedback-level state
   (e.g., a small bonus for issuing a probe that produces a *decisive*
   same-bank/different-bank signal, computed from the real `row_hit`/`clk`
   trace — not a hint about which candidate is correct), (b) apply only in
   training configs, never in eval/benchmark scoring, and (c) never be able to
   push reward above what a real, budget-respecting, actually-successful
   trajectory would earn. Keep the success-vs-format reward-weight split
   (`reward.success_weight`/`format_weight` in the training config) as the
   template for how a new shaping weight would be added — small, separate,
   clearly labeled.
3. Re-run `--dry-run`-equivalent validation (extend `dry_run()` in
   `scripts/train_grpo.py`, or write a multi-turn analog) with the P26
   reference policy after every curriculum change, exactly as the existing
   note in `configs/training/grpo_qwen8b.yaml` already insists for Tier 0/1
   curriculum edits.

**Admission gate.** Reference policy (P26) reaches its calibrated
success-rate window on every band; shaping terms provably vanish from eval
scoring; no shaping term is reachable that yields reward without a real flip.

---

## P26 — Reference probing policy + admission gate

**Goal.** Prove the environment's disclosed signals are *sufficient* — before
training anything — by hand-writing a deterministic policy that solves Tier 2a
using exactly the same tools/signals an LLM policy would have, and nothing
else.

**Tasks.**
1. Implement the bank-identification procedure from §0.2 as a real, tested
   algorithm: for each candidate, open the target row, `RD(candidate)`, then
   `RD(target)` **in a separate `dram.issue`**, and classify same-bank vs.
   different-bank from whether the target re-read forced a new `ACT` — i.e. the
   **`public_counters.acts` delta** and/or **`last_action.cycle_delta`** (or,
   under `full_trace`, the presence of `ACT`/`PRE` events). Do **not** key off
   the access's `row_hit`, which is always `true` (§0.2). Then, among same-bank
   candidates, spend the real hammer
   budget (`HAMMER` compact form) attempting each, checking
   `feedback.new_public_flips` after each attempt, stopping on success.
2. This becomes both:
   - `scripts/verify_phase2X_tier2.py`'s admission fixture (a "CI hammer
     fixture"-style deterministic policy, analogous to
     `CIHammerFixturePolicy` in `rowhammer_env/llm/__init__.py`, explicitly
     *not* advertised as the LLM policy — matches the P19 no-mock discipline
     of keeping a scripted fixture clearly labeled as a test fixture);
   - the reference success-rate baseline P25's curriculum bands are calibrated
     against, exactly as `BAND_WINDOW`/reference-policy calibration already
     works for `any_flip`/`target_row` (`compiler.py:22-27`).
3. **Run this before investing in P24's RL harness.** If it can't solve Tier
   2a `easy` within a reasonable budget, the task/budget/candidate-window
   parameters need adjusting *before* GRPO training is attempted — cheaper to
   discover with a deterministic script than with GPU-hours of RL.

**Admission gate.** Reference policy solves `easy`/`medium` within budget
across many seeds with a differential trace showing it actually used the
timing signal (a control that ignores the `acts`/`cycle_delta` timing and just
probes/hammers candidates in random order should do markedly worse or fail
budget, proving the signal is load-bearing, not incidental).

---

## 4. Open questions to resolve with the user before P24 is fully committed

1. **Tier 2b redefinition — RESOLVED (2026-07-11, user decision).**
   `hidden_target` is redefined to the real-attacker-knowledge model: the policy
   is given the victim's own numeric address and numeric candidate addresses (a
   real attacker knows their own allocations), and the hidden variable becomes
   the **address→bank mapping**, kept a per-episode secret so physical adjacency
   must be reverse-engineered from the timing channel (DRAMA threat model), not
   disclosed. This requires a realistic, source-traced secret address mapping
   (`IMPLEMENTATION_PLAN_V3.md` P24) — without it, a disclosed victim address +
   public row stride makes `victim ± row_stride` a trivial computation, not a
   discovery. The fully-opaque, zero-addressability variant is dropped (§5).
2. **TRL version capability on the training host** — verify what multi-turn/
   custom-rollout support actually exists before committing P24 to a specific
   implementation shape; this dev checkout has no `trl` installed to check
   directly.
3. **Budget sizing for Tier 2a** — deliberately left to P26's reference
   policy rather than guessed here; don't hardcode `tool_calls`/`acts` numbers
   into configs until the reference policy's real call count is measured.

## 5. Non-goals (explicitly out of scope for this plan)

- Fabricating a row-distance-dependent signal not grounded in the simulator's
  real physics (§0.2) — would violate SPEC §2.
- Any reward path that scores success from anything but
  `_trusted_success()`/real `new_public_flips` (SPEC §9).
- Solving the *fully-opaque, zero-addressability* `hidden_target` (an opaque
  handle with no numeric address and no addressable neighbors at all) — dropped
  in favour of the real-attacker-knowledge redefinition (§4.1); it asks for
  stricter-than-real knowledge and would need a fabricated row-distance signal.
  (Tier 2b as *redefined* — known address, secret bank mapping — **is** in scope.)
