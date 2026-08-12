# Audit log — trusted reward / physics

Compiled audit findings for the trusted-reward / budget-honesty / physics
components of rhdram-env. Each entry records the component audited, the files and
functions in scope, the @spec contract, and the findings ranked by severity.

> **Revision note (re-verification pass).** Every finding below was independently
> re-checked against the code, the built worker (`build/phase2/ramulator_worker`),
> and `profiles/ddr4_vts25_v1/profile.json`. All 15 findings describe real code
> behavior. Four entries were **corrected** because the original mechanism,
> severity rationale, or arithmetic did not survive checking — each carries a
> `**Corrected:**` block naming what changed:
>
> - *predicates Finding 1* — severity High → **Medium**; the "difficulty bands are
>   bypassable" rationale is unsound and has been replaced.
> - *predicates Finding 2* — the mechanism was wrong (it blamed a dead code path)
>   and the proposed stopgap therefore does not fix the live bug. Rewritten.
> - *predicates Finding 7* — removed a sentence asserting the `acts` axis is fully
>   guarded, which contradicts the budget component's Finding 1.
> - *disturbance Finding 1* — mechanism confirmed against the real worker, but the
>   exploit sizing was wrong in both directions. Recomputed.
>
> The three component sections were originally produced independently and were not
> reconciled with each other; cross-references have been added where one section's
> conclusion depends on another's.
>
> **Test baseline at the time of this pass:** `2 failed, 262 passed, 6 skipped,
> 2 errors`. The two errors are a missing `configs/training/grpo_qwen8b.yaml`; the
> two failures are pre-existing and unrelated to anything below. No code was
> modified by either pass.
>
> **Open semantics have since been settled by the maintainer** — see
> [Settled decisions](#settled-decisions) at the end. Findings whose fix depended
> on one of those calls now name the chosen semantics inline.

---

## PoC triage — scope for the proof-of-concept training run

**Goal this triage is scoped against:** the environment must be able to train a
policy to *discover* RowHammer adjacency, and the physics must stay recognisably
RowHammer. It does **not** need every family correct, every disclosure level
airtight, or full modelling fidelity.

**What that scope actually covers.** `configs/training/grpo_curriculum.yaml` trains
exactly three families — `known_target_anybit` (tier 0), `bounded_sweep` (tier 2a),
`hidden_adjacency` (tier 2b). The policy-facing action space is `SYSTEM_PROMPT`
(`llm/grpo_env.py:261`): `dram.info`, `dram.read`, `dram.write`, `dram.issue`
(`RD`/`WR`/`WAIT`/`repeat`/`HAMMER`), `episode.finish`. **`script.run` is in
`TOOL_SCHEMAS` but is not named in the trained prompt.** A finding is in PoC scope
only if it is reachable from that action space, on one of those three families.

### Fix before training (1)

**disturbance F1 — RowPress dwell.** The only finding that is both a reward-hacking
vector on a trained family and reachable from a *documented* primitive. `WAIT` is
listed in `SYSTEM_PROMPT`; interleaving it with alternating RDs converts the acts
budget — which *is* the tier-2 difficulty knob — into a near-free resource.
Measured on shipped `bounded_sweep_easy` budgets, seed 7, true aggressors:

```
WAIT-interleaved:  success, acts_used    320, cycles 32,014,833, tool_calls  4
plain HAMMER:      success, acts_used  8,000, cycles    461,199, tool_calls  2
budgets:                  acts 20,000, cycles 240,000,000, tool_calls 40
```

A **25x activation discount**, at 13% of the cycle budget. The easy band is 4
candidates / 2 aggressors = 6 pairs; blind brute force costs 6 x 320 = 1,920 acts
and 6 x 32M = 192M cycles — **inside every shipped budget**. This confirms the
hypothesis Finding 1 flagged as unverified: the exploit defeats the
`use_timing=False` differential control of `@spec:train-reference-policy` on the
easy band, i.e. on the first discovery stage of the curriculum. (Medium/hard remain
protected by the cycle budget: 120 pairs x 32M ≈ 3.8B cycles.)

Note the finding's own "Reproduction is not budget-reachable" caveat is scoped to
the 5M-cycle families and does **not** hold for the discovery families, which budget
240M cycles.

Fix is the one named in the finding: match any `PRE*` prefix as a close, and settle
every open bank in the `(channel, rank)` for the all-bank form.

### Fix if cheap (2)

- **predicates F1 + F4 together** — tier 0 is on the training path and
  `_target_row_flip` credits a flip in a different physical row. Repointing the six
  families at `_target_bankrow_flip` plus carrying channel/rank (or asserting them
  single) is a few lines, and makes tier 0 score the same way tiers 2a/2b already
  do. Cost is re-seeding the tests named in F1.
- **budget F1 pre-issue guard, plus refusing a step once the episode is `done`** —
  see the *Missed* note under that finding. On the trained path the leak is one
  activation out of 15,000–130,000, which is immaterial to training; the reason to
  do it is that it makes budget honesty a property of the env rather than of
  whichever driver happens to honour `done`.
- **budget F2 — `HAMMER` with no sweep count** — in PoC scope after all, and the
  finding under-rates it for an *LLM* policy specifically. Every trained family
  hammers via `HAMMER`, and a policy that omits `pairs` gets a silent no-op with
  `reward=0` and no error — indistinguishable from a hammer that simply did not
  reach threshold, so there is nothing in the trajectory to learn from. Raising
  `BAD_SCHEMA` turns a silent dead turn into a correctable one. Cheapest item here.

### Written off for the PoC

Each is real; none is reachable-and-material on the three trained families from the
trained action space. Downsides are stated at each finding inline and summarised
here.

| Finding | Why out of PoC scope | Downside of not fixing |
|---|---|---|
| predicates F2 — `_pattern_target` | `pattern_target` is not in the curriculum | The family is scored wrong wherever it *is* used; reward 1.0 for a byte that demonstrably does not read the requested value. Blocks decision C. Highest-integrity defect left standing. |
| predicates F3 + disturbance F5 — column anchoring | **partly in scope after all** — see step 5 of the PoC ordering | The row-aligned anchor is one line and is a *no-op for every current caller* (all compiler candidates and the reference policy already hammer row-aligned addresses), so take it. What stays written off is the secret-mapper half (disturbance F2): under `RoBaRaCoChRowXOR`, row-aligning still lands the flip in the wrong bank's linear address. So `target_cell` / `pattern_target` become robust to a non-aligned probe, but `dram.read` fidelity on the two discovery families does not improve. |
| predicates F5 + decision B — latching | the discarded-win path needs `script.run`, absent from the trained prompt | A policy that discovers `script.run` gets reward decided by which predicate its family happens to use on identical trusted history. Latent, not dormant. |
| predicates F6 — degenerate mask/value | no shipped config reaches it | Config schema still accepts silently-unwinnable or free-win tasks. Fails silent, not closed. |
| disturbance F2 — secret-mapper flip address | reward is correct (`flipped_row_keys` is mapper-agnostic); only the overlay is wrong | `dram.read` is unusable as a verification channel on exactly the two discovery families — a successful flip reads as clean at the disclosed address, and a phantom flip is readable at `aggressor ± stride`. The policy still gets truthful `new_public_flips`, so training works; introspection does not. Also the honest fix needs a worker `ENCODE` op that does not exist. |
| disturbance F3 + decision C — `note_write` | realism only | The `all_ones` strata and `direction.bias_strength` in the profile stay unreachable; the latent model is "single/double x constant", not "x data pattern". Half of a fitted, validated profile is discarded. `@spec:tool-dram-write`'s `note_write` pointer stays false. |
| disturbance F4 + decision D — oracle clear | already parked; `mitigation: none` is the default | `@spec:mitigation-oracle`'s "faithful port" claim stays untrue. Owed a Drift entry regardless. |
| disturbance F6 — multi-bit byte collapse | ~1.5% of flips, accounting only | `new_public_flips` over-reports vs committed state; real multiplicity capped below `MAX_FLIPPED_BITS_PER_ROW`. |
| decision E — `oracle_refreshes` gate | no shipped config pairs `oracle` with a hidden victim | A one-line guard left undone; the safety of `oracle` + `hidden_target` rests on convention with nothing enforcing it. |
| decision F — reserved known row | `any_flip` / `profile_generalization` are calibration/eval families, not trained | Those two keep a ~3.8x band-independent shortcut, so their difficulty bands do not mean what they claim. Matters when you next quote a calibration number, not before. |
| *missed*: write poisons the overlay | affects `target_cell` / `pattern_target` | One `dram.write` makes those families permanently unwinnable *and* makes `dram.read` permanently lie about that byte. Unphysical: memory that can never be disturbed again. |
| *missed*: `episode.finish` uncharged | trivial | The terminal tool call is free; `tool_calls` accounting is not exact. |
| disturbance "Minor / latent" items | all dead or harmless today | `:546`'s unsourced `200.0` and unguarded `nrh == 0` are the two worth keeping on a list; the rest are cosmetic. |

### What writing these off costs, in one place

1. **SPEC.md is left knowingly wrong.** disturbance F1 (until fixed), F3, F4, and
   predicates F3 / disturbance F5 are divergences from *tagged* contracts. Deciding
   not to fix code does not discharge the SPEC discipline obligation — each one is
   owed a **Drift** entry naming the divergence, or the repo's own rule ("if code
   and SPEC.md disagree, that's a bug") stops being true and the spec stops being
   usable as the source of truth. This is the single largest cost of the write-off
   and it is cheap to pay.
2. **The environment is trustworthy for three families and not for the rest.** Any
   later result quoted from `target_cell`, `pattern_target`, `any_flip`,
   `profile_generalization`, or `mitigation_aware` is unsound until the relevant
   finding is closed. Worth stating wherever family-level results are reported.
3. **`dram.read` is not a reliable observation channel anywhere.** Between
   disturbance F2 (wrong bank under a secret mapper), F5 (column spill), and the
   write-poisoning bug, the read path disagrees with trusted state in three
   independent ways. Training does not depend on it; any human debugging the
   environment does.
4. **The write-offs are load-bearing on assumptions that are one config edit away
   from false** — single channel/rank (predicates F4), `script.run` absent from the
   prompt (predicates F5, budget F1), no `oracle` + hidden-victim pairing
   (decision E), `pattern_target` unused (predicates F2). None is enforced in code.
   Adding the fail-closed asserts is much cheaper than the fixes and removes the
   silent-failure mode from all four.

---

## Component: budget enforcement + command expansion

**Audited files / functions**

- `rowhammer_env/phase5_env.py` — `_charge`, `_issue_acts_ceiling`
- `rowhammer_env/phase2_env.py` — `_issue`, `expand_commands`
  (supporting: `_repeat_count`, `_TimingDigest`, `_digest_addr_key`, `_addr_value`)
- Cross-checked against `rowhammer_env/phase4_env.py` (`_from_worker`, `reset`,
  `_disturbance_overrides`) — phase4 does **not** override `step`/`_issue`/
  `_issue_acts_ceiling`, so phase2's `_issue` is the live per-primitive path and
  phase4's `_from_worker` commits disturbance (`disturbance.consume`) per primitive.

**@spec contracts**

- `@spec:rl-budgets` (SPEC.md §, "resource budgets and enforcement")
- `@spec:tool-dram-issue` ("command list + compact expansion")
- `@spec:invariant-budget-honesty` ("no over-budget credit")
- `@spec:invariant-trusted-reward`, `@spec:invariant-no-leakage`,
  `@spec:invariant-determinism` (cross-cutting)

---

### Finding 1 — Budget-honesty boundary: one over-budget activation is issued (and its flip credited) when the `acts` budget is exactly 0 (or already negative) — **Medium**

> **PoC: fix if cheap.** Immaterial to training on its own (one activation out of
> 15,000+); worth doing because it moves budget honesty from the driver into the env.
> Take fix (a) only — see *Corrections to this finding* below.

**Location:** `phase2_env.py:361-363` (post-issue ceiling check in `_issue`);
`phase5_env.py:210-219` (`_issue_acts_ceiling`); `phase5_env.py:228-233` (`_charge`).

**Mechanism.** The ceiling is
`acts_ceiling = self._acts_prev + max(0, budget_remaining["acts"])`. In `_issue`
the guard runs *after* the primitive has already been sent to the worker and its
disturbance committed:

```python
last = self._from_worker(self._worker.call(req))   # disturbance.consume() already ran
...
if acts_ceiling is not None and int(last.public_counters.get("acts", 0)) >= acts_ceiling:
    budget_truncated = True
    break
```

For `remaining >= 1` this is correct (the crossing primitive is genuinely within
budget). But when `budget_remaining["acts"] == 0`, `acts_ceiling == self._acts_prev`,
and the *first* primitive is issued unconditionally: an RD produces an ACT, `acts`
becomes `_acts_prev + 1 >= ceiling`, the loop truncates — but the ACT was already
sent, `disturbance.consume` already accrued it, and any resulting flip is now in
`self.disturbance.flips`, so `_trusted_success()` credits it in `phase5.step`. This
violates `@spec:invariant-budget-honesty` ("Activations the `acts` budget cannot pay
for are never issued to the worker"). `max(0, ...)` produces the same behavior for an
already-negative remaining.

**Reachable state.** `_charge` (`phase5_env.py:228`) treats exhaustion as
`budget_remaining["acts"] < 0`, so landing `acts` at exactly 0 leaves the episode
alive (`done=False`). The `dram.issue` path can't itself end at live-0 (it truncates
on `acts >= ceiling`), but `dram.read`/`dram.write` are **not** ceiling-guarded and
are charged only post-hoc in `_charge`; a read/write that brings `acts` remaining to
exactly 0 leaves the episode alive, and the next `dram.issue` then over-issues one
primitive as above.

**Why it's wrong.** The guard is a *post*-issue check, so it can never prevent the
boundary primitive; the code comment "The primitive just issued is within budget" is
false whenever remaining ≤ 0. Practical blast radius is small (a single activation
rarely produces a committed flip), but the invariant is absolute.

**Minimal fix (choose one):**

- (a) Make the ceiling a *pre*-issue guard: track running cumulative acts across the
  loop seeded from the pre-issue `_acts_prev`, and `break` at the *top* of the loop
  when `acts_so_far >= acts_ceiling`, so nothing is sent once the ceiling is reached.
- (b) Simpler and defensible: in `_charge`, terminate on
  `budget_remaining["acts"] <= 0` so a 0-acts episode never survives to a subsequent
  issue. One-character change, matches "exhaustion → done"; only removes the ability
  to issue free WAITs / `dram.info` at exactly-0 acts.

**Corrections to this finding.**

1. **The blast radius is not "a single activation".** It is one over-budget
   activation *per `dram.issue` call*, unbounded by the acts budget and capped only
   by `tool_calls`, because nothing refuses a step once the episode is `done` and
   `_acts_prev` advances with each leak. Measured (`target_cell`, `acts: 10`,
   alternating aggressors so every call opens a new row):

   ```
   call 10: acts=11 remaining=-1  BUDGET_EXCEEDED
   call 15: acts=16 remaining=-6  BUDGET_EXCEEDED
   call 19: acts=20 remaining=-10 BUDGET_EXCEEDED   <- 2x the budget, exposure 10->20
   ```

   (Holding the same aggressor row masks this: the follow-up RD is a row hit and
   emits no ACT. Alternating rows is what exposes it.) Finding 7's closing claim
   that "both axes leak exactly one boundary call" is therefore wrong for `acts`.
2. **Take fix (a), not (b).** (b) only addresses `remaining == 0`; it does nothing
   about the negative-remaining case above, so it does not close the invariant. It
   is also not a behaviour-free one-character change — it terminates every episode
   that lands exactly on its acts budget, before any final `dram.read` /
   `episode.finish`.
3. **The real missing guard is a post-`done` step refusal.** `phase2_env.step` /
   `phase5_env.step` never gate on episode state, so budget honesty is a property of
   whichever driver is in use. The rollout loops (`llm/rollout.py`,
   `llm/multiturn_rollout.py`) do stop on `done`; `script_sandbox._handle_tool` does
   not, and forwards `obs.done` to the script without acting on it. Fix (a) plus the
   refusal makes the invariant hold at the env boundary, which is where
   `@spec:invariant-budget-honesty` locates it.

---

### Finding 2 — `HAMMER` with no sweep count silently expands to nothing — **Low**

> **PoC: fix — cheapest item in the audit.** Under-rated here for an LLM policy:
> every trained family hammers via `HAMMER`, and a silent zero-sweep no-op is
> indistinguishable in the trajectory from a hammer that fell short. Raise
> `BAD_SCHEMA`.

**Location:** `expand_commands`, `phase2_env.py:157` —
`pairs = _repeat_count(command, keys=("pairs","count","repeat"), default=0)`.

**Scenario.** A policy issues `{"op": "HAMMER", "rows": [A, B]}` (omitting
`pairs`/`count`/`repeat`). `pairs` defaults to **0**, the inner loop emits zero RDs,
and the HAMMER contributes nothing. If it's the only command, `primitives == []`, the
`_issue` loop never runs, and `_issue` returns the initial no-op `last` observation
(empty feedback, `reward=0`, no error) at `phase2_env.py:333,382`.

**Why it's a rough edge.** Primitive `RD`/`WR`/`WAIT` default `repeat` to **1**
(`phase2_env.py:162`), so a policy reasonably expects `HAMMER` to do *something* by
default; instead it's a silent no-op with no diagnostic. Not a spec violation
(`@spec:tool-dram-issue` always shows `N` given) and not a leakage/reward problem, but
the inconsistent default is a footgun.

**Minimal fix.** Either default HAMMER's count to 1 for consistency, or raise
`BAD_SCHEMA` when a HAMMER supplies no sweep count. **Open question:** is a zero-sweep
HAMMER intended to be a legal no-op? If yes, leave it; if no, apply one of the two.

---

### Sections verified correct

- **`expand_commands` bounds/expansion.** `MAX_ISSUE_ACTIVATIONS` (2,000,000) is
  enforced incrementally in `_emit` (`phase2_env.py:142`), so both a large `repeat`
  and a large `pairs × len(rows)` HAMMER raise `ILLEGAL_COMMAND`; negative counts raise
  `BAD_SCHEMA` (`_repeat_count`, line 115); non-`RD/WR/WAIT/HAMMER` ops raise
  `ILLEGAL_COMMAND`; `ACT/PRE/REF/RFM` are correctly non-issuable. HAMMER's alternation
  (`for pair: for row in rows: RD`) is the bit-for-bit double-sided sequence
  `@spec:tool-dram-issue` describes.
- **Per-primitive worker driving + budget/disturbance accounting** run on the true
  expanded event count (one worker call per primitive, disturbance committed per call),
  consistent with `@spec:rl-budgets`. Aside from the Finding-1 boundary, the ceiling
  correctly bounds a bulk HAMMER to ~`remaining` activations.
- **`_charge` accounting** is monotone-safe (`max(0, ...)` clamps, no counter
  increases), charges tool_calls/cycles/acts from trusted `public_counters`, and only
  credits reward via `_trusted_success()` → `rewards.success_for(...)` — no
  policy-claim path. Consistent with `@spec:invariant-trusted-reward`.
- **`_TimingDigest` / `_digest_addr_key` leak-safety.** `per_addr` keys are the token
  the policy itself supplied (handle id, or logical/bare-int address), never a resolved
  hidden linear address; the `else` branch only resolves a linear address for physical
  inputs, which fail closed in `_addr_value` under non-physical disclosure before
  reaching the worker. `absorb` derives everything from real events (counts/clocks
  only). Consistent with `@spec:invariant-no-leakage` / `@spec:timing-channel`.
- **Determinism / episode reset.** `_acts_prev`, `budget_remaining`, `success`,
  `_compiled`, and handle lists are all reset in `phase5.reset` / `_register_handles`;
  the worker is rebuilt per reset so the acts counter starts at 0, making
  `_acts_prev = 0` correct. Consistent with `@spec:invariant-determinism`.

**Drift note.** Neither Finding 1 nor Finding 2 is currently listed in the SPEC.md
Drift section. No code was modified during this audit.

---

## Component: trusted per-family success predicates

**Audited files / functions**

- `rowhammer_env/rewards/predicates.py` (whole module) — `_target_row_flip`,
  `_target_bankrow_flip`, `_any_flip`, `_target_cell_flip`, `_pattern_target`,
  `PREDICATES`, `success_for`
- Read for verification (not in scope, but the predicates' inputs and callers):
  `rowhammer_env/tasks/compiler.py` (`FAMILIES`, `TaskSpec.compile`, `CompiledTask`,
  `disturbance_overrides`, `_int_or_none`), `rowhammer_env/disturbance.py`
  (`_hammer`, `_maybe_flip`, `_flip`, `_flip_positions`, `_victim`, `note_write`,
  `apply`, `restore`, `consume`), `rowhammer_env/phase5_env.py` (`_trusted_success`,
  `step`, `_script`, `_objective_and_target`, `_physical_target`),
  `rowhammer_env/phase4_env.py` (`_from_worker`, `_decode`),
  `rowhammer_env/geometry.py`, `rowhammer_env/mappers.py`,
  `rowhammer_env/tasks/disclosure.py` (`AddressResolver`),
  `cpp/simulator_service/ramulator_worker.cpp` (`decode`),
  `configs/ramulator/p1_external_ddr4.py`, `spec/schemas/task.schema.json`

**@spec contracts**

- `@spec:rl-reward` (SPEC.md:126, "sparse, trusted reward")
- `@spec:task-families` (SPEC.md:367, "each maps to exactly one success predicate")
- `@spec:invariant-trusted-reward` (SPEC.md:600) — names this file as an
  enforcement point
- `@spec:invariant-no-leakage`, `@spec:invariant-budget-honesty`,
  `@spec:invariant-determinism` (cross-cutting)
- Original bundle: `spec/SPEC.md` §7 families 4–7, §9 reward

Findings were reproduced at the engine level (real `DisturbanceEngine` on the
admitted `ddr4_vts25_v1` profile, DDR4 fixture geometry, no worker build needed).

---

### Finding 1 — `_target_row_flip` is bank-blind: credits a flip in a *different physical row* — **Medium**

> **PoC: fix if cheap (with F4).** `known_target_anybit` is tier 0 of the trained
> curriculum, so this predicate decides real training reward. Repointing the six
> families at `_target_bankrow_flip` is a few lines; the cost is re-seeding the
> tests named below.

**Location:** `rowhammer_env/rewards/predicates.py:23` —
`any(addr // dist.row_bytes == task.target_row for addr in dist.flips)`.

**Mechanism.** Under RoBaRaCoCh, `Row` is the most-significant mapped field, so
`addr // row_stride` recovers the row *index* and discards channel/rank/bankgroup/
bank. With the admitted DDR4 geometry (`bankgroup=4, bank=4`, stride 131072), **16
distinct physical rows share every row index**. The compiled task carries the true
victim bank (`CompiledTask.target_bank` / `target_bankgroup`) and the engine records
the full decoded key in `flipped_row_keys` — this predicate uses neither.

**Reproduced.** `known_target_anybit`, seed 13 → `target_row=20566`, victim key
`(0,0,0,0,20566)`. Double-siding rows 20565/20567 **in bankgroup 3 / bank 2** (a
legal address: `mapping: physical` lets the policy name the bank, and `bank_count`
is in the public geometry block per `@spec:env-geometry`):

```
SUCCESS after 50450 double-sided pairs on bg3/bank2
flip 2695716864 -> addr // row_bytes == 20566, decodes to {bg:3, bank:2, row:20566}
flipped_row_keys = {(0,0,3,2,20564), (0,0,3,2,20566), (0,0,3,2,20568)}
target key (0,0,0,0,20566) present? False
success_for -> True        # reward 1.0
```

**Why it's wrong.** `spec/SPEC.md:113` (family 5) is "any cell flips in a target
row". A flip in bank 2 for a task whose disclosed victim is bank 0 is a *different
row*, and the disclosed target is a full physical coordinate (`_physical_target`,
`phase5_env.py:319`, emits bank/bankgroup/row). Six families route through this
predicate: `known_target_anybit`, `target_row`, `hidden_target`,
`unknown_adjacency`, `mitigation_aware`, `low_disclosure`.

**Corrected: this is a false-positive bug, not a reward-hacking vector.** The
original entry ranked this High on a "second-order damage" argument — that because
every bank's row-`R` victim draws an *independent* lognormal threshold
(`_sample_threshold`, `disturbance.py:512`, keys on the full 5-tuple), a policy
spreading hammering across all 16 banks wins on the **minimum of 16 draws** and so
bypasses the `@spec:task-difficulty-bands` calibration. That argument is unsound:
spreading also multiplies the activation cost by 16, and the threshold reduction
does not come close to paying for it.

```
hisasa double|all_zeros, sigma_within_chip = 0.143
median of a single draw    19,114
E[min of 16 draws]         14,900          (0.78x)
ACTs to the first flip:  16 x 2 x 14,900 = 476,000
             vs bank 0:   1 x 2 x 19,114 =  38,000
```

Under any `acts` budget, spreading across banks is a **~12x loss**. A policy has no
way to observe per-bank thresholds, so it cannot select the lucky bank either. The
bands are not bypassable this way, and there is no strategy here for RL to find.

What remains — and what still justifies fixing it — is that the predicate credits
reward for a flip in a row the task did not ask for. The disclosed target is a full
physical coordinate; a flip in bank 2 for a bank-0 target is a wrong answer scored
as right. That is a correctness/trust defect in the reward signal, ranked **Medium**.

Note the same predicate has a second false-positive path that does *not* depend on
the bank: the `row 20569` flip in the run above belongs to the row-20568 victim and
is credited to row 20569 purely by linear arithmetic. See Finding 3's "related
corruption" and the disturbance component's Finding 5.

**Minimal fix.** Point all six families at `_target_bankrow_flip`. Verified this
works unchanged for the public mapper: `target_addr = target_row * row_stride`
decodes to `bankgroup=0, bank=0, column=0`, exactly the `(0,0,0,0,target_row)`
default the compiler already stores. Apply together with Finding 4 — switching to
the decoded key while channel/rank stay hardcoded leaves the key half-wrong.

Tests that seed `dist.flips` directly and would need to seed `flipped_row_keys`
instead: `tests/test_phase13.py:150,157,163,165,172`, `tests/test_phase11.py:100`,
and the `flips`-poking assertions in `tests/test_phase14.py`. (The original entry
named `tests/test_hidden_adjacency.py` here; that file already asserts on
`flipped_row_keys` and needs no change.)

**Interaction with Finding 5 (resolved by decision B).** `flipped_row_keys` is
never cleared by a write while `flips` is, so on its own this fix would silently
decide whether an overwriting write can retract a win for these six families.
Decision B settles it: success latches, so retraction cannot change reward whichever
container the predicate reads. Land the latch first anyway, so this change is
reward-neutral by construction rather than by argument.

---

### Finding 2 — `_pattern_target` credits success for a byte that does not read the requested value — **High**

> **PoC: written off.** `pattern_target` is not in `grpo_curriculum.yaml`.
> **Downside:** the highest-integrity defect in the audit stays live — reward 1.0
> for a byte that demonstrably does not read the requested value (reproduced: the
> predicate assumes `0x08`, `dram.read` returns `0xf7`). Any result from this family
> is unsound until fixed, and decision C stays blocked behind it.

**Location:** `rowhammer_env/rewards/predicates.py:56-60` — `byte = 1 << flipped_bit`.

**Corrected: the mechanism originally given here is a dead code path.** The first
pass attributed this to the engine's `1->0` direction — `note_write` recording an
`all_ones` row, `_victim` setting `direction = "1->0"`. That path cannot execute:
per the disturbance component's Finding 3, `note_write` has **no production
caller**, so `_row_pattern` is permanently empty, `_victim_pattern` always returns
`"all_zeros"`, and `direction` is always `"0->1"`. Read literally, the two entries
together said this bug was unreachable. It is not. The real path is simpler and is
live today.

**Mechanism (corrected).** The predicate reconstructs the victim byte as
`1 << flipped_bit`, i.e. it assumes the *stored* byte is `0x00`. The engine never
asserts that. Storage is the worker's own functional memory (`bytes_`,
`ramulator_worker.cpp:200,214`) and the disturbance overlay is applied on read as
`out[i] ^= 1 << bit` (`apply`, `disturbance.py:215`, via `phase4_env.py:180`). So
the byte a policy actually reads is `stored ^ (1 << bit)` for *whatever* is stored.
A plain `dram.write` of `0xFF` — an ordinary, legal tool call needing no engine
state at all — makes `stored = 0xFF`, so the byte reads `0xFE` while the predicate
assumes `0x01`. The engine's `direction` and `data_pattern` fields are not involved
at any point.

**Reproduced (real worker + real engine, `RowHammerDisturbanceEnv`, seed 1).**
`dram.write` `0xFF` over the victim byte, then `HAMMER` rows 9/11:

```
engine flips at target: bit 0 · direction 0->1 · data_pattern all_zeros
dram.read byte actually returns: 0xfe   (bit 0 == 0)
byte the predicate assumed:      0x01
_pattern_target -> True                  # reward 1.0
```

Note `direction 0->1` and `data_pattern all_zeros` — exactly the "safe" state the
original entry assumed made the predicate correct. Reward 1.0 for "the byte reads 1
under mask 1" when the byte demonstrably reads 0 there.

**Why it's wrong.** `spec/SPEC.md:115` — "success if a target mask/value appears
after disturbance". Nothing appeared. Not a violation of
`@spec:invariant-trusted-reward` in the letter (no policy claim is read), but it
substitutes an assumption for the trusted memory state the invariant points at.

**Fix.** The honest fix needs the actual byte, which the `(task, dist)` signature
cannot reach — this is an interface change: pass a trusted read (worker `READ` +
`dist.apply`) into the predicate.

**The stopgap originally proposed here does not work.** It was
`base = 0xFF if victim.data_pattern == "all_ones" else 0x00`, `byte = base ^ (1 <<
flipped_bit)`. In the reproduction above `data_pattern` is `"all_zeros"`, so it
computes `base = 0x00`, `byte = 0x01`, and still returns `True`. It reconstructs
the byte from an engine-side pattern table that is empty in production, not from
what is stored. Do not use it, including as a temporary measure — it would close
the ticket without changing the behavior.

**Sequencing (corrected).** This finding is live *now* and does not wait on the
disturbance component's Finding 3. Fixing Finding 3 (wiring `note_write`) adds a
second, independent way to reach the same wrong answer (a genuine `1->0` victim),
so Finding 3 must not land before this one — but this one stands alone.

---

### Finding 3 — one benign probe makes `target_cell` / `pattern_target` permanently unwinnable — **Medium**

> **PoC: fix — the arithmetic half only.** Row-aligning the victim anchor
> (`base = request_addr - (request_addr % row_bytes)` before `± distance * row_bytes`)
> closes both this finding and the cross-row spill, and needs none of disturbance
> F2's `ENCODE` work. It is a **no-op for every current caller** — measured
> `victim.addr % row_bytes == 64` only when deliberately hammering a non-aligned
> address; the compiler's candidates and `ReferenceProbePolicy` are all row-aligned,
> so no existing behaviour or test moves. What stays written off is the
> secret-mapper half: row-aligning still yields the wrong bank's linear address
> under `RoBaRaCoChRowXOR` (disturbance F2).
> **Correction to an earlier note in this file:** "clamp flip offsets to
> `[0, row_bytes)`" is a **no-op** — `_flip_positions` (`disturbance.py:564`) already
> draws `rng.randrange(self.row_bytes)`. The spill comes from `victim.addr` carrying
> the aggressor's intra-row offset, so the anchor is the only thing worth changing.
> **Correction:** decision A was not open. `@spec:sim-exposure-flip` (SPEC.md:296-304)
> already states "cell 0 is always column 0 / `first_bit`", so this is code
> contradicting a tagged contract — a Drift entry, not a semantics call.

**Location:** `predicates.py:46` and `:56` read `dist.flips[task.target_addr]`; root
cause is `Victim.addr` being pinned at first touch (`disturbance.py:470-494` —
`if victim is not None: return victim`, the address is never revisited).

**Mechanism.** `victim.addr` is `request_addr ± d*row_bytes`, so it inherits the
*column* of whatever address the policy happened to touch first. `_flip_positions`
(`disturbance.py:564`) puts cell 0 at `victim.addr + 0`, so the flip's linear
address is column-shifted and `target_addr` (column 0) is never populated.

**Reproduced.** `target_cell`, bit 3, seed 13:

- Hammer column 0 throughout → success after 2499 pairs, flip lands at
  `target_addr + 0`. Correct.
- Issue **one** RD at column 8 of row R+1 first (a legal probe — column 8 is inside
  the same row), then hammer column 0 for 200,000 pairs →
  `target_addr in flips == False`, `success_for` never True. Episode is unwinnable,
  reward 0, no error, no diagnostic.

**Related corruption (same root cause).** Because `victim.addr` is not row-aligned
when a non-bank-0 address is hammered, secondary multiplicity cells at
`victim.addr + byte_off` spill past the row window. In the Finding-1 run, one flip
belonging to the row-20568 victim was recorded at an address decoding to **row
20569** (`{bg:1, bank:2, row:20569, column:40}`). That is an independent
false-positive path for Finding 1's linear check, and it also makes a `dram.read`
of row 20569 show a flip that belongs to row 20568.

**Fix (decision A settled: column 0).** Normalize the victim's cell anchor to the
row's own column-0 base at victim creation and keep flip offsets inside
`[0, row_bytes)`. The alternative — having the predicate look up the trusted victim
by row key and compare `(bit, column offset)` — is rejected: it needs a predicate
signature change and a redefinition of "the target cell" in every task config, for
realism the research question does not currently require.

`@spec:sim-exposure-flip` must record the chosen semantics ("a flip's recorded
address is the victim row's column-0 base plus an offset within `[0, row_bytes)`")
as part of the same change, since it is currently silent on this.

---

### Finding 4 — `_target_bankrow_flip` hardcodes channel 0 / rank 0 — **Medium**

> **PoC: fix if cheap (with F1).** Correct today on `p1_external_ddr4.py`. If the
> repoint in F1 lands, take the assert form at minimum — it is the difference
> between a config change failing closed and every discovery episode silently
> scoring 0.

**Location:** `predicates.py:35` —
`key = (0, 0, task.target_bankgroup, task.target_bank, task.target_row)`.

**Mechanism.** The compiler reads only `bankgroup`/`bank` out of the DECODE
`addr_vec` (`compiler.py:371-372`), though the worker returns every level
(`ramulator_worker.cpp:179-183`). Today this is safe: `p1_external_ddr4.py` has one
controller and `rank=1`, so flip keys really are `(0,0,·,·,·)`.

**Why it's wrong.** `@spec:env-geometry` / `@spec:sim-standard-model` advertise the
engine as standard-generic with DDR5/HBM2 adapters, and `Geometry._row_stride`
explicitly derives from the reported level set "for *any* standard". The first
multi-rank or multi-channel config makes the key never match, and every discovery
episode (`bounded_sweep`, `hidden_adjacency`) silently returns reward 0 with no
error — a fail-open-to-zero, indistinguishable from a policy that simply failed.

**Minimal fix.** Carry `channel`/`rank` from the decode into `CompiledTask` and use
them in the key; or assert `level_sizes["channel"] == level_sizes["rank"] == 1` at
compile time and fail closed with `TaskConfigError`.

---

### Finding 5 — `flips` and `flipped_row_keys` disagree after a write; success is not latched — **Medium-Low**

> **PoC: written off.** The discarded-win path runs through `script.run`, which is
> in `TOOL_SCHEMAS` but is not named in the trained `SYSTEM_PROMPT`; through `step`
> the rollout loops honour `done` and stop.
> **Downside:** a policy that finds `script.run` gets reward decided by which
> predicate its family happens to use on identical trusted event history. The write-
> off rests on a prompt, not on code — nothing prevents `script.run` from being
> reached. See also the missed finding below: a write does not merely retract a win,
> it permanently poisons the row's overlay, which latching would hide rather than fix.

**Location:** `disturbance.py:269-272` (`consume` → `restore` clears `dist.flips`)
vs `disturbance.py:352` (`flipped_row_keys` is never cleared); success recomputed
per call at `phase5_env.py:122` and `:204`.

**Mechanism.** A `WR` overlapping the victim erases entries from `dist.flips` but
leaves `flipped_row_keys` populated. Success is recomputed, never latched.

**Scenario.** Via `step` this is mostly hidden (success is evaluated after every
call and terminates the episode). Via `script.run` it is not — though not for the
reason originally given. Each brokered `rh.*` call *does* go through
`phase5.step` (`script_sandbox.py:626`), so success is evaluated per inner call;
but the broker does not stop on `obs.done` (it just forwards the observation), the
script keeps running, and `_script` then **overwrites** `self.success` with a fresh
end-of-script evaluation at `phase5_env.py:204`. Net effect is as described: a
mid-script win is discarded if the state no longer satisfies the predicate. A script that
hammers to a flip and then writes to the victim row scores **0** for the
`_target_row_flip` / `_target_cell_flip` / `_pattern_target` / `_any_flip` families
and **1** for the `_target_bankrow_flip` families — identical trusted event history,
opposite reward, decided only by which predicate the family happens to use.

**Fix (decision B settled: latch, a flip is not retractable).** Latch success in
the env — once `_trusted_success()` returns true for an episode it stays true.
This matches `@spec:rl-episode-termination` ("the success predicate becomes true"
*ends* the episode, so there is no legitimate "after" in which to retract it) and
removes the erase-your-own-win foot-gun. It also makes the reward independent of
whether an episode is driven through `step` or `script.run`, which is the actual
defect here.

Concretely: `self.success` must become sticky in `phase5_env.py` — `step` at :122
and `_script` at :204 both currently *overwrite* it, and `_script`'s overwrite is
what discards a mid-script win. `flips` and `flipped_row_keys` may then continue to
disagree without affecting reward, but they should still be reconciled (a write
should either retract from both or neither) so that `dram.read` fidelity and the
predicate agree on what memory holds. Retraction-from-both is the honest choice for
the *overlay*; latching is the choice for the *reward*. These are not in conflict.

---

### Finding 6 — degenerate `mask` / `value` configs fail open or fail silent — **Low**

> **PoC: written off.** Unreachable from any shipped config.
> **Downside:** the task schema keeps accepting silently-unwinnable (`value: 0`,
> `value: 0x100`) and free-win (`mask: 0`) configs. Fails silent rather than closed,
> so the failure mode is "this task never trains" with no diagnostic.

**Location:** guard at `predicates.py:60` (`task.target_value != 0`); root cause is
unvalidated input in `TaskSpec.from_config` (`compiler.py:285` `_int_or_none`,
`:395-396` `& 0xFF`) plus `objective: {additionalProperties: true}` in
`spec/schemas/task.schema.json`.

**Scenarios** (none reachable from a shipped config today — latent):

- `value: 0` → the `!= 0` guard makes the task unwinnable forever, silently.
- `mask: 0` with nonzero value → `(byte & 0) == (value & 0)` is always true, so
  *any* flip in the target byte scores regardless of bit — the pattern objective
  becomes free.
- `value: 0x100` → `& 0xFF` silently truncates to 0 → unwinnable.

**Minimal fix.** Validate in `TaskSpec.from_config` — `mask != 0`,
`0 <= value <= 0xFF`, `value & mask == value` — raising `TaskConfigError`, then
delete the ad-hoc `!= 0` guard from the predicate, which currently papers over
unvalidated input.

---

### Finding 7 — `tool_calls` budget is charged *after* execution, so an over-budget flip is credited — **Low** (adjacent component)

> **PoC: fix if cheap (with budget F1).**
> **Correction:** the spec citation over-claims. `@spec:invariant-budget-honesty`
> (SPEC.md:613-616) is written entirely about the `acts` axis — "Activations the
> `acts` budget cannot pay for are never issued to the worker" — and says nothing
> about `tool_calls`. The code is not violating that invariant here. Making this a
> contract needs a SPEC.md edit, not just a code fix; the audit's framing skips that.

**Location:** `phase5_env.py:120-127` — `super().step()` runs, *then* `_charge`.

**Mechanism.** The call that drives `tool_calls` to `-1` has already executed on the
worker and its flips are already committed; lines 122-125 then set `reward = 1.0` on
top of the `BUDGET_EXCEEDED` error. A flip the `tool_calls` budget could not pay for
is credited, contra `@spec:invariant-budget-honesty`.

**Relationship to the budget-enforcement audit above.** Same shape as that
component's Finding 1, on a different axis: a post-hoc check that cannot prevent
the boundary call.

**Corrected.** The original entry closed with "the `acts` axis *is* properly
guarded within `dram.issue` … so no over-budget credit was found on that path".
That is wrong, and contradicts the budget component's Finding 1 in this same
document. `_issue_acts_ceiling` plus the per-primitive break at
`phase2_env.py:361-364` bounds a *bulk* expansion correctly, but because the guard
runs after the worker call it cannot prevent the boundary primitive, and at
`budget_remaining["acts"] == 0` the ceiling equals `_acts_prev` so the very first
primitive is over budget. Both axes leak exactly one boundary call; fix them
together (both are pre-dispatch guards).

**Minimal fix.** Refuse the step when `tool_calls` remaining is 0, before dispatch.

---

### Open question (not asserted as a bug) — the reserved known row under `_any_flip`

> **PoC: written off.** `any_flip` / `profile_generalization` are calibration and
> eval families; neither is in `grpo_curriculum.yaml`, and the trained families are
> all `target_kind: "known"` so the reserved row is their own target.
> **Downside:** those two families keep a ~3.8x band-independent shortcut, so their
> difficulty bands do not measure what they claim. Costs nothing until you next
> quote a calibration number from them.
> **Correction to decision F's fix location:** it proposes fixing in `_victim`
> ("apply the fixed threshold only when the reserved row *is* this task's target"),
> but the engine has no notion of the task's target — `known_target_row` *is* that
> notion. The cause is `compiler.py:352`, which deliberately installs
> `engine_known_row = RESERVED_KNOWN_ROW` for sampled families. The fix belongs in
> the compiler, plus letting the engine accept "no known row".

For sampled families the compiler still installs the fixed-threshold known target at
`RESERVED_KNOWN_ROW = 10` (`compiler.py:352`), so hammering rows 9/11 produces a
flip at the deterministic `known_threshold` (5000 in the fixture) rather than a
profile-sampled one. `_any_flip` credits it — literally correct per `spec/SPEC.md`
family 4, but it gives `any_flip` and `profile_generalization` a band-independent
shortcut, while their bands were calibrated against the sampled victim.

Sizing: hisasa `double|all_zeros` has a fixed known threshold of 5,000 against a
sampled median of 19,114 — a **~3.8× shortcut**, available to any policy that
happens to hammer low row indices.

**Resolved (decision F: the reserved row is inert unless it is the task's own
target).** In `_victim` (`disturbance.py:470-494`), apply the fixed calibrated
threshold only when the reserved row *is* this task's target; otherwise sample it
like any other row. The deterministic phase-4..9 fixtures keep their reproducible
known target (they set it as the target), and the graded families stop having a
free lane.

**Band calibration is not affected.** The reference used to calibrate the bands
already hammers the *sampled* victim, not the reserved row —
`_engine_any_flip_success` (`scripts/verify_phase13.py:198-201`) compiles the task
and hammers `ct.target_row`, which for a sampled family is drawn from
`randrange(1024, row_count - 1024)` and so is never row 10. The windows were
therefore calibrated against a world in which this shortcut does not exist; closing
it makes the environment match the calibration rather than invalidating it.
Monte-Carlo over the hisasa lognormal (`mu 9.858`, `sigma_within 0.143`,
`sigma_between 0.763`), optimal single-row play, reserved row removed:

```
easy    150,000 acts -> P(flip) 0.996   window (0.85, 1.0)   ok
medium   20,000 acts -> P(flip) 0.524   window (0.25, 0.75)  ok
hard      5,000 acts -> P(flip) 0.044   window (0.0,  0.20)  ok
```

All three stay inside `BAND_WINDOW`. Re-run `verify_phase13` to confirm on the real
engine, but do not plan for a re-calibration.

**No training impact.** `any_flip` and `profile_generalization` are calibration /
evaluation families; neither appears in `configs/training/grpo_curriculum.yaml`,
whose stages are `tier0_known_target`, `tier2a_bounded_sweep_{easy,medium,hard}`
and `tier2b_hidden_adjacency_{easy,medium,hard}`. The shortcut is not on any path
the policy currently trains on.

---

### Sections verified correct

- **`_any_flip`** matches `spec/SPEC.md:112` exactly. The engine is rebuilt per
  `reset` (`phase4_env.py:106`), so there is no cross-episode flip residue.
- **`_target_bankrow_flip`'s decoded-key approach** is the right idea and correct
  under the secret `RoBaRaCoChRowXOR` mapper — modulo Finding 4's hardcoded
  channel/rank.
- **`@spec:invariant-trusted-reward`** holds in the letter: no predicate touches
  policy-supplied input (no logs, stdout, claims, or `episode.finish` assertions),
  and `episode.finish` (`phase5_env.py:115`) returns 1.0 only if the predicate
  already holds. Findings 1 and 2 are wrong-condition bugs, not claim-trusting bugs.
- **`success_for` guards.** `None` task/engine fail closed to reward 0; an
  unregistered family raises `KeyError` rather than defaulting (no silent fallback);
  `PREDICATES` covers all 12 `FAMILIES` keys, and aliases (`known_target`) are
  canonicalized before `CompiledTask.family` is set, so no alias can reach the
  lookup.
- **`@spec:invariant-determinism`.** Predicates are pure functions of engine state —
  no RNG, no clock, no I/O, no memoized state across episodes.
- **`@spec:invariant-no-leakage`.** The module's only output is a bool consumed as
  `reward`/`done`; nothing derived from hidden coordinates, the mapper secret, or
  handle internals reaches an observation through it.

**Drift note.** None of Findings 1–7 is currently listed in the SPEC.md Drift
section (which holds one unrelated doc-naming entry). No code was modified during
this audit.

---

## Component: disturbance engine (event → flip physics)

**Audited files / functions**

- `rowhammer_env/disturbance.py` (whole module) — `DisturbanceEngine.__init__`,
  `consume`, `_hammer`, `_maybe_flip`, `_flip`, `_close_bank`, `_settle_dwell`,
  `_rowpress_factor`, `_refresh`, `_oracle_on_act`, `_victim`, `_victim_pattern`,
  `_module_offset`, `_sample_threshold`, `_sample_first_bit`, `_multiplicity_bits`,
  `_flip_positions`, `apply`, `restore`, `note_write`, `_row_key_of_addr`,
  `known_threshold`, `known_single_threshold`, `_known_target_key`
- Read for verification (callers / event producers, not in scope):
  `rowhammer_env/phase4_env.py` (`_from_worker`, `reset`, `_disturbance_overrides`,
  `_decode`), `rowhammer_env/phase2_env.py` (`_issue`, `_from_worker`),
  `rowhammer_env/phase5_env.py` (`_active_mapper`, `_disturbance_overrides`),
  `rowhammer_env/rewards/predicates.py`, `rowhammer_env/geometry.py`,
  `rowhammer_env/standards.py`, `rowhammer_env/profiles.py`,
  `rowhammer_env/mitigations.py`, `rowhammer_env/tasks/disclosure.py`,
  `rowhammer_env/tools/addressing.py`, `rowhammer_env/tasks/compiler.py`
  (`FAMILIES`, `compile`, `BAND_ACTS`, `LEGACY_BUDGETS`),
  `cpp/ramulator_extensions/issued_event_recorder.cpp`,
  `cpp/simulator_service/ramulator_worker.cpp`,
  `third_party/ramulator2/.../dram/impl/DDR4.cpp` (command names),
  `.../controller/refresh/impl/all_bank.cpp`,
  `.../controller/plugin/impl/oracle_rh.cpp`,
  `profiles/ddr4_vts25_v1/profile.json`, `configs/ramulator/p1_external_ddr4.py`

**@spec contracts**

- `@spec:sim-disturbance-engine` (SPEC.md:278) — event → flip overlay
- `@spec:sim-latent-vulnerability` (SPEC.md:286) — per-row latent state
- `@spec:sim-exposure-flip` (SPEC.md:296) — exposure accumulation / flip transition
- `@spec:sim-rowpress` (SPEC.md:307) — open-row dwell
- `@spec:sim-refresh-decay` (SPEC.md:314) — refresh window decay
- `@spec:sim-known-target` (SPEC.md:322) — fixed calibrated reference row
- `@spec:sim-profile-loading` (SPEC.md:339) — admitted, signed profiles only
- `@spec:mitigation-oracle` (SPEC.md:455) — faithful OracleRH port
- Also touches `@spec:tool-dram-read` / `@spec:tool-dram-write` (SPEC.md:166, which
  names `apply`/`note_write`/`restore`), `@spec:env-secret-mapper` (SPEC.md:260)
- Cross-cutting: `@spec:invariant-trusted-reward`, `@spec:invariant-no-leakage`,
  `@spec:invariant-budget-honesty`, `@spec:invariant-determinism`

Findings 1 and 2 were reproduced against the **real built worker**
(`build/phase2/ramulator_worker` + `p2_external_ddr4.yaml`), not synthesized events.

---

### Finding 1 — RowPress dwell is never settled by a real precharge; a `WAIT`-separated activation counts as ~18.7 hammers — **High**

*(Title corrected: the ~18.7× is the per-activation RowPress factor, not an
end-to-end budget discount. See "Exposure sizing — corrected" below for the actual
budget impact, which is far smaller.)*

> **PoC: FIX — the one blocking item.** See the triage section for the measured
> in-budget exploit on `bounded_sweep_easy` (320 acts vs 8,000; blind brute force of
> the 4-candidate window fits inside every shipped budget). `WAIT` is a documented
> primitive in the trained `SYSTEM_PROMPT`, so this is directly reachable by the
> policy being trained, on the families being trained.
> **Scope correction:** this finding's "Reproduction is not budget-reachable" caveat
> is true only of the 5M-cycle families. The discovery families budget 240M cycles,
> where the reproduction fits with ~8x headroom.

**Location:** `disturbance.py:264` (the `PRE` branch in `consume`),
`disturbance.py:356-364` (`_close_bank`); interacts with the ACT-driven fallback at
`disturbance.py:290`.

**Mechanism.** `consume` routes closes on `op == "PRE" | "PREab" | "PREA"`. Against
the admitted DDR4 worker **neither branch ever fires**:

- The per-bank precharge Ramulator emits is named **`PREpb`**
  (`DDR4.cpp:72` — `{"ACT","PREpb","PREab","RD","WR","RDA","WRA","REFab"}`), matching
  none of the three literals. `"PRE"` and `"PREA"` are names no Ramulator DRAM model
  produces.
- `PREab` *is* emitted, but only from the AllBank refresh request, whose `addr_vec`
  is rank-scoped — so the event carries `bankgroup: -1, bank: -1`. `_close_bank`
  builds `bank_key = (0, 0, -1, -1)`, which is never a key in `self._open`.

Live worker output confirming both:

```
Counter({'ACT': 40, 'RD': 40, 'PREpb': 39})
PREpb {'op':'PREpb','clk':40,'bankgroup':0,'bank':0,'row':11,...}
PREab {'op':'PREab','clk':9364,'bankgroup':-1,'bank':-1,...}
```

Dwell is therefore only ever settled by the fallback in `_hammer:290` — *next ACT to
a different row in the same bank* — which measures ACT→ACT, not ACT→PRE. Any idle
time in between (the row is physically closed, by `PREpb` or by the refresh `PREab`)
is counted as open-row dwell.

**Reproduced.** Real worker + real engine, seed 1, `known_threshold = 5000`,
alternating `RD row 9` / `RD row 11` with `WAIT 100000` between them:

```
iters 135  acts 270
Victim(threshold=5000, left=135.0, right=135.0, bonus=4752.3, flipped=True)
flips 1
```

The victim flips after **270 activations** against a calibrated hcfirst of 5000 —
exposure is `270 + 4752 bonus`. The identical hammer *without* `WAIT` accrues
`bonus 0.0` and does not flip at 800 ACTs. `WAIT` is an admitted policy primitive
(`@spec:tool-dram-issue`) and one `dram.issue` can expand thousands of them, so
tool-call budgets do not constrain it.

**Why it's wrong.** Directly contradicts `@spec:sim-rowpress` ("Applied only for
profiles that characterise RowPress; ordinary back-to-back traffic accrues no
bonus") — here traffic to a *closed* row accrues the saturated 18.67× bonus. It also
undercuts `@spec:invariant-budget-honesty` in substance: a flip is credited on 270
ACTs that the calibrated model prices at 5000.

**Exposure sizing — corrected.** The original entry's numbers were wrong in both
directions and one of them contradicted its own next paragraph. Recomputed:

Bonus accrues at saturation, so the optimal dwell is exactly
`ROWPRESS_DWELL_SATURATION` and the rate is `17.667 / 100_000 ≈ 1.77e-4` bonus per
cycle (hisasa). Two caps apply, not one:

1. the episode `cycles` budget, and
2. **the refresh window** — `_refresh` zeroes `victim.bonus` (along with
   `left`/`right`) at every window boundary, measured at 8192 REFabs ×
   tREFI 9348 ≈ **76.6M cycles**. Bonus cannot accumulate across a boundary, so
   the ceiling on usable bonus is `76.6e6 × 1.77e-4 ≈ 13,500`, however large the
   cycle budget is.

Against the shipped configs:

- **5M-cycle families** (`known_target_anybit`, `target_row`, `pattern_target`,
  … — all shipped configs set `cycles: 5000000`) → ~883 bonus. On a 5000 hcfirst
  that is 4,117 ACTs instead of 5,000: a **1.2× discount**, not the 2.4× originally
  claimed. Marginal, and irrelevant to families whose `acts` budget (100,000) is
  already 20× the threshold.
- **`bounded_sweep` / `hidden_adjacency`** (`cycles: 240000000`) → capped at
  **~13,500 per refresh window**, not the ~42,600 originally claimed, which
  ignored the decay the original entry itself noted two paragraphs later. Still
  material — hisasa's double-sided median is 19,114 and its floor is 5,000 — and
  reachable within those families' `acts` budgets (~766 ACTs buys the full
  13,500), so this remains the highest-impact case.

**Unverified claim, retained as a hypothesis only.** The original entry asserted
this is "enough to blind-brute-force the easy band's 4-candidate window **without
using the timing channel at all**", defeating the `use_timing=False` differential
control of `@spec:train-reference-policy`. That was never demonstrated and its
arithmetic came from the 42,600 figure now corrected. It is also structurally
constrained: `_settle_dwell` only fires on an ACT to a *different row in the same
bank*, so a candidate in the wrong bank accrues no bonus at all — the bug lowers
the cost of hammering a *correct* candidate rather than removing the need to find
one. **Measure this before sizing the fix**; do not treat the control as broken on
this entry's word.

**Reproduction is not budget-reachable.** The 270-ACT run above uses
`WAIT 100000` between activations, i.e. ~27M cycles — over every shipped
5M-cycle budget. It demonstrates the physics defect, not an in-budget exploit.

Separately, and correctly noted in the original: at 5M cycles the window boundary
is never reached, so the decay promised by `@spec:sim-refresh-decay` never fires
at all for non-discovery families.

**Minimal fix.** Match the real closing commands: treat any op starting with `"PRE"`
as a close, and for the all-bank form (`bank == -1`) settle *every* open bank in that
`(channel, rank)` rather than looking up a `(…, -1, -1)` key.

**Test gap.** The unit tests cannot catch this: `tests/test_phase14.py:49`
synthesizes `{"op": "PRE", ...}`, a command name the worker never emits.

---

### Finding 2 — under a secret mapper the flip is written to a linear address in the wrong bank — **High**

> **PoC: written off.** The reward is correct — `_target_bankrow_flip` reads
> `flipped_row_keys`, which is mapper-agnostic — and the policy still receives a
> truthful `new_public_flips` count in feedback, so the training signal is intact.
> **Downside:** `dram.read` is unusable as a verification channel on exactly the two
> discovery families. A successful flip reads clean at the disclosed victim address;
> a phantom flip is readable at `aggressor ± stride`; and the anchor is
> hammer-order dependent. Training does not depend on this; anyone debugging the
> environment does.
> **Correction to the fix:** "an injected `decode`/encode against the active mapper"
> is not available. The worker publishes only a `DECODE` op
> (`issued_event_recorder.cpp`, `set_decoder`) — there is no encode — and
> `mappers.is_python_projectable` returns True *only* for the public `RoBaRaCoCh`,
> by design. This needs a new worker `ENCODE` op or an inverse search, which makes
> it the largest item in the plan, not a "minimal fix", and it gates steps 4-5.

**Location:** `disturbance.py:298-300` (`victim_addr = request_addr ± distance *
self.row_bytes`), justified by the comment at `disturbance.py:296`.

**Mechanism.** That comment ("RoBaRaCoCh places Row as the most-significant field, so
`request_addr` decodes to `row`, so ±d row strides land on the neighbours' column 0")
is **false for `RoBaRaCoChRowXOR`** — the per-episode secret mapper
(`@spec:env-secret-mapper`) that the discovery families run by construction.

**Reproduced.** Secret mapper `RoBaRaCoChRowXOR{xor_offset: 1}`, victim
`(bg0, bank0, row 5000)` at true linear `655360000`; hammering its true same-bank
neighbours (located via the worker `DECODE` op):

```
flipped_row_keys {(0, 0, 0, 0, 5000)}      # trusted state correct → reward fires
flip addrs [655482880]                      # == aggr_lo + row_stride
decode(655482880) = {bankgroup: 3, bank: 3, row: 5000}
read at true victim addr 655360000 → all zeros
```

**Why it's wrong.** The *reward* is right — `_target_bankrow_flip` reads
`flipped_row_keys`, which is mapper-agnostic, so that half of P24 works as designed.
But `apply` / `restore` / `result.public_flips` are all keyed off `victim.addr`, so:

- `dram.read` at the victim's **disclosed** logical address shows no flip, breaking
  `@spec:tool-dram-read` ("Reads return base64 bytes with any disturbance flips
  applied") for `bounded_sweep` and `hidden_adjacency`;
- a phantom flip appears in an unrelated bank, readable there;
- the anchor is **hammer-order dependent** — reaching the victim from `row-1` gives
  `A_lo + stride`, from `row+1` gives `A_hi - stride`, and under the XOR mapper those
  are different addresses (they coincide under RoBaRaCoCh).

**Leakage assessment (flagged as a question, not asserted).** Probably not
exploitable: the phantom address is `aggressor ± stride`, which the policy already
knows, and flips only appear where it hammered. But it is a hidden-state-derived
write to an address the policy can read, so it deserves an explicit ruling rather
than an assumption.

**Minimal fix.** The engine needs the victim row's *true* linear address rather than
deriving it arithmetically — the clean route is the one the compiler already uses: an
injected `decode`/encode against the active mapper, so `Victim.addr` is the true
column-0 address of the decoded key. Failing closed (no overlay when the mapper is
not Python-projectable) is worse than today, since it loses `dram.read` fidelity
either way.

---

### Finding 3 — `note_write` is dead code in every production path; the whole `all_ones` half of the model is unreachable — **Medium**

> **PoC: written off (decision C deferred).** Realism only; no training-path effect.
> **Downside:** the profile's `single|all_ones`, `double|all_ones` strata and its
> `direction.bias_strength` stay unreachable, so the latent model is "single/double
> x constant" rather than "x data pattern" — half of a fitted, validated profile
> discarded. `@spec:tool-dram-write`'s `note_write` pointer (SPEC.md:171-172) stays
> false and is owed a Drift entry. The ordering constraint still holds if this is
> ever revived: predicates F2 must land first.

**Location:** `disturbance.py:227` (`note_write`); the gap is in
`phase4_env.py:171-191` (`_from_worker`), which calls `consume` and `apply` and
nothing else.

**Mechanism.** `note_write` is called only from `tests/test_phase14.py:131,306`. No
production path invokes it — `phase2_env`'s write path never touches the engine's
pattern table. Therefore `_row_pattern` is permanently empty, `_victim_pattern`
(`disturbance.py:496`) always returns `"all_zeros"`, and every victim in every
episode gets `direction = "0->1"` (`disturbance.py:484`).

**Consequences.** The profile's `single|all_ones` and `double|all_ones` strata, and
its `direction.bias_strength`, are unreachable. Stratum selection degenerates to the
aggressor axis alone, so `@spec:sim-latent-vulnerability`'s "single/double × data
pattern" is really "single/double × constant".

**Why it's wrong.** Diverges from `@spec:sim-latent-vulnerability` ("Stratum … is
inferred from the issued ACT neighbourhood **and the victim region's written
pattern**") and from `@spec:tool-dram-write`, whose **Defined in:** pointer names
`note_write` explicitly (SPEC.md:171-172). Not in the Drift section.

**Cross-reference — corrected.** The original entry claimed this made the
predicates audit's Finding 2 (`_pattern_target`) unreachable, and that fixing this
one would "activate" it. That is wrong: Finding 2 is reachable today via a plain
`dram.write` of `0xFF`, with `direction == "0->1"` and `data_pattern ==
"all_zeros"` — the engine's pattern table is not on its path at all (reproduced
there against the real worker). What is true is the reverse dependency: wiring
`note_write` adds a *second*, independent way to reach the same wrong answer (a
genuine `1->0` victim). **Fix predicates Finding 2 before this one**, or this
change widens a live bug.

**Minimal fix (decision C settled: wire it in).** Call
`self.disturbance.note_write(addr, raw_bytes)` in `phase4_env._from_worker` on the
`WR` branch, before `restore`. Note `consume` has the request's `addr`/`size` but
*not* the bytes — those live in the tool args, not the worker echo — so the call
belongs in `_from_worker`, or `consume`'s signature has to grow.

The profile already carries the `all_ones` strata and `direction.bias_strength`
this activates, so leaving it dead discards modelling fidelity that is already paid
for and already validated. **Strict ordering constraint:** predicates Finding 2 must
land first — wiring this while `_pattern_target` still assumes a zero-seeded byte
adds a second, independent route to the same false reward.

---

### Finding 4 — oracle counters are cleared once per refresh *window*, not per all-bank refresh — **Medium**

> **PoC: written off — already parked by decision D.** `mitigation: none` is the
> default and no trained task enables `oracle`.
> **Downside:** `@spec:mitigation-oracle`'s "faithful port" claim stays untrue. The
> Drift entry decision D calls for is owed whether or not the code is touched.

**Location:** `disturbance.py:421-424`, which sits *below* the window-boundary early
return at `disturbance.py:412` (`if count % self.refresh_window != 0: return`).

**Mechanism.** The oracle-counter clear therefore runs every **8192nd** REFab.
`oracle_rh.cpp:79-88` clears `m_table` on **every** REFab
(`is_refreshing && bank_targets == BankTarget::All`), and `@spec:mitigation-oracle`
states "counters cleared on all-bank refresh".

**Concrete divergence.** With tREFI ≈ 9364 cycles (measured: REFabs at clk 9380 and
18728) and tRC ≈ 40 cycles, at most ~234 ACTs to one row fit between consecutive
REFabs. The real plugin at the default `tRH = 2000`
(`disturbance.py:163`, `max(1, known_threshold * 2 // 5)`) would essentially **never**
fire a VRR; the Python port fires one every 2000 ACTs. The port is materially *more*
protective than the thing it claims to be a faithful port of.

**Minimal fix.** Hoist the oracle-counter clear above the window-boundary early
return.

**Deliberately flagged, not mechanically fixed.** Applying that one-line hoist makes
the `oracle` mitigation close to a no-op at the current `tRH` default and will move
the phase-7 protection fixture. This is a `tRH`-calibration decision (and `tRH`
defaults are explicitly listed under SPEC.md "Open / unsettled"), not an edit to make
in isolation.

**Resolution (decision D: mitigations are out of scope).** Mitigation work is
parked until the base environment is correct. Do **not** hoist the clear and do
**not** retune `tRH` as part of this audit's remediation. Instead:

1. Record this divergence as a **SPEC.md Drift entry** — the Python port is
   knowingly more protective than `oracle_rh.cpp`, so `@spec:mitigation-oracle`'s
   "faithful port" claim does not currently hold. Flagged, not resolved, is the
   honest state.
2. Exclude `mitigation_aware` from calibration and from the training curriculum.
   This costs nothing: `mitigation: none` is already the default, no
   `configs/training/` file references the family, and
   `configs/tasks/mitigation_aware_oracle.yaml` is the only shipped task that
   enables `oracle`. Its only other references are `scripts/verify_phase8.py:45`
   and `scripts/verify_phase13.py:169`.

Worth knowing whenever mitigations come back into scope: `oracle` today is a
hand-written Python re-implementation, not the shipped plugin
(`mitigations.py:83-89` — `execution="python_reference_port"`,
`ramulator_impl=None`, so no plugin is ever injected into the worker YAML even
though `oracle_rh.cpp` is in the tree). Whether to keep porting or to drive the
real plugin is the first question to revisit then; it is not a question to answer
now.

---

### Finding 5 — flip cells are anchored to the aggressor's column offset, not the victim row's column 0 — **Medium**

> **PoC: fix the row-aligned anchor; defer the mapper half.** This finding's own
> "Minimal fix" — `base = request_addr - (request_addr % self.row_bytes)` — is the
> right change and is *not* entangled with F2's missing `ENCODE` op: it is pure
> arithmetic on the linear address. The offset clamp in the same sentence is a
> no-op (`_flip_positions` already draws inside `[0, row_bytes)`); the anchor is the
> whole fix. Verified a no-op for every current caller, since the compiler's
> candidates and the reference policy are already row-aligned.

**Location:** `disturbance.py:298-300` (anchor), `disturbance.py:339-340` (write at
`victim.addr + byte_off`), `disturbance.py:564` (cell 0 at offset 0).

**Mechanism.** `_flip_positions` documents "Cell 0 is always column 0 /
`first_bit`", but cell 0 is written at `victim.addr + 0`, and `victim.addr` inherits
the aggressor request's intra-row offset. The offset is latched permanently by
whichever ACT first creates the victim (`_victim:470-473` returns early on a cache
hit and never revisits the address).

**Scenario.** Hammer `target_addr - row_bytes + 64` instead of
`target_addr - row_bytes`: the flip lands at `target_addr + 64`. `_target_row_flip`
still passes (`addr // row_bytes` is unchanged), but `_target_cell_flip`
(`dist.flips.get(task.target_addr) == task.target_bit`) and `_pattern_target` both
fail — the `target_cell` / `pattern_target` families become unsolvable for a policy
that hammers at a non-row-aligned address.

**Not currently triggered:** the compiler's candidates and the reference policy are
all row-aligned. An LLM policy picking arbitrary in-row addresses would hit it.

**Cross-reference.** Same root cause as the predicates audit's Finding 3 (one benign
column-8 probe makes `target_cell` permanently unwinnable) and its "related
corruption" note (multiplicity cells spilling past the row window into the next
row's address space). Fix once, at the engine, and both resolve.

**Minimal fix (decision A settled: column 0).** Row-align the anchor —
`base = request_addr - (request_addr % self.row_bytes)` before applying
`± distance * row_bytes` — and clamp flip offsets to `[0, row_bytes)` so
multiplicity cells cannot spill into the next row's address space. Subsumed by
Finding 2's fix, which must resolve the true column-0 address through the active
mapper anyway; do them as one change.

---

### Finding 6 — multiple flipped bits in one byte silently collapse — **Low**

> **PoC: written off.** ~1.5% of flips, accounting-only divergence.
> **Downside:** `new_public_flips` over-reports relative to committed state, and real
> multiplicity is capped below `MAX_FLIPPED_BITS_PER_ROW`. Since `new_public_flips`
> is the feedback channel the trained policy actually reads, a small over-report is
> present in the training signal — bounded and unbiased in direction, but present.

**Location:** `disturbance.py:340` — `self.flips[victim.addr + byte_off] = bit`,
against `self.flips: dict[int, int]` (one bit per byte address,
`disturbance.py:176`).

**Mechanism.** `_flip_positions` dedupes on the `(byte, bit)` pair, so `(0, 3)` and
`(0, 5)` both survive the `seen` filter. The loop then assigns `self.flips[addr]`
twice, keeping only the last, while `victim.flipped_bits` and `result.new_flips` each
increment twice. State and accounting diverge, and `apply` (`disturbance.py:215`) can
only ever flip one bit per byte — capping real multiplicity below
`MAX_FLIPPED_BITS_PER_ROW`.

**Severity.** Low: with `row_bytes = 131072` the byte-collision rate at 64 bits is
~1.5%. But it is `@spec:sim-exposure-flip`'s multiplicity contract being quietly
under-delivered, and `new_public_flips` over-reporting relative to committed state.

**Minimal fix.** Make the overlay a bitmask (`dict[int, int]` of OR-ed masks,
`out[i] ^= mask` in `apply`), or dedupe `_flip_positions` on byte offset alone.

---

### Open question (not asserted as a bug) — does `oracle_refreshes` leak `tRH`, and through it the calibrated hcfirst?

> **PoC: written off (decision E deferred).** No shipped config pairs `oracle` with
> a hidden victim, and no trained task enables `oracle` at all.
> **Downside:** a one-line fail-closed guard left undone, so the safety of an
> `oracle` + `hidden_target` pairing rests on convention with nothing enforcing it —
> and `@spec:invariant-no-leakage` (SPEC.md:606-609) names thresholds explicitly.
> This is the cheapest of the write-offs to reverse.

`phase4_env.py:184` writes `oracle_refreshes` into feedback, and
`Disclosure.project_feedback` (`disclosure.py:105-108`) strips it **only** at
`reward_only`. Since `tRH` defaults to `max(1, known_threshold * 2 // 5)`
(`disturbance.py:163`), a policy that counts ACTs between increments recovers
`known_threshold` exactly.

Nothing leaks today: the only shipped oracle task is `mitigation_aware`, whose
`victim: exact` disclosure already publishes `known_threshold` in reset metadata
(`phase4_env.py:157-159`). But `@spec:invariant-no-leakage` names **thresholds**
explicitly, and nothing prevents a config pairing `mitigation: oracle` with
`hidden_target` / `low_disclosure`.

**Resolved (decision E: gate it, fail closed).** Add `oracle_refreshes` to the keys
`Disclosure.project_feedback` strips whenever `expose_victim()` is false, alongside
the existing `public_flips` guard at `disclosure.py:103`. It costs nothing on any
current task (the only oracle carrier already discloses the threshold) and it stops
the config schema from silently accepting a leaking `oracle` + `hidden_target`
pairing. "No shipped config does the dangerous thing" is a convention with nothing
enforcing it; `@spec:invariant-no-leakage` names thresholds explicitly.

Note this is independent of decision D — the guard should land even while
mitigations are out of scope, precisely because it is what makes re-enabling them
safe later.

---

### Minor / latent (no action required today)

- **`disturbance.py:257-262`** — ACT events in a response whose request op is not
  `RD`/`WR` are silently dropped (`request_addr is None`). Verified that `WAIT`
  drains currently contain only `PREab`/`REFab`, so nothing is lost today — but it
  fails *open* on fidelity rather than closed.
- **`disturbance.py:266`** — `op.startswith("REF")` / `("RFM")` feed one rank-wide
  counter, so a per-bank `REFpb` or an `RFMab`/`RFMpb` would advance the JEDEC window
  and clear the whole rank's exposure. Dead for DDR4; live the moment
  `standards.py`'s DDR5_RFM adapters are admitted.
- **`disturbance.py:137`** — `and not domain["temperature_extrapolation"]` is dead;
  line 132 already raised if it were true.
- **`disturbance.py:174`** — `int(known_first_bit) & 0x7` silently wraps an
  out-of-range bit instead of failing closed. Harmless only because
  `compiler.py:355` applies the identical mask, so both sides agree by coincidence.
- **`disturbance.py:546`** — `gain = 1.0 + (mean_flips / nrh) * 200.0`. The `200.0`
  is an unsourced calibration constant with no spec pointer or `@spec:` tag, and only
  `mult[0]` (the first operating point) is ever read. `nrh == 0` would raise
  `ZeroDivisionError`; the empty-list case is guarded, the zero case is not.
- **`disturbance.py:205`** (`known_single_threshold`) — `aggr` is unpacked and unused.

---

### Sections verified correct

- **Profile admission** (`__init__:109-142`) — validation flag, standard/geometry
  cross-check, extrapolation and temperature-domain guards all fail closed with
  `PROFILE_REJECTED:`-prefixed messages that `phase4_env.py:111` maps to the stable
  error code. Matches `@spec:sim-profile-loading`.
- **Determinism.** Every RNG is `random.Random(sha256("seed:...").digest()[:8])`;
  `_module_offset`'s cache is a pure memo; `_flip_positions` regenerates the full
  sequence each call so positions are stable as multiplicity grows. A fresh engine is
  built per `reset` (`phase4_env.py:106`), so there is no cross-episode state.
  Matches `@spec:invariant-determinism`.
- **Exposure sign / geometry** (`_hammer:297-311`) — the `left`/`right` labelling is
  correct (an ACT at `row+d` increments the victim's `left`), and with the ±1/±2
  blast a single aggressor can only ever touch one side of a given victim, so
  single-sided hammering cannot be miscounted as double-sided. The
  `min(l,r)*2 + bonus` vs `max(l,r) + bonus` split matches `@spec:sim-exposure-flip`.
- **`_row_key_of_addr:588-619`** — field order (column → rank → bankgroup → bank,
  row as MSB) matches `AddressMapper.placement` exactly, including the
  prefetch-adjusted column span. Correct under the public mapper. (Both of its
  callers are currently dead per Finding 3.)
- **Trusted reward.** The engine exposes only `flips` and `flipped_row_keys`, both
  written solely from decoded worker events; nothing in `consume` reads
  policy-supplied text, logs, or claims. `@spec:invariant-trusted-reward` holds.
- **`_oracle_on_act:445-450`** — the fire condition (`count++`, fire at `>= tRH`,
  reset to 0, refresh the aggressor's blast neighbours) matches `oracle_rh.cpp`
  exactly. Only the clear-on-refresh half diverges (Finding 4).
- **Boundary handling.** `consume([])`, empty `note_write`, `restore` over a
  zero-size range, and the `victim_row < 0` / `victim_addr < 0` guards are all sound;
  `_flip_positions` cannot spin (64 max cells against ~1M available).

**Drift note.** Findings 1, 3 and 4 are behavioral divergences from *tagged* spec
sections and none is listed in the SPEC.md Drift section. If the code side is judged
wrong, the fixes belong in `disturbance.py` (plus `phase4_env.py` for Finding 3), not
in SPEC.md. No code was modified during this audit.

---

## Missed by the original passes

Found while re-checking the above; not covered by any finding in the three
component sections.

### M1 — a write permanently poisons the victim row's overlay — **Medium** (PoC: written off)

**Location:** `disturbance.py:223-225` (`restore`) against `disturbance.py:335-338`
(`_flip`'s `n_bits <= victim.flipped_bits` early return) and `disturbance.py:570`
(`_flip_positions` returns `cells[victim.flipped_bits:n_bits]`).

**Mechanism.** `restore` pops entries from `self.flips` but leaves `victim.flipped`
and `victim.flipped_bits` untouched. `_flip_positions` then never re-emits a cell
index below `flipped_bits`, so a cell erased by a write is never rewritten no matter
how much further exposure accrues.

**Reproduced** (`target_cell`, seed 1, real worker):

```
after hammer:                     flips@target = 6, flipped_bits = 1, success = True
after dram.write of 0x00 there:   flips@target = None,               success = False
after 400,000 further pairs
  (flipped_bits saturates at 64,
   exposure 402,500 vs threshold 5,000):
                                  flips@target = None, read = 00000000, success = False
```

**Why it matters.** This is a *third* independent route to Finding 3's "one benign
call makes the episode unwinnable", and neither decision A (anchoring) nor decision
B (latching) closes it — latching would hide it from reward while leaving
`dram.read` permanently wrong about that byte. It is also unphysical: a DRAM cell
that can never be disturbed again after being rewritten.

**Fix.** Reset the victim's `flipped` / `flipped_bits` for the cells `restore`
actually cleared, so the row can re-accumulate. Reconciling `flipped_row_keys` at
the same time is the "retract from both" half of decision B.

### M2 — the env never refuses a step after `done` — **Medium** (PoC: fix with budget F1)

Covered in the corrections under the budget component's Finding 1. Recorded here
separately because it is the enabling condition for that finding's real blast
radius, and because it is what makes `script.run` a budget hole rather than a
bounded one.

### M3 — `episode.finish` is not charged — **Low** (PoC: written off)

`phase5_env.py:113-117` returns before `_charge`, and `phase2_env`'s
`episode.finish` branch is likewise uncharged. The terminal tool call is free, so
`tool_calls` accounting is off by one for every episode that finishes explicitly.
Harmless today; it means `tool_calls` is not exactly the number of calls made.

---

## Settled decisions

Semantics the spec left open, now decided by the maintainer. Each names the
findings it unblocks and the SPEC.md section that must be updated alongside the
code, per SPEC.md discipline.

> **Read with the corrections at each finding.** Two entries below did not survive
> re-checking: **A** was never open (`@spec:sim-exposure-flip` already mandates
> column 0, so this is Drift, not a semantics call), and **F**'s fix is described at
> the wrong layer (the cause is `compiler.py:352`, not `_victim`). Under the PoC
> triage, **C**, **D**, **E** and **F** are all deferred and **A** is reduced to its
> offset clamp; only that clamp and the reward-condition half of the table are in
> scope now.

| | Question | Decision | Unblocks | SPEC.md |
|---|---|---|---|---|
| **A** | Is "the target cell" column 0, or the cell the engine chose? | **Column 0.** Row-align the victim anchor; keep flip offsets inside `[0, row_bytes)`. | predicates F3, disturbance F5 | `@spec:sim-exposure-flip` — currently silent, must state the anchoring rule |
| **B** | Is a flip retractable by an overwriting write? | **No — latch success.** Once true for an episode it stays true. Overlay retraction and reward latching are separate and may both hold. | predicates F5, and the semantics half of predicates F1 | `@spec:rl-episode-termination` — make the latch explicit |
| **C** | Should `note_write` be wired in? | **Yes.** Ordering constraint: predicates F2 first. | disturbance F3 | already specified (`@spec:tool-dram-write` names `note_write`) — this closes drift rather than creating it |
| **D** | What is the right `tRH` default? | **Mitigations are out of scope.** Do not hoist the clear, do not retune `tRH`. Record as Drift, exclude `mitigation_aware` from calibration and curriculum. | nothing — explicitly parked | new **Drift** entry; leave `tRH` under "Open / unsettled" |
| **E** | Gate `oracle_refreshes` on `expose_victim()`? | **Yes, fail closed.** Lands independently of D. | disturbance open question | `@spec:disclosure-leakage-guard` |
| **F** | Is the reserved known row inert when it is not the task's target? | **Yes.** Sample it normally unless it is the task's own target. | predicates open question | `@spec:sim-known-target` — scope the fixed threshold to the target case |

## Implementation ordering — PoC scope

This is the ordering to execute now. The full ordering below it is retained for
when the environment is taken past the proof of concept.

1. **disturbance F1 — `PRE*` close matching.** The only blocking item. Match any op
   starting with `"PRE"` as a close; for the all-bank form (`bank == -1`) settle
   every open bank in that `(channel, rank)`. Then re-run the measurement in the
   triage section: the WAIT-interleaved run on `bounded_sweep_easy` should cost the
   same order of activations as the plain HAMMER, not 25x fewer. `tests/test_phase14.py:49`
   synthesizes `{"op": "PRE"}`, a name the worker never emits — update it to `PREpb`
   or the test still cannot catch a regression.
2. **budget F2 — `HAMMER` without a sweep count raises `BAD_SCHEMA`.** One line.
3. **predicates F1 + F4 — repoint the six families at `_target_bankrow_flip`**, and
   either carry channel/rank from the decode or assert them single at compile time.
   Re-seed the tests F1 names.
4. **budget F1 fix (a) + refuse a step once `done`** (M2). Pre-dispatch guard on
   both axes.
5. **Row-align the victim anchor** (disturbance F5 / predicates F3, arithmetic half
   of decision A): `base = request_addr - (request_addr % self.row_bytes)` before
   applying `± distance * row_bytes`. Closes the "one benign probe bricks the
   episode" path *and* the cross-row spill. Needs none of disturbance F2's `ENCODE`
   work, and is a no-op for every current caller — do **not** also add an offset
   clamp, `_flip_positions` already draws inside `[0, row_bytes)`.
6. **Record Drift entries** for everything left unfixed that diverges from a tagged
   contract: predicates F3 / disturbance F5 vs `@spec:sim-exposure-flip`, disturbance
   F3 vs `@spec:tool-dram-write`, disturbance F4 vs `@spec:mitigation-oracle`. Not
   optional — SPEC.md discipline treats an unrecorded divergence as a bug in the
   repo's own terms, and a written-off fix is still a divergence.

Steps 1–5 are each small and independent; none needs a decision from the Settled
table, and none touches the `ENCODE`-op work that gates the deferred plan.

### Handoff notes — read before starting

**Test baseline.** Run `python -m pytest tests/ -q`, **not** bare `pytest` — root
collection pulls in `third_party/` and dies with 72 collection errors before running
anything. Expected green-state baseline is `2 failed, 262 passed, 6 skipped,
2 errors`:

- `test_phase0.py::test_old_hidden_project_files_are_absent` fails only because
  `.pytest_cache` exists (running pytest creates it). Pre-existing, self-inflicted.
- `test_probe_shaping.py::test_shaping_is_opt_in_and_off_by_default` — pre-existing.
- both errors are a missing `configs/training/grpo_qwen8b.yaml`.

Any *other* red is caused by the change. Do not "fix" the two pre-existing failures
as part of this work.

**Step 1 must be followed by the reference gate.** The bands were calibrated against
an engine in which the RowPress bug was live, so removing the bonus could in
principle move them. `ReferenceProbePolicy` (`llm/policies.py`) emits no `WAIT` —
only `scripts/verify_phase{2,14}.py` do — so no bonus it relies on should be lost,
but confirm rather than assume: `python3 -m unittest tests.test_curriculum` must
still clear `reference_min_success: 1.0` on every stage. If a band drops, that is a
calibration decision to escalate, **not** a reason to soften the fix.

**Step 1's regression test.** `tests/test_phase14.py:49` synthesizes
`{"op": "PRE", ...}`, a command name the worker never emits. A prefix check makes
that test keep passing while still not exercising the real path — change it to
`PREpb`, and add a case for the all-bank form (`bankgroup: -1, bank: -1`), or the
bug can silently return.

**Step 3 interacts with a deferred decision.** Finding 1 says to land the success
latch (decision B) first so the repoint is "reward-neutral by construction". The PoC
ordering defers latching, so that guarantee does not hold. The direction of the
change is safe — `flipped_row_keys` is never cleared, so the repoint can only make
a win *stickier*, never lose one, which is the same direction latching moves — but
it is a behaviour change, not a no-op. Say so in the change report rather than
inheriting Finding 1's claim.

**Reproducing the step-1 measurement.** The triage numbers came from an ad-hoc
script that is not in the repo. To re-measure: build `RowHammerTaskEnv` on
`configs/tasks/bounded_sweep_easy.yaml` with its **shipped** budgets (no override),
`reset(seed=7)`, take the true aggressors as
`[ct.target_addr + c.offset for c in ct.candidates if c.is_aggressor]`, then compare
two strategies to first success — `{"op":"HAMMER","rows":aggr,"pairs":2000}` versus
blocks of `RD aggr[0] / WAIT 100000 / RD aggr[1] / WAIT 100000` — reporting
`initial_budgets - budget_remaining` for `acts` and `cycles`. Consider committing
this as `scripts/verify_rowpress_close.py` so the fix has a permanent guard.

**Repo discipline (from CLAUDE.md, not restated elsewhere here).** No hacks or
workarounds — if a fix cannot be done properly, stop and say so. Tag implementation
and tests with `# @spec:<tag>` for the contract each touches. Update SPEC.md in the
same change where architecture or the RL formulation moves. Do not run `git add` or
`git commit` — propose them. Report anything fragile explicitly.

## Implementation ordering — full (deferred past the PoC)

Grouped so shared root causes are fixed once, and so no fix widens a bug a later
fix was going to close. Steps 1–5 are the base environment; nothing here depends
on the mitigation work parked under decision D.

1. **`_pattern_target`** (predicates F2) — live now, independent of everything else,
   and a hard prerequisite for step 4. Needs the interface change (pass a trusted
   read into the predicate); the `data_pattern` stopgap does not work.
2. **Engine address anchoring** (decision A) — disturbance F5 + F2; predicates F3
   falls out of the same change. Resolve the victim row's true column-0 address
   through the active mapper instead of `request_addr ± d * row_bytes`.
3. **Budget boundary guards** — budget F1 + predicates F7. Both are pre-dispatch
   refusals on the same defect shape; one change.
4. **Physics fidelity** — disturbance F1 (`PRE*` close matching), then disturbance
   F3 (`note_write`, decision C). Re-measure the discovery-family exposure numbers
   after F1; the sizing in that finding is corrected but the "defeats the
   `use_timing=False` control" claim is still unverified.
5. **Reward-condition correctness** — predicates F1 + F4 together (decoded key with
   channel/rank carried through), latching per decision B, decision F's known-row
   scoping, decision E's disclosure gate, and predicates F6 config validation.
   Re-run `verify_phase13` afterwards to confirm the bands still land in window
   (they are expected to — see decision F).
6. **Parked** — disturbance F4 and everything else mitigation-related, until the
   base environment is correct and mitigations are back in scope.

Loose ends that are not decisions and not blocked, listed so they are not lost:
disturbance F6 (multi-bit byte collapse), and the "Minor / latent" items —
`disturbance.py:137` dead guard, `:174` silent bit wrap, `:205` unused unpack,
`:546` unsourced `200.0` constant and unguarded `nrh == 0`.
