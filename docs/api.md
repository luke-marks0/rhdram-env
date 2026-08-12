# RowHammer OpenEnv Policy API

Policies connect over the OpenEnv websocket client and receive
`Phase2Observation` objects. Reward is sparse and trusted: a policy cannot earn
reward by declaring success; `episode.finish` returns `0` until the simulator and
disturbance engine have produced the task condition.

## Tool Surface

- `dram.info`: disclosed address forms, commands, mitigations, task metadata,
  and remaining public budget.
- `dram.read`: read bytes from a disclosed logical, physical, or handle address.
- `dram.write`: write base64 bytes to a disclosed address.
- `dram.issue`: issue `RD`, `WR`, or `WAIT` commands through the Ramulator worker.
- `script.run`: run `python-rh-sdk` code in the P18 OS sandbox; `rh` calls are
  brokered back to the same tools. Bounded by the per-call `timeout_ms` arg (a
  wall-clock safety deadline) and the same episode budgets as its brokered inner
  calls — there is no separate `script_ms` budget.
- `episode.finish`: terminate the episode and receive reward from trusted state.

`rowhammer_env.llm.TOOL_SCHEMAS` exposes this surface as function/tool schemas
for tool-calling chat backends. `OpenAICompatibleToolPolicy` can call any
OpenAI-compatible chat-completions endpoint configured with
`RHD_LLM_CHAT_COMPLETIONS_URL`, `RHD_LLM_MODEL`, and optionally
`RHD_LLM_API_KEY`.

## Discovery families and the secret address mapping

Discovery families (`bounded_sweep` and `hidden_adjacency`) hide which candidate
rows are same-bank physical neighbours of the victim. For these, the
DRAM **address→bank mapping is a per-episode secret**: the worker runs the
authored `RoBaRaCoChRowXOR` mapper (RoBaRaCoCh decode plus a Row→Bank XOR with a
seedable offset), so `victim ± row_stride` lands in a *different* bank and
adjacency cannot be computed from a numeric address — it must be reverse-engineered
through the bank-conflict timing channel (`public_counters.acts` /
`last_action.cycle_delta` / the `timing_digest`, disclosed under `full_trace`). The
mapper identity/offset is never disclosed in any observation, error, handle name,
or digest. The disclosed geometry (`dram.info.geometry`) is unchanged — the row
stride is identical to the public mapper. Success is scored on the trusted decoded
victim flip, never on a policy read or claim. See
`docs/adr-0004-secret-address-mapping.md`.

The server-internal `DECODE` request (true coordinates under the active mapper) and
its inverse `ENCODE` (the linear address of a set of coordinates) are used only by
the task compiler, to build candidate sets, and by the disturbance model, to anchor
each victim row's flips at that row's own column-0 address. Neither is in the policy
tool surface and neither can be reached through `step`.

## Budgets, activation accounting, and termination

Every episode carries a resource budget (SPEC §8), disclosed in `dram.info` and in
each observation's `budget_remaining`:

- `tool_calls` — number of `step()` / tool invocations (one per policy action).
- `acts` — cumulative **DRAM row activations** (ACTs) issued, counted from the
  trusted issued-event stream (`public_counters.acts`). This is the physically
  load-bearing budget: read-disturbance is caused by ACTs to rows adjacent to the
  victim, and the empirical flip threshold `hcfirst` (sampled from the VTS25
  real-chip profile; ≈5000 activations double-sided here) is *itself* an activation
  count.
- `cycles` — simulated DRAM-controller cycles (a wall-clock proxy).

**Why an episode has a budget.** It models a real attacker's finite effort.
Read-disturbance is not free: an aggressor row must be activated `hcfirst` times
*before an auto-refresh restores the leaked charge*. The engine models that refresh
decay (SPEC §5, "Refresh/decay"): at each JEDEC refresh-window boundary the rank's
accumulated exposure is cleared, so a hammer that is too slow or spread across too
many rows is refreshed away and must re-accumulate, while a fast burst that crosses
`hcfirst` inside one window flips. The `acts` budget is the complementary cap on the
*total* activation effort an episode may spend. Together they make the real attack
problem — *concentrate enough activations, fast enough, on the right neighbour rows*
— the thing the policy must solve, rather than an unbounded grind.

**Activation-budget enforcement.** A single `dram.issue` may not spend more
activations than the remaining `acts` budget. Because the server expands compact
forms (`HAMMER`, `repeat`) and issues them one primitive at a time, it stops issuing
once the budget is reached: a `HAMMER` that would exceed the budget is truncated at
the budget and returns `BUDGET_EXCEEDED`. The over-budget activations never reach the
disturbance model, so **no flip is credited that the budget could not pay for**. A
policy therefore cannot brute-force a flip by hammering every candidate in one call;
on the discovery families it must use the bank-conflict timing channel to spend its
activation budget on the true same-bank neighbours. Families that do not budget
activations (the known-target legacy configs) are unconstrained. The numbers are
calibrated, not arbitrary: the flip threshold is the empirical `hcfirst`; the graded
`BAND_ACTS` budgets are sized so the flippable fraction is a smooth function of the
budget (three difficulty windows); the discovery-family budgets are the reference
probing policy's measured probe+hammer cost plus headroom.

**Termination** (SPEC §9): the success predicate becomes true, the policy calls
`episode.finish`, a budget is exhausted, an unrecoverable simulator/sandbox error
occurs, or the maximum simulated cycle count is reached.

## CI Fixture Policy

`CIHammerFixturePolicy` is a deterministic test fixture for CI. It is not
advertised as an LLM policy; it exists to prove the HTTP adapter, rollouts, and
metrics work without a network model dependency.
