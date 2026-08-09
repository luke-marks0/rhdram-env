# Decisions

A running log of design decisions that resolve ambiguities or `Drift`
entries in `SPEC.md`. Each entry records what was decided, the options that
were weighed, and why — so the reasoning survives even after the code and
spec agree.

---

## 2026-08-09 — Policy-issuable ops are `RD`/`WR`/`WAIT` only (resolves SPEC.md Drift #1)

**Decision.** Narrow the design bundle to the implemented op set instead of
widening the implementation. `spec/schemas/action.schema.json`'s command `op` enum
and `spec/SPEC.md §8`'s command form now list `RD | WR | WAIT` (+ the compact
`HAMMER`), and §8 states explicitly that `ACT`/`PRE`/`REF`/`RFM` are
controller-generated, are rejected with `ILLEGAL_COMMAND`, and appear only in the
issued-event stream. `spec/examples/action.issue.json` (was an `ACT`/`WAIT`/`PRE`
list) and `spec/examples/observation.reset.json` (`capabilities.commands`) were
updated to match. No behavior changed: the worker and `expand_commands` already
admitted exactly this set; `tests/test_issue_expansion.py` now pins the rejection
of all four controller ops.

**Context.** The bundle advertised `ACT | PRE | RD | WR | REF | RFM | WAIT` as
policy-issuable, while `cpp/simulator_service/ramulator_worker.cpp` (`issue`) and
`rowhammer_env/phase2_env.py` (`expand_commands`) admit only `RD`/`WR`/`WAIT` plus
the compact forms. Root `SPEC.md` already documented the narrow set and flagged the
mismatch as Drift #1.

**Options considered.**

- **(a) Widen the implementation** — accept `ACT`/`PRE`/`REF`/`RFM` from the policy.
  - *For:* matches the literal bundle text; a raw-command interface is a more
    expressive action space, and refresh/RFM-adjacent patterns could be expressed
    directly rather than implied.
  - *Against:* Ramulator's frontend takes memory *requests* (`Read`/`Write`); ACT,
    PRE, REF, and RFM are emitted by the controller and refresh/mitigation logic in
    response to that stream. There is no honest way to inject them without going
    around the controller — which would be exactly the mock/bypass path the no-mock
    invariant forbids, and would corrupt the issued-event stream that the
    disturbance model treats as ground truth. It is also unfaithful to the threat
    model: a real attacker issues loads/stores (and hopes for a scheduling
    outcome), and does not command rows open or closed.
  - *Note:* rejecting these ops before ticking Ramulator is load-bearing for D3 —
    an illegal command produces no issued events and cannot change disturbance
    state.

- **(b) Narrow the bundle** *(chosen)* — the schema/§8 describe the implemented
  policy-facing set, and the controller-generated ops are documented as such.
  - *For:* code and spec agree by cutting the side that was never implementable;
    keeps the invariant that everything on the tool surface really executes;
    removes a schema that would validate an action the worker always rejects.
  - *Against:* the bundle's original command form is lost as a statement of intent;
    the `@spec:sim-not-modeled` non-goal list is now the only record that a
    raw-command action space was ever contemplated (it already names
    "policy-issuable ACT/PRE/REF/RFM (controller-generated only)").

**Rationale for choosing (b).** The narrow set is not a shortcut — it is what the
simulator can model truthfully. `acts` remains the load-bearing budget precisely
because activations are a *consequence* of the request stream the policy controls,
which is the property that makes discovered patterns meaningful. Publishing a
schema that accepts commands the environment must reject is a correctness bug in
the contract, not a feature gap.

**Follow-up not taken.** The worker's rejection message
(`"<op> is not admitted in Phase 2"`) still names a phase; rewording it requires a
C++ rebuild and is already scoped as part of the P29 phase-name purge
(`spec/P29_RELEASE_READINESS.md §2.5`). The error *code* (`ILLEGAL_COMMAND`) is
unchanged and is what the contract and tests depend on.

---

## 2026-08-09 — Remove the `script_ms` budget (resolves SPEC.md Drift #4)

**Decision.** Drop `script_ms` entirely from the budget contract rather than
implementing it as a tracked cumulative counter. It is removed from the task
schema, `spec/SPEC.md §10`, the `@spec:rl-budgets` axis list, every
`configs/tasks/*.yaml`, the `spec/examples/*` task fixtures, `docs/api.md`, and
the `tests/test_probe_signal.py` budget fixture. The `script.run` tool itself is
unchanged.

**Context.** `script_ms` was declared as a fourth budget axis ("CPU budget for
`script.run`; `0` disables the script path") in the spec, schema, and all
configs, but nothing read it: `RowHammerTaskEnv._charge`
(`rowhammer_env/phase5_env.py`) charges only `tool_calls`/`cycles`/`acts`, and
`script.run` was bounded instead by its own per-call `timeout_ms` arg, a
`max_calls` cap tied to the remaining `tool_calls` budget, and — because a
script's inner `rh.*` calls are brokered back through `env.step` — the normal
`acts`/`cycles`/`tool_calls` charging. The documented `0 = disabled` gate was
also never implemented (the script path is unconditionally in `ALLOWED_TOOLS`).
This was logged as `SPEC.md` Drift #4.

**Options considered.**

- **(a) Implement `script_ms` properly** — a deterministic simulated-compute
  counter accumulated in `_charge`, plus honoring `0 = disables the script path`.
  - *For:* keeps the four-axis contract every config/schema/test already assumes;
    gives an independent lever on *offline compute* (a script can solve hidden
    structure with few tool calls and short wall-time, which `acts`/`cycles`/
    `tool_calls` do not price); would make the existing `script_ms: 0` in every
    config actually take effect if "off" was the intent; preserves optionality for
    compute-bounded training.
  - *Against:* there is no honest deterministic definition of "CPU-ms" — real CPU
    time is host-dependent and cannot drive termination/reward without breaking RL
    reproducibility and the "reward only from trusted deterministic state"
    principle, so a faithful implementation is exactly what is disallowed; a fixed
    per-op proxy is arbitrary and, if proportional to inner calls, redundant with
    `tool_calls`; adds a new counter/exhaustion path/gate/tests for a lever nothing
    uses; enabling the `0 = disabled` gate flips live behavior (scripts on -> off
    everywhere); the `0` overload (disabled vs. zero-remaining) stays awkward.

- **(b) Remove `script_ms` entirely** *(chosen)* — rely on the per-call
  `timeout_ms` deadline plus the load-bearing `tool_calls`/`acts`/`cycles`
  budgets.
  - *For:* most honest and simplest — the script path's real cost is already
    conserved (inner calls charge the real budgets; `timeout_ms` + `max_calls`
    bound runaway scripts), so `script_ms` adds no safety today; code and spec
    agree by cutting the dead side; sidesteps the determinism trap entirely.
  - *Against:* forecloses pricing offline compute without re-adding a field later;
    broad (if mechanical) diff across configs/spec/docs/test; leaves no per-task
    switch to disable the script path (none worked anyway) — if that capability is
    ever wanted it should be an explicit `allow_script` flag, not an overloaded
    budget of `0`.

**Rationale for choosing (b).** The load-bearing constraints (`acts` is the
physically meaningful one; `cycles`/`tool_calls` cap effort and turn count) plus
`timeout_ms` already bound the script path, so `script_ms` was redundant for
safety. A cumulative CPU-ms budget could not be implemented honestly without
introducing host-dependent nondeterminism into termination and reward, which the
project's reproducibility and trusted-reward principles forbid. Removing the dead
contract is the clarity-over-cleverness, honesty-over-convenience choice.

**Follow-up not taken.** No per-task script gating was added. If disabling the
script path per task becomes a requirement, add an explicit `allow_script`
disclosure/flag rather than reviving a budget-value sentinel.
