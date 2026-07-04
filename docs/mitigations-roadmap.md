# Mitigations: current state, integration decision, and roadmap

Status: proposal / open issues. Owner: TBD. Supersedes nothing; feeds a future
P16 follow-up.

This document records (1) how mitigations work in this repo today, (2) the
issues blocking a full mitigation menu, (3) a decision on *how* to integrate the
real Ramulator mitigations, and (4) answers to two design questions: can
mitigations be combined, and is TRR (and the other "basic" ones) implemented.

---

## 1. Architecture recap (why this is subtle)

The environment is deliberately **two-layer** (SPEC §4–5, P11 design decision):

- **C++ (Ramulator worker)** simulates the real DRAM **command stream** —
  timing, scheduling, row-buffer state, refresh, address mapping. It emits every
  issued `ACT/PRE/RD/WR/REF/RFM` with decoded coordinates. **Ramulator does not
  model bit flips.**
- **Python (`rowhammer_env/disturbance.py::DisturbanceEngine`)** consumes that
  issued-event stream and applies a **statistical flip model** calibrated from
  real chip data (the VTS25 profile). **All bit flips happen here.**

Consequence: a Ramulator **mitigation** protects a victim only by **changing the
command stream** — inserting preventive refreshes, or throttling/deferring
activations. That change only becomes "protection" once the **Python flip model
correctly interprets it** and resets the affected rows' accumulated exposure.

This split is why "just turn on the C++ mitigation" is not automatic.

---

## 2. Current state — what we actually have

Admitted and working (`rowhammer_env/mitigations.py::ADMITTED`):

| Name | Implementation | Runs where | Status |
|---|---|---|---|
| `none` | baseline auto-refresh only | worker config | real |
| `oracle` | port of Ramulator `OracleRH` | **Python** (`DisturbanceEngine`) | real, validated vs `oracle_rh.cpp` (P16 gate proves it prevents the target flip) |

Known but **not admitted** (fail closed with `UNAVAILABLE_CAPABILITY`, hidden from
`dram.info`): `para`, `twice`, `graphene`, `blockhammer`, `hydra`, `rrs`, `aqua`,
`rfm`, `prac`, `custom`.

The C++ implementations for most of these already ship in the vendored tree
(`third_party/ramulator2/src/ramulator/controller/...`) — see the inventory in §6.

### Issue 2a — the Ramulator-plugin injection path is dead code
`mitigations.py::ramulator_controller_plugins` only injects a controller plugin
when a capability has `ramulator_impl` set **and** the source mentions it. Both
admitted mitigations (`none`, `oracle`) leave `ramulator_impl=None` (oracle runs
in Python), so `worker_config_for_mitigation`'s YAML-injection branch never
executes today. It is untested scaffolding for future mitigations.

### Issue 2b — Python refresh consumption is too coarse for real mitigations
`DisturbanceEngine.consume` (`disturbance.py:246-253`) already routes `REF/RFM`
events to `_refresh`, **but** `_refresh` (`disturbance.py:380`) only models
**all-bank auto-refresh** as a statistical window decay. It does **not** honor a
**targeted, row-addressed** preventive refresh — which is exactly what real
mitigations emit. The oracle sidesteps this by special-casing targeted victim-row
restoration in Python (`_oracle_on_act`). This is the core engineering gap.

### Issue 2c — the API is single-mitigation only
The task/config surface is `mitigation: {name, params}` (one mitigation). Real
hardware stacks defenses (see §5), and Ramulator supports it natively — the API
does not.

---

## 3. Decision — how to integrate (the "best way")

**Decision: wire Ramulator's real C++ plugins/controllers (do NOT reimplement
each mitigation in Python), and generalize the Python refresh/restoration path
once so new mitigations flow through it.**

### Why integration, not Python reimplementation
This is not just a preference — it is what the spec requires:

- SPEC §2 constraint 3: *"Use Ramulator 2.1 for … controller functionality
  wherever available."*
- SPEC §6: *"Use a Ramulator 2.1 implementation directly if it exists and passes
  conformance tests. If a mitigation exists only in another branch/version or
  paper artifact, port it as real code with paper-to-code tests…"*

Since Ramulator ships PARA, Graphene, TWiCe, Hydra, RRS, AQUA, RFM, three TRR
variants, OracleRH (plugins) and BlockHammer, PRAC (controllers), the spec-aligned
path is to **use those directly**. Reimplementing them all in Python would:
duplicate validated C++, risk drift, and contradict SPEC §6. `oracle`-in-Python is
the sanctioned exception — it is the *idealized reference* defender and was still
validated against `oracle_rh.cpp`; keep it as the ground-truth reference.

### The refinement that makes this scale
Rather than special-casing each mitigation in `_refresh`, **generalize refresh
consumption once**: any issued `REF/RFM/VRR`-class event that decodes to a
specific row should restore *that row's* exposure (bank+row keyed), while
undecoded/all-bank refreshes keep the existing window-decay behavior. Then adding
a mitigation reduces to:

1. set `ramulator_impl` (+ param mapping) in `mitigations.py`, and
2. write its conformance/differential test.

No per-mitigation flip-model code. Combined mitigations then compose for free
(the model just sees the union of emitted refreshes/deferrals).

### Two mechanism classes (integration effort differs)
- **Refresh-inserting** (OracleRH, PARA, Graphene, TWiCe, TRR×3, RFM, Hydra):
  protection = extra targeted refreshes → needs the generalized targeted-refresh
  handling above.
- **Activation-throttling / deferring** (BlockHammer, PRAC): protection = fewer /
  delayed `ACT`s → this already shows up in the ACT stream and refresh timing, so
  it is expected to need little or no new flip-model logic. *(Needs verification
  that the worker surfaces deferred/dropped ACTs faithfully.)*
- **Row-remapping** (AQUA, RRS, Hydra): these physically relocate rows. The
  Python exposure map + functional overlay are keyed by (bank,row)/logical addr
  and do **not** currently track relocation. These are the **hardest** and need a
  separate design note before admission — do them last.

### Alternative considered and rejected
- *Pure-Python reimplementation of every mitigation* — rejected (violates SPEC §6,
  duplicates validated code, drift risk).
- *Move the whole flip model into C++ (`cpp/disturbance/`)* — a valid long-term
  option explicitly deferred by the P11 design decision; out of scope here. Would
  make mitigation integration trivial but is a large rewrite.

---

## 4. Admission is one-at-a-time (why we don't just "enable all")

Even with the mechanism generalized, each mitigation must pass P16's gate before
being advertised (SPEC §6, P7 discipline):

1. a **differential trace** showing it changes the issued command stream the way
   the paper/impl says (e.g. inserts preventive refreshes / defers ACTs);
2. a **matched-budget** check: a task that succeeds under `none` must **fail**
   under the mitigation at the same budget;
3. parameter consistency — the mitigation's threshold must be coherent with the
   profile's HCfirst distribution, and the mapping from "row the mitigation
   refreshed" → "victim the flip model tracks" must be exact.

Skipping this risks advertising a defense that only *looks* effective. So "all
mitigations" is a sequence of small validated units, not a single switch.

---

## 5. Can mitigations run at the same time? — Yes (with an API change)

Real commercial DRAM does combine defenses (e.g. **RFM + PRAC**, **TRR + RFM**).
Ramulator supports this natively:

- **Plugins stack.** `controller_plugins` in the worker YAML is a **list**
  (already present in `build/phase2/p2_external_ddr4.yaml:219`), and
  `worker_config_for_mitigation` appends to it. So e.g. `RFMManager` + `Graphene`
  or `TRR` + `RFM` can run together.
- **Controllers are exclusive.** `BlockHammer` and `PRAC` are *controllers*
  (`controller/impl/*_controller.cpp`), one per controller. So `BlockHammer` +
  `PRAC` cannot combine, but a controller (e.g. `PRAC`) **can** combine with
  plugins (e.g. `RFM`) — which matches how PRAC+RFM is deployed in hardware.

**What's needed:** extend the config surface from `mitigation: {name, params}` to
accept a **list** (or `{controller, plugins:[...]}`), validate the combination
(reject two controllers; allow controller + N plugins), and generate one worker
YAML that sets the controller `impl` and appends all plugin entries. The Python
flip model needs no extra work beyond the generalized refresh consumption in §3 —
it sees the union of all emitted refreshes/deferrals.

---

## 6. Is TRR (and the other basic ones) implemented?

**In Ramulator: yes. Wired in this repo: no** (they fail closed).

Vendored Ramulator inventory (`third_party/ramulator2/src/ramulator/controller/`):

| Ramulator name | Kind | Class | In our repo |
|---|---|---|---|
| `IdealTRR` (`ideal_trr.cpp`) | plugin | refresh | not wired |
| `SamsungTRR` (`samsung_trr.cpp`) | plugin | refresh | not wired |
| `HynixTRR` (`hynix_trr.cpp`) | plugin | refresh | not wired |
| `PARA` (`para.cpp`) | plugin | refresh (probabilistic) | not wired |
| `Graphene` (`graphene.cpp`) | plugin | refresh (counter) | not wired |
| `TWiCeIdeal` (`twice.cpp`) | plugin | refresh (counter) | not wired |
| `OracleRH` (`oracle_rh.cpp`) | plugin | refresh (ideal) | **ported to Python** |
| `Hydra` (`hydra.cpp`) | plugin | refresh + tracking | not wired |
| `RRS` (`rrs.cpp`) | plugin | row-remap | not wired (hardest) |
| `AQUA` (`aqua.cpp`) | plugin | row-remap | not wired (hardest) |
| `RFMManager` (`rfm_manager.cpp`) | plugin | refresh mgmt | not wired |
| `BlockHammer` (`blockhammer_controller.cpp`) | controller | throttle | not wired |
| `PRAC` (`prac_controller.cpp`) | controller | throttle/backoff | not wired |

So "basic" TRR is available (three vendor-flavored variants), plus the standard
academic suite. None except the oracle reference is currently exposed.

---

## 7. Proposed roadmap (ordered)

1. **Generalize refresh consumption** (`disturbance.py`): honor targeted,
   row-decoded `REF/RFM/VRR` restorations; keep all-bank window decay for
   undecoded refreshes. Add a unit test (targeted refresh restores only its row).
2. **Exercise the injection path** (`mitigations.py` / `phase4_env.py`): add a
   test that `worker_config_for_mitigation` writes a valid plugin-augmented YAML
   and the worker boots with it. (Closes Issue 2a.)
3. **Admit PARA first** (simple, plugin present): set `ramulator_impl="PARA"` +
   param map (`threshold`, `seed`), write the differential + matched-budget
   conformance test. This is the reference pattern for the rest.
4. **Admit the refresh-class batch** one at a time: Graphene → TWiCe → TRR (Ideal
   → vendor variants) → RFM. Each is `ramulator_impl` + param map + conformance
   test.
5. **Verify throttle-class** (BlockHammer, PRAC): confirm the worker surfaces
   deferred/dropped ACTs; likely little flip-model change, but validate.
6. **Combined mitigations**: extend the config to a controller + plugin list;
   validate PRAC+RFM and a TRR+RFM combo.
7. **Row-remapping** (AQUA, RRS, Hydra): separate design note for tracking row
   relocation in the exposure map / functional overlay before admission.

---

## 8. Open questions

- Do Ramulator's plugin-inserted preventive refreshes surface through the P11
  `on_issue` recorder with a **decoded row** (needed for targeted restoration), or
  only as bank/all-bank events? Determines how much of §3's generalization is
  possible per mitigation. *Verify per plugin.*
- For throttle-class mitigations, does deferring an `ACT` in Ramulator change the
  **issued** stream the flip model sees, or only internal scheduling? (It must be
  the issued stream.)
- Row-remapping: can the functional-memory overlay follow a relocated row without
  leaking the mapping to the policy (SPEC §8 non-leakage)?
