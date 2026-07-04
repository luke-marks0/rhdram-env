# ADR 0002 — Standard generalization via per-standard adapters (P15)

Status: Accepted (Phase 15)
Resolves: defect F (DDR4 hard-gate + hardcoded geometry; profile schema
DDR4-shaped; no path for other standards).

## Context

Before P15 the disturbance engine was hard-gated to DDR4
(`disturbance.py`: `if profile["standard"] != "DDR4": raise "Phase 4 admits DDR4
only"`), and two DDR4 assumptions were baked in as constants:

- `STANDARD_BLAST = {"DDR4": [(1, 1.0)]}` and `REFRESH_COMMANDS_PER_WINDOW = 8192`
  were engine-level literals rather than per-standard values.
- `geometry._row_stride()` summed the bit widths of the DDR level set
  (`rank`, `bankgroup`, `bank`) by name. For a standard with a different level
  structure — HBM2 has a `PseudoChannel` level and no `Rank` — this would compute
  the wrong linear row stride.

The profile package (`schema_version: 1`) carried only DDR4-relevant fields, with
no place to declare the standard-specific dimensions HBM needs (pseudo-channel,
die stacking, on-die ECC, RFM/VRR refresh semantics).

Ramulator 2.1.0 already ships real DDR5/HBM2 (and DDR5_RFM/VRR) DRAM models, and
the worker's `IssuedEventRecorder` already publishes the real `DRAMSpec` geometry
(`standard`, level names/sizes, prefetch, `tx_bytes`) for whatever config it
loads. So the simulator side is standard-generic already; only the Python engine
and profile schema needed to stop assuming DDR4.

## Decision

Make the engine **parameterized by standard**, driven by the geometry the worker
reports and a small per-standard adapter, and version the profile schema.

- **Standard facts** (`profile_builder/standards.py`): a source-traceable table of
  each standard's read-disturbance-relevant, *public* facts — blast neighbourhood,
  JEDEC refresh divisor, RFM/VRR, on-die ECC, pseudo-channel, die stacking. These
  are standard properties, not fitted calibration, so they live in the profile
  package's own package and are stamped into every profile (see schema v2). Covers
  DDR4, DDR4_VRR, DDR5, DDR5_RFM/VRR/RFM_VRR, and HBM2.
- **Standard model** (`rowhammer_env/standards.py`): `StandardModel.from_geometry`
  turns those facts + the real geometry into the concrete parameters the engine
  uses (blast set, refresh window, row stride) and selects the Ramulator
  `dram.impl` + controller per standard. It fails closed on an unsupported
  standard and on a geometry whose levels contradict the standard's dimensions.
- **Generic geometry** (`geometry.py`): `row_stride` now sums the bit widths of
  every non-Channel level below Row read straight from the reported
  `level_names[1:row_index]`, matching Ramulator's `RoBaRaCoCh`
  (`addr_mapper_base.cpp`) for any standard. DDR4 is byte-identical (131072); HBM2
  correctly includes its pseudo-channel bit (32768).
- **Engine** (`disturbance.py`): the DDR4 hard-gate is gone. The engine checks the
  profile's standard has an admitted adapter, cross-checks it against the
  geometry's standard (**no parameter pooling** — a DDR4-fitted profile is never
  run on a DDR5/HBM2 geometry; that fails closed with `PROFILE_REJECTED`), builds a
  `StandardModel`, and sources `blast`/`refresh_window` from it.
- **Profile schema v2**: `build.py` emits `schema_version: 2` with additive
  `topology`, `refresh`, and `standard_dimensions` blocks derived from the standard
  facts. The DDR4 package carries the DDR4 values (no HBM dimensions). The loader
  and engine read the blocks defensively, so a legacy v1 package (no blocks) falls
  back to the same adapter defaults — backward compatible.
- **HBM2 ingestion** (`profile_builder/ingest/hbm2.py`): wired to the pinned
  `hbm2_read_disturbance` source, but the source is `deferred_pending_license_
  and_hash`, so every entry point fails closed with `UNAVAILABLE_CAPABILITY`. No
  HBM2 calibration is fabricated; the capability is simply absent until the source
  is admitted.

A full C++ port of the engine remains out of scope (see ADR 0001); the statistical
model stays in Python and is fed the real issued-event stream.

## Consequences

- The same engine drives any admitted standard; adding one is a data change (a new
  `StandardFacts` entry + an admitted empirical profile of that standard), not an
  engine rewrite.
- A second, genuinely different standard (HBM2, with a pseudo-channel level and
  on-die ECC) instantiates the generic engine core from its real geometry, and the
  Python address projection reproduces Ramulator's own decode for it (differential
  trace in `verify_phase15.py`) — proving no DDR4 constant leaks.
- No parameter pooling: a profile fitted on one standard cannot run on another's
  geometry, and each DDR4 chip family keeps its own held-out validation split.
- `verify_phase15.py` + `tests/test_phase15.py` gate all of the above; the DDR4
  path is unchanged (the P0→P14 chain still passes).
- `configs/ramulator/p2_external_{ddr5,hbm2}.py` and the `build_phase2` export make
  a real second-standard geometry reachable through the same worker binary; these
  are used only for geometry/adapter checks, never for reward (no second-standard
  profile is admitted).
