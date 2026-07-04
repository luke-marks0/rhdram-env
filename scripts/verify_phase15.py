#!/usr/bin/env python3
"""Phase 15 admission gate: standards & profile generalization (fixes defect F).

The disturbance engine is no longer hard-gated to DDR4. Its standard-specific
behaviour — the linear row stride, the blast neighbourhood, the refresh window,
the Ramulator ``dram.impl``/controller — is derived from the real ``DRAMSpec``
geometry plus a per-standard adapter (:mod:`rowhammer_env.standards`), and the
profile package now carries the standard's dimensions in a versioned schema
(``schema_version: 2``). A second, genuinely different standard (HBM2, which adds
a PseudoChannel level and on-die ECC) instantiates the same generic machinery
with no DDR4 constants, while every empirical profile is admitted independently
and a profile fitted on one standard is never pooled onto another.

Checks (see spec/IMPLEMENTATION_PLAN_V2.md §P15):

- **DDR4 regression**: the whole P0→P14 stack still passes unchanged (chained),
  and the engine's DDR4 adapter matches the committed profile's declared standard
  dimensions (blast ±1, 8192-refresh window, no HBM dimensions).
- **Schema v2**: the committed profile carries topology/refresh/standard_dimensions
  consistent with the adapter, and the loader stays backward-compatible with a
  v1-shaped package (missing blocks default to the standard's values).
- **Second standard, no DDR4 leak**: real DDR5 and HBM2 worker geometries drive the
  generic engine core; the Python address projection reproduces Ramulator's own
  decode (differential trace) for the second standard; derived constants differ
  from DDR4 and HBM2's extra dimensions are honoured.
- **No parameter pooling**: the DDR4 profile fails closed against a non-DDR4
  geometry; the HBM2 profile/source stays deferred and fails closed; each DDR4
  chip family keeps its own held-out validation.
"""

from __future__ import annotations

import copy
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_phase2  # noqa: E402

from profile_builder.ingest.hbm2 import require_admitted, source_status  # noqa: E402
from profile_builder.errors import SourceUnavailable  # noqa: E402
from profile_builder.package.build import verify_package  # noqa: E402
from profile_builder.standards import as_profile_blocks, facts_for  # noqa: E402
from rowhammer_env.disturbance import DisturbanceEngine  # noqa: E402
from rowhammer_env.geometry import Geometry  # noqa: E402
from rowhammer_env.profiles import ADMITTED_PROFILES, DEFERRED_PROFILES, load_profile  # noqa: E402
from rowhammer_env.standards import SUPPORTED_STANDARDS, StandardModel  # noqa: E402
from rowhammer_env.tools.addressing import AddressMapper  # noqa: E402
from rowhammer_env.worker_protocol import WorkerClient, WorkerRequest  # noqa: E402

WORKER = ROOT / "build/phase2/ramulator_worker"
DDR4_CONFIG = ROOT / "build/phase2/p2_external_ddr4.yaml"
SECOND_STANDARD_CONFIGS = {
    "DDR5": ROOT / "build/phase2/p2_external_ddr5.yaml",
    "HBM2": ROOT / "build/phase2/p2_external_hbm2.yaml",
}


# --- worker helpers -------------------------------------------------------------

def worker_geometry(config: pathlib.Path) -> Geometry:
    """The real DRAMSpec geometry a worker reports for a given config."""
    w = WorkerClient(WORKER, config)
    try:
        payload = w.call(WorkerRequest("INFO", "info", ()))
        if not payload.get("ok"):
            raise SystemExit(f"worker INFO failed for {config.name}: {payload.get('error')}")
        return Geometry(payload["geometry"])
    finally:
        w.close()


def worker_act_decode(config: pathlib.Path, addr: int) -> dict[str, int]:
    """Issue a RD at ``addr`` and return the decoded coords of its ACT event."""
    w = WorkerClient(WORKER, config)
    try:
        payload = w.call(WorkerRequest("ISSUE", "a1", ("RD", str(addr))))
        if not payload.get("ok"):
            raise SystemExit(f"worker ISSUE failed for {config.name}: {payload.get('error')}")
        acts = [e for e in payload.get("events", []) if e.get("op") == "ACT"]
        if not acts:
            raise SystemExit(f"{config.name}: RD at {addr} issued no ACT event")
        return acts[0]
    finally:
        w.close()


# --- checks ---------------------------------------------------------------------

def check_ddr4_adapter_consistency() -> None:
    """The engine's DDR4 adapter matches the committed profile's declared standard."""
    geo = worker_geometry(DDR4_CONFIG)
    if geo.standard != "DDR4":
        raise SystemExit(f"DDR4 config reported standard {geo.standard}")
    sm = StandardModel.from_geometry(geo)
    if sm.blast != ((1, 1.0),):
        raise SystemExit(f"DDR4 blast changed: {sm.blast}")
    if sm.refresh_window != 8192:
        raise SystemExit(f"DDR4 refresh window changed: {sm.refresh_window}")
    if sm.ramulator_impl() != ("DDR4", "GenericDDR"):
        raise SystemExit(f"DDR4 impl selection changed: {sm.ramulator_impl()}")

    profile = verify_package()
    if str(profile.get("schema_version")) != "2":
        raise SystemExit(f"committed profile is not schema v2: {profile.get('schema_version')}")
    expected = as_profile_blocks(facts_for("DDR4"))
    for key in ("topology", "refresh", "standard_dimensions"):
        if profile.get(key) != expected[key]:
            raise SystemExit(f"profile {key} does not match the DDR4 adapter: {profile.get(key)}")
    dims = profile["standard_dimensions"]
    if any(dims[k] for k in ("pseudo_channel", "die_stacking", "on_die_ecc", "subarray_resolved")):
        raise SystemExit(f"DDR4 profile leaked an HBM dimension: {dims}")
    print("  DDR4: engine adapter matches the committed schema-v2 profile "
          "(blast ±1, 8192-refresh, no HBM dimensions)")


def check_schema_v2_backward_compat() -> None:
    """A v1-shaped profile (no standard blocks) yields identical engine behaviour.

    The engine reads the v2 ``topology`` block defensively (``profile.get`` with a
    default), so a legacy v1 package — which has no topology/refresh blocks — falls
    back to the same standard-adapter values. Prove that by building the engine
    against a stripped (v1-shaped) copy of the admitted profile and checking it
    matches the full v2 engine.
    """
    geo = worker_geometry(DDR4_CONFIG)
    eng_v2 = DisturbanceEngine(geometry=geo, seed=15)

    legacy = copy.deepcopy(load_profile("ddr4_vts25_v1"))
    for key in ("schema_version", "topology", "refresh", "standard_dimensions"):
        legacy.pop(key, None)
    orig_load = DisturbanceEngine.__init__.__globals__["load_profile"]
    DisturbanceEngine.__init__.__globals__["load_profile"] = lambda _pid: copy.deepcopy(legacy)
    try:
        eng_v1 = DisturbanceEngine(geometry=geo, seed=15)
    finally:
        DisturbanceEngine.__init__.__globals__["load_profile"] = orig_load

    if tuple(eng_v1.blast) != tuple(eng_v2.blast) or eng_v1.refresh_window != eng_v2.refresh_window:
        raise SystemExit(
            f"v1-shaped profile diverged from v2 (blast {eng_v1.blast} vs {eng_v2.blast}, "
            f"refresh {eng_v1.refresh_window} vs {eng_v2.refresh_window})"
        )
    if eng_v1.known_threshold != eng_v2.known_threshold:
        raise SystemExit("v1-shaped profile changed the calibrated threshold")
    print("  schema v2: standard blocks are additive; a v1-shaped package falls back to "
          "the same adapter defaults (identical blast/refresh/threshold)")


def _fingerprint(geo: Geometry) -> tuple:
    return (geo.standard, geo.prefetch, geo.tx_bytes, tuple(geo.level_names),
            tuple(sorted(geo.level_sizes.items())))


def check_second_standard_no_ddr4_leak() -> None:
    """DDR5 and HBM2 drive the generic engine core with no DDR4 constants."""
    ddr4_geo = worker_geometry(DDR4_CONFIG)
    ddr4_fp = _fingerprint(ddr4_geo)

    for standard, config in SECOND_STANDARD_CONFIGS.items():
        if not config.is_file():
            raise SystemExit(f"missing {standard} worker config {config} (run build_phase2)")
        geo = worker_geometry(config)
        if geo.standard != standard:
            raise SystemExit(f"{standard} config reported standard {geo.standard}")
        if standard not in SUPPORTED_STANDARDS:
            raise SystemExit(f"{standard} has no admitted adapter")
        # The second-standard geometry must genuinely differ from DDR4's, so a
        # leaked DDR4 constant would give the wrong answer somewhere below.
        if _fingerprint(geo) == ddr4_fp:
            raise SystemExit(f"{standard} geometry is identical to DDR4; not a distinct standard")

        sm = StandardModel.from_geometry(geo)  # the generic engine's standard core
        mapper = AddressMapper(geo)  # generic physical<->linear projection

        if sm.ramulator_impl()[0] != standard:
            raise SystemExit(f"{standard} selected the wrong dram.impl: {sm.ramulator_impl()}")

        # HBM2's extra dimensions must be honoured (pseudo-channel + on-die ECC),
        # and DDR's must not appear on HBM (and vice-versa).
        facts = facts_for(standard)
        has_pseudo = "pseudochannel" in geo.level_names
        if facts.pseudo_channel != has_pseudo:
            raise SystemExit(f"{standard} pseudo-channel dimension mismatch")
        if standard == "HBM2" and not (facts.pseudo_channel and facts.on_die_ecc and facts.die_stacking):
            raise SystemExit("HBM2 adapter dropped a standard-specific dimension")

        # Differential decode: the generic projection must reproduce Ramulator's own
        # addr_vec for this second standard, proving the geometry math is not
        # DDR4-shaped. Exercise a non-trivial coordinate within the device.
        coords = _sample_coords(geo)
        addr = mapper.encode(coords)
        event = worker_act_decode(config, addr)
        for key, want in coords.items():
            if key == "channel":
                continue
            if int(event.get(key, -1)) != want:
                raise SystemExit(
                    f"{standard} differential decode mismatch on {key}: "
                    f"worker={event.get(key)} projection={want} (addr {addr})"
                )
        redecode = mapper.decode(addr)
        if any(redecode[k] != coords.get(k, 0) for k in coords):
            raise SystemExit(f"{standard} encode/decode round-trip failed")

        print(f"  {standard}: generic engine core built from real geometry "
              f"(row_bytes={sm.row_bytes}, impl={sm.ramulator_impl()[0]}, "
              f"pseudo_channel={facts.pseudo_channel}); differential decode matches Ramulator")


def _sample_coords(geo: Geometry) -> dict[str, int]:
    """A concrete in-range physical coordinate for a differential-decode probe."""
    sizes = geo.level_sizes
    coords = {"channel": 0, "row": min(5, sizes.get("row", 1) - 1), "column": 0}
    for level in ("rank", "pseudochannel", "bankgroup", "bank"):
        if level in sizes and sizes[level] > 1:
            coords[level] = min(1, sizes[level] - 1) if level != "bank" else min(3, sizes[level] - 1)
    return coords


def check_no_parameter_pooling() -> None:
    """A DDR4 profile is never run on another standard's geometry; HBM2 stays deferred."""
    # 1. The DDR4-fitted profile fails closed against a non-DDR4 geometry.
    for standard, config in SECOND_STANDARD_CONFIGS.items():
        geo = worker_geometry(config)
        try:
            DisturbanceEngine(geometry=geo, seed=15, profile_id="ddr4_vts25_v1")
        except ValueError as exc:
            if not str(exc).startswith("PROFILE_REJECTED:"):
                raise SystemExit(f"{standard}: DDR4 profile rejected with the wrong code: {exc}")
        else:
            raise SystemExit(f"{standard}: DDR4 profile was pooled onto a {standard} geometry")

    # 2. The HBM2 empirical profile is deferred, absent from capability discovery,
    #    and its ingestion fails closed.
    if "hbm2_read_disturbance" in ADMITTED_PROFILES:
        raise SystemExit("HBM2 profile is advertised as admitted while its source is deferred")
    if "hbm2_read_disturbance" not in DEFERRED_PROFILES:
        raise SystemExit("HBM2 profile is not recorded as deferred")
    for pid in ("hbm2_read_disturbance", "hbm2_read_disturbance_v1"):
        try:
            load_profile(pid)
        except ValueError as exc:
            if not str(exc).startswith("UNAVAILABLE_CAPABILITY:"):
                raise SystemExit(f"deferred profile {pid} failed with the wrong code: {exc}")
        else:
            raise SystemExit(f"deferred profile {pid} loaded")
    if source_status().admitted:
        raise SystemExit("HBM2 source is marked admitted in the manifest")
    try:
        require_admitted()
    except SourceUnavailable:
        pass
    else:
        raise SystemExit("HBM2 ingestion did not fail closed while deferred")

    # 3. No cross-chip pooling: every DDR4 chip family keeps its own held-out chips
    #    disjoint from its training chips (independent validation, SPEC §7 family 10).
    profile = load_profile("ddr4_vts25_v1")
    for family, fam in profile["fit"]["families"].items():
        holdout = set(fam["chips_holdout"])
        train = set(fam["chips_train"])
        if holdout & train:
            raise SystemExit(f"family {family}: held-out chips overlap training chips (pooling)")
    print("  no pooling: DDR4 profile fails closed on DDR5/HBM2 geometry; HBM2 source deferred + "
          "fail-closed; per-chip held-out splits disjoint")


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase14.py"], cwd=ROOT, check=True)
    build_phase2.ensure_second_standard_configs()
    print("phase15 checks:")
    check_ddr4_adapter_consistency()
    check_schema_v2_backward_compat()
    check_second_standard_no_ddr4_leak()
    check_no_parameter_pooling()
    print("phase15 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
