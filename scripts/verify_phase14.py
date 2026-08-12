#!/usr/bin/env python3
"""Phase 14 admission gate: disturbance fidelity (fixes defects B, C, E).

The toy overlay is replaced by a **profile-driven** read-disturbance model:
correct stratum selection, hierarchical threshold variance, direction &
multiplicity, RowPress dwell, profile-driven blast radius, and refresh/decay with
a real target-row-refresh oracle. Everything is driven by the *real* admitted
``ddr4_vts25_v1`` profile and the real DDR4 geometry the worker reports.

Checks (spec/IMPLEMENTATION_PLAN_V2.md §P14, spec/TEST_PLAN.md D4–D10):

- **D5 / D4**: the known fixture flips at its calibrated threshold and the
  below-threshold control does not, across fixed seeds, through the real worker.
- **D6**: single- vs double-sided exposure differs per the profile strata.
- **D7**: open-row dwell (RowPress) lowers the hammers-to-flip, and only for
  RowPress-supporting profiles.
- **D8**: a full refresh window decays exposure; the Python ``oracle`` matches
  Ramulator's ``OracleRH`` on a differential ACT trace and protects the target.
- **D9**: out-of-domain temperatures fail closed; supported ones are admitted.
- **D10**: flips persist through reads and refresh until a write restores them.
- **Statistical validity**: the hierarchical sampler reproduces the profile's
  quantiles and variance components.
"""

from __future__ import annotations

import base64
import math
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv, RowHammerTaskEnv  # noqa: E402
from rowhammer_env.disturbance import DisturbanceEngine  # noqa: E402
from rowhammer_env.geometry import Geometry  # noqa: E402
from rowhammer_env.tools.addressing import AddressMapper  # noqa: E402

SEED = 14


# --- worker helpers -------------------------------------------------------------

def rd(addr: int) -> dict:
    return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}


def issue(env, commands: list[dict]):
    obs = env.step(Phase2Action(tool="dram.issue", args={"commands": commands}))
    if obs.error:
        raise SystemExit(f"unexpected issue error: {obs.error}")
    return obs


def read_byte(env, addr: int) -> int:
    obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": addr}, "length": 1}))
    if obs.error:
        raise SystemExit(f"unexpected read error: {obs.error}")
    return base64.b64decode(obs.data_b64 or "")[0]


def write_byte(env, addr: int, value: int) -> None:
    obs = env.step(
        Phase2Action(
            tool="dram.write",
            args={"addr": {"kind": "logical", "addr": addr}, "data_b64": base64.b64encode(bytes([value])).decode()},
        )
    )
    if obs.error:
        raise SystemExit(f"unexpected write error: {obs.error}")


def hammer(env, left: int, right: int, pairs: int) -> None:
    for _ in range(pairs):
        issue(env, [rd(left), rd(right)])


def real_geometry() -> Geometry:
    """The exact DDR4 geometry the worker publishes (real DRAMSpec decode)."""
    env = RowHammerDisturbanceEnv()
    obs = env.reset(seed=SEED, episode_id="phase14_geo")
    if obs.error:
        raise SystemExit(f"worker reset failed: {obs.error}")
    geo = env.disturbance.geometry
    env.close()
    return geo


def new_engine(geo: Geometry, **kwargs) -> DisturbanceEngine:
    return DisturbanceEngine(geometry=geo, row_encoder=AddressMapper(geo).encode, seed=SEED, **kwargs)


def act(row: int, *, clk: int = 0) -> dict:
    return {"op": "ACT", "channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": row, "row_hit": False, "clk": clk}


def pre(*, clk: int = 0) -> dict:
    """A per-bank precharge, as DDR4 names it."""
    return {"op": "PREpb", "channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "clk": clk}


def refab() -> dict:
    return {"op": "REFab", "channel": 0, "rank": 0, "row": -1, "clk": 0}


def edrd(addr: int) -> dict:
    return {"op": "RD", "addr": addr, "size": 64}


# --- checks ---------------------------------------------------------------------

def check_known_flip_control() -> None:
    """D5 known-flip fixture + D4 below-threshold control, through the worker."""
    for seed in (SEED, SEED + 1, SEED + 5):
        env = RowHammerDisturbanceEnv()
        env.reset(seed=seed, episode_id=f"phase14_flip_{seed}")
        d = env.disturbance
        target, left, right = d.target_addr, d.target_addr - d.row_bytes, d.target_addr + d.row_bytes
        if read_byte(env, target) != 0:
            raise SystemExit("fresh target byte was not zero")
        hammer(env, left, right, d.known_threshold // 2)
        if read_byte(env, target) != 1:
            raise SystemExit(f"D5: known fixture did not flip (seed {seed})")
        env.close()

        control = RowHammerDisturbanceEnv()
        control.reset(seed=seed, episode_id=f"phase14_ctrl_{seed}")
        d = control.disturbance
        hammer(control, d.target_addr - d.row_bytes, d.target_addr + d.row_bytes, d.known_threshold // 2 - 1)
        if read_byte(control, d.target_addr) != 0:
            raise SystemExit(f"D4: below-threshold control flipped (seed {seed})")
        control.close()
    print("  D5/D4: known fixture flips at threshold; below-threshold control does not (3 seeds)")


def check_persistence() -> None:
    """D10 — a flip persists through reads and refresh until a write restores it."""
    env = RowHammerDisturbanceEnv()
    env.reset(seed=SEED, episode_id="phase14_persist")
    d = env.disturbance
    target, left, right = d.target_addr, d.target_addr - d.row_bytes, d.target_addr + d.row_bytes
    hammer(env, left, right, d.known_threshold // 2)
    if read_byte(env, target) != 1:
        raise SystemExit("D10: setup flip did not occur")
    # Repeated reads and a long WAIT (many auto-refreshes) must not correct it.
    for _ in range(20):
        if read_byte(env, target) != 1:
            raise SystemExit("D10: flip did not persist across reads")
    issue(env, [{"op": "WAIT", "cycles": 200000}])
    if read_byte(env, target) != 1:
        raise SystemExit("D10: flip did not persist across refresh")
    write_byte(env, target, 0)
    if read_byte(env, target) != 0:
        raise SystemExit("D10: write did not restore the flipped cell")
    env.close()
    print("  D10: flip persists across reads + refresh, restored only by a write")


def check_single_vs_double(geo: Geometry) -> None:
    """D6 — single- and double-sided exposure differ per the profile strata."""
    eng = new_engine(geo)
    if not eng.known_single_threshold > eng.known_threshold:
        raise SystemExit("D6: single threshold not above double threshold")
    tr = eng.known_target_row
    key = (0, 0, 0, 0, tr)
    left, right = eng.target_addr - eng.row_bytes, eng.target_addr + eng.row_bytes

    double = new_engine(geo)
    for _ in range(double.known_threshold // 2):
        double.consume([act(tr - 1)], edrd(left))
        double.consume([act(tr + 1)], edrd(right))
    if not double.victims[key].flipped:
        raise SystemExit("D6: double-sided did not flip at double threshold")

    single = new_engine(geo)
    for _ in range(single.known_threshold):  # 2x the double threshold, one side only
        single.consume([act(tr - 1)], edrd(left))
    if single.victims[key].flipped:
        raise SystemExit("D6: single-sided flipped below its single-stratum threshold")
    print(
        f"  D6: double flips at {eng.known_threshold}; single needs "
        f"{eng.known_single_threshold} (ratio {eng.known_single_threshold / eng.known_threshold:.1f})"
    )


def _pairs_to_flip(eng: DisturbanceEngine, dwell: int) -> int:
    tr = eng.known_target_row
    key = (0, 0, 0, 0, tr)
    left, right = eng.target_addr - eng.row_bytes, eng.target_addr + eng.row_bytes
    clk = 0
    for n in range(1, 20000):
        eng.consume([act(tr - 1, clk=clk)], edrd(left))
        clk += dwell
        eng.consume([pre(clk=clk), act(tr + 1, clk=clk)], edrd(right))
        clk += dwell
        eng.consume([pre(clk=clk)], edrd(left))
        clk += 10
        if eng.victims[key].flipped:
            return n
    raise SystemExit("RowPress fixture never flipped")


def check_rowpress(geo: Geometry) -> None:
    """D7 — open-row dwell reduces hammers-to-flip, only for RowPress profiles."""
    short = _pairs_to_flip(new_engine(geo), dwell=10)
    long = _pairs_to_flip(new_engine(geo), dwell=200000)
    if not long * 3 < short:
        raise SystemExit(f"D7: RowPress dwell did not reduce hammers-to-flip ({long} vs {short})")

    unsupported = new_engine(geo)
    unsupported.profile["fit"]["families"][unsupported.family]["rowpress"]["supported"] = False
    if unsupported._rowpress_factor(200000) != 1.0:
        raise SystemExit("D7: RowPress applied to a non-RowPress profile")
    print(f"  D7: RowPress cuts hammers-to-flip {short} -> {long}; ignored when unsupported")


def check_refresh_decay(geo: Geometry) -> None:
    """D8 (decay) — a full refresh window resets accumulated exposure."""
    eng = new_engine(geo)
    tr = eng.known_target_row
    key = (0, 0, 0, 0, tr)
    left, right = eng.target_addr - eng.row_bytes, eng.target_addr + eng.row_bytes
    for _ in range(eng.known_threshold // 2 - 3):
        eng.consume([act(tr - 1)], edrd(left))
        eng.consume([act(tr + 1)], edrd(right))
    if eng.victims[key].left <= 0:
        raise SystemExit("D8: no exposure accumulated before refresh")
    for _ in range(eng.refresh_window):
        eng.consume([refab()], {"op": "WAIT", "addr": 0, "size": 0})
    if eng.victims[key].left != 0 or eng.victims[key].right != 0:
        raise SystemExit("D8: refresh window did not decay exposure")
    for _ in range(3):
        eng.consume([act(tr - 1)], edrd(left))
        eng.consume([act(tr + 1)], edrd(right))
    if eng.victims[key].flipped:
        raise SystemExit("D8: leftover hammer flipped after refresh reset")
    print(f"  D8: a full {eng.refresh_window}-refresh window decays accumulated exposure")


class _OracleRHReference:
    """Transcription of Ramulator OracleRH's ACT/VRR logic (oracle_rh.cpp).

    Per-aggressor activation counter; issue a victim-row refresh once the count
    reaches ``tRH`` and reset it. Used only as the differential reference.
    """

    def __init__(self, tRH: int) -> None:
        self.tRH = tRH
        self.counts: dict[tuple[int, int], int] = {}
        self.vrr = 0

    def on_act(self, bank: int, row: int) -> None:
        key = (bank, row)
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] >= self.tRH:
            self.counts[key] = 0
            self.vrr += 1


def check_oracle_differential(geo: Geometry) -> None:
    """D8 (oracle) — Python oracle matches OracleRH and protects the target."""
    eng = new_engine(geo, mitigation="oracle")
    ref = _OracleRHReference(eng.tRH)
    tr = eng.known_target_row
    left, right = eng.target_addr - eng.row_bytes, eng.target_addr + eng.row_bytes
    produced = 0
    for _ in range(eng.known_threshold):
        produced += eng.consume([act(tr - 1)], edrd(left)).oracle_refreshes
        ref.on_act(0, tr - 1)
        produced += eng.consume([act(tr + 1)], edrd(right)).oracle_refreshes
        ref.on_act(0, tr + 1)
    if produced == 0:
        raise SystemExit("D8: oracle never fired")
    if produced != ref.vrr:
        raise SystemExit(f"D8: oracle diverged from OracleRH ({produced} vs {ref.vrr} VRRs)")
    if eng.target_addr in eng.flips:
        raise SystemExit("D8: oracle failed to protect the target")

    # End-to-end protection through the real worker (mirrors the M2 semantics).
    env = RowHammerTaskEnv(mitigation={"name": "oracle", "params": {}})
    env.reset(seed=SEED, episode_id="phase14_oracle")
    d = env.disturbance
    refreshes = 0
    for _ in range(d.known_threshold):
        obs = issue(env, [rd(d.target_addr - d.row_bytes), rd(d.target_addr + d.row_bytes)])
        refreshes += int(obs.feedback.get("oracle_refreshes", 0))
        if obs.reward:
            raise SystemExit("D8: oracle allowed target reward")
    if read_byte(env, d.target_addr) != 0 or refreshes == 0:
        raise SystemExit("D8: oracle did not protect the target through the worker")
    env.close()
    print(f"  D8: Python oracle == OracleRH on the ACT trace ({produced} VRRs); target protected")


def check_temperature_domain(geo: Geometry) -> None:
    """D9 — out-of-domain temperatures fail closed; supported ones are admitted."""
    try:
        new_engine(geo, temperature=85)
    except ValueError as exc:
        if not str(exc).startswith("PROFILE_REJECTED:"):
            raise SystemExit(f"D9: wrong error for out-of-domain temperature: {exc}")
    else:
        raise SystemExit("D9: out-of-domain temperature was admitted")

    # Fail closed over the wire with a stable PROFILE_REJECTED code.
    env = RowHammerDisturbanceEnv(temperature=85)
    obs = env.reset(seed=SEED, episode_id="phase14_temp")
    if not obs.error or obs.error["code"] != "PROFILE_REJECTED":
        raise SystemExit("D9: worker did not fail closed on out-of-domain temperature")
    if new_engine(geo, temperature=50).temperature != 50:
        raise SystemExit("D9: supported temperature not admitted")
    print("  D9: 85C rejected (PROFILE_REJECTED); 50C admitted")


def check_statistical_validity(geo: Geometry) -> None:
    """Sampled thresholds reproduce the profile's quantiles and variance model."""
    family, stratum = "hisasa", "double|all_zeros"
    params = new_engine(geo).profile["fit"]["families"][family]["strata"][stratum]["hcfirst_lognormal"]
    module_means: list[float] = []
    within_sigmas: list[float] = []
    all_logs: list[float] = []
    for m in range(80):
        eng = DisturbanceEngine(
            geometry=geo, row_encoder=AddressMapper(geo).encode, seed=7000 + m, family=family, stratum=stratum
        )
        logs = [math.log(eng._sample_threshold((0, 0, 0, 0, r), "double", "all_zeros")) for r in range(3, 500)]
        module_means.append(statistics.mean(logs))
        within_sigmas.append(statistics.pstdev(logs))
        all_logs.extend(logs)

    within = statistics.mean(within_sigmas)
    between = statistics.pstdev(module_means)
    median_ratio = math.exp(statistics.median(all_logs)) / params["median"]
    if abs(within - params["sigma_within_chip"]) > 0.03:
        raise SystemExit(f"stat: sigma_within not reproduced ({within:.3f} vs {params['sigma_within_chip']})")
    if abs(between - params["sigma_between_chip"]) > 0.15:
        raise SystemExit(f"stat: sigma_between not reproduced ({between:.3f} vs {params['sigma_between_chip']})")
    if not (1.0 / 1.5 < median_ratio < 1.5):
        raise SystemExit(f"stat: median ratio {median_ratio:.2f} outside the profile row-median gate")
    print(
        f"  stat: sampler recovers sigma_within={within:.3f}, sigma_between={between:.3f}, "
        f"median ratio {median_ratio:.2f}"
    )


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase11.py"], cwd=ROOT, check=True)
    print("phase14 checks:")
    geo = real_geometry()
    check_known_flip_control()
    check_persistence()
    check_single_vs_double(geo)
    check_rowpress(geo)
    check_refresh_decay(geo)
    check_oracle_differential(geo)
    check_temperature_domain(geo)
    check_statistical_validity(geo)
    print("phase14 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
