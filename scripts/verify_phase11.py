#!/usr/bin/env python3
"""Phase 11 admission gate: real issued-event stream (fixes defect A; resolves D).

The disturbance model must consume the *actual post-schedule* DRAM commands the
Ramulator worker issues (ACT/PRE/RD/WR/REF) with decoded coordinates and
row-buffer state — not frontend read/write completions that ignore locality.

Checks (see spec/IMPLEMENTATION_PLAN_V2.md §P11 and spec/TEST_PLAN.md D3):

- **D3**: rejected/illegal commands produce no issued events and no exposure change.
- **Row-buffer locality**: reads to a single open row collapse to ~1 ACT and
  cannot flip; the *same* reads interleaved into row conflicts issue many ACTs
  and flip. Same read budget, opposite outcome — driven only by issued ACTs.
- **A-regression**: the fixed double-sided pattern that flipped before still
  flips, now driven by ACT counts; the single-open-row control does not.
- **Deterministic replay**: a fixed seed + command sequence yields identical
  issued-event streams and flips.
"""

from __future__ import annotations

import base64
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv  # noqa: E402

SEED = 11


def rd(addr: int) -> dict:
    return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}


def issue(env: RowHammerDisturbanceEnv, commands: list[dict]):
    obs = env.step(Phase2Action(tool="dram.issue", args={"commands": commands}))
    if obs.error:
        raise SystemExit(f"unexpected issue error: {obs.error}")
    return obs


def read_byte(env: RowHammerDisturbanceEnv, addr: int) -> int:
    obs = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": addr}, "length": 1}))
    if obs.error:
        raise SystemExit(f"unexpected read error: {obs.error}")
    return base64.b64decode(obs.data_b64 or "")[0]


def acts_of(obs) -> int:
    return int(obs.public_counters.get("acts", 0))


def fresh_env() -> tuple[RowHammerDisturbanceEnv, int, int, int, int]:
    env = RowHammerDisturbanceEnv()
    obs = env.reset(seed=SEED, episode_id="phase11")
    if obs.error:
        raise SystemExit(obs.error)
    d = env.disturbance
    target = d.target_addr
    return env, target, target - d.row_bytes, target + d.row_bytes, d.known_threshold


def check_d3() -> None:
    """Rejected/illegal commands must not emit events or change exposure."""
    env, target, left, right, _ = fresh_env()
    # Establish some real exposure first.
    issue(env, [rd(left), rd(right)])
    before_victims = {k: (v.left, v.right, v.flipped) for k, v in env.disturbance.victims.items()}
    before_flips = dict(env.disturbance.flips)
    if not before_victims:
        raise SystemExit("D3 setup produced no exposure to compare against")

    # A command outside the admitted set is rejected before Ramulator ticks.
    bad = env.step(Phase2Action(tool="dram.issue", args={"commands": [{"op": "ACT", "addr": {"kind": "logical", "addr": left}}]}))
    if not bad.error or bad.error["code"] != "ILLEGAL_COMMAND":
        raise SystemExit("illegal command did not fail closed with ILLEGAL_COMMAND")

    # A malformed request rejected inside the worker must also emit no events.
    oversized = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": left}, "length": 100000}))
    if not oversized.error or oversized.error["code"] != "BAD_SCHEMA":
        raise SystemExit("oversized read did not fail closed with BAD_SCHEMA")
    if oversized.feedback.get("trace_tail"):
        raise SystemExit("rejected request leaked issued events into the trace")

    after_victims = {k: (v.left, v.right, v.flipped) for k, v in env.disturbance.victims.items()}
    after_flips = dict(env.disturbance.flips)
    if after_victims != before_victims or after_flips != before_flips:
        raise SystemExit("rejected commands changed disturbance exposure (D3 violated)")
    env.close()
    print("  D3: rejected commands emit no events and leave exposure unchanged")


def _run_blocked(pairs: int) -> tuple[int, bool, int]:
    """Read `left` `pairs` times, then `right` `pairs` times (row hits, no conflict)."""
    env, target, left, right, threshold = fresh_env()
    last = None
    for _ in range(pairs):
        last = issue(env, [rd(left)])
    for _ in range(pairs):
        last = issue(env, [rd(right)])
    flipped = read_byte(env, target) != 0
    acts = acts_of(last)
    env.close()
    return acts, flipped, threshold


def _run_alternating(pairs: int) -> tuple[int, bool, tuple[int, int]]:
    """Interleave `left`/`right` `pairs` times (each read is a row conflict -> ACT)."""
    env, target, left, right, _ = fresh_env()
    last = None
    for _ in range(pairs):
        last = issue(env, [rd(left), rd(right)])
    victim = env.disturbance.victims.get((0, 0, 0, 0, env.disturbance.known_target_row))
    counts = (victim.left, victim.right) if victim else (0, 0)
    flipped = read_byte(env, target) != 0
    acts = acts_of(last)
    env.close()
    return acts, flipped, counts


def check_locality() -> None:
    _, _, _, _, threshold = fresh_env()
    pairs = threshold // 2  # interleaved: exposure = 2 * pairs = threshold -> flip

    blocked_acts, blocked_flip, _ = _run_blocked(pairs)
    alt_acts, alt_flip, (vl, vr) = _run_alternating(pairs)

    total_reads = 2 * pairs
    if blocked_flip:
        raise SystemExit("blocked single-row access flipped the victim (row-buffer locality broken)")
    if not alt_flip:
        raise SystemExit("interleaved double-sided access did not flip the victim")
    # Both patterns issue the same number of reads; only the interleaved one
    # produces the row conflicts (ACTs) that drive disturbance.
    if not (blocked_acts * 10 < alt_acts):
        raise SystemExit(f"blocked ACTs ({blocked_acts}) not << interleaved ACTs ({alt_acts})")
    if blocked_acts >= total_reads // 4:
        raise SystemExit(f"blocked access issued too many ACTs ({blocked_acts}) for {total_reads} reads")
    if vl < pairs or vr < pairs:
        raise SystemExit(f"victim exposure not driven by ACT counts (left={vl}, right={vr}, pairs={pairs})")
    print(
        f"  locality: {total_reads} reads blocked -> {blocked_acts} ACTs, no flip; "
        f"interleaved -> {alt_acts} ACTs, flip (victim L/R={vl}/{vr})"
    )
    print("  A-regression: fixed double-sided pattern flips via ACTs; single-row control does not")


def check_determinism() -> None:
    """Same seed + command sequence -> identical issued-event stream and flips."""

    def run() -> tuple[list, dict]:
        env, target, left, right, _ = fresh_env()
        ops: list = []
        for _ in range(40):
            obs = issue(env, [rd(left), rd(right)])
            ops.append(tuple((e["op"], e.get("row"), e.get("row_hit")) for e in obs.feedback["trace_tail"]))
        read_byte(env, target)
        flips = dict(env.disturbance.flips)
        env.close()
        return ops, flips

    ops_a, flips_a = run()
    ops_b, flips_b = run()
    if ops_a != ops_b:
        raise SystemExit("issued-event stream was not deterministic across identical runs")
    if flips_a != flips_b:
        raise SystemExit("flip state was not deterministic across identical runs")
    print(f"  determinism: identical issued-event stream and flips across replays ({len(ops_a)} steps)")


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase4.py"], cwd=ROOT, check=True)
    print("phase11 checks:")
    check_d3()
    check_locality()
    check_determinism()
    print("phase11 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
