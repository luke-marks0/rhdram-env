#!/usr/bin/env python3
"""Exercise rhdram-env as an RL environment: real rollouts through reset/step.

Drives the OpenEnv-style RowHammerTaskEnv with a simple baseline policy and a
set of controls that prove the reward comes from real simulated DRAM state
(Ramulator-issued events -> disturbance engine), not from anything the policy
asserts. Demonstrates phases 4-8: disturbance flips, trusted sparse reward,
budgets, the oracle mitigation, and the sandboxed script.run path.

Requires the fetched sources and built worker:
    python3 -B scripts/fetch_phase1_sources.py
    python3 -B scripts/build_phase2.py
Then:
    python3 -B scripts/rl_run.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerTaskEnv


def rd(addr: int) -> dict:
    return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}


def double_sided_pair(target_addr: int, row_bytes: int) -> Phase2Action:
    """One tool call issuing two RDs: the two rows adjacent to the target.

    ``row_bytes`` is the logical stride between physically adjacent rows,
    derived from the simulator geometry (``env.disturbance.row_bytes``), so the
    two aggressors decode to the target's real neighbours.
    """
    return Phase2Action(
        tool="dram.issue",
        args={"commands": [rd(target_addr - row_bytes), rd(target_addr + row_bytes)]},
    )


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_reset(obs) -> dict:
    meta = obs.metadata
    dist = meta.get("disturbance", {})
    print(f"  task_id        : {meta.get('task_id')}")
    print(f"  family         : {meta.get('task_family')}")
    print(f"  objective      : {meta.get('objective')}")
    print(f"  disclosure     : {meta.get('disclosure')}")
    print(f"  profile        : {meta.get('profile')}  mitigation={meta.get('mitigation')}")
    print(f"  target (disc.) : {meta.get('target')}")
    print(f"  disturbance    : family={dist.get('family')} stratum={dist.get('stratum')} "
          f"known_target_row={dist.get('known_target_row')} known_threshold={dist.get('known_threshold')}")
    print(f"  budgets        : {meta.get('budget_remaining')}")
    return meta


def rollout_known_target(seed: int = 5) -> None:
    banner("EPISODE A  -  known_target_anybit  (baseline policy should SUCCEED)")
    env = RowHammerTaskEnv()
    obs = env.reset(seed=seed, episode_id="rl_known_target")
    assert not obs.error, obs.error
    meta = show_reset(obs)

    target_addr = meta["target"]["addr"]
    threshold = meta["disturbance"]["known_threshold"]
    row_bytes = env.disturbance.row_bytes
    # Baseline policy: double-sided hammer until the trusted reward fires.
    max_pairs = threshold  # generous cap; flip expected at ~threshold/2 pairs
    print(f"\n  policy: double-sided hammer rows around addr {target_addr} "
          f"(aggressors {target_addr - row_bytes} / {target_addr + row_bytes})")
    step = 0
    final = obs
    for step in range(1, max_pairs + 1):
        final = env.step(double_sided_pair(target_addr, row_bytes))
        if final.error:
            print(f"  step {step}: ERROR {final.error}")
            break
        if step % 500 == 0 or final.done:
            br = final.metadata.get("budget_remaining", {})
            print(f"  step {step:>5}: cycle={final.cycle:>9} reward={final.reward} "
                  f"done={final.done} new_flips={final.feedback.get('new_public_flips')} "
                  f"budget(tool_calls={br.get('tool_calls')}, cycles={br.get('cycles')})")
        if final.done:
            break

    print(f"\n  RESULT: reward={final.reward} done={final.done} after {step} hammer steps")
    if final.feedback.get("public_flips"):
        print(f"  public_flips: {final.feedback['public_flips']}")
    assert final.reward == 1.0 and final.done, "expected trusted success reward"
    env.close()
    print("  -> trusted sparse reward fired ONLY after a real simulated flip. PASS")


def control_finish_before_flip(seed: int = 5) -> None:
    banner("EPISODE B  -  negative control: under-hammer then finish (reward MUST stay 0)")
    env = RowHammerTaskEnv()
    obs = env.reset(seed=seed, episode_id="rl_control")
    assert not obs.error, obs.error
    target_addr = obs.metadata["target"]["addr"]
    threshold = obs.metadata["disturbance"]["known_threshold"]

    row_bytes = env.disturbance.row_bytes
    pairs = threshold // 2 - 1  # one pair short of the flip threshold
    print(f"  hammering {pairs} pairs (exposure {pairs * 2} < threshold {threshold}) ...")
    last = None
    for _ in range(pairs):
        last = env.step(double_sided_pair(target_addr, row_bytes))
        assert not last.error, last.error
    print(f"  after under-hammering: reward={last.reward} done={last.done} "
          f"new_flips so far reported={last.feedback.get('new_public_flips')}")
    finished = env.step(Phase2Action(tool="episode.finish", args={}))
    print(f"  episode.finish -> reward={finished.reward} done={finished.done}")
    assert finished.reward == 0.0 and finished.done, "control must not be rewarded"
    env.close()
    print("  -> no flip => no reward, even though the policy 'finished'. PASS")


def control_oracle_mitigation(seed: int = 5) -> None:
    banner("EPISODE C  -  oracle mitigation: same hammer, flip MUST be prevented (reward 0)")
    env = RowHammerTaskEnv(mitigation={"name": "oracle", "params": {}})
    obs = env.reset(seed=seed, episode_id="rl_oracle")
    assert not obs.error, obs.error
    target_addr = obs.metadata["target"]["addr"]
    threshold = obs.metadata["disturbance"]["known_threshold"]

    row_bytes = env.disturbance.row_bytes
    pairs = threshold  # hammer well past the unmitigated threshold
    print(f"  hammering {pairs} pairs (2x the unmitigated flip point) under oracle mitigation ...")
    last = obs
    refreshes = 0
    for _ in range(pairs):
        last = env.step(double_sided_pair(target_addr, row_bytes))
        if last.error:
            print(f"  ERROR {last.error}")
            break
        refreshes += int(last.feedback.get("oracle_refreshes", 0) or 0)
        if last.done:
            break
    print(f"  RESULT: reward={last.reward} done={last.done} oracle_refreshes_total={refreshes}")
    assert last.reward == 0.0, "oracle mitigation must prevent the flip"
    env.close()
    print("  -> mitigation refreshed the victim before any flip => reward 0. PASS")


def rollout_script_path(seed: int = 6) -> None:
    banner("EPISODE D  -  sandboxed script.run path (rh_sdk broker) should SUCCEED")
    env = RowHammerTaskEnv()
    obs = env.reset(seed=seed, episode_id="rl_script")
    assert not obs.error, obs.error
    target_addr = obs.metadata["target"]["addr"]
    threshold = obs.metadata["disturbance"]["known_threshold"]
    left, right = target_addr - env.disturbance.row_bytes, target_addr + env.disturbance.row_bytes

    code = (
        "from rh_sdk import rh\n"
        f"for _ in range({threshold // 2}):\n"
        f"    rh.issue(commands=[{{'op':'RD','addr':{left}}}, {{'op':'RD','addr':{right}}}])\n"
    )
    print("  submitting a restricted rh_sdk script (one double-sided hammer loop)...")
    out = env.step(Phase2Action(tool="script.run", args={"language": "python-rh-sdk", "code": code}))
    assert not out.error, out.error
    print(f"  script.run -> reward={out.reward} done={out.done} "
          f"tool_calls={out.feedback['script']['tool_calls']}")
    assert out.reward == 1.0, "script path should reach trusted success"

    # And the sandbox must reject host access.
    bad = env.step(Phase2Action(tool="script.run", args={"language": "python-rh-sdk", "code": "import os\n"}))
    print(f"  malicious 'import os' -> error={bad.error}")
    assert bad.error and bad.error["code"] == "SANDBOX_VIOLATION", "sandbox must fail closed"
    env.close()
    print("  -> script path is trace-equivalent AND blocks host access. PASS")


def main() -> int:
    print("rhdram-env RL run :: OpenEnv RowHammerTaskEnv reset/step rollouts")
    rollout_known_target()
    control_finish_before_flip()
    control_oracle_mitigation()
    rollout_script_path()
    banner("ALL RL ROLLOUTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
