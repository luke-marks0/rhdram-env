#!/usr/bin/env python3
"""Repro guard for the RowPress close-matching defect (@spec:sim-rowpress).

``DisturbanceEngine.consume`` used to route closes on the op names ``PRE`` /
``PREA``, which Ramulator's DDR4 never emits (it emits ``PREpb`` and ``PREab``),
and to look ``PREab`` up under its literal ``bankgroup``/``bank`` of -1. Open-row
dwell was therefore only ever settled by the next ACT in the same bank, so a
WAIT between two activations counted as a multi-second open row and earned the
saturated RowPress bonus — making WAIT-separated hammering an order of magnitude
cheaper, in activations, than a real double-sided hammer.

This drives the real worker on the shipped ``bounded_sweep_easy`` band under its
shipped budgets and compares activations-to-first-flip for a plain double-sided
``HAMMER`` against WAIT-separated activations of the same two aggressors. With
the defect present the WAIT form flips after ~280 activations against the
HAMMER's 8000; with dwell settled by the real precharge it does not flip at all
inside the episode's budgets.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env import Phase2Action, RowHammerTaskEnv

WORKER = ROOT / "build/phase2/ramulator_worker"
TASK_CONFIG = ROOT / "configs/tasks/bounded_sweep_easy.yaml"
SEED = 7
HAMMER_PAIRS = 2000
WAIT_CYCLES = 100_000
WAIT_BLOCK_ITERATIONS = 100
MAX_TOOL_CALLS = 64


@dataclass
class Run:
    """Outcome of driving one command block until success or budget exhaustion."""

    success: bool
    acts: int
    cycle: int
    calls: int
    error: dict[str, str] | None


def _require_runtime() -> None:
    if not WORKER.is_file():
        raise SystemExit("Phase 2 worker is not built; run scripts/build_phase2.py first")


def _aggressors(env: RowHammerTaskEnv) -> list[int]:
    ct = env._compiled
    assert ct is not None
    aggressors = [ct.target_addr + c.offset for c in ct.candidates if c.is_aggressor]
    if len(aggressors) != 2:
        raise SystemExit(f"expected a double-sided aggressor pair, got {aggressors}")
    return aggressors


def _drive(label: str, block: list[dict], task: dict) -> Run:
    """Issue ``block`` repeatedly on a fresh episode until a flip or a spent budget."""
    env = RowHammerTaskEnv(task=task)
    try:
        env.reset(seed=SEED, episode_id=f"rowpress_close_{label}")
        acts = 0
        cycle = 0
        for call in range(1, MAX_TOOL_CALLS + 1):
            obs = env.step(Phase2Action(tool="dram.issue", args={"commands": block}))
            acts = int(obs.public_counters.get("acts", acts))
            cycle = int(obs.cycle)
            if obs.error:
                return Run(False, acts, cycle, call, obs.error)
            if obs.done and obs.reward:
                return Run(True, acts, cycle, call, None)
        return Run(False, acts, cycle, MAX_TOOL_CALLS, None)
    finally:
        env.close()


def _wait_block(aggressors: list[int]) -> list[dict]:
    block: list[dict] = []
    for _ in range(WAIT_BLOCK_ITERATIONS):
        for addr in aggressors:
            block.append({"op": "RD", "addr": addr})
            block.append({"op": "WAIT", "cycles": WAIT_CYCLES})
    return block


def main() -> int:
    _require_runtime()
    task = yaml.safe_load(TASK_CONFIG.read_text())

    probe = RowHammerTaskEnv(task=task)
    try:
        probe.reset(seed=SEED, episode_id="rowpress_close_addrs")
        aggressors = _aggressors(probe)
    finally:
        probe.close()

    hammer = _drive("hammer", [{"op": "HAMMER", "rows": aggressors, "pairs": HAMMER_PAIRS}], task)
    print(f"  HAMMER: success={hammer.success} acts={hammer.acts} cycles={hammer.cycle} calls={hammer.calls}")
    if not hammer.success:
        raise SystemExit(f"double-sided HAMMER baseline did not flip: {hammer}")

    wait = _drive("wait", _wait_block(aggressors), task)
    print(f"  WAIT:   success={wait.success} acts={wait.acts} cycles={wait.cycle} calls={wait.calls} error={wait.error}")

    # Same order of magnitude: WAIT-separated activations must not buy a flip at a
    # small fraction of the honest double-sided activation cost.
    floor = hammer.acts // 4
    if wait.success:
        if wait.acts < floor:
            raise SystemExit(
                f"RowPress close matching regressed: WAIT-separated activations flipped after "
                f"{wait.acts} acts, under the {floor}-act floor set by the {hammer.acts}-act HAMMER "
                f"baseline. Open-row dwell is not being settled by the real precharge."
            )
        print(f"  WAIT flipped at {wait.acts} acts, at or above the {floor}-act floor.")
    else:
        # Only a spent task budget proves the WAIT form was driven as far as the
        # episode allows; stopping on this script's own call cap would prove nothing.
        if wait.error is None:
            raise SystemExit(
                f"WAIT run stopped on the script's {MAX_TOOL_CALLS}-call cap rather than on a task "
                f"budget after {wait.acts} acts; raise MAX_TOOL_CALLS — the comparison is vacuous."
            )
        print(f"  WAIT exhausted the episode budget after {wait.acts} acts with no flip ({wait.error['code']}).")

    print("rowpress close-matching verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
