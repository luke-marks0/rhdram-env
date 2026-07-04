from __future__ import annotations

import base64
import pathlib
import unittest

from rowhammer_env import Phase2Action, Phase2Observation, RowHammerTaskEnv
from rowhammer_env.script_sandbox import RestrictedScriptBroker, sandbox_attestation


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
CONFIG = ROOT / "build/phase2/p2_external_ddr4.yaml"


class _DummyEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def step(self, action: Phase2Action) -> Phase2Observation:
        self.calls.append((action.tool, action.args))
        return Phase2Observation(
            reward=0.0,
            done=False,
            metadata={"x": 1},
            feedback={"trace_tail": [{"op": "RD", "row": 7}]},
        )


class SandboxRuntimeTests(unittest.TestCase):
    def test_runtime_attestation_blocks_host_surfaces(self) -> None:
        attestation = sandbox_attestation()
        self.assertEqual(attestation["runtime"], "unshare+bwrap")
        for key in ("host_fs_blocked", "proc_pagemap_blocked", "dev_mem_blocked", "dev_kvm_blocked", "network_blocked"):
            self.assertIs(attestation[key], True, key)
        # Positive control: a bound path must be readable, so the *_blocked results
        # above cannot all be True merely because nothing exists in the probe env.
        self.assertIs(attestation["control_usr_readable"], True)

    def test_broker_uses_ipc_and_returns_observations(self) -> None:
        env = _DummyEnv()
        result = RestrictedScriptBroker(env, max_calls=2).run(
            "from rh_sdk import rh\n"
            "obs = rh.info()\n"
            "print(obs['metadata']['x'])\n"
        )
        self.assertEqual(env.calls, [("dram.info", {})])
        self.assertEqual(result["tool_calls"], 1)
        self.assertIn("1", result["stdout_tail"])
        self.assertEqual(result["observations"][-1]["feedback"]["trace_tail"][0]["row"], 7)

    def test_escape_attempts_fail_closed(self) -> None:
        scripts = [
            "import os\n",
            "open('/etc/hostname').read()\n",
            "print.__self__.__import__('os')\n",
            "from rh_sdk import not_rh\n",
        ]
        for code in scripts:
            env = RowHammerTaskEnv()
            obs = env.step(Phase2Action(tool="script.run", args={"code": code, "timeout_ms": 500}))
            self.assertEqual(obs.error["code"], "SANDBOX_VIOLATION", code)
            env.close()

    def test_wall_time_limit_returns_script_timeout(self) -> None:
        env = RowHammerTaskEnv()
        obs = env.step(Phase2Action(tool="script.run", args={"code": "while True:\n    pass\n", "timeout_ms": 200}))
        self.assertEqual(obs.error["code"], "SCRIPT_TIMEOUT")
        env.close()


@unittest.skipUnless(WORKER.is_file() and CONFIG.is_file(), "Phase 2 worker/config not built")
class SandboxWorkerTests(unittest.TestCase):
    def test_script_trace_matches_direct_tool_sequence(self) -> None:
        direct = RowHammerTaskEnv()
        obs = direct.reset(seed=18, episode_id="p18_direct")
        self.assertIsNone(obs.error)
        dist = direct.disturbance
        assert dist is not None
        left = dist.target_addr - dist.row_bytes
        right = dist.target_addr + dist.row_bytes
        direct_obs = direct.step(
            Phase2Action(
                tool="dram.issue",
                args={
                    "commands": [
                        {"op": "RD", "addr": {"kind": "logical", "addr": left}},
                        {"op": "RD", "addr": {"kind": "logical", "addr": right}},
                    ]
                },
            )
        )
        direct.close()

        script = RowHammerTaskEnv()
        obs = script.reset(seed=18, episode_id="p18_script")
        self.assertIsNone(obs.error)
        code = (
            "from rh_sdk import rh\n"
            f"rh.issue(commands=[{{'op':'RD','addr':{left}}}, {{'op':'RD','addr':{right}}}])\n"
        )
        script_obs = script.step(Phase2Action(tool="script.run", args={"code": code, "timeout_ms": 5000}))
        self.assertIsNone(script_obs.error)
        script_trace = script_obs.feedback["script"]["observations"][-1]["feedback"]["trace_tail"]
        self.assertEqual(script_trace, direct_obs.feedback["trace_tail"])
        script.close()

    def test_malicious_script_does_not_corrupt_simulator_state(self) -> None:
        env = RowHammerTaskEnv()
        obs = env.reset(seed=18, episode_id="p18_corrupt")
        self.assertIsNone(obs.error)
        target = env.disturbance.target_addr
        before = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": target}, "length": 1}))
        bad = env.step(Phase2Action(tool="script.run", args={"code": "import os\n", "timeout_ms": 500}))
        after = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": target}, "length": 1}))
        self.assertEqual(bad.error["code"], "SANDBOX_VIOLATION")
        self.assertEqual(base64.b64decode(before.data_b64 or ""), base64.b64decode(after.data_b64 or ""))
        env.close()


if __name__ == "__main__":
    unittest.main()
