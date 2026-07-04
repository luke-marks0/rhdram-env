from __future__ import annotations

import asyncio
import json
import pathlib
import socket
import subprocess
import sys
import time
import unittest
import urllib.request

import yaml

from rowhammer_env import RowHammerTaskEnv
from rowhammer_env.llm import CIHammerFixturePolicy, ClaimSuccessFixturePolicy, RolloutConfig, TOOL_SCHEMAS, run_episode
from rowhammer_env.llm.policies import OpenAICompatibleToolPolicy, ToolCall
from rowhammer_env.llm.rollout import run_curriculum
from rowhammer_env.observability import summarize_episodes


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"
CONFIG = ROOT / "build/phase2/p2_external_ddr4.yaml"


class Phase19UnitTests(unittest.TestCase):
    def test_tool_schema_covers_openenv_tools(self) -> None:
        self.assertEqual(
            {schema["function"]["name"] for schema in TOOL_SCHEMAS},
            {"dram.info", "dram.read", "dram.write", "dram.issue", "script.run", "episode.finish"},
        )

    def test_openai_compatible_parser_extracts_tool_call(self) -> None:
        policy = OpenAICompatibleToolPolicy(url="http://127.0.0.1/unused", model="tool-model")
        call = policy._parse_tool_call(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "dram.info", "arguments": "{}"}},
                            ]
                        }
                    }
                ]
            }
        )
        self.assertEqual(call, ToolCall("dram.info", {}))

    def test_reset_accepts_dynamic_task(self) -> None:
        env = RowHammerTaskEnv(task={"family": "known_target_anybit"})
        obs = env.reset(seed=19, task={"family": "profile_generalization", "split": "eval"})
        self.assertIsNone(obs.error)
        self.assertEqual(obs.metadata["task_family"], "profile_generalization")
        self.assertEqual(obs.metadata["disturbance"]["family"], "axmicr")
        env.close()

    def test_dynamic_task_keeps_explicit_profile_override(self) -> None:
        env = RowHammerTaskEnv(profile_id="hbm2_read_disturbance_v1")
        env._configure_task({"family": "known_target_anybit", "profile": "ddr4_vts25_v1"})
        self.assertEqual(env.profile_id, "hbm2_read_disturbance_v1")
        env.close()


@unittest.skipUnless(WORKER.is_file() and CONFIG.is_file(), "Phase 2 worker/config not built")
class Phase19HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.port = _free_port()
        except OSError as exc:
            raise unittest.SkipTest(f"local sockets unavailable: {exc}") from exc
        cls.proc = _start_server(cls.port)
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        _stop_server(cls.proc)

    def test_http_policy_success_and_claim_control(self) -> None:
        task = {"family": "known_target_anybit"}
        success = asyncio.run(
            run_episode(RolloutConfig(base_url=self.base_url, seed=19, task=task, episode_id="t19_success", max_steps=2), CIHammerFixturePolicy())
        )
        control = asyncio.run(
            run_episode(RolloutConfig(base_url=self.base_url, seed=19, task=task, episode_id="t19_control", max_steps=1), ClaimSuccessFixturePolicy())
        )
        self.assertTrue(success.success)
        self.assertEqual(success.reward, 1.0)
        self.assertEqual(control.reward, 0.0)
        self.assertTrue(control.done)

    def test_eval_metrics_are_reproducible_on_heldout_split(self) -> None:
        task = yaml.safe_load((ROOT / "configs/tasks/profile_generalization_eval.yaml").read_text())
        first = summarize_episodes(
            asyncio.run(
                run_curriculum(
                    base_url=self.base_url,
                    policy_factory=lambda: CIHammerFixturePolicy(probe_pairs=8),
                    tasks=[task],
                    seeds=[21, 22],
                    max_steps=2,
                )
            )
        )
        second = summarize_episodes(
            asyncio.run(
                run_curriculum(
                    base_url=self.base_url,
                    policy_factory=lambda: CIHammerFixturePolicy(probe_pairs=8),
                    tasks=[task],
                    seeds=[21, 22],
                    max_steps=2,
                )
            )
        )
        self.assertEqual(first.to_json(), second.to_json())
        self.assertIn("eval", first.by_split)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(port: int) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-m",
            "uvicorn",
            "rowhammer_env.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_for_health(f"http://127.0.0.1:{port}", proc)
    return proc


def _stop_server(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)


def _wait_for_health(base_url: str, proc: subprocess.Popen[str]) -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise RuntimeError(f"uvicorn exited early\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("server did not become healthy")


if __name__ == "__main__":
    unittest.main()
