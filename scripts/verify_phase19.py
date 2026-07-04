#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.llm import CIHammerFixturePolicy, ClaimSuccessFixturePolicy, RolloutConfig, TOOL_SCHEMAS, run_episode
from rowhammer_env.llm.rollout import run_curriculum
from rowhammer_env.observability import summarize_episodes


def _require_runtime() -> None:
    if not (ROOT / "build/phase2/ramulator_worker").is_file() or not (ROOT / "build/phase2/p2_external_ddr4.yaml").is_file():
        raise SystemExit("Phase 2 worker/config is not built; run scripts/build_phase2.py first")
    missing = [name for name in ("fastapi", "uvicorn", "websockets", "fastmcp", "requests") if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"missing P17 HTTP dependencies: {missing}")


def main() -> int:
    _require_runtime()
    port = _free_port()
    proc = _start_server(port)
    base_url = f"http://127.0.0.1:{port}"
    try:
        _check_tool_schema()
        asyncio.run(_check_reward_and_control(base_url))
        asyncio.run(_check_eval_reproducible(base_url))
        _check_training_script(base_url)
    finally:
        _stop_server(proc)
    print("phase19 verification passed")
    return 0


async def _check_reward_and_control(base_url: str) -> None:
    task = {"family": "known_target_anybit"}
    success = await run_episode(RolloutConfig(base_url=base_url, seed=19, task=task, episode_id="p19_success", max_steps=2), CIHammerFixturePolicy())
    if not success.success or success.reward != 1.0:
        raise SystemExit(f"fixture policy did not earn trusted reward: {success.as_metrics_input()}")
    control = await run_episode(RolloutConfig(base_url=base_url, seed=19, task=task, episode_id="p19_control", max_steps=1), ClaimSuccessFixturePolicy())
    if control.reward != 0.0 or not control.done:
        raise SystemExit(f"finish-without-flip control was rewarded: {control.as_metrics_input()}")
    print("  reward: HTTP policy earns reward after real flip; claim-only control earns 0")


async def _check_eval_reproducible(base_url: str) -> None:
    task = yaml.safe_load((ROOT / "configs/tasks/profile_generalization_eval.yaml").read_text())
    seeds = [21, 22]
    first = summarize_episodes(await run_curriculum(base_url=base_url, policy_factory=lambda: CIHammerFixturePolicy(probe_pairs=8), tasks=[task], seeds=seeds, max_steps=2))
    second = summarize_episodes(await run_curriculum(base_url=base_url, policy_factory=lambda: CIHammerFixturePolicy(probe_pairs=8), tasks=[task], seeds=seeds, max_steps=2))
    if first.to_json() != second.to_json():
        raise SystemExit(f"held-out eval metrics were not reproducible: {first.to_json()} != {second.to_json()}")
    if first.episodes != len(seeds) or "eval" not in first.by_split:
        raise SystemExit(f"held-out eval metrics missing split accounting: {first.as_dict()}")
    print(f"  eval: reproducible held-out metrics {first.to_json()}")


def _check_training_script(base_url: str) -> None:
    result = subprocess.run(
        [sys.executable, "-B", "scripts/train_phase19_policy.py", "--base-url", base_url, "--seed", "19", "--episodes", "2"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    data = json.loads(result.stdout)
    if not data.get("improved") or data.get("rewards") != [0.0, 1.0]:
        raise SystemExit(f"training example did not show a useful reward update: {data}")
    print(f"  training: policy update signal {json.dumps(data, sort_keys=True)}")


def _check_tool_schema() -> None:
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    expected = {"dram.info", "dram.read", "dram.write", "dram.issue", "script.run", "episode.finish"}
    if names != expected:
        raise SystemExit(f"tool schema mismatch: {names}")
    print("  tools: LLM tool schema covers the OpenEnv tool surface")


def _start_server(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["MAX_CONCURRENT_ENVS"] = "4"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
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
        env=env,
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
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise RuntimeError(f"uvicorn exited early\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)
    proc.terminate()
    stdout, stderr = proc.communicate(timeout=5)
    raise RuntimeError(f"server did not become healthy: {last_error}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main())
