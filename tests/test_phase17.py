import asyncio
import base64
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import unittest
import urllib.request

from rowhammer_env import Phase2Action, RowHammerTaskEnv


ROOT = pathlib.Path(__file__).resolve().parents[1]
# The policy-side SDK lives in the top-level sdk/ dir (mirrors how the repo puts
# rowhammer_env on the cwd path); make it importable when running the suite.
if str(ROOT / "sdk") not in sys.path:
    sys.path.insert(0, str(ROOT / "sdk"))
WORKER = ROOT / "build/phase2/ramulator_worker"
CONFIG = ROOT / "build/phase2/p2_external_ddr4.yaml"
HAS_WORKER = WORKER.is_file() and CONFIG.is_file()
HAS_HTTP_DEPS = all(importlib.util.find_spec(name) for name in ("fastapi", "uvicorn", "websockets", "fastmcp", "requests"))


def _rd(addr: int) -> dict[str, object]:
    return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}


def _write(addr: int, payload: bytes) -> Phase2Action:
    return Phase2Action(
        tool="dram.write",
        args={"addr": {"kind": "logical", "addr": addr}, "data_b64": base64.b64encode(payload).decode()},
    )


def _read(addr: int, length: int) -> Phase2Action:
    return Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": addr}, "length": length})


@unittest.skipUnless(HAS_WORKER, "Phase 2 worker/config not built")
@unittest.skipUnless(HAS_HTTP_DEPS, "P17 HTTP dependencies not installed")
class Phase17OpenEnvHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.port = _free_port()
        except OSError as exc:
            raise unittest.SkipTest(f"local sockets unavailable: {exc}") from exc
        env = os.environ.copy()
        env["MAX_CONCURRENT_ENVS"] = "4"
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cls.proc = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-m",
                "uvicorn",
                "rowhammer_env.server.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls.port),
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        _wait_for_health(cls.base_url, cls.proc)

    @classmethod
    def tearDownClass(cls) -> None:
        proc = getattr(cls, "proc", None)
        if proc is None:
            return
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=5)

    def test_websocket_step_matches_in_process_reward_and_trace_shape(self) -> None:
        seed = 17
        direct = RowHammerTaskEnv(task={"family": "known_target_anybit"})
        direct_reset = direct.reset(seed=seed, episode_id="p17_direct")
        assert direct.disturbance is not None
        target = direct.disturbance.target_addr
        row_bytes = direct.disturbance.row_bytes
        action = Phase2Action(tool="dram.issue", args={"commands": [_rd(target - row_bytes), _rd(target + row_bytes)]})
        expected = direct.step(action)
        direct.close()

        async def run_client():
            from rowhammer_env.client import RowHammerClient

            async with RowHammerClient(base_url=self.base_url) as client:
                reset = await client.reset(seed=seed, episode_id="p17_ws")
                result = await client.step(action)
                return reset, result

        reset, result = asyncio.run(run_client())
        self.assertEqual(result.reward, expected.reward)
        self.assertEqual(result.done, expected.done)
        self.assertEqual(result.observation.cycle, expected.cycle)
        self.assertIn("trace_tail", result.observation.feedback)
        # SPEC §8 initial-observation fields live in metadata, which OpenEnv strips
        # from the wire; verify the info-mirror carries them and matches in-process.
        self.assertEqual(reset.observation.metadata.get("objective"), direct_reset.metadata.get("objective"))
        self.assertEqual(reset.observation.info, reset.observation.metadata)
        self.assertIn("budget_remaining", result.observation.metadata)

    def test_base64_and_env_error_round_trip(self) -> None:
        data_b64 = base64.b64encode(b"p17!").decode()
        addr = {"kind": "logical", "addr": 8192}

        async def run_client():
            from rowhammer_env.client import RowHammerClient

            async with RowHammerClient(base_url=self.base_url) as client:
                await client.reset(seed=17, episode_id="p17_b64")
                write = await client.step(Phase2Action(tool="dram.write", args={"addr": addr, "data_b64": data_b64}))
                read = await client.step(Phase2Action(tool="dram.read", args={"addr": addr, "length": 4}))
                bad = await client.step(Phase2Action(tool="dram.issue", args={}))
                return write, read, bad

        write, read, bad = asyncio.run(run_client())
        self.assertIsNone(write.observation.error)
        self.assertEqual(base64.b64decode(read.observation.data_b64 or ""), b"p17!")
        self.assertEqual(bad.observation.error["code"], "BAD_SCHEMA")

    def test_concurrent_websocket_sessions_are_isolated(self) -> None:
        addr = 8192
        wait = Phase2Action(tool="dram.issue", args={"commands": [{"op": "WAIT", "cycles": 50}]})

        async def run_clients():
            from rowhammer_env.client import RowHammerClient

            async with RowHammerClient(base_url=self.base_url) as a, RowHammerClient(base_url=self.base_url) as b:
                await a.reset(seed=17, episode_id="p17_a")
                await b.reset(seed=18, episode_id="p17_b")
                await a.step(_write(addr, b"AAAA"))
                await b.step(_write(addr, b"BBBB"))
                await a.step(wait)
                read_a = await a.step(_read(addr, 4))
                read_b = await b.step(_read(addr, 4))
                return await a.state(), await b.state(), read_a, read_b

        state_a, state_b, read_a, read_b = asyncio.run(run_clients())
        # Hidden state (per-session memory) does not bleed across sessions: each
        # sees only its own write to the same address.
        self.assertEqual(base64.b64decode(read_a.observation.data_b64 or ""), b"AAAA")
        self.assertEqual(base64.b64decode(read_b.observation.data_b64 or ""), b"BBBB")
        # Session bookkeeping is independent too.
        self.assertEqual(state_a.episode_id, "p17_a")
        self.assertEqual(state_b.episode_id, "p17_b")
        self.assertGreater(state_a.step_count, state_b.step_count)
        self.assertGreater(state_a.cycle, state_b.cycle)

    def test_wire_task_selection_drives_the_curriculum(self) -> None:
        # An orchestrator selects a different family per episode over the wire; the
        # served default is known_target_anybit, so any_flip proves task selection.
        async def run_client():
            from rowhammer_env.client import RowHammerClient

            async with RowHammerClient(base_url=self.base_url) as client:
                reset = await client.reset(seed=17, episode_id="p17_task", task={"family": "any_flip"})
                bad = await client.reset(seed=17, episode_id="p17_bad", task={"family": "no_such_family"})
                return reset, bad

        reset, bad = asyncio.run(run_client())
        self.assertIsNone(reset.observation.error)
        self.assertEqual(reset.observation.metadata.get("task_family"), "any_flip")
        self.assertEqual(reset.observation.metadata.get("objective", {}).get("type"), "any_flip")
        # any_flip hides the victim: no target coordinates leak into the wire obs.
        self.assertNotIn("target", reset.observation.metadata)
        # A malformed task config fails closed with a stable SPEC §8 code.
        self.assertEqual(bad.observation.error["code"], "BAD_SCHEMA")

    def test_malformed_action_maps_to_stable_error_code(self) -> None:
        # A structurally-malformed action (no tool) must surface a SPEC §8 code over
        # the wire (BAD_SCHEMA), not OpenEnv's transport-level VALIDATION_ERROR.
        async def run_wire():
            from websockets.asyncio.client import connect

            async with connect(f"ws://127.0.0.1:{self.port}/ws") as ws:
                await ws.send(json.dumps({"type": "reset", "data": {"seed": 17}}))
                reset = json.loads(await ws.recv())
                await ws.send(json.dumps({"type": "step", "data": {"args": {}}}))
                response = json.loads(await ws.recv())
                return reset, response

        reset, response = asyncio.run(run_wire())
        self.assertEqual(reset["type"], "observation")
        self.assertEqual(response["type"], "observation")
        self.assertEqual(response["data"]["observation"]["error"]["code"], "BAD_SCHEMA")

    def test_rh_sdk_surface_drives_the_server(self) -> None:
        import rh_sdk
        from rh_sdk import RowHammerSDK, connect

        from rowhammer_env.client import RowHammerClient

        self.assertTrue(hasattr(rh_sdk, "RowHammerSDK"))

        async def run_client():
            client = connect(self.base_url)
            self.assertIsInstance(client, RowHammerClient)
            async with client:
                sdk = RowHammerSDK(client)
                await client.reset(seed=17, episode_id="p17_sdk")
                return await sdk.info()

        info = asyncio.run(run_client())
        self.assertIn("address_forms", info.observation.metadata)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


if __name__ == "__main__":
    unittest.main()
