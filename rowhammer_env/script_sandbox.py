from __future__ import annotations

import json
import os
import pathlib
import resource
import select
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from .phase2_env import Phase2Action, Phase2Observation


class ScriptError(ValueError):
    code = "SANDBOX_VIOLATION"


class ScriptViolation(ScriptError):
    code = "SANDBOX_VIOLATION"


class ScriptTimeout(ScriptError):
    code = "SCRIPT_TIMEOUT"


class SandboxUnavailable(ScriptError):
    code = "UNAVAILABLE_CAPABILITY"


CHILD_RUNNER = r"""
import ast
import builtins
import contextlib
import json
import sys
import types

PROTO_IN = sys.stdin
PROTO_OUT = sys.stdout
RH_METHODS = {"info", "read", "write", "issue", "finish"}
SAFE_CALLS = {"abs", "all", "any", "bool", "bytes", "dict", "enumerate", "float", "int", "len", "list", "max", "min", "print", "range", "str", "sum", "tuple", "zip"}
FORBIDDEN_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "object",
    "open",
    "setattr",
    "type",
    "vars",
}


def _send(obj):
    PROTO_OUT.write(json.dumps(obj, separators=(",", ":")) + "\n")
    PROTO_OUT.flush()


class _BoundedWriter:
    def __init__(self, limit, tail_limit):
        self.limit = int(limit)
        self.tail_limit = int(tail_limit)
        self.total = 0
        self.parts = []

    def write(self, data):
        data = str(data)
        self.total += len(data.encode("utf-8", "replace"))
        if self.total > self.limit:
            raise RuntimeError("script stdout/stderr limit exceeded")
        self.parts.append(data)
        joined = "".join(self.parts)
        if len(joined) > self.tail_limit:
            joined = joined[-self.tail_limit:]
        self.parts = [joined]
        return len(data)

    def flush(self):
        return None

    def tail(self):
        return "".join(self.parts)


class _Rh:
    def __init__(self, max_calls):
        self.max_calls = int(max_calls)
        self.calls = 0
        self.observations = []

    def _call(self, method, kwargs):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("script tool-call limit exceeded")
        _send({"type": "tool", "method": method, "kwargs": kwargs})
        line = PROTO_IN.readline()
        if not line:
            raise RuntimeError("broker closed the IPC channel")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "broker rejected the tool call"))
        observation = response.get("observation", {})
        self.observations.append(observation)
        self.observations = self.observations[-8:]
        return observation

    def info(self, **kwargs):
        return self._call("info", kwargs)

    def read(self, **kwargs):
        return self._call("read", kwargs)

    def write(self, **kwargs):
        return self._call("write", kwargs)

    def issue(self, **kwargs):
        return self._call("issue", kwargs)

    def finish(self, **kwargs):
        return self._call("finish", kwargs)


def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name == "rh_sdk":
        return sys.modules["rh_sdk"]
    raise ImportError("imports are disabled in the policy sandbox")


def _validate_policy(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            raise RuntimeError("imports are disabled in the policy sandbox")
        if isinstance(node, ast.ImportFrom):
            names = {alias.name for alias in node.names}
            if node.module != "rh_sdk" or names != {"rh"}:
                raise RuntimeError("only 'from rh_sdk import rh' is allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise RuntimeError("dunder attribute access is disabled in the policy sandbox")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_NAMES:
                raise RuntimeError(f"name is not available in the policy sandbox: {node.id}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in SAFE_CALLS:
                    raise RuntimeError(f"call is not available in the policy sandbox: {func.id}")
            elif isinstance(func, ast.Attribute):
                if not (isinstance(func.value, ast.Name) and func.value.id == "rh" and func.attr in RH_METHODS):
                    raise RuntimeError("only rh_sdk broker method calls are available")
            else:
                raise RuntimeError("dynamic calls are disabled in the policy sandbox")
        if isinstance(
            node,
            (
                ast.AsyncFor,
                ast.AsyncFunctionDef,
                ast.AsyncWith,
                ast.Await,
                ast.ClassDef,
                ast.Delete,
                ast.FunctionDef,
                ast.Global,
                ast.Lambda,
                ast.Nonlocal,
                ast.Try,
                ast.With,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            raise RuntimeError(f"statement is not available in the policy sandbox: {type(node).__name__}")


def _main():
    config = json.loads(PROTO_IN.readline())
    rh = _Rh(config.get("max_calls", 0))

    rh_sdk = types.ModuleType("rh_sdk")
    rh_sdk.rh = rh
    sys.modules["rh_sdk"] = rh_sdk

    safe_builtins = {
        "__import__": _limited_import,
        "abs": builtins.abs,
        "all": builtins.all,
        "any": builtins.any,
        "bool": builtins.bool,
        "bytes": builtins.bytes,
        "dict": builtins.dict,
        "enumerate": builtins.enumerate,
        "float": builtins.float,
        "int": builtins.int,
        "len": builtins.len,
        "list": builtins.list,
        "max": builtins.max,
        "min": builtins.min,
        "print": builtins.print,
        "range": builtins.range,
        "str": builtins.str,
        "sum": builtins.sum,
        "tuple": builtins.tuple,
        "zip": builtins.zip,
    }
    globals_ = {"__builtins__": safe_builtins}
    stdout = _BoundedWriter(config.get("stdout_limit", 65536), config.get("stdout_tail", 4096))
    stderr = _BoundedWriter(config.get("stdout_limit", 65536), config.get("stdout_tail", 4096))
    try:
        tree = ast.parse(config.get("code", ""), "<policy>", "exec")
        _validate_policy(tree)
        compiled = compile(tree, "<policy>", "exec")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compiled, globals_, {})
    except BaseException as exc:
        _send({"type": "violation", "message": f"{type(exc).__name__}: {exc}"})
        return 1
    _send(
        {
            "type": "result",
            "result": {
                "stdout_tail": stdout.tail(),
                "stderr_tail": stderr.tail(),
                "tool_calls": rh.calls,
                "observations": rh.observations,
            },
        }
    )
    return 0


raise SystemExit(_main())
"""


PROBE_RUNNER = r"""
import json
import socket


def blocked_path(path):
    try:
        with open(path, "rb") as handle:
            handle.read(1)
        return False
    except OSError:
        return True


def network_blocked():
    try:
        with socket.create_connection(("127.0.0.1", 9), timeout=0.2):
            return False
    except OSError:
        return True


print(
    json.dumps(
        {
            "runtime": "unshare+bwrap",
            "host_fs_blocked": blocked_path("/home/arch/repos/rhdram-env/README.md"),
            "proc_pagemap_blocked": blocked_path("/proc/pagemap"),
            "dev_mem_blocked": blocked_path("/dev/mem"),
            "dev_kvm_blocked": blocked_path("/dev/kvm"),
            "network_blocked": network_blocked(),
        },
        sort_keys=True,
    )
)
"""


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = 2
    memory_bytes: int = 256 * 1024 * 1024
    wall_time_s: float = 5.0
    stdout_bytes: int = 64 * 1024
    stdout_tail_bytes: int = 4096
    protocol_bytes: int = 2 * 1024 * 1024
    processes: int = 4096
    open_files: int = 64


@dataclass(frozen=True)
class SandboxRuntime:
    unshare: str
    bwrap: str
    python: str

    @classmethod
    def detect(cls) -> "SandboxRuntime":
        unshare = shutil.which("unshare")
        bwrap = shutil.which("bwrap")
        python = "/usr/bin/python" if pathlib.Path("/usr/bin/python").is_file() else shutil.which("python3")
        if not unshare or not bwrap or not python:
            raise SandboxUnavailable("approved sandbox runtime is unavailable")
        runtime = cls(unshare=unshare, bwrap=bwrap, python=python)
        attestation = runtime.attest()
        required = ("host_fs_blocked", "proc_pagemap_blocked", "dev_mem_blocked", "dev_kvm_blocked", "network_blocked")
        if not all(attestation.get(key) is True for key in required):
            raise SandboxUnavailable(f"sandbox runtime failed attestation: {attestation}")
        return runtime

    def command(self, code: str) -> list[str]:
        return [
            self.unshare,
            "--net",
            "--user",
            "--map-root-user",
            self.bwrap,
            "--unshare-ipc",
            "--unshare-pid",
            "--unshare-uts",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind-try",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--clearenv",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTHONHASHSEED",
            "0",
            "--chdir",
            "/tmp",
            "--",
            self.python,
            "-I",
            "-S",
            "-c",
            code,
        ]

    def attest(self) -> dict[str, Any]:
        limits = SandboxLimits(wall_time_s=2.0)
        proc = subprocess.run(
            self.command(PROBE_RUNNER),
            input="",
            text=True,
            capture_output=True,
            timeout=limits.wall_time_s,
            preexec_fn=_limit_process(limits),
        )
        if proc.returncode != 0:
            raise SandboxUnavailable((proc.stderr or proc.stdout or "sandbox probe failed")[:500])
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise SandboxUnavailable(f"sandbox probe returned invalid JSON: {exc}") from exc
        return data


_RUNTIME: SandboxRuntime | None = None


def sandbox_attestation() -> dict[str, Any]:
    return _runtime().attest()


def _runtime() -> SandboxRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = SandboxRuntime.detect()
    return _RUNTIME


def _limit_process(limits: SandboxLimits):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.stdout_bytes, limits.stdout_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (limits.processes, limits.processes))
        os.setsid()

    return apply


class RestrictedScriptBroker:
    """Run policy Python in an OS-isolated child and broker rh_sdk calls over IPC."""

    def __init__(
        self,
        env: Any,
        max_calls: int = 10_000,
        timeout_ms: int = 5000,
        runtime: SandboxRuntime | None = None,
    ) -> None:
        self.env = env
        self.max_calls = max(0, int(max_calls))
        self.limits = SandboxLimits(wall_time_s=max(0.001, int(timeout_ms) / 1000.0))
        self.runtime = runtime or _runtime()
        self.calls = 0

    def run(self, code: str) -> dict[str, Any]:
        if len(code) > 200_000:
            raise ScriptViolation("script exceeds maximum length")
        proc = subprocess.Popen(
            self.runtime.command(CHILD_RUNNER),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=_limit_process(self.limits),
        )
        assert proc.stdin is not None and proc.stdout is not None
        config = {
            "code": code,
            "max_calls": self.max_calls,
            "stdout_limit": self.limits.stdout_bytes,
            "stdout_tail": self.limits.stdout_tail_bytes,
        }
        try:
            self._write(proc, config)
            return self._serve(proc)
        finally:
            self._terminate(proc)
            self._close_pipes(proc)

    def _serve(self, proc: subprocess.Popen[str]) -> dict[str, Any]:
        assert proc.stdout is not None
        deadline = time.monotonic() + self.limits.wall_time_s
        protocol_bytes = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timeout(proc)
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                raise self._timeout(proc)
            line = proc.stdout.readline()
            if line == "":
                stderr = self._stderr_tail(proc)
                code = proc.poll()
                if code is not None and code < 0 and -code in (signal.SIGKILL, signal.SIGXCPU):
                    raise ScriptTimeout("script exceeded sandbox resource limits")
                raise ScriptViolation(f"sandbox process exited before result: {stderr or code}")
            protocol_bytes += len(line.encode("utf-8", "replace"))
            if protocol_bytes > self.limits.protocol_bytes:
                raise ScriptViolation("sandbox protocol output limit exceeded")
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScriptViolation(f"sandbox protocol violation: {exc}") from exc
            kind = message.get("type")
            if kind == "tool":
                self._handle_tool(proc, message)
            elif kind == "result":
                return dict(message.get("result") or {})
            elif kind == "violation":
                raise ScriptViolation(str(message.get("message", "sandbox violation")))
            else:
                raise ScriptViolation(f"sandbox protocol sent unknown message type: {kind}")

    def _handle_tool(self, proc: subprocess.Popen[str], message: dict[str, Any]) -> None:
        self.calls += 1
        if self.calls > self.max_calls:
            raise ScriptViolation("script tool-call limit exceeded")
        method = str(message.get("method", ""))
        kwargs = dict(message.get("kwargs") or {})
        tool = {
            "info": "dram.info",
            "read": "dram.read",
            "write": "dram.write",
            "issue": "dram.issue",
            "finish": "episode.finish",
        }.get(method)
        if tool is None:
            raise ScriptViolation(f"rh.{method} is not available")
        try:
            args = self._normalize_args(method, kwargs)
            obs = self.env.step(Phase2Action(tool=tool, args=args))
        except (KeyError, TypeError, ValueError) as exc:
            self._write(proc, {"ok": False, "error": str(exc)})
            return
        self._write(proc, {"ok": True, "observation": self._observation(tool, obs)})

    def _write(self, proc: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            proc.stdin.flush()
        except BrokenPipeError as exc:
            raise ScriptViolation("sandbox IPC channel closed") from exc

    def _observation(self, tool: str, obs: Phase2Observation) -> dict[str, Any]:
        return {
            "tool": tool,
            "reward": obs.reward,
            "done": obs.done,
            "error": obs.error,
            "feedback": obs.feedback,
            "metadata": obs.metadata,
            "data_b64": obs.data_b64,
            "cycle": obs.cycle,
            "last_action": obs.last_action,
            "public_counters": obs.public_counters,
        }

    def _normalize_args(self, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if method == "read":
            return {"addr": self._addr(kwargs["addr"]), "length": kwargs.get("length", 1)}
        if method == "write":
            return {"addr": self._addr(kwargs["addr"]), "data_b64": kwargs["data_b64"]}
        if method == "issue":
            return {"commands": [self._command(c) for c in kwargs["commands"]]}
        return kwargs

    def _command(self, command: dict[str, Any]) -> dict[str, Any]:
        out = dict(command)
        if "addr" in out:
            out["addr"] = self._addr(out["addr"])
        return out

    def _addr(self, addr: Any) -> dict[str, Any]:
        if isinstance(addr, int):
            return {"kind": "logical", "addr": addr}
        if isinstance(addr, dict) and addr.get("kind") == "logical":
            return addr
        raise ScriptViolation("only logical addresses are available to scripts")

    def _timeout(self, proc: subprocess.Popen[str]) -> ScriptTimeout:
        self._terminate(proc)
        return ScriptTimeout("script exceeded wall-time limit")

    def _stderr_tail(self, proc: subprocess.Popen[str]) -> str:
        if proc.stderr is None:
            return ""
        try:
            return proc.stderr.read(500)
        except OSError:
            return ""

    def _terminate(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.kill()
            proc.wait(timeout=1)
        except OSError:
            return
        except subprocess.TimeoutExpired:
            return

    def _close_pipes(self, proc: subprocess.Popen[str]) -> None:
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is None or pipe.closed:
                continue
            try:
                pipe.close()
            except OSError:
                pass
