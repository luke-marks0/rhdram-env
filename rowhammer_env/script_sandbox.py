from __future__ import annotations

import ast
from typing import Any

from .phase2_env import Phase2Action


class ScriptViolation(ValueError):
    code = "SANDBOX_VIOLATION"


class RestrictedScriptBroker:
    """Compile a tiny rh_sdk Python subset into brokered environment calls."""

    def __init__(self, env: Any, max_calls: int = 10_000) -> None:
        self.env = env
        self.max_calls = max_calls
        self.calls = 0

    def run(self, code: str) -> dict[str, Any]:
        tree = ast.parse(code)
        observations: list[dict[str, Any]] = []
        for stmt in tree.body:
            observations.extend(self._stmt(stmt))
        return {"stdout_tail": "", "tool_calls": self.calls, "observations": observations[-8:]}

    def _stmt(self, stmt: ast.stmt) -> list[dict[str, Any]]:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "rh_sdk":
            if any(alias.name != "rh" for alias in stmt.names):
                raise ScriptViolation("only 'from rh_sdk import rh' is allowed")
            return []
        if isinstance(stmt, ast.Expr):
            return [self._call(stmt.value)]
        if isinstance(stmt, ast.For):
            if not isinstance(stmt.target, ast.Name) or stmt.orelse:
                raise ScriptViolation("only simple for-loops are allowed")
            rng = self._range(stmt.iter)
            out: list[dict[str, Any]] = []
            for _ in range(rng):
                for child in stmt.body:
                    out.extend(self._stmt(child))
            return out
        raise ScriptViolation(f"statement is not allowed: {type(stmt).__name__}")

    def _range(self, node: ast.AST) -> int:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) == 1
        ):
            raise ScriptViolation("loops must be 'for _ in range(N)'")
        n = ast.literal_eval(node.args[0])
        if not isinstance(n, int) or n < 0 or n > self.max_calls:
            raise ScriptViolation("loop bound is outside script limits")
        return n

    def _call(self, node: ast.AST) -> dict[str, Any]:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "rh"
        ):
            raise ScriptViolation("only rh_sdk broker calls are allowed")
        method = node.func.attr
        if method not in {"info", "read", "write", "issue", "finish"}:
            raise ScriptViolation(f"rh.{method} is not available")
        if node.args:
            raise ScriptViolation("positional arguments are not allowed")
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        return self._tool(method, kwargs)

    def _tool(self, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.calls > self.max_calls:
            raise ScriptViolation("script tool-call limit exceeded")
        tool = {
            "info": "dram.info",
            "read": "dram.read",
            "write": "dram.write",
            "issue": "dram.issue",
            "finish": "episode.finish",
        }[method]
        args = self._normalize_args(method, kwargs)
        obs = self.env.step(Phase2Action(tool=tool, args=args))
        return {
            "tool": tool,
            "reward": obs.reward,
            "done": obs.done,
            "error": obs.error,
            "feedback": obs.feedback,
            "data_b64": obs.data_b64,
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
