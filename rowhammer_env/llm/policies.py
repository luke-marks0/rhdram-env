from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .tools import TOOL_SCHEMAS


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


class ToolPolicy(Protocol):
    def next_tool(self, observation: Any, trajectory: list[dict[str, Any]]) -> ToolCall:
        ...


class OpenAICompatibleToolPolicy:
    """Tool-call adapter for a real OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or (
            "You are controlling a RowHammer research environment. Use only tool calls. "
            "Reward is assigned only by the environment."
        )
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "OpenAICompatibleToolPolicy":
        url = os.environ.get("RHD_LLM_CHAT_COMPLETIONS_URL")
        model = os.environ.get("RHD_LLM_MODEL")
        if not url or not model:
            raise ValueError("RHD_LLM_CHAT_COMPLETIONS_URL and RHD_LLM_MODEL are required")
        return cls(url=url, model=model, api_key=os.environ.get("RHD_LLM_API_KEY"))

    def next_tool(self, observation: Any, trajectory: list[dict[str, Any]]) -> ToolCall:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "observation": _observation_dict(observation),
                            "recent_steps": trajectory[-4:],
                        },
                        sort_keys=True,
                    ),
                },
            ],
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
        }
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM tool backend request failed: {exc}") from exc
        return self._parse_tool_call(body)

    def _parse_tool_call(self, body: dict[str, Any]) -> ToolCall:
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM tool backend returned no choices")
        message = dict(choices[0].get("message") or {})
        calls = message.get("tool_calls") or []
        if not calls:
            return ToolCall("episode.finish", {})
        function = dict(calls[0].get("function") or {})
        name = str(function.get("name") or "")
        args_raw = function.get("arguments") or "{}"
        args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        return ToolCall(name, args)


class ClaimSuccessFixturePolicy:
    """CI fixture control: claims completion without creating a flip."""

    def next_tool(self, observation: Any, trajectory: list[dict[str, Any]]) -> ToolCall:
        del observation, trajectory
        return ToolCall("episode.finish", {})


class CIHammerFixturePolicy:
    """CI fixture policy that uses disclosed addresses to produce real rewards."""

    def __init__(self, *, probe_pairs: int = 256) -> None:
        self.probe_pairs = int(probe_pairs)
        self._used = False
        self._last_metadata: dict[str, Any] = {}

    def next_tool(self, observation: Any, trajectory: list[dict[str, Any]]) -> ToolCall:
        del trajectory
        if self._used:
            return ToolCall("episode.finish", {})
        self._used = True
        metadata = _metadata(observation)
        self._last_metadata = metadata
        pair = self._aggressor_pair(metadata)
        pairs = self._pair_count(metadata)
        commands = []
        for _ in range(pairs):
            commands.append({"op": "RD", "addr": {"kind": "logical", "addr": pair[0]}})
            commands.append({"op": "RD", "addr": {"kind": "logical", "addr": pair[1]}})
        return ToolCall("dram.issue", {"commands": commands})

    def _pair_count(self, metadata: dict[str, Any]) -> int:
        threshold = (metadata.get("disturbance") or {}).get("known_threshold")
        if isinstance(threshold, int) and threshold > 0:
            return max(1, threshold // 2)
        return max(1, self.probe_pairs)

    def _aggressor_pair(self, metadata: dict[str, Any]) -> tuple[int, int]:
        target = metadata.get("target")
        if isinstance(target, dict) and target.get("kind") == "physical" and "row" in target:
            row = int(target["row"])
            addr = int(target.get("addr", 0))
            row_bytes = addr // row if row > 0 else 131072
            return (addr - row_bytes, addr + row_bytes)
        row_bytes = 131072
        return (2048 * row_bytes, 2050 * row_bytes)


class TrainableHammerPolicy(CIHammerFixturePolicy):
    """Tiny reward-updated policy used by the P19 training example."""

    def __init__(self, *, initial_pairs: int = 1) -> None:
        super().__init__(probe_pairs=initial_pairs)
        self.pairs = int(initial_pairs)

    def _pair_count(self, metadata: dict[str, Any]) -> int:
        del metadata
        return max(1, self.pairs)

    def update(self, *, reward: float, observation: Any) -> dict[str, Any]:
        before = self.pairs
        metadata = _metadata(observation)
        threshold = (metadata.get("disturbance") or self._last_metadata.get("disturbance") or {}).get("known_threshold")
        if reward <= 0.0 and isinstance(threshold, int) and threshold > 0:
            self.pairs = max(self.pairs + 1, threshold // 2)
        return {"pairs_before": before, "pairs_after": self.pairs, "reward": reward}


def _metadata(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "metadata"):
        return dict(observation.metadata)
    if isinstance(observation, dict):
        return dict(observation.get("metadata") or {})
    return {}


def _observation_dict(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return dict(observation)
    return {"repr": repr(observation)}
