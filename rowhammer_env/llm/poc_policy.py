"""PoC control policies and a timing ablation applied only at the policy boundary."""
from __future__ import annotations

import json
from typing import Any

from .grpo_env import hint_actions
from .policies import ClaimSuccessFixturePolicy, ReferenceProbePolicy, ToolCall


class KnownTargetControl:
    def __init__(self, *, below_threshold: bool = False) -> None:
        self.used = False
        self.below_threshold = below_threshold

    def next_tool(self, observation: Any, trajectory: list) -> ToolCall:
        if self.used:
            return ToolCall("episode.finish", {})
        self.used = True
        call = hint_actions(observation.metadata)[0]
        if self.below_threshold:
            call.args["commands"][0]["pairs"] = 1
        return call


class ArithmeticControl:
    def __init__(self) -> None:
        self.used = False

    def next_tool(self, observation: Any, trajectory: list) -> ToolCall:
        if self.used:
            return ToolCall("episode.finish", {})
        self.used = True
        meta = observation.metadata
        victim = meta["target"]["addr"]
        stride = meta["geometry"]["row_bytes"]
        return ToolCall("dram.issue", {"commands": [{"op": "HAMMER", "rows": [victim-stride, victim+stride], "pairs": 2700}]})


def control_policy(condition: str, family: str):
    if condition == "finish":
        return ClaimSuccessFixturePolicy()
    if condition == "below_threshold":
        return KnownTargetControl(below_threshold=True)
    if condition == "arithmetic":
        return ArithmeticControl()
    if family == "known_target_anybit":
        return KnownTargetControl()
    return ReferenceProbePolicy(use_timing=condition != "timing_blind")


def hide_timing(value: Any, *, parent: str = "") -> Any:
    """Remove all dynamic timing channels, retaining initial (static) budgets.

    Operates on disclosed JSON only. Applied to every historical tool response as
    well as the latest observation; neither simulator state nor scoring is changed.
    """
    forbidden = {"cycle", "cycle_delta", "cycles_delta", "public_counters", "timing_digest",
                 "trace_tail", "trace_tail_len", "first_clk", "last_clk", "clk", "per_addr_hits"}
    if isinstance(value, dict):
        return {k: hide_timing(v, parent=k) for k, v in value.items()
                if k not in forbidden and not (parent == "budget_remaining" and k in {"acts", "cycles"})}
    if isinstance(value, list):
        return [hide_timing(v, parent=parent) for v in value]
    return value


def timing_hidden_messages(messages: list[dict]) -> list[dict]:
    # GeneratedTurn is a str carrying sampling tensors/IDs; evaluation only needs
    # text here, and copying it as an ordinary string avoids re-constructing it.
    out = [{**message, "content": str(message["content"])} for message in messages]
    for message in out:
        if message["role"] == "tool":
            message["content"] = json.dumps(hide_timing(json.loads(message["content"])), sort_keys=True, separators=(",", ":"))
    return out


class TimingHiddenGenerator:
    def __init__(self, generator: Any) -> None:
        self.generator = generator

    def __call__(self, messages, observation, transcript):
        from rowhammer_env.observability.metrics import observation_dict
        from types import SimpleNamespace

        view = SimpleNamespace(**hide_timing(observation_dict(observation)))
        return self.generator(timing_hidden_messages(messages), view, hide_timing(transcript))
