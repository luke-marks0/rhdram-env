from __future__ import annotations

from typing import Any

from .openenv_source import load_openenv_client_types
from .phase2_env import Phase2Action, Phase2Observation, Phase2State


EnvClient, StepResult = load_openenv_client_types()


class RowHammerClient(EnvClient[Phase2Action, Phase2Observation, Phase2State]):
    """Typed websocket client for the RowHammer OpenEnv server."""

    def _step_payload(self, action: Phase2Action | dict[str, Any]) -> dict[str, Any]:
        if isinstance(action, Phase2Action):
            return action.model_dump()
        return dict(action)

    def _parse_result(self, payload: dict[str, Any]) -> Any:
        obs_data = dict(payload.get("observation", {}))
        wire_metadata = obs_data.get("wire_metadata")
        if wire_metadata and not obs_data.get("metadata"):
            obs_data["metadata"] = wire_metadata
        obs_data["reward"] = payload.get("reward")
        obs_data["done"] = payload.get("done", False)
        observation = Phase2Observation.model_validate(obs_data)
        # The server folds ``metadata`` into ``info`` on the wire (OpenEnv strips
        # ``metadata``); restore it so wire observations read like in-process ones.
        if observation.info and not observation.metadata:
            observation.metadata = dict(observation.info)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> Phase2State:
        return Phase2State.model_validate(payload)

    async def tool(self, name: str, **args: Any) -> Any:
        return await self.step(Phase2Action(tool=name, args=args))

    async def info(self) -> Any:
        return await self.step(Phase2Action(tool="dram.info", args={}))

    async def read(self, addr: dict[str, Any], length: int = 1) -> Any:
        return await self.step(Phase2Action(tool="dram.read", args={"addr": addr, "length": length}))

    async def write(self, addr: dict[str, Any], data_b64: str) -> Any:
        return await self.step(Phase2Action(tool="dram.write", args={"addr": addr, "data_b64": data_b64}))

    async def issue(self, commands: list[dict[str, Any]]) -> Any:
        return await self.step(Phase2Action(tool="dram.issue", args={"commands": commands}))

    async def finish(self) -> Any:
        return await self.step(Phase2Action(tool="episode.finish", args={}))
