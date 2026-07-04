from __future__ import annotations

from typing import Any

from rowhammer_env.client import RowHammerClient
from rowhammer_env.phase2_env import Phase2Action


class RowHammerSDK:
    """Small policy-facing wrapper matching the restricted script broker surface."""

    def __init__(self, client: RowHammerClient) -> None:
        self.client = client

    async def info(self) -> Any:
        return await self.client.step(Phase2Action(tool="dram.info", args={}))

    async def read(self, *, addr: dict[str, Any], length: int = 1) -> Any:
        return await self.client.step(Phase2Action(tool="dram.read", args={"addr": addr, "length": length}))

    async def write(self, *, addr: dict[str, Any], data_b64: str) -> Any:
        return await self.client.step(Phase2Action(tool="dram.write", args={"addr": addr, "data_b64": data_b64}))

    async def issue(self, *, commands: list[dict[str, Any]]) -> Any:
        return await self.client.step(Phase2Action(tool="dram.issue", args={"commands": commands}))

    async def finish(self) -> Any:
        return await self.client.step(Phase2Action(tool="episode.finish", args={}))


def connect(base_url: str = "http://localhost:8000", **kwargs: Any) -> RowHammerClient:
    return RowHammerClient(base_url=base_url, **kwargs)


rh = RowHammerSDK
