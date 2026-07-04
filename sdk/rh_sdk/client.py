from __future__ import annotations

from typing import Any

from rowhammer_env.client import RowHammerClient


class RowHammerSDK:
    """Policy-facing wrapper matching the restricted script broker surface.

    Delegates to :class:`RowHammerClient` so scripts, external policies, and the
    in-sandbox ``rh_sdk`` broker all share one tool surface.
    """

    def __init__(self, client: RowHammerClient) -> None:
        self.client = client

    async def info(self) -> Any:
        return await self.client.info()

    async def read(self, *, addr: dict[str, Any], length: int = 1) -> Any:
        return await self.client.read(addr, length)

    async def write(self, *, addr: dict[str, Any], data_b64: str) -> Any:
        return await self.client.write(addr, data_b64)

    async def issue(self, *, commands: list[dict[str, Any]]) -> Any:
        return await self.client.issue(commands)

    async def finish(self) -> Any:
        return await self.client.finish()


def connect(base_url: str = "http://localhost:8000", **kwargs: Any) -> RowHammerClient:
    return RowHammerClient(base_url=base_url, **kwargs)


rh = RowHammerSDK
