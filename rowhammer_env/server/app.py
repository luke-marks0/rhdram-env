from __future__ import annotations

import os
from typing import Any

from rowhammer_env import Phase2Action, Phase2Observation, RowHammerTaskEnv
from rowhammer_env.openenv_source import load_openenv_http_server


_HTTPEnvServer, _create_app, _ConcurrencyConfig = load_openenv_http_server()


def make_env() -> RowHammerTaskEnv:
    """Factory used by OpenEnv; each session receives a fresh worker episode."""
    return RowHammerTaskEnv()


def create_rowhammer_app(
    *,
    max_concurrent_envs: int | None = None,
    session_timeout_s: float | None = None,
) -> Any:
    if max_concurrent_envs is None:
        max_concurrent_envs = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

    concurrency_config = _ConcurrencyConfig(
        max_concurrent_envs=max_concurrent_envs,
        session_timeout=session_timeout_s,
    )
    return _create_app(
        make_env,
        Phase2Action,
        Phase2Observation,
        env_name="rowhammer_env",
        concurrency_config=concurrency_config,
    )


app = create_rowhammer_app()


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
