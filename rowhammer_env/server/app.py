from __future__ import annotations

import json
import os
import pathlib
from typing import Any

from fastapi import FastAPI

from rowhammer_env import Phase2Action, Phase2Observation, RowHammerTaskEnv
from rowhammer_env.openenv_source import load_openenv_http_server


_HTTPEnvServer, _ConcurrencyConfig, _ServerMode = load_openenv_http_server()


def _default_task() -> dict[str, Any] | None:
    """Optional server-pinned default task from env JSON or a JSON file."""
    raw = os.getenv("RH_TASK") or os.getenv("ROW_HAMMER_TASK_JSON")
    if raw:
        try:
            task = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"task environment JSON is not valid: {exc}") from exc
        if not isinstance(task, dict):
            raise SystemExit("task environment JSON must be an object")
        return task

    task_file = os.getenv("ROW_HAMMER_TASK_FILE")
    if not task_file:
        return None
    try:
        task = json.loads(pathlib.Path(task_file).read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ROW_HAMMER_TASK_FILE is not valid JSON: {exc}") from exc
    if not isinstance(task, dict):
        raise SystemExit("ROW_HAMMER_TASK_FILE must contain a JSON object")
    return task


def make_env(task: dict[str, Any] | None = None) -> RowHammerTaskEnv:
    """Factory used by OpenEnv; each session receives a fresh worker episode.

    The default task can be pinned with the ``RH_TASK`` env var (a JSON task
    config); an orchestrator can also override it per-episode by passing ``task``
    to ``reset`` over the WebSocket transport.
    """
    return RowHammerTaskEnv(task=task if task is not None else _default_task())


def create_rowhammer_app(
    *,
    task: dict[str, Any] | None = None,
    max_concurrent_envs: int | None = None,
    session_timeout_s: float | None = None,
    mode: str | None = None,
) -> FastAPI:
    if max_concurrent_envs is None:
        max_concurrent_envs = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))
    if mode is None:
        mode = os.getenv("RH_SERVER_MODE", "simulation")
    # Validate up front so a bad RH_SERVER_MODE fails at startup with a clear
    # error; ServerMode is a str-enum accepted directly by register_routes.
    server_mode = _ServerMode(str(mode).lower())

    concurrency_config = _ConcurrencyConfig(
        max_concurrent_envs=max_concurrent_envs,
        session_timeout=session_timeout_s,
    )

    def factory() -> RowHammerTaskEnv:
        return make_env(task=task)

    server = _HTTPEnvServer(
        factory,
        Phase2Action,
        Phase2Observation,
        concurrency_config=concurrency_config,
    )
    app = FastAPI(title="RowHammer OpenEnv HTTP API", version="1.0.0")
    # Episodes are stateful ONLY over the WebSocket /ws transport (RowHammerClient
    # uses it). The HTTP POST /reset and /step endpoints are stateless — each
    # request builds and closes a fresh env, so they cannot carry an episode.
    # Set RH_SERVER_MODE=production to drop the stateless /reset, /step, /state
    # routes and expose only /ws, /health, /schema, /metadata, /mcp.
    server.register_routes(app, mode=server_mode)
    return app


app = create_rowhammer_app()


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
