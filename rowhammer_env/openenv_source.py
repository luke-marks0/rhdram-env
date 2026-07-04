from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
OPENENV_SRC = ROOT / "third_party/openenv/src"
ENV_SERVER = OPENENV_SRC / "openenv/core/env_server"


def _package(name: str, path: pathlib.Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load(name: str, path: pathlib.Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_openenv_source() -> pathlib.Path:
    if not ENV_SERVER.is_dir():
        raise ModuleNotFoundError("third_party/openenv source is required")

    src = str(OPENENV_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)

    # Keep the synthetic package path used by the early phases. Importing the
    # package __init__ pulls in the HTTP server and therefore FastAPI; loading
    # just the package shells lets Phase1/Phase2 types work without P17 deps.
    _package("openenv", ENV_SERVER.parents[1])
    _package("openenv.core", ENV_SERVER.parent)
    _package("openenv.core.env_server", ENV_SERVER)
    return OPENENV_SRC


def load_openenv_server_types() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    ensure_openenv_source()
    types_mod = _load("openenv.core.env_server.types", ENV_SERVER / "types.py")
    interfaces_mod = _load("openenv.core.env_server.interfaces", ENV_SERVER / "interfaces.py")
    return (
        interfaces_mod.Environment,
        types_mod.Action,
        types_mod.Observation,
        types_mod.State,
    )


def load_openenv_http_server() -> tuple[Any, Any, Any]:
    load_openenv_server_types()
    try:
        http_server = _load("openenv.core.env_server.http_server", ENV_SERVER / "http_server.py")
        types_mod = _load("openenv.core.env_server.types", ENV_SERVER / "types.py")
    except ModuleNotFoundError as exc:
        if exc.name in {"fastapi", "uvicorn", "fastmcp"}:
            raise ModuleNotFoundError(
                "OpenEnv HTTP serving requires fastapi, uvicorn, and fastmcp; install the P17 runtime dependencies"
            ) from exc
        raise
    return http_server.HTTPEnvServer, types_mod.ConcurrencyConfig, types_mod.ServerMode


def load_openenv_client_types() -> tuple[Any, Any]:
    load_openenv_server_types()
    try:
        env_client = _load("openenv.core.env_client", OPENENV_SRC / "openenv/core/env_client.py")
        client_types = _load("openenv.core.client_types", OPENENV_SRC / "openenv/core/client_types.py")
    except ModuleNotFoundError as exc:
        if exc.name in {"websockets", "requests"}:
            raise ModuleNotFoundError(
                "OpenEnv websocket clients require websockets and requests; install the P17 runtime dependencies"
            ) from exc
        raise
    return env_client.EnvClient, client_types.StepResult
