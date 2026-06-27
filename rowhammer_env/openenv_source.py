from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV_SERVER = ROOT / "third_party/openenv/src/openenv/core/env_server"


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


def load_openenv_server_types() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    if not ENV_SERVER.is_dir():
        raise ModuleNotFoundError("third_party/openenv source is required for Phase 1")

    _package("openenv", ENV_SERVER.parents[1])
    _package("openenv.core", ENV_SERVER.parent)
    _package("openenv.core.env_server", ENV_SERVER)
    types_mod = _load("openenv.core.env_server.types", ENV_SERVER / "types.py")
    interfaces_mod = _load("openenv.core.env_server.interfaces", ENV_SERVER / "interfaces.py")
    return (
        interfaces_mod.Environment,
        types_mod.Action,
        types_mod.Observation,
        types_mod.State,
    )
