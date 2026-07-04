from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping


ROOT = pathlib.Path(__file__).resolve().parents[1]
RAMULATOR = ROOT / "third_party/ramulator2"


@dataclass(frozen=True)
class MitigationCapability:
    """Validated mitigation capability exposed through ``dram.info``.

    P16 admits mitigations one at a time. Only entries in ``ADMITTED`` are
    advertised; known SPEC names outside that set fail closed.
    """

    name: str
    implementation: str
    execution: str
    params: tuple[str, ...] = ()
    conformance: str = ""
    ramulator_impl: str | None = None

    def as_public(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "implementation": self.implementation,
            "execution": self.execution,
            "params": list(self.params),
        }
        if self.conformance:
            out["conformance"] = self.conformance
        return out


ALIASES = {
    "OracleRH": "oracle",
    "TWiCeIdeal": "twice",
    "RFMManager": "rfm",
    "BlockHammer": "blockhammer",
    "PRAC": "prac",
}

KNOWN_MITIGATIONS = (
    "none",
    "oracle",
    "para",
    "twice",
    "graphene",
    "blockhammer",
    "hydra",
    "rrs",
    "aqua",
    "rfm",
    "prac",
    "custom",
)

ADMISSION_ORDER = (
    "oracle",
    "para",
    "graphene",
    "twice",
    "blockhammer",
    "prac",
    "hydra",
    "rrs",
    "aqua",
    "rfm",
)

ADMITTED: dict[str, MitigationCapability] = {
    "none": MitigationCapability(
        name="none",
        implementation="baseline_refresh_only",
        execution="worker_config",
        conformance="M1",
    ),
    "oracle": MitigationCapability(
        name="oracle",
        implementation="OracleRH",
        execution="python_reference_port",
        params=("tRH",),
        conformance="P14 OracleRH differential",
        ramulator_impl="OracleRH",
    ),
}


def normalize_name(name: str | None) -> str:
    raw = "none" if name is None else str(name)
    return ALIASES.get(raw, raw.lower())


def normalize_mitigation(config: Mapping[str, Any] | None) -> dict[str, Any]:
    config = config or {"name": "none", "params": {}}
    return {"name": normalize_name(str(config.get("name", "none"))), "params": dict(config.get("params") or {})}


def require_admitted_mitigation(config: Mapping[str, Any] | str | None) -> MitigationCapability:
    if isinstance(config, str):
        name = normalize_name(config)
    else:
        name = normalize_mitigation(config).get("name", "none")
    capability = ADMITTED.get(name)
    if capability is None:
        raise ValueError(f"UNAVAILABLE_CAPABILITY:{name}")
    return capability


def public_mitigation_capabilities() -> list[dict[str, Any]]:
    """Capabilities safe to expose to policies.

    The list intentionally contains admitted mitigations only. Known but
    unvalidated names such as PARA/Graphene/BlockHammer stay absent from
    discovery and fail closed when requested explicitly.
    """

    return [ADMITTED[name].as_public() for name in ("none", *ADMISSION_ORDER) if name in ADMITTED]


def unavailable_mitigation_names() -> tuple[str, ...]:
    return tuple(name for name in KNOWN_MITIGATIONS if name not in ADMITTED)


def _ramulator_source_mentions(symbol: str) -> bool:
    if not RAMULATOR.is_dir():
        return False
    for path in RAMULATOR.glob("src/**/*.cpp"):
        try:
            if symbol in path.read_text(errors="ignore"):
                return True
        except OSError:
            continue
    return False


def ramulator_controller_plugins(config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Return Ramulator controller-plugin config for an admitted mitigation.

    The checked-out Ramulator tree may lag the plan and lack the real mitigation
    plugins. In that case the mitigation remains admitted only if it has an
    existing validated execution path, and no unavailable plugin is injected into
    the worker YAML.
    """

    mitigation = normalize_mitigation(config)
    capability = require_admitted_mitigation(mitigation)
    if capability.ramulator_impl is None or not _ramulator_source_mentions(capability.ramulator_impl):
        return []
    plugin: dict[str, Any] = {"impl": capability.ramulator_impl}
    if capability.name == "oracle" and "tRH" in mitigation["params"]:
        plugin["tRH"] = int(mitigation["params"]["tRH"])
    return [plugin]


def worker_config_for_mitigation(
    base_config_path: pathlib.Path,
    mitigation: Mapping[str, Any] | None,
    *,
    out_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    """Generate or select a worker YAML for the requested mitigation.

    Mitigations without an available Ramulator plugin use the base worker config
    and rely on their admitted Python-side conformance path. Unknown or
    unvalidated mitigations fail closed before a worker starts.
    """

    mitigation = normalize_mitigation(mitigation)
    plugins = ramulator_controller_plugins(mitigation)
    if not plugins:
        return base_config_path

    import yaml

    out_dir = out_dir or base_config_path.parent / "mitigations"
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(repr((base_config_path, mitigation)).encode()).hexdigest()[:12]
    out_path = out_dir / f"{base_config_path.stem}.{mitigation['name']}.{digest}.yaml"

    config = yaml.safe_load(base_config_path.read_text())
    for controller in config["memory_system"]["controllers"]:
        controller_plugins = controller.setdefault("controller_plugins", [])
        for plugin in plugins:
            if not any(existing.get("impl") == plugin["impl"] for existing in controller_plugins):
                controller_plugins.append(dict(plugin))
    out_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return out_path
