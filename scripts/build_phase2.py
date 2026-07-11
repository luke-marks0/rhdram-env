#!/usr/bin/env python3
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import build_phase1
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
RAMULATOR = ROOT / "third_party/ramulator2"
OUT = ROOT / "build/phase2"
EXTENSIONS = ROOT / "cpp/ramulator_extensions"

# The worker instantiates the issued-event capture plugin (P11). The plugin is
# only linked into the worker binary, so it cannot be referenced from the Python
# `ramulator export` API (which knows only plugins compiled into libramulator.so).
# Instead we inject it directly into the worker's YAML, which the worker's own
# factory — with the plugin registered — resolves at load time.
PLUGIN_NAME = "IssuedEventRecorder"

# Second-standard worker configs (P15). Built so a real DDR5/HBM2 geometry — from
# the true DRAMSpec, not DDR4 constants — is reachable through the same worker for
# the standard-generalization checks. No empirical profile of these standards is
# admitted, so they are used only for geometry/adapter checks, not for reward.
SECOND_STANDARD_CONFIGS = {
    "p2_external_ddr5.yaml": ROOT / "configs/ramulator/p2_external_ddr5.py",
    "p2_external_hbm2.yaml": ROOT / "configs/ramulator/p2_external_hbm2.py",
}


def _inject_plugin(config: dict) -> dict:
    for controller in config["memory_system"]["controllers"]:
        plugins = controller.setdefault("controller_plugins", [])
        if not any(p.get("impl") == PLUGIN_NAME for p in plugins):
            plugins.append({"impl": PLUGIN_NAME})
    return config


def write_worker_config(phase1_yaml: pathlib.Path, out_yaml: pathlib.Path) -> None:
    config = _inject_plugin(yaml.safe_load(phase1_yaml.read_text()))
    out_yaml.write_text(yaml.safe_dump(config, sort_keys=False))


def export_worker_config(py_config: pathlib.Path, out_yaml: pathlib.Path) -> None:
    """Export a Ramulator Python config to YAML and inject the capture plugin.

    Used for the second-standard configs, whose YAML is produced directly from the
    committed ``configs/ramulator/*.py`` recipe (the DDR4 path reuses the phase-1
    export). Idempotent: safe to call when the YAML already exists.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RAMULATOR / "python")
    tmp_yaml = out_yaml.with_suffix(".raw.yaml")
    subprocess.run(
        [sys.executable, "-B", "-m", "ramulator", "export", str(py_config), "-o", str(tmp_yaml)],
        cwd=ROOT,
        env=env,
        check=True,
    )
    config = _inject_plugin(yaml.safe_load(tmp_yaml.read_text()))
    out_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    tmp_yaml.unlink(missing_ok=True)


def ensure_second_standard_configs() -> None:
    """Build the DDR5/HBM2 worker YAMLs if absent (used by verify_phase15)."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, py_config in SECOND_STANDARD_CONFIGS.items():
        out_yaml = OUT / name
        if not out_yaml.is_file():
            export_worker_config(py_config, out_yaml)


def main() -> int:
    build_phase1.main()
    OUT.mkdir(parents=True, exist_ok=True)

    write_worker_config(ROOT / "build/phase1/p1_external_ddr4.yaml", OUT / "p2_external_ddr4.yaml")
    for name, py_config in SECOND_STANDARD_CONFIGS.items():
        export_worker_config(py_config, OUT / name)

    subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-O2",
            "-I",
            str(RAMULATOR / "src"),
            "-I",
            str(EXTENSIONS),
            "cpp/simulator_service/ramulator_worker.cpp",
            str(EXTENSIONS / "issued_event_recorder.cpp"),
            str(EXTENSIONS / "row_xor_addr_mapper.cpp"),
            "-L",
            str(RAMULATOR),
            f"-Wl,-rpath,{RAMULATOR}",
            "-lramulator",
            "-o",
            str(OUT / "ramulator_worker"),
        ],
        cwd=ROOT,
        check=True,
    )
    print("phase2 build complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
