#!/usr/bin/env python3
from __future__ import annotations

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


def write_worker_config(phase1_yaml: pathlib.Path, out_yaml: pathlib.Path) -> None:
    config = yaml.safe_load(phase1_yaml.read_text())
    for controller in config["memory_system"]["controllers"]:
        plugins = controller.setdefault("controller_plugins", [])
        if not any(p.get("impl") == PLUGIN_NAME for p in plugins):
            plugins.append({"impl": PLUGIN_NAME})
    out_yaml.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> int:
    build_phase1.main()
    OUT.mkdir(parents=True, exist_ok=True)

    write_worker_config(ROOT / "build/phase1/p1_external_ddr4.yaml", OUT / "p2_external_ddr4.yaml")

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
