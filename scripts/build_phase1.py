#!/usr/bin/env python3
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
RAMULATOR = ROOT / "third_party/ramulator2"
OUT = ROOT / "build/phase1"


def run(args: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def main() -> int:
    if not RAMULATOR.is_dir():
        raise SystemExit("missing third_party/ramulator2; run scripts/fetch_phase1_sources.py")
    if not (RAMULATOR / "libramulator.so").is_file():
        cmake = shutil.which("cmake")
        if not cmake:
            raise SystemExit("cmake is required to build Ramulator 2.1")
        build_dir = ROOT / "build/ramulator2"
        generator = ["-G", "Ninja"] if shutil.which("ninja") else []
        run([cmake, "-S", str(RAMULATOR), "-B", str(build_dir), *generator, "-DRAMULATOR_PYTHON_BINDINGS=OFF"])
        run([cmake, "--build", str(build_dir), "--target", "ramulator"])

    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "p1_external_ddr4.yaml").is_file():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(RAMULATOR / "python")
        run([
            sys.executable,
            "-B",
            "-m",
            "ramulator",
            "export",
            "configs/ramulator/p1_external_ddr4.py",
            "-o",
            str(OUT / "p1_external_ddr4.yaml"),
        ], env=env)

    rpath = str(RAMULATOR)
    run([
        "g++",
        "-std=c++20",
        "-O2",
        "-I",
        str(RAMULATOR / "src"),
        "cpp/simulator_service/ramulator_external_smoke.cpp",
        "-L",
        str(RAMULATOR),
        f"-Wl,-rpath,{rpath}",
        "-lramulator",
        "-o",
        str(OUT / "ramulator_external_smoke"),
    ])
    print("phase1 build complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
