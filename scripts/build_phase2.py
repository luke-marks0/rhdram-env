#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys

import build_phase1


ROOT = pathlib.Path(__file__).resolve().parents[1]
RAMULATOR = ROOT / "third_party/ramulator2"
OUT = ROOT / "build/phase2"


def main() -> int:
    build_phase1.main()
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-O2",
            "-I",
            str(RAMULATOR / "src"),
            "cpp/simulator_service/ramulator_worker.cpp",
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

