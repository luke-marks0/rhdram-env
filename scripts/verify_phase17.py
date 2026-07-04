#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing P17 dependency: {name}; install with `python -m pip install -r requirements.txt`")


def run(*args: str) -> None:
    subprocess.run([sys.executable, "-B", *args], cwd=ROOT, check=True)


def main() -> int:
    for name in ("fastapi", "uvicorn", "websockets"):
        require_module(name)
    worker = ROOT / "build/phase2/ramulator_worker"
    config = ROOT / "build/phase2/p2_external_ddr4.yaml"
    if not worker.is_file() or not config.is_file():
        raise SystemExit("Phase 2 worker/config is not built; run scripts/build_phase2.py first")

    run("-m", "unittest", "tests.test_phase17")
    print("phase 17 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
