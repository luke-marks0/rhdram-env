#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_release.py"], cwd=ROOT, check=True)
    print("phase20 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
