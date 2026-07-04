#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]

# Every dependency the P17 contract tests actually import. Checking the full set
# here (not just fastapi/uvicorn/websockets) keeps the gate honest: the server
# cannot even start without fastmcp, and the tests self-skip when any of these is
# absent — so a partial install must fail the gate, not green-light with 0 tests.
REQUIRED = ("fastapi", "uvicorn", "websockets", "fastmcp", "requests")


def require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(
            f"missing P17 dependency: {name}; install with `python -m pip install -r requirements.txt`"
        )


def main() -> int:
    for name in REQUIRED:
        require_module(name)
    worker = ROOT / "build/phase2/ramulator_worker"
    config = ROOT / "build/phase2/p2_external_ddr4.yaml"
    if not worker.is_file() or not config.is_file():
        raise SystemExit("Phase 2 worker/config is not built; run scripts/build_phase2.py first")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_phase17")
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if not result.wasSuccessful():
        raise SystemExit("phase 17 contract tests failed")
    # A gate that passes while every test was skipped is worse than useless — the
    # deps and worker are required above, so any skip means the environment did not
    # actually exercise the HTTP transport. Fail closed.
    if result.testsRun == 0 or result.skipped:
        raise SystemExit(
            f"phase 17 tests did not run (ran={result.testsRun}, skipped={len(result.skipped)}); "
            "the HTTP serving path was not exercised"
        )
    print("phase 17 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
