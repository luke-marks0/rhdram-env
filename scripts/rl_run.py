#!/usr/bin/env python3
"""P19 HTTP policy run: launch separately, then train through OpenEnv.

Example:
    python3 -m rowhammer_env.server.app
    python3 -B scripts/rl_run.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the P19 reward-updated policy example")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    return subprocess.call(
        [
            sys.executable,
            "-B",
            "scripts/train_phase19_policy.py",
            "--base-url",
            args.base_url,
            "--seed",
            str(args.seed),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
