#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.llm.training import train_hammer_pairs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="P19 reward-updated HTTP policy example")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()
    result = asyncio.run(
        train_hammer_pairs(
            base_url=args.base_url,
            task={"family": "known_target_anybit"},
            seed=args.seed,
            episodes=args.episodes,
            initial_pairs=1,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
