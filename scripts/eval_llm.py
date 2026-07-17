#!/usr/bin/env python3
"""Evaluate any chat model against the RowHammer env with no training.

Drives real multi-turn episodes against a frozen model behind an OpenAI-compatible
``/v1/chat/completions`` endpoint, scoring the trusted sparse episode reward.

Serve a model however you like, e.g. vLLM:

    python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes

then point this at it:

    export RHD_LLM_CHAT_COMPLETIONS_URL=http://127.0.0.1:8000/v1/chat/completions
    export RHD_LLM_MODEL=Qwen/Qwen3-8B

    # a single task:
    python3 -B scripts/eval_llm.py --task configs/tasks/bounded_sweep_easy.yaml --seeds 1-16

    # every band of a curriculum config, on its held-out eval seeds:
    python3 -B scripts/eval_llm.py --config configs/training/grpo_curriculum.yaml

Ollama, TGI, OpenAI, etc. all work — anything speaking the chat-completions tool API.
The RowHammer env server is launched automatically (override with --base-url to reuse one).
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.llm.curriculum import load_curriculum  # noqa: E402
from rowhammer_env.llm.grpo_env import launch_server  # noqa: E402
from rowhammer_env.llm.policies import OpenAICompatibleToolPolicy  # noqa: E402
from rowhammer_env.llm.rollout import RolloutConfig, run_episode  # noqa: E402


def parse_seeds(spec: str) -> list[int]:
    """``"1-16"`` -> 1..16 inclusive; ``"1,4,9"`` -> those seeds; the two combine."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def jobs_from_config(config_path: pathlib.Path, seeds_override: list[int] | None) -> list[tuple[str, dict, list[int]]]:
    """``(label, task, seeds)`` per unique curriculum task, over the eval seeds.

    Seeds come from ``--seeds`` if given, else the config's ``eval.seed_range``/``eval.seeds``.
    """
    cfg = yaml.safe_load(config_path.read_text())
    if seeds_override is not None:
        seeds = seeds_override
    else:
        eval_cfg = cfg.get("eval") or {}
        rng = eval_cfg.get("seed_range")
        if rng:
            seeds = list(range(int(rng[0]), int(rng[1]) + 1))
        elif eval_cfg.get("seeds"):
            seeds = [int(s) for s in eval_cfg["seeds"]]
        else:
            raise SystemExit("--config has no 'eval:' seeds; pass --seeds")
    seen: set[str] = set()
    jobs: list[tuple[str, dict, list[int]]] = []
    for stage in load_curriculum(cfg):
        for path in stage.tasks:
            if path in seen:
                continue
            seen.add(path)
            task = yaml.safe_load((ROOT / path).read_text())
            jobs.append((pathlib.Path(path).name, task, seeds))
    return jobs


async def _run(base_url: str, task: dict | None, seeds: list[int], max_turns: int) -> list[float]:
    rewards: list[float] = []
    for seed in seeds:
        policy = OpenAICompatibleToolPolicy.from_env()
        cfg = RolloutConfig(base_url=base_url, seed=seed, task=task, episode_id=f"eval_{seed}", max_steps=max_turns)
        result = await run_episode(cfg, policy)
        rewards.append(float(result.reward))
        print(f"  seed={seed:<4} turns={len(result.trajectory):<4} reward={result.reward}")
    return rewards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--task", type=pathlib.Path, help="a single task config under configs/tasks/")
    src.add_argument("--config", type=pathlib.Path, help="a curriculum config: eval every stage's task")
    ap.add_argument("--seeds", default=None, help='e.g. "1-16" or "1,4,9" (default: 1-8 for --task, eval seeds for --config)')
    ap.add_argument("--max-turns", type=int, default=200, help="turn cap per episode (multi-turn)")
    ap.add_argument("--base-url", default=None, help="reuse a running env server instead of launching one")
    args = ap.parse_args()

    seeds_override = parse_seeds(args.seeds) if args.seeds else None
    if args.config:
        jobs = jobs_from_config(args.config, seeds_override)
    else:
        task = yaml.safe_load((ROOT / args.task).read_text())
        jobs = [(args.task.name, task, seeds_override or parse_seeds("1-8"))]

    server = None
    base_url = args.base_url
    if base_url is None:
        server = launch_server(root=str(ROOT), host="127.0.0.1", max_concurrent_envs=1, mode="production")
        base_url = server.base_url
        print(f"launched env server at {base_url}")
    try:
        all_rewards: list[float] = []
        for label, task, seeds in jobs:
            print(f"\nevaluating {label} over {len(seeds)} seeds (no training)...")
            rewards = asyncio.run(_run(base_url, task, seeds, args.max_turns))
            solved = sum(1 for r in rewards if r >= 1.0)
            print(f"  {label}: solved {solved}/{len(rewards)} = {solved / len(rewards):.1%}")
            all_rewards.extend(rewards)
        solved = sum(1 for r in all_rewards if r >= 1.0)
        print(f"\ntotal: solved {solved}/{len(all_rewards)} = {solved / len(all_rewards):.1%}  (mean {sum(all_rewards) / len(all_rewards):.3f})")
        return 0
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
