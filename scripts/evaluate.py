#!/usr/bin/env python3
"""Evaluate every experimental condition on identical held-out seeds, trusted reward only.

    # controls + reference only (no GPU needed):
    python scripts/evaluate.py --config configs/training.yaml --out runs/eval --scripted-only
    # full comparison including base and trained policies + a timing-hidden probe:
    python scripts/evaluate.py --config configs/training.yaml --out runs/eval \
        --base --adapter runs/grpo1/checkpoints/final/adapter --split benchmark

Conditions (PoC scope "experimental controls"): reference solver, finish-only,
below-threshold, timing-blind, the untrained base model, the trained policy, and the
trained policy re-evaluated with the timing channel hidden. Evaluation uses the training
rollout implementation and the sparse trusted reward — no probe shaping — on the same
seeds for every condition. Scripted conditions need no model stack; ``--base``/
``--adapter`` pull in torch.
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.poc import PoCEnv, load_task  # noqa: E402
from rowhammer_env.training import config as configlib  # noqa: E402
from rowhammer_env.training.artifacts import RunWriter, condition_report  # noqa: E402
from rowhammer_env.training.reference import (  # noqa: E402
    BelowThresholdControl, FinishOnlyControl, ReferenceSolver, TimingBlindControl,
)
from rowhammer_env.training.rollout import run_episode  # noqa: E402

SCRIPTED = {
    "reference": ReferenceSolver,
    "finish_only": FinishOnlyControl,
    "below_threshold": BelowThresholdControl,
    "timing_blind": TimingBlindControl,
}


def timing_hidden(task: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``task`` with the timing channel removed (feedback: summarized_counts),
    so the timing_digest never reaches the policy — the timing-hidden ablation."""
    variant = copy.deepcopy(task)
    disclosure = dict(variant.get("disclosure") or {})
    disclosure["feedback"] = "summarized_counts"
    variant["disclosure"] = disclosure
    variant["id"] = f"{variant.get('id', 'task')}_timing_hidden"
    return variant


def eval_condition(env, tasks, stages, seeds, policy_factory, *, max_turns, task_map=None):
    """Run one condition across every (stage, seed) and return the episode results."""
    results = []
    for stage in stages:
        task = (task_map or tasks)[stage]
        for seed in seeds:
            results.append(run_episode(
                env, task=task, seed=seed, stage=stage, policy=policy_factory(),
                max_turns=max_turns, shaping_weight=0.0,
            ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "benchmark"])
    parser.add_argument("--seeds", default=None, help="override, e.g. 10000-10031")
    parser.add_argument("--adapter", default=None, help="trained LoRA adapter directory")
    parser.add_argument("--base", action="store_true", help="also evaluate the untrained base model")
    parser.add_argument("--scripted-only", action="store_true", help="skip all model conditions")
    args = parser.parse_args()

    config = configlib.load_config(args.config)
    writer = RunWriter(args.out)
    stages = list(configlib.STAGES)
    tasks = {s: load_task(p) for s, p in zip(stages, configlib.TASK_PATHS)}
    max_turns = int(config["rollout"]["max_turns"])
    if args.seeds:
        lo, hi = (int(x) for x in args.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = configlib.seeds(config, args.split)

    env = PoCEnv(task=tasks[stages[0]])
    results_by_condition: dict[str, list] = {}

    def record(name: str, results: list) -> None:
        results_by_condition[name] = results
        for r in results:
            writer.log_episode(r, condition=name)
            writer.log_trajectory(r, condition=name)
        rate = sum(1 for r in results if r.success >= 1.0) / max(1, len(results))
        print(f"  {name:24s} success {rate:.2%} over {len(results)} episodes")

    try:
        print(f"scripted conditions on {args.split} seeds {seeds[0]}..{seeds[-1]}")
        for name, cls in SCRIPTED.items():
            record(name, eval_condition(env, tasks, stages, seeds, cls, max_turns=max_turns))

        if not args.scripted_only and (args.base or args.adapter):
            from rowhammer_env.training.policy import QwenPolicy

            policy = QwenPolicy(config)
            if args.base:
                print("base model")
                record("base_model", eval_condition(env, tasks, stages, seeds, lambda: policy, max_turns=max_turns))
            if args.adapter:
                policy.load_adapter(args.adapter)
                print("trained model")
                record("trained_model", eval_condition(env, tasks, stages, seeds, lambda: policy, max_turns=max_turns))
                print("trained model, timing hidden")
                hidden = {s: timing_hidden(t) for s, t in tasks.items()}
                record("trained_timing_hidden",
                       eval_condition(env, tasks, stages, seeds, lambda: policy, max_turns=max_turns, task_map=hidden))
    finally:
        env.close()

    report = condition_report(results_by_condition)
    report["_meta"] = {"split": args.split, "seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds)}
    writer.write_json("metrics.json", report)
    print(f"\nwrote {writer.path('metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
