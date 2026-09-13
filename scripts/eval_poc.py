#!/usr/bin/env python3
"""Evaluate scoped controls or an HF/LoRA model, preserving every episode locally."""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import uuid

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.poc import PoCEnv, TASK_PATHS, load_task, resolved_config, seed_list, validate_config
from rowhammer_env.llm.multiturn_rollout import ToolPolicyGenerator, run_training_episode, run_training_episode_local
from rowhammer_env.llm.poc_policy import TimingHiddenGenerator, control_policy, timing_hidden_messages
from rowhammer_env.llm.rollout import RolloutConfig
from rowhammer_env.observability.experiment import RunArtifacts, episode_record, file_hash

CONTROL_CONDITIONS = ("reference", "finish", "below_threshold", "timing_blind", "arithmetic")
MODEL_CONDITIONS = ("untrained", "sft", "trained", "trained_timing_hidden")


def model_generator(cfg, adapter=None, model_name=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from train_grpo import HFCompletionGenerator

    set_seed(int(cfg["grpo"]["seed"]))
    name = model_name or cfg["model"]["name"]
    revision = cfg["model"].get("revision", "main")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda")
    model.eval()
    generator = HFCompletionGenerator(model, tokenizer, max_new_tokens=cfg["rollout"]["max_turn_tokens"],
        max_turn_tokens=cfg["rollout"]["max_turn_tokens"], enable_thinking=False,
        max_prompt_tokens=cfg["rollout"]["max_prompt_tokens"], temperature=cfg["grpo"]["temperature"],
        top_p=cfg["grpo"]["top_p"], top_k=cfg["grpo"]["top_k"])
    return generator, {"model": name, "revision": getattr(model.config, "_commit_hash", revision)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/training/poc.yaml")
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--conditions", default=",".join(CONTROL_CONDITIONS))
    ap.add_argument("--split", choices=("benchmark", "validation", "train"), default="benchmark")
    ap.add_argument("--seed-limit", type=int, help="diagnostic subset; recorded, not a complete benchmark")
    ap.add_argument("--adapter", type=pathlib.Path)
    ap.add_argument("--model")
    ap.add_argument("--base-url", help="use fresh stateful WS sessions instead of local native workers")
    ap.add_argument("--export-sft", action="store_true", help="save reference messages (training seeds only)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    conditions = args.conditions.split(",")
    if any(c not in CONTROL_CONDITIONS + MODEL_CONDITIONS for c in conditions):
        ap.error("unknown condition")
    if args.seed_limit is not None and args.seed_limit < 1:
        ap.error("--seed-limit must be positive")
    if args.export_sft and (args.split != "train" or conditions != ["reference"]):
        ap.error("SFT export requires --split train --conditions reference")
    if "untrained" in conditions and args.adapter:
        ap.error("untrained condition must not load an adapter")
    if any(c in {"sft", "trained", "trained_timing_hidden"} for c in conditions) and not args.adapter:
        ap.error("trained/SFT conditions require --adapter")
    if args.base_url:
        from rowhammer_env.llm.grpo_env import disclose_metadata

        metadata = asyncio.run(disclose_metadata(args.base_url, 1, load_task(TASK_PATHS[0])))
        if metadata.get("policy_surface") != "poc":
            ap.error("evaluation requires a PoC server (RH_POC=1)")
    generator, model_identity = None, {}
    if any(c in MODEL_CONDITIONS for c in conditions):
        generator, model_identity = model_generator(cfg, args.adapter, args.model)
    identity = {"conditions": conditions, "split": args.split, "seed_limit": args.seed_limit,
                "transport": "websocket" if args.base_url else "local", "evaluation_shaping_weight": 0.0,
                "model": model_identity, "adapter": str(args.adapter) if args.adapter else None}
    if args.adapter:
        identity["adapter_sha256"] = {p.name: file_hash(p) for p in sorted(args.adapter.glob("*")) if p.is_file()}
    bundle = RunArtifacts(args.output, config=resolved_config(cfg), identity=identity)
    try:
        for path, stage in zip(TASK_PATHS, cfg["curriculum"], strict=True):
            task = load_task(path)
            seeds = seed_list(stage if args.split == "train" else cfg["eval" if args.split == "validation" else "benchmark"])
            if args.seed_limit:
                seeds = seeds[:args.seed_limit]
            for condition in conditions:
                if condition == "below_threshold" and task["family"] != "known_target_anybit":
                    continue
                if condition == "timing_blind" and task["family"] == "known_target_anybit":
                    continue
                if condition == "arithmetic" and task["family"] != "hidden_adjacency":
                    continue
                rewards = []
                for seed in seeds:
                    if condition in MODEL_CONDITIONS:
                        from transformers import set_seed

                        set_seed(int(cfg["grpo"]["seed"]) ^ seed)
                    gen = generator if condition in MODEL_CONDITIONS else ToolPolicyGenerator(control_policy(condition, task["family"]))
                    if condition == "trained_timing_hidden":
                        gen = TimingHiddenGenerator(gen)
                    episode_id = f"poc_{uuid.uuid4().hex}"
                    if args.base_url:
                        rollout = asyncio.run(run_training_episode(
                            RolloutConfig(args.base_url, seed, task, episode_id), gen, max_turns=cfg["rollout"]["max_turns"]))
                    else:
                        env = PoCEnv(task=task)
                        try:
                            rollout = run_training_episode_local(env, gen, seed=seed, task=task,
                                episode_id=episode_id, max_turns=cfg["rollout"]["max_turns"])
                        finally:
                            env.close()
                    if rollout.result.initial_observation.get("metadata", {}).get("policy_surface") != "poc":
                        raise RuntimeError("evaluation requires a PoC server (RH_POC=1)")
                    messages = timing_hidden_messages(rollout.messages) if condition == "trained_timing_hidden" else rollout.messages
                    record = episode_record(rollout.result, condition=condition, messages=messages)
                    if condition in MODEL_CONDITIONS:
                        record["policy_sampling_seed"] = int(cfg["grpo"]["seed"]) ^ seed
                    bundle.append(record)
                    if args.export_sft:
                        if rollout.reward != 1.0:
                            raise RuntimeError(f"reference failed on {path} seed {seed}; SFT export aborted")
                        with (bundle.path / "sft.jsonl").open("a") as stream:
                            stream.write(json.dumps({"messages": messages, "seed": seed, "task_id": task["id"]}) + "\n")
                    rewards.append(rollout.reward)
                    print(f"{condition} {pathlib.Path(path).stem} seed={seed} reward={rollout.reward} turns={len(rollout.trajectory)}", flush=True)
                print(f"  {condition}: {sum(r == 1.0 for r in rewards)}/{len(rewards)}", flush=True)
        bundle.finish()
    except BaseException as exc:
        bundle.finish(status="failed", error=str(exc))
        raise
    print(f"Saved {bundle.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
