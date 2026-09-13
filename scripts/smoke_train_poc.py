#!/usr/bin/env python3
"""GPU gate: real GRPO update, LoRA reload, full-state resume, and sparse evaluation."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.llm.runtime import check_training_stack


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/training/poc.yaml")
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--adapter", type=pathlib.Path, help="recommended: initialize from the supervised warm-start adapter")
    ap.add_argument("--stage", default="bounded_easy")
    args = ap.parse_args()
    stack = check_training_stack()
    args.output.mkdir(parents=True, exist_ok=False)

    def run(*parts):
        subprocess.run([sys.executable, "-B", *map(str, parts)], cwd=ROOT, check=True)

    first, resumed = args.output / "first", args.output / "resumed"
    command = [ROOT / "scripts/train_grpo.py", "--config", args.config, "--stage", args.stage,
               "--smoke", "--output-dir", first]
    if args.adapter:
        command.extend(["--resume-adapter", args.adapter])
    run(*command)
    checkpoint = first / "checkpoint-1"
    for file in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "adapter_model.safetensors"):
        if not (checkpoint / file).is_file():
            raise RuntimeError(f"checkpoint missing resume state: {file}")
    run(ROOT / "scripts/train_grpo.py", "--config", args.config, "--stage", args.stage, "--smoke",
        "--output-dir", resumed, "--resume-from-checkpoint", checkpoint)
    for path in (first, resumed):
        evidence = json.loads((path / "parameter_update.json").read_text())
        if not evidence["parameters_changed"] or not evidence["finite_trainable_parameters"]:
            raise RuntimeError(f"no finite optimizer update in {path}")
        records = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines()]
        episode_ids = [r["episode_id"] for r in records]
        if len(episode_ids) != len(set(episode_ids)):
            raise RuntimeError("rollouts reused an episode session identity")
        if not any(r["external_tokens"] > 0 for r in records if r["condition"] == "train"):
            raise RuntimeError("smoke never exercised a multi-turn tool response; inspect warm-start policy")
    resume_evidence = json.loads((resumed / "parameter_update.json").read_text())
    if resume_evidence["start_step"] != 1 or resume_evidence["end_step"] != 2:
        raise RuntimeError("resume did not restore and advance trainer state")

    # Compare every saved LoRA tensor after fresh model/adapter construction. This
    # is separate from resuming the training object above.
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    cfg = yaml.safe_load(args.config.read_text())
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], revision=cfg["model"].get("revision", "main"),
                                               torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, resumed)
    disk = load_file(str(resumed / "adapter_model.safetensors"))
    loaded = get_peft_model_state_dict(model)
    if disk.keys() != loaded.keys() or not all(torch.equal(disk[k], loaded[k].cpu()) for k in disk):
        raise RuntimeError("reloaded adapter differs from saved tensors")
    del model, base
    run(ROOT / "scripts/eval_poc.py", "--config", args.config, "--output", args.output / "reloaded_eval",
        "--conditions", "trained", "--adapter", resumed, "--split", "validation", "--seed-limit", 2)
    summary = json.loads((args.output / "reloaded_eval/summary.json").read_text())
    if summary["score"] != "trusted_reward == 1.0":
        raise RuntimeError("reloaded evaluation did not use sparse trusted scoring")
    (args.output / "report.json").write_text(json.dumps({"passed": True, "stack": stack,
        "grpo_parameter_updates": True, "adapter_reload_exact": True, "optimizer_resume": resume_evidence,
        "fresh_sessions": True, "multi_turn_mask_exercised": True, "evaluation": summary}, indent=2) + "\n")
    print(f"GPU training smoke PASS: {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
