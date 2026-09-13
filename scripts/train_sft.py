#!/usr/bin/env python3
"""Action-only LoRA cold start from native reference trajectories on training seeds."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.llm.multiturn_rollout import build_masked_completion
from rowhammer_env.llm.runtime import check_training_stack
from rowhammer_env.poc import seed_list, validate_config
from rowhammer_env.observability.experiment import file_hash, provenance, snapshot_sources


def encode_demonstration(row, tokenizer, *, context_limit):
    messages = row["messages"]
    prompt, completion, mask = build_masked_completion(messages, tokenizer, n_prompt_messages=2, enable_thinking=False)
    ids = prompt + completion
    if len(ids) > context_limit:
        raise ValueError(f"demonstration exceeds context ({len(ids)} > {context_limit}); do not truncate away its successful action")
    labels = [-100] * len(prompt) + [token if flag else -100 for token, flag in zip(completion, mask, strict=True)]
    if all(label == -100 for label in labels):
        raise ValueError("demonstration contains no trainable assistant tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/training/poc.yaml")
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--resume-from-checkpoint")
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    stack = check_training_stack()
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    from rowhammer_env.llm.runtime import evidence_callback
    import torch

    set_seed(cfg["grpo"]["seed"])
    if args.output.exists() and (args.output / "sft_run.json").exists() and not args.resume_from_checkpoint:
        raise FileExistsError("SFT run already exists; use a new output or resume a checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"].get("revision", "main"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    allowed = set().union(*(set(seed_list(stage)) for stage in cfg["curriculum"]))
    if not rows or any(row.get("seed") not in allowed for row in rows):
        raise ValueError("SFT input must be nonempty and contain only training seeds")
    encoded = [encode_demonstration(row, tokenizer, context_limit=cfg["rollout"]["max_prompt_tokens"]) for row in rows]
    dataset = Dataset.from_list(encoded)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], revision=cfg["model"].get("revision", "main"),
                                                torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(**{k: v for k, v in cfg["peft"].items() if k != "enabled"},
                                           task_type="CAUSAL_LM", bias="none"))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"configuration": cfg, "stack": stack, "provenance": provenance(),
                "data_sha256": file_hash(args.data), "episodes": len(rows),
                "model_revision": getattr(model.config, "_commit_hash", None),
                "max_steps": args.max_steps, "resume_from_checkpoint": args.resume_from_checkpoint}
    manifest_path = args.output / ("sft_resume.json" if args.resume_from_checkpoint else "sft_run.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.resume_from_checkpoint:
        snapshot_sources(args.output, manifest["provenance"])
    trainer = Trainer(model=model, processing_class=tokenizer, train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(output_dir=str(args.output), learning_rate=1e-4, num_train_epochs=1,
            max_steps=args.max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=16,
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_ratio=0.05, lr_scheduler_type="cosine", max_grad_norm=1.0,
            logging_steps=1, save_steps=20, save_total_limit=2, report_to="none", seed=cfg["grpo"]["seed"]),
        callbacks=[evidence_callback(args.output, require_update=True)])
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output))
    trainer.save_state()
    tokenizer.save_pretrained(args.output)
    print(f"Saved SFT adapter: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
