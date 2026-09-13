#!/usr/bin/env python3
"""CPU integration checks for actual TRL/PEFT/masking; GPU smoke is a separate gate.

Uses a real, tiny randomly initialized Qwen3 and the real native simulator. This
tests library mechanics, not pretrained-model performance. No fabricated rewards.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from rowhammer_env.llm.runtime import check_training_stack, evidence_callback


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=pathlib.Path)
    ap.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/training/poc.yaml")
    ap.add_argument("--tokenizer", type=pathlib.Path, help="optional already-downloaded copy of the pinned model tokenizer")
    args = ap.parse_args()
    stack = check_training_stack(require_cuda=False)
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from rowhammer_env.llm.grpo_env import build_messages, launch_server
    from rowhammer_env.llm.multiturn_rollout import ToolPolicyGenerator, run_training_episode_local
    from rowhammer_env.llm.poc_policy import control_policy
    from rowhammer_env.poc import ROOT as repo_root, PoCEnv, TASK_PATHS, load_task
    from train_grpo import build_rows, make_multiturn_rollout_func, render_prompts
    from train_sft import encode_demonstration

    torch.set_num_threads(2)
    cfg = yaml.safe_load(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or cfg["model"]["name"], revision=cfg["model"].get("revision", "main"))
    tokenizer.pad_token = tokenizer.eos_token
    # Exercise every canonical GRPO argument against the actual pinned API.
    GRPOConfig(**{**cfg["grpo"], "output_dir": str(args.output / "configuration_check"),
                  "use_cpu": True, "bf16": False})
    demonstrations = []
    lengths = []
    for path in TASK_PATHS:
        task = load_task(path)
        env = PoCEnv(task=task)
        try:
            rollout = run_training_episode_local(env, ToolPolicyGenerator(control_policy("reference", task["family"])),
                                                seed=1, task=task, max_turns=40)
        finally:
            env.close()
        assert rollout.reward == 1.0, path
        encoded = encode_demonstration({"messages": rollout.messages}, tokenizer,
                                       context_limit=cfg["rollout"]["max_prompt_tokens"])
        # Verify each complete tool-result span is masked using the real tokenizer.
        for i, message in enumerate(rollout.messages):
            if message["role"] != "tool":
                continue
            before = tokenizer.apply_chat_template(rollout.messages[:i], tokenize=True, add_generation_prompt=False)
            after = tokenizer.apply_chat_template(rollout.messages[:i+1], tokenize=True, add_generation_prompt=False)
            assert set(encoded["labels"][len(before):len(after)]) == {-100}, "tool tokens entered SFT labels"
        lengths.append({"task": path, "turns": len(rollout.trajectory), "tokens": len(encoded["input_ids"]),
                        "assistant_tokens": sum(t != -100 for t in encoded["labels"])})
        demonstrations.append(encoded)

    small_config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=32768, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

    def base_model():
        set_seed(42)
        return Qwen3ForCausalLM(small_config)

    def adapted_model():
        return get_peft_model(base_model(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                     task_type="CAUSAL_LM", lora_dropout=0.0))

    # Run the actual custom GRPO hook, with model-sampled completions and fresh WS
    # sessions. Four random tokens cannot implement this task: the real reward
    # should be zero and correctly produce no parameter update with beta=0.
    server = launch_server(root=str(repo_root), env_overrides={"RH_POC": "1"}, max_concurrent_envs=4)
    try:
        rows = build_rows(server.base_url, [({"config": TASK_PATHS[0]}, 1), ({"config": TASK_PATHS[0]}, 2)])
        render_prompts(rows, tokenizer, False)
        model = adapted_model()
        lookup = {r["prompt"]: (r["seed"], json.loads(r["task_json"])) for r in rows}
        hook, reward, probe_reward = make_multiturn_rollout_func(server.base_url, tokenizer, model, lookup,
            max_turns=2, max_new_tokens=4, max_turn_tokens=4, max_prompt_tokens=4096,
            enable_thinking=False, temperature=1.0, top_p=1.0, top_k=0,
            concurrency=2, artifact_dir=args.output / "grpo", zero_variance_patience=0,
            emit_probe_shaping=True, shaping_kind="unique_candidate_probe")
        grpo_path = args.output / "grpo"
        grpo_path.mkdir()
        trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[reward, probe_reward], rollout_func=hook,
            train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(rows),
            callbacks=[evidence_callback(grpo_path)],
            args=GRPOConfig(output_dir=str(grpo_path), use_cpu=True, bf16=False, num_generations=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=2, generation_batch_size=2,
                max_steps=1, max_completion_length=4096, temperature=1.0, beta=0.0,
                logging_steps=1, report_to="none", save_strategy="no", weight_decay=0.0,
                loss_type="dapo", scale_rewards="batch", reward_weights=[1.0, 0.1],
                num_generations_eval=1, per_device_eval_batch_size=2, eval_steps=1, eval_strategy="steps"))
        trainer.train()
        grpo_evidence = json.loads((grpo_path / "parameter_update.json").read_text())
        assert not grpo_evidence["parameters_changed"], "equal zero rewards unexpectedly updated policy"
        validation = [json.loads(line) for line in (grpo_path / "rollouts.jsonl").read_text().splitlines()
                      if json.loads(line)["condition"] == "validation"]
        assert len(validation) == 2 and all(r["probe_shaping"] == 0.0 for r in validation)
    finally:
        server.stop()

    # Real supervised gradients on real successful tool trajectories, then exact
    # adapter reload and optimizer/scheduler/RNG resume from step 1 to step 2.
    sft_path = args.output / "sft"
    sft_path.mkdir()
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    # Full-window tokenization/masking was checked above for every task. Optimizer
    # mechanics need only a short real trajectory: the full 151k-token vocabulary
    # makes a long FP32 CPU backward need several GB even for this tiny network.
    data = Dataset.from_list(demonstrations[:1])

    def sft_trainer(path, model):
        return Trainer(model=model, processing_class=tokenizer, train_dataset=data, data_collator=collator,
            callbacks=[evidence_callback(path, require_update=True)],
            args=TrainingArguments(output_dir=str(path), use_cpu=True, bf16=False, max_steps=2,
                learning_rate=1e-3, per_device_train_batch_size=1, gradient_accumulation_steps=1,
                save_steps=1, logging_steps=1, report_to="none", seed=42))

    trained = adapted_model()
    sft_trainer(sft_path, trained).train()
    trained.save_pretrained(sft_path / "final")
    reloaded = PeftModel.from_pretrained(base_model(), sft_path / "final")
    expected = get_peft_model_state_dict(trained)
    restored = get_peft_model_state_dict(reloaded)
    assert expected.keys() == restored.keys()
    assert all(torch.equal(expected[k].cpu(), restored[k].cpu()) for k in expected), "adapter reload changed tensors"
    resumed_path = args.output / "resumed"
    resumed_path.mkdir()
    resumed = adapted_model()
    sft_trainer(resumed_path, resumed).train(resume_from_checkpoint=str(sft_path / "checkpoint-1"))
    resume_evidence = json.loads((resumed_path / "parameter_update.json").read_text())
    assert resume_evidence["start_step"] == 1 and resume_evidence["end_step"] == 2
    resumed_state = get_peft_model_state_dict(resumed)
    assert all(torch.equal(expected[k].cpu(), resumed_state[k].cpu()) for k in expected), "resumed optimizer/RNG diverged"
    report = {"passed": True, "scope": "CPU libraries, real tokenizer, native simulator, masking, optimizer, checkpoint resume",
              "gpu_training_validated": False, "model": "tiny randomly initialized Qwen3 (library integration only)",
              "stack": stack, "reference_context_lengths": lengths, "grpo_zero_reward_update": grpo_evidence,
              "canonical_grpo_config_accepted": True, "finite_sparse_validation": True,
              "sft_resume_update": resume_evidence, "adapter_reload_exact": True, "resume_matches_uninterrupted": True}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
