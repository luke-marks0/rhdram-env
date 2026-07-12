#!/usr/bin/env python3
"""TRL GRPO training against the RowHammer OpenEnv environment.

A small (default 4B) instruction model is trained with Group Relative Policy
Optimization (GRPO). Each dataset row is one *task instance* (a task config +
seed); the environment discloses its objective/target at reset, which is baked
into the prompt. For every prompt GRPO samples ``num_generations`` completions;
each completion is parsed into a tool-call sequence and **replayed through the
real OpenEnv server** (``rowhammer_env.llm.grpo_env``). Reward is the trusted,
sparse episode reward — ``1.0`` only when the simulator + disturbance engine
produce a real flip satisfying the objective, ``0.0`` otherwise. No mock reward,
no reward from model-declared success (SPEC §2/§9).

The reward-from-trusted-state rollout path is the same one the P19 gate
verifies (``run_episode`` + the sparse trusted reward in ``RowHammerTaskEnv``).

Usage
-----
    # 1) (optional) let the trainer launch its own server, or start one yourself:
    #    python3 -m rowhammer_env.server.app
    # 2) run training:
    python3 -B scripts/train_grpo.py --config configs/training/grpo_qwen8b.yaml

    # dry run: build the dataset + score the reference hammer, no model load:
    python3 -B scripts/train_grpo.py --config configs/training/grpo_qwen8b.yaml --dry-run

Requires the training extras (``requirements-train.txt``) and the P17 HTTP
runtime (``requirements.txt``), plus a built Phase-2 worker (``build/phase2``).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import inspect
import json
import os
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.llm.grpo_env import (  # noqa: E402
    RolloutItem,
    build_messages,
    completion_text,
    disclose_metadata,
    evaluate_rewards,
    hint_actions,
    launch_server,
    parse_actions,
    summarize_actions,
)
from rowhammer_env.llm.multiturn_rollout import (  # noqa: E402  (torch-free)
    run_training_episode,
    to_grpo_example,
)
from rowhammer_env.llm.rollout import RolloutConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: pathlib.Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {path} must be a mapping")
    return cfg


def resolve_task(entry: dict) -> tuple[dict | None, str]:
    """Resolve a task-list entry into a (task_config, label) pair.

    An entry is either ``{config: path/to.yaml}`` (a full SPEC §10 task config)
    or ``{family: <name>}`` (the shorthand the env also accepts).
    """
    if "config" in entry:
        task_path = (ROOT / entry["config"]).resolve()
        task = yaml.safe_load(task_path.read_text())
        if not isinstance(task, dict):
            raise SystemExit(f"task config {task_path} must be a mapping")
        return task, str(task.get("id") or task.get("family") or task_path.stem)
    if "family" in entry:
        return {"family": str(entry["family"])}, str(entry["family"])
    raise SystemExit(f"task entry needs 'config' or 'family': {entry}")


# --------------------------------------------------------------------------- #
# Dataset: one row per (task, seed). Prompts baked from the disclosed reset obs.
# --------------------------------------------------------------------------- #
async def _build_rows(base_url: str, task_entries: list[dict], seeds: list[int]) -> list[dict]:
    rows: list[dict] = []
    for entry in task_entries:
        task, label = resolve_task(entry)
        for seed in seeds:
            metadata = await disclose_metadata(base_url, seed, task)
            rows.append(
                {
                    "messages": build_messages(metadata),
                    "seed": int(seed),
                    "task_json": json.dumps(task) if task is not None else "",
                    "task_id": str(metadata.get("task_id") or label),
                    "family": str(metadata.get("task_family") or label),
                }
            )
    return rows


def build_rows(base_url: str, task_entries: list[dict], seeds: list[int]) -> list[dict]:
    return asyncio.run(_build_rows(base_url, task_entries, seeds))


def render_prompts(rows: list[dict], tokenizer, enable_thinking: bool) -> None:
    """Bake each row's chat messages into a text ``prompt`` for GRPO.

    ``enable_thinking`` is passed straight to the chat template. For Qwen3 this
    toggles the ``<think>`` reasoning block; other templates ignore the kwarg.
    Pre-rendering (rather than a conversational dataset) gives version-independent
    control of the thinking toggle across TRL releases.
    """
    for row in rows:
        messages = row.pop("messages")
        try:
            row["prompt"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Template doesn't accept enable_thinking; render without it.
            row["prompt"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )


# --------------------------------------------------------------------------- #
# Reward functions (trusted, sparse) + optional format shaping
# --------------------------------------------------------------------------- #
def make_reward_functions(base_url: str, *, concurrency: int, max_steps: int, record: bool = False):
    def reward_env_success(completions, seed=None, task_json=None, family=None, task_id=None, **_):
        texts = [completion_text(c) for c in completions]
        items: list[RolloutItem] = []
        parsed: list[list] = []
        for i, text in enumerate(texts):
            actions = parse_actions(text)
            parsed.append(actions)
            task = json.loads(task_json[i]) if task_json and task_json[i] else None
            items.append(
                RolloutItem(
                    seed=int(seed[i]) if seed else 0,
                    task=task,
                    actions=actions,
                    episode_id=f"grpo_reward_{i}_{int(seed[i]) if seed else 0}",
                )
            )
        rewards = evaluate_rewards(base_url, items, concurrency=concurrency, max_steps=max_steps)
        if record:
            # Stash each scored rollout for the wandb callback to flush on_log.
            from rowhammer_env.llm.wandb_logging import record_rollout

            for i, (text, actions, r) in enumerate(zip(texts, parsed, rewards)):
                summary = summarize_actions(actions)
                record_rollout(
                    {
                        "family": family[i] if family else "",
                        "seed": int(seed[i]) if seed else 0,
                        "task_id": task_id[i] if task_id else "",
                        "reward_env": float(r),
                        "reward_format": 1.0 if actions else 0.0,
                        "completion": text,
                        **summary,
                    }
                )
        return rewards

    def reward_format(completions, **_):
        # Small dense shaping: reward a well-formed, parseable tool-call response.
        out = []
        for c in completions:
            actions = parse_actions(completion_text(c))
            out.append(1.0 if actions else 0.0)
        return out

    reward_env_success.__name__ = "env_success"
    reward_format.__name__ = "format_ok"
    return reward_env_success, reward_format


# --------------------------------------------------------------------------- #
# Multi-turn training rollouts (P27) — instance-only (needs torch/trl/GPU).
#
# The single-shot path above parses one completion into an action list and replays
# it. The discovery families (bounded_sweep / hidden_adjacency) need genuine
# turn-by-turn interaction: probe, read the timing digest, narrow candidates, hammer.
# ``rowhammer_env.llm.multiturn_rollout`` owns the torch-free loop + completion-mask
# assembly (unit-tested on the host); this section is the thin trl/torch binding.
#
# Reward stays trusted and sparse: the whole trajectory earns the final episode reward
# from ``_trusted_success()`` (``to_grpo_example``'s ``reward``), never anything derived
# from model text. GRPO's group-relative advantage is per full rollout.
# --------------------------------------------------------------------------- #
class HFCompletionGenerator:
    """One assistant completion per turn from a HF causal LM (instance-only).

    Renders the running transcript with the model's chat template, samples a short
    continuation, and returns its text. ``torch``/``transformers`` are imported lazily
    inside ``__call__`` so this file still imports (and ``--dry-run`` still runs) on a
    host without them.
    """

    def __init__(self, model, tokenizer, *, max_new_tokens: int, enable_thinking: bool, temperature: float) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.temperature = float(temperature)

    def __call__(self, messages, observation, transcript):
        del observation, transcript
        import torch

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


def make_multiturn_rollout_func(
    base_url, tokenizer, model, prompt_to_meta, *, max_turns, max_new_tokens, enable_thinking, temperature
):
    """A TRL GRPO ``rollout_func`` that runs a real multi-turn episode per prompt.

    Replaces the trainer's single-shot generation: for each prompt it drives the live
    OpenEnv server through ``run_training_episode`` with an :class:`HFCompletionGenerator`
    (the training model), then returns the token-level masked completion + the trusted
    episode reward via ``to_grpo_example``.

    TRL's contract (verified against TRL 1.8.0): the trainer calls
    ``rollout_func(prompts, trainer)`` *positionally* — the second argument is the
    ``GRPOTrainer`` itself, **not** the dataset columns. So ``seed``/``task`` are
    recovered per prompt from ``prompt_to_meta`` (built from the pre-rendered dataset
    rows; every generation of a given prompt shares its seed/task). The returned dict
    uses TRL's keys ``prompt_ids``/``completion_ids``/``logprobs`` plus a token-level
    ``completion_mask`` (assistant spans only) and a ``trusted_reward`` extra column the
    reward function reads. A prompt missing from the lookup fails **loudly** (never a
    silent wrong-episode reward).
    """

    def _logprobs(prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        # Teacher-forcing pass over prompt+completion → per-completion-token logprob
        # under the current policy. Autoregressive equivalence: each assistant token was
        # generated left-to-right by this same model given exactly this left context, so
        # a single forward pass reproduces the sampling logprobs (temperature 1.0). Tool
        # tokens' logprobs are computed too but are masked out of the loss.
        import torch

        ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
        with torch.no_grad():
            logits = model(ids).logits[0]
        logp = torch.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids)
        return [float(logp[start + t - 1, tok]) for t, tok in enumerate(completion_ids)]

    def rollout_func(prompts, trainer):
        del trainer  # the model/tokenizer are captured; the trainer handle is unused
        prompt_ids_b: list[list[int]] = []
        completion_ids_b: list[list[int]] = []
        completion_mask_b: list[list[int]] = []
        logprobs_b: list[list[float]] = []
        trusted_reward_b: list[float] = []
        for prompt in prompts:
            if prompt not in prompt_to_meta:
                raise RuntimeError(
                    "multi-turn rollout_func: prompt not found in the task lookup — the "
                    "dataset 'prompt' column must match the rollout prompts verbatim. "
                    "Check render_prompts()/chat-template consistency."
                )
            seed, task = prompt_to_meta[prompt]
            generator = HFCompletionGenerator(
                model, tokenizer, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking, temperature=temperature
            )
            config = RolloutConfig(base_url=base_url, seed=seed, task=task, episode_id=f"grpo_mt_{seed}")
            rollout = asyncio.run(run_training_episode(config, generator, max_turns=max_turns))
            example = to_grpo_example(rollout, tokenizer)
            prompt_ids_b.append(example["prompt_ids"])
            completion_ids_b.append(example["completion_ids"])
            completion_mask_b.append(example["completion_mask"])
            logprobs_b.append(_logprobs(example["prompt_ids"], example["completion_ids"]))
            trusted_reward_b.append(example["reward"])
        return {
            "prompt_ids": prompt_ids_b,
            "completion_ids": completion_ids_b,
            "completion_mask": completion_mask_b,
            "logprobs": logprobs_b,
            "trusted_reward": trusted_reward_b,
        }

    def reward_trusted(completions, trusted_reward=None, **_):
        # The reward is the trusted episode reward carried from the rollout — not
        # re-derived from completion text (SPEC §9).
        if trusted_reward is None:
            return [0.0] * len(completions)
        return [float(r) for r in trusted_reward]

    reward_trusted.__name__ = "env_success_multiturn"
    return rollout_func, reward_trusted


# --------------------------------------------------------------------------- #
# Weights & Biases: autorun rollout + metric monitoring when wandb.enabled.
# --------------------------------------------------------------------------- #
def setup_wandb(cfg: dict, grpo_cfg: dict) -> tuple[list, bool]:
    """Configure wandb logging from the ``wandb:`` config section.

    Returns ``(callbacks, enabled)``. When enabled, this:
      * points the run at the configured project/entity/run name/mode,
      * flips on TRL's own prompt/completion table (``log_completions``), and
      * returns a callback that logs the per-rollout table + custom metrics.

    It mutates ``grpo_cfg`` in place (``report_to`` + completion-logging keys);
    unknown keys are dropped later by the GRPOConfig filter, so this stays
    portable across TRL versions. Fails soft: a missing wandb install just
    disables logging with a warning instead of aborting the run.
    """
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return [], False
    try:
        import wandb  # noqa: F401

        from rowhammer_env.llm.wandb_logging import make_rollout_logger_callback
    except ImportError:
        print("warning: wandb not installed (pip install wandb); disabling wandb logging")
        return [], False

    os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg.get("project", "rowhammer-grpo")))
    if wandb_cfg.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", str(wandb_cfg["entity"]))
    if wandb_cfg.get("run_name"):
        os.environ.setdefault("WANDB_NAME", str(wandb_cfg["run_name"]))
    if wandb_cfg.get("mode"):  # "online" | "offline" | "disabled"
        os.environ.setdefault("WANDB_MODE", str(wandb_cfg["mode"]))

    grpo_cfg["report_to"] = "wandb"
    grpo_cfg.setdefault("log_completions", True)
    grpo_cfg.setdefault("num_completions_to_print", int(wandb_cfg.get("num_completions_to_print", 8)))
    grpo_cfg.setdefault("wandb_log_unique_prompts", True)

    callback = make_rollout_logger_callback(max_table_rows=int(wandb_cfg.get("max_table_rows", 32)))
    print(f"wandb logging enabled (project={os.environ['WANDB_PROJECT']})")
    return [callback], True


# --------------------------------------------------------------------------- #
# Dry run: no model, just prove the data + reward pipeline end-to-end.
# --------------------------------------------------------------------------- #
def dry_run(base_url: str, task_entries: list[dict], seeds: list[int], concurrency: int, max_steps: int) -> int:
    rows = build_rows(base_url, task_entries, seeds)
    print(f"built {len(rows)} task-instance rows")
    print("--- sample prompt (messages) ---")
    print(json.dumps(rows[0]["messages"], indent=2)[:2000])

    # Score the reference double-sided hammer (should earn 1.0 on known targets).
    items: list[RolloutItem] = []
    for i, row in enumerate(rows):
        task = json.loads(row["task_json"]) if row["task_json"] else None
        meta = asyncio.run(disclose_metadata(base_url, row["seed"], task))
        items.append(
            RolloutItem(seed=row["seed"], task=task, actions=hint_actions(meta), episode_id=f"grpo_dry_{i}")
        )
    rewards = evaluate_rewards(base_url, items, concurrency=concurrency, max_steps=max_steps)
    for row, r in zip(rows, rewards):
        print(f"  {row['family']:<28} seed={row['seed']:<3} reference-hammer reward={r}")
    print(f"reference-hammer mean reward = {sum(rewards) / len(rewards):.3f}")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--base-url", default=None, help="override env.base_url (skip launching a server)")
    parser.add_argument("--dry-run", action="store_true", help="build data + score the reference hammer, no training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg.get("env", {})
    task_entries = cfg.get("tasks") or [{"family": "known_target_anybit"}]
    seeds = [int(s) for s in cfg.get("seeds", list(range(1, 9)))]
    concurrency = int(env_cfg.get("concurrency", env_cfg.get("max_concurrent_envs", 8)))
    max_steps = int(env_cfg.get("max_steps", 4))
    rollout_cfg = cfg.get("rollout", {})
    multi_turn = bool(rollout_cfg.get("multi_turn", False))
    max_turns = int(rollout_cfg.get("max_turns", 64))

    # ---- server: connect to an existing one or launch our own ---------------
    server = None
    base_url = args.base_url or env_cfg.get("base_url")
    if base_url is None:
        if not env_cfg.get("launch_server", True):
            raise SystemExit("no env.base_url set and env.launch_server is false")
        server = launch_server(
            root=str(ROOT),
            host=str(env_cfg.get("server_host", "127.0.0.1")),
            port=env_cfg.get("server_port"),
            max_concurrent_envs=int(env_cfg.get("max_concurrent_envs", 8)),
            mode=str(env_cfg.get("server_mode", "production")),
        )
        base_url = server.base_url
        print(f"launched OpenEnv server at {base_url}")

    try:
        if args.dry_run:
            return dry_run(base_url, task_entries, seeds, concurrency, max_steps)

        # Heavy training imports live here so --dry-run needs no torch/trl.
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        model_cfg = cfg.get("model", {})
        model_name = str(model_cfg.get("name", "Qwen/Qwen3-4B"))
        enable_thinking = bool(model_cfg.get("enable_thinking", False))
        print(f"model={model_name} enable_thinking={enable_thinking}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        rows = build_rows(base_url, task_entries, seeds)
        render_prompts(rows, tokenizer, enable_thinking)
        dataset = Dataset.from_list([{k: r[k] for k in ("prompt", "seed", "task_json", "task_id", "family")} for r in rows])
        print(f"dataset: {len(dataset)} task-instance prompts")

        import torch

        dtype_name = str(model_cfg.get("torch_dtype", "bfloat16"))
        torch_dtype = getattr(torch, dtype_name, torch.bfloat16)
        model_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if model_cfg.get("attn_implementation"):
            model_kwargs["attn_implementation"] = str(model_cfg["attn_implementation"])
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Optional LoRA (the practical default for a 4B model on one GPU).
        peft_config = None
        peft_cfg = cfg.get("peft", {})
        if peft_cfg.get("enabled", True):
            from peft import LoraConfig

            peft_config = LoraConfig(
                r=int(peft_cfg.get("r", 16)),
                lora_alpha=int(peft_cfg.get("lora_alpha", 32)),
                lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=peft_cfg.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
            )

        grpo_cfg = dict(cfg.get("grpo", {}))
        grpo_cfg.setdefault("output_dir", "runs/grpo_rowhammer")

        # Monitoring: autorun wandb rollout/metric logging when wandb.enabled.
        # Must run before make_reward_functions so the reward fn records rollouts.
        wandb_callbacks, wandb_enabled = setup_wandb(cfg, grpo_cfg)

        reward_env, reward_format = make_reward_functions(
            base_url, concurrency=concurrency, max_steps=max_steps, record=wandb_enabled
        )
        reward_cfg = cfg.get("reward", {})
        if multi_turn:
            # One trusted reward func for the whole trajectory (no format shaping — the
            # rollout already enforces well-formed turns by construction).
            reward_weights = [float(reward_cfg.get("success_weight", 1.0))]
        else:
            reward_weights = [float(reward_cfg.get("success_weight", 1.0)), float(reward_cfg.get("format_weight", 0.1))]

        # Only forward keys GRPOConfig actually defines, so a newer/older TRL
        # doesn't reject the file (fail loud on genuinely unknown keys instead).
        valid = {f.name for f in dataclasses.fields(GRPOConfig)}
        # reward_weights is derived from the reward: section, not the grpo: section.
        grpo_cfg.pop("reward_weights", None)
        unknown = set(grpo_cfg) - valid
        if unknown:
            print(f"warning: dropping GRPO keys unsupported by installed TRL: {sorted(unknown)}")
        training_args = GRPOConfig(
            reward_weights=reward_weights,
            **{k: v for k, v in grpo_cfg.items() if k in valid},
        )

        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=wandb_callbacks or None,
        )
        if multi_turn:
            # Genuine turn-by-turn trajectories via a TRL custom rollout. Fail closed if
            # the installed TRL has no rollout_func hook — never silently fall back to
            # single-shot (that would train on a different, easier problem; SPEC §2).
            if "rollout_func" not in inspect.signature(GRPOTrainer.__init__).parameters:
                raise SystemExit(
                    "rollout.multi_turn requires a TRL build exposing GRPOTrainer(rollout_func=...); "
                    f"installed trl {getattr(__import__('trl'), '__version__', '?')} does not. "
                    "Upgrade TRL (see IMPLEMENTATION_PLAN_V3 §2 open decision #1) or unset rollout.multi_turn."
                )
            # TRL calls rollout_func(prompts, trainer) with only the prompt strings, so
            # map each pre-rendered prompt back to its (seed, task) for the episode.
            prompt_to_meta = {
                r["prompt"]: (int(r["seed"]), json.loads(r["task_json"]) if r["task_json"] else None)
                for r in rows
            }
            rollout_func, reward_trusted = make_multiturn_rollout_func(
                base_url,
                tokenizer,
                model,
                prompt_to_meta,
                max_turns=max_turns,
                max_new_tokens=int(grpo_cfg.get("max_completion_length", 256)),
                enable_thinking=enable_thinking,
                temperature=float(grpo_cfg.get("temperature", 1.0)),
            )
            print(f"multi-turn GRPO rollout enabled (max_turns={max_turns})")
            trainer = GRPOTrainer(
                reward_funcs=[reward_trusted],
                rollout_func=rollout_func,
                **trainer_kwargs,
            )
        else:
            trainer = GRPOTrainer(reward_funcs=[reward_env, reward_format], **trainer_kwargs)
        trainer.train()
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"saved policy to {training_args.output_dir}")
        return 0
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
