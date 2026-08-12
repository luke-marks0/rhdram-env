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

Usage
-----
    # 1) (optional) let the trainer launch its own server, or start one yourself:
    #    python3 -m rowhammer_env.server.app
    # 2) run training:
    python3 -B scripts/train_grpo.py --config configs/training/grpo_curriculum.yaml

    # dry run: build the dataset + score the reference hammer, no model load:
    python3 -B scripts/train_grpo.py --config configs/training/grpo_curriculum.yaml --dry-run

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
import uuid

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.llm.curriculum import (  # noqa: E402  (torch-free)
    curriculum_task_seed_pairs,
    load_curriculum,
)
from rowhammer_env.llm.grpo_env import (  # noqa: E402
    HINT_LEVELS,
    RolloutItem,
    build_messages,
    completion_text,
    disclose_metadata,
    evaluate_rewards,
    hint_actions,
    launch_server,
    parse_actions,
    set_hint_level,
    summarize_actions,
)
from rowhammer_env.llm.multiturn_rollout import (  # noqa: E402  (torch-free)
    render_tool_call,
    run_batched_training_episodes,
    run_training_episode,
    to_grpo_example,
)
from rowhammer_env.llm.policies import ToolCall  # noqa: E402  (torch-free)
from rowhammer_env.llm.rollout import RolloutConfig  # noqa: E402
from rowhammer_env.llm.shaping import (  # noqa: E402  (torch-free)
    probe_shaping_reward,
    shaping_weights,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: pathlib.Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"config {path} must be a mapping")
    return cfg


def resolve_task(entry: dict) -> tuple[dict | None, str]:
    """Resolve a task-list entry: ``{config: path}`` (a full SPEC §10 task config)
    or ``{family: name}`` (shorthand the env also accepts).
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
def task_seed_pairs(cfg: dict, task_entries: list[dict], seeds: list[int]) -> list[tuple[dict, int]]:
    """Ordered ``(task_entry, seed)`` pairs the dataset is built from.

    A ``curriculum:`` block orders pairs by stage (easiest first); otherwise it's the
    plain ``tasks x seeds`` cartesian product.
    """
    if cfg.get("curriculum"):
        stages = load_curriculum(cfg)
        return [({"config": path}, seed) for path, seed in curriculum_task_seed_pairs(stages)]
    return [(entry, seed) for entry in task_entries for seed in seeds]


def filter_curriculum_to_stage(cfg: dict, stage_name: str) -> dict:
    """Return a copy of ``cfg`` whose curriculum is just the named stage."""
    entries = cfg.get("curriculum") or []
    match = [e for e in entries if str(e.get("name")) == stage_name]
    if not match:
        names = [str(e.get("name")) for e in entries]
        raise SystemExit(f"--stage {stage_name!r} not in curriculum; have {names}")
    cfg = dict(cfg)
    cfg["curriculum"] = match
    return cfg


def _parse_seed_spec(spec: dict) -> list[int]:
    """Seeds from an ``eval:`` block via ``seeds:`` or ``seed_range: [start, stop]``."""
    rng = spec.get("seed_range")
    if rng is not None:
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise SystemExit(f"eval.seed_range must be [start, stop], got {rng!r}")
        return list(range(int(rng[0]), int(rng[1]) + 1))
    seeds = spec.get("seeds")
    if seeds:
        return [int(s) for s in seeds]
    raise SystemExit("eval: needs 'seeds' or 'seed_range'")


def eval_task_seed_pairs(cfg: dict, task_entries: list[dict]) -> list[tuple[dict, int]]:
    """Held-out ``(task_entry, seed)`` pairs from the ``eval:`` block, empty if absent.

    Evaluates the current config's tasks (curriculum stages or plain ``tasks``) over the
    held-out eval seeds.
    """
    eval_cfg = cfg.get("eval")
    if not eval_cfg:
        return []
    eval_seeds = _parse_seed_spec(eval_cfg)
    if cfg.get("curriculum"):
        seen: set[str] = set()
        entries: list[dict] = []
        for stage in load_curriculum(cfg):
            for path in stage.tasks:
                if path not in seen:
                    seen.add(path)
                    entries.append({"config": path})
    else:
        entries = task_entries
    return [(entry, seed) for entry in entries for seed in eval_seeds]


async def _build_rows(base_url: str, pairs: list[tuple[dict, int]]) -> list[dict]:
    rows: list[dict] = []
    for entry, seed in pairs:
        task, label = resolve_task(entry)
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


def build_rows(base_url: str, pairs: list[tuple[dict, int]]) -> list[dict]:
    return asyncio.run(_build_rows(base_url, pairs))


def render_prompts(rows: list[dict], tokenizer, enable_thinking: bool) -> None:
    """Bake each row's chat messages into a text ``prompt`` for GRPO.

    Pre-rendering (rather than a conversational dataset) gives version-independent
    control of the ``enable_thinking`` toggle across TRL releases.
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
# assembly; this section is the thin trl/torch binding.
# --------------------------------------------------------------------------- #
class HFCompletionGenerator:
    """One assistant completion per turn from a HF causal LM (instance-only, sequential).

    ``torch``/``transformers`` are imported lazily inside ``__call__`` so this file
    still imports (and ``--dry-run`` still runs) on a host without them.
    """

    # A graceful, parseable end-of-episode turn, emitted instead of generating when
    # the running transcript has no context room left for another turn.
    _FINISH_TURN = render_tool_call(ToolCall("episode.finish", {}))

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_new_tokens: int,
        enable_thinking: bool,
        temperature: float,
        top_p: float | None = None,
        top_k: int = 20,
        max_turn_tokens: int | None = None,
        max_prompt_tokens: int | None = None,
        response_margin: int = 32,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.enable_thinking = bool(enable_thinking)
        # Per-turn cap: a non-thinking turn is one short tool call (~80 tokens), so 160
        # bounds a rambling turn without spending the whole multi-turn budget at once. A
        # thinking turn must fit the <think>...</think> reasoning plus the tool call, so
        # it needs a generous default instead. Honour an explicit max_turn_tokens first.
        if max_turn_tokens is not None:
            self.max_new_tokens = int(max_turn_tokens)
        elif self.enable_thinking:
            self.max_new_tokens = max(int(max_new_tokens), 1024)
        else:
            self.max_new_tokens = min(int(max_new_tokens), 160)
        self.temperature = float(temperature)
        # Qwen's own guidance differs by mode: thinking wants a wider nucleus (~0.95;
        # tight sampling makes reasoning degenerate into repetition), non-thinking ~0.8.
        self.top_p = float(top_p) if top_p is not None else (0.95 if self.enable_thinking else 0.8)
        self.top_k = int(top_k)
        # Hard context budget: an untrained policy that never emits episode.finish grows
        # the transcript every turn until the rendered prompt overflows the model's
        # positional window (generate() then returns corrupt tokens). Cap the running
        # prompt and end the episode gracefully instead. Defaults to the model's own
        # positional limit; a training run will usually set this lower.
        cfg_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        self.max_prompt_tokens = int(max_prompt_tokens or cfg_ctx or 32768)
        self.response_margin = int(response_margin)

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
        prompt_len = int(inputs["input_ids"].shape[1])
        room = self.max_prompt_tokens - prompt_len - self.response_margin
        if room <= 0:
            return self._FINISH_TURN
        turn_new_tokens = min(self.max_new_tokens, room)
        # Generating from a model mid-training is a trap: gradient checkpointing forces
        # use_cache=False and .train() leaves dropout on, and that no-KV-cache path
        # yields corrupt tokens. Switch to eval + KV cache + GC off only for generate,
        # then restore training state exactly.
        was_training = self.model.training
        gc_enabled = bool(getattr(self.model, "is_gradient_checkpointing", False))
        self.model.eval()
        if gc_enabled and hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=turn_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            if gc_enabled and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if was_training:
                self.model.train()
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


# Shared graceful end-of-episode turn for the batched generators below.
_FINISH_TURN = render_tool_call(ToolCall("episode.finish", {}))


def _render_prompt(tokenizer, messages, enable_thinking):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _turn_cap(max_turn_tokens, enable_thinking, max_new_tokens):
    """Per-turn generation cap, matching HFCompletionGenerator's mode-aware default."""
    if max_turn_tokens is not None:
        return int(max_turn_tokens)
    return max(int(max_new_tokens), 1024) if enable_thinking else min(int(max_new_tokens), 160)


class VLLMBatchedGenerator:
    """Batched multi-turn generator backed by a vLLM engine — one ``generate()`` over
    the prompts of ALL active episodes per tick (vLLM's continuous batching). Reuses
    the trainer's colocated engine; the caller syncs policy weights into it before
    generation. Over-budget episodes get a graceful ``episode.finish`` instead.
    """

    def __init__(self, llm, tokenizer, *, max_turn_tokens, enable_thinking, temperature, top_p, top_k, max_prompt_tokens, response_margin=32):
        self.llm = llm
        self.tokenizer = tokenizer
        self.enable_thinking = bool(enable_thinking)
        self.max_turn_tokens = int(max_turn_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.max_prompt_tokens = int(max_prompt_tokens) if max_prompt_tokens else 32768
        self.response_margin = int(response_margin)

    def __call__(self, requests):
        from vllm import SamplingParams

        texts = [_FINISH_TURN] * len(requests)
        prompts, params, slots = [], [], []
        for i, req in enumerate(requests):
            prompt = _render_prompt(self.tokenizer, req.messages, self.enable_thinking)
            n_tok = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            room = self.max_prompt_tokens - n_tok - self.response_margin
            if room <= 0:
                continue  # leave as the graceful finish turn
            prompts.append(prompt)
            params.append(SamplingParams(
                n=1, temperature=self.temperature, top_p=self.top_p,
                top_k=self.top_k, max_tokens=min(self.max_turn_tokens, room),
            ))
            slots.append(i)
        if prompts:
            outputs = self.llm.generate(prompts, params, use_tqdm=False)
            for slot, output in zip(slots, outputs):
                texts[slot] = output.outputs[0].text
        return texts


class HFBatchedGenerator:
    """Batched multi-turn generator using the training model's own ``generate()`` over
    a left-padded batch of all active episodes' prompts (no vLLM, no weight sync) —
    the fallback throughput path when the trainer has no reusable vLLM engine.
    """

    def __init__(self, model, tokenizer, *, max_turn_tokens, enable_thinking, temperature, top_p, top_k, max_prompt_tokens, response_margin=32):
        self.model = model
        self.tokenizer = tokenizer
        self.enable_thinking = bool(enable_thinking)
        self.max_turn_tokens = int(max_turn_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.max_prompt_tokens = int(max_prompt_tokens) if max_prompt_tokens else 32768
        self.response_margin = int(response_margin)

    def __call__(self, requests):
        import torch

        texts = [_FINISH_TURN] * len(requests)
        prompts, slots = [], []
        for i, req in enumerate(requests):
            prompt = _render_prompt(self.tokenizer, req.messages, self.enable_thinking)
            n_tok = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if self.max_prompt_tokens - n_tok - self.response_margin <= 0:
                continue
            prompts.append(prompt)
            slots.append(i)
        if not prompts:
            return texts

        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"  # decoder-only batch generation needs left pad
        try:
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        finally:
            self.tokenizer.padding_side = prev_side

        was_training = self.model.training
        gc_enabled = bool(getattr(self.model, "is_gradient_checkpointing", False))
        self.model.eval()
        if gc_enabled and hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_turn_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            if gc_enabled and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if was_training:
                self.model.train()
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, slot in enumerate(slots):
            texts[slot] = self.tokenizer.decode(gen[j], skip_special_tokens=True)
        return texts


def make_multiturn_rollout_func(
    base_url,
    tokenizer,
    model,
    prompt_to_meta,
    *,
    max_turns,
    max_new_tokens,
    enable_thinking,
    temperature,
    max_turn_tokens=None,
    max_prompt_tokens=None,
    concurrency=8,
    emit_probe_shaping=False,
):
    """A TRL GRPO ``rollout_func`` that runs real multi-turn episodes, batched.

    Replaces the trainer's single-shot generation: the batch of prompts runs as
    concurrent multi-turn episodes (:func:`run_batched_training_episodes`, in waves
    of ``concurrency`` live env sessions), with ONE batched model call per tick — a
    :class:`VLLMBatchedGenerator` when the trainer exposes a reusable colocated vLLM
    engine (weights synced each step via ``trainer._move_model_to_vllm``), else a
    :class:`HFBatchedGenerator`. Each finished rollout becomes the token-level masked
    completion + trusted episode reward via ``to_grpo_example``.

    TRL calls ``rollout_func(prompts, trainer)`` positionally (verified against TRL
    1.8.0) — the second argument is the ``GRPOTrainer`` itself, not dataset columns —
    so ``seed``/``task`` are recovered per prompt from ``prompt_to_meta`` (built from
    the pre-rendered dataset rows). The returned dict uses TRL's keys
    ``prompt_ids``/``completion_ids``/``logprobs`` plus a token-level assistant-span
    mask under TRL's ``env_mask`` key (1 = model tokens, 0 = external/tool tokens —
    a ``completion_mask`` key would be silently ignored), and two extra columns the
    reward functions read: ``trusted_reward`` (the sparse episode reward) and
    ``probe_shaping`` (the bounded P28 probe-decisiveness bonus, computed from the
    rollout's trusted timing digests — :mod:`rowhammer_env.llm.shaping`). A prompt
    missing from the lookup fails loudly rather than risk a silent wrong-episode
    reward.

    Returns ``(rollout_func, reward_trusted, reward_probe_shaping)``; the caller adds
    ``reward_probe_shaping`` only when shaping is enabled.
    """

    def _logprobs(prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
        # Teacher-forcing pass over prompt+completion -> per-completion-token logprob
        # under the current policy (reproduces the sampling logprobs at temperature
        # 1.0, since each token was generated left-to-right from this same context).
        import torch

        ids = torch.tensor([prompt_ids + completion_ids], device=model.device)
        was_training = model.training
        model.eval()  # match the eval-mode sampling distribution (no dropout)
        try:
            with torch.no_grad():
                logits = model(ids).logits[0]
        finally:
            if was_training:
                model.train()
        logp = torch.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids)
        return [float(logp[start + t - 1, tok]) for t, tok in enumerate(completion_ids)]

    logged_backend: list[str] = []

    def _pick_batch_generator(trainer):
        """Reuse the trainer's colocated vLLM engine if usable, else batched HF.

        vLLM is only used when the engine exists AND its weights can be synced
        (``_move_model_to_vllm``) — stale weights would train on rollouts that don't
        reflect the current policy.
        """
        llm = getattr(trainer, "llm", None)
        sync = getattr(trainer, "_move_model_to_vllm", None)
        top_p = 0.95 if enable_thinking else 0.8
        cap = _turn_cap(max_turn_tokens, enable_thinking, max_new_tokens)
        if llm is not None and not callable(sync) and not logged_backend:
            print(
                "warning: a vLLM engine is present but no _move_model_to_vllm() weight-sync "
                "was found on this TRL — using batched HF generation instead. The colocated "
                "vLLM engine is then reserving GPU memory for nothing; set grpo.use_vllm: "
                "false to reclaim it."
            )
        if llm is not None and callable(sync):
            sync()  # push current policy weights into the vLLM engine (once per step)
            gen = VLLMBatchedGenerator(
                llm, tokenizer, max_turn_tokens=cap, enable_thinking=enable_thinking,
                temperature=temperature, top_p=top_p, top_k=20, max_prompt_tokens=max_prompt_tokens,
            )
            backend = "vllm-batched"
        else:
            gen = HFBatchedGenerator(
                model, tokenizer, max_turn_tokens=cap, enable_thinking=enable_thinking,
                temperature=temperature, top_p=top_p, top_k=20, max_prompt_tokens=max_prompt_tokens,
            )
            backend = "hf-batched"
        if not logged_backend:
            logged_backend.append(backend)
            print(f"multi-turn rollout backend: {backend} (concurrency={concurrency}, max_turns={max_turns})")
        return gen

    def rollout_func(prompts, trainer):
        batch_gen = _pick_batch_generator(trainer)
        # A unique episode_id per rollout so concurrent same-seed generations get
        # distinct live env sessions (same deterministic task, independent exploration).
        specs: list[tuple[int, dict | None, str]] = []
        for prompt in prompts:
            if prompt not in prompt_to_meta:
                raise RuntimeError(
                    "multi-turn rollout_func: prompt not found in the task lookup — the "
                    "dataset 'prompt' column must match the rollout prompts verbatim. "
                    "Check render_prompts()/chat-template consistency."
                )
            seed, task = prompt_to_meta[prompt]
            specs.append((seed, task, f"grpo_mt_{uuid.uuid4().hex[:12]}"))

        # Run in waves within the server's concurrent-session budget; each wave batches
        # its generation across all its still-active episodes.
        rollouts: list = [None] * len(specs)
        for start in range(0, len(specs), max(1, concurrency)):
            wave = specs[start : start + max(1, concurrency)]
            wave_rollouts = asyncio.run(
                run_batched_training_episodes(base_url, wave, batch_gen, max_turns=max_turns)
            )
            for k, rollout in enumerate(wave_rollouts):
                rollouts[start + k] = rollout

        prompt_ids_b: list[list[int]] = []
        completion_ids_b: list[list[int]] = []
        env_mask_b: list[list[int]] = []
        logprobs_b: list[list[float]] = []
        trusted_reward_b: list[float] = []
        probe_shaping_b: list[float] = []
        for rollout in rollouts:
            example = to_grpo_example(rollout, tokenizer, enable_thinking=enable_thinking)
            mask = example["completion_mask"]  # 1 = assistant/model token, 0 = tool/env
            logprobs = _logprobs(example["prompt_ids"], example["completion_ids"])
            # Zero logprobs on env/tool tokens, matching TRL's own tool-loop convention;
            # they're excluded from the loss by env_mask anyway.
            logprobs = [lp if m == 1 else 0.0 for lp, m in zip(logprobs, mask)]
            prompt_ids_b.append(example["prompt_ids"])
            completion_ids_b.append(example["completion_ids"])
            env_mask_b.append(mask)
            logprobs_b.append(logprobs)
            trusted_reward_b.append(example["reward"])
            # Bounded, outcome-neutral shaping computed here (torch-free) from the
            # rollout's trusted timing digests — never from completion text (SPEC §9).
            if emit_probe_shaping:
                probe_shaping_b.append(probe_shaping_reward(rollout))
        out = {
            "prompt_ids": prompt_ids_b,
            "completion_ids": completion_ids_b,
            "env_mask": env_mask_b,
            "logprobs": logprobs_b,
            "trusted_reward": trusted_reward_b,
        }
        # Only surface the shaping column when it's actually used.
        if emit_probe_shaping:
            out["probe_shaping"] = probe_shaping_b
        return out

    def reward_trusted(completions, trusted_reward=None, **_):
        # The reward is the trusted episode reward carried from the rollout — not
        # re-derived from completion text (SPEC §9).
        if trusted_reward is None:
            return [0.0] * len(completions)
        return [float(r) for r in trusted_reward]

    def reward_probe_shaping(completions, probe_shaping=None, **_):
        if probe_shaping is None:
            return [0.0] * len(completions)
        return [float(r) for r in probe_shaping]

    reward_trusted.__name__ = "env_success_multiturn"
    reward_probe_shaping.__name__ = "probe_shaping"
    return rollout_func, reward_trusted, reward_probe_shaping


# --------------------------------------------------------------------------- #
# Weights & Biases: autorun rollout + metric monitoring when wandb.enabled.
# --------------------------------------------------------------------------- #
def setup_wandb(cfg: dict, grpo_cfg: dict) -> tuple[list, bool]:
    """Configure wandb logging from the ``wandb:`` config section.

    Returns ``(callbacks, enabled)``. Mutates ``grpo_cfg`` in place (``report_to`` +
    completion-logging keys). Fails soft: a missing wandb install just disables
    logging with a warning instead of aborting the run.
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
def dry_run(base_url: str, pairs: list[tuple[dict, int]], concurrency: int, max_steps: int) -> int:
    rows = build_rows(base_url, pairs)
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
    parser.add_argument("--stage", default=None, help="train only this curriculum stage by name")
    parser.add_argument("--resume-adapter", default=None, help="load a LoRA adapter as init weights (stage chaining)")
    parser.add_argument("--output-dir", default=None, help="override grpo.output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.stage:
        cfg = filter_curriculum_to_stage(cfg, args.stage)
    env_cfg = cfg.get("env", {})
    task_entries = cfg.get("tasks") or [{"family": "known_target_anybit"}]
    seeds = [int(s) for s in cfg.get("seeds", list(range(1, 9)))]
    pairs = task_seed_pairs(cfg, task_entries, seeds)
    eval_pairs = eval_task_seed_pairs(cfg, task_entries)
    if cfg.get("curriculum"):
        stages = load_curriculum(cfg)
        plan = " -> ".join(f"{s.name}({len(s.tasks)}x{len(s.seeds)})" for s in stages)
        print(f"curriculum ({len(stages)} stages): {plan}")
    if eval_pairs:
        print(f"held-out eval: {len(eval_pairs)} task-instance prompts")
    concurrency = int(env_cfg.get("concurrency", env_cfg.get("max_concurrent_envs", 8)))
    max_steps = int(env_cfg.get("max_steps", 4))
    rollout_cfg = cfg.get("rollout", {})
    multi_turn = bool(rollout_cfg.get("multi_turn", False))
    max_turns = int(rollout_cfg.get("max_turns", 64))
    # None -> the generator falls back to the model's own max_position_embeddings.
    # Set lower in the config to bound per-step memory.
    max_prompt_tokens = rollout_cfg.get("max_prompt_tokens")
    max_prompt_tokens = int(max_prompt_tokens) if max_prompt_tokens else None
    # None -> the generator defaults by mode (generous for reasoning, tight otherwise).
    max_turn_tokens = rollout_cfg.get("max_turn_tokens")
    max_turn_tokens = int(max_turn_tokens) if max_turn_tokens else None

    # Module-wide so the baked dataset prompt and the rollout prompt match. "full"
    # discloses a copyable answer (zero within-group reward variance -> zero GRPO
    # advantage); "geometry"/"none" force the model to actually search.
    hint_level = str(cfg.get("prompt", {}).get("hint_level", "full"))
    set_hint_level(hint_level)
    print(f"prompt hint_level={hint_level} (of {HINT_LEVELS})")

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
            return dry_run(base_url, pairs, concurrency, max_steps)

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

        _cols = ("prompt", "seed", "task_json", "task_id", "family")
        rows = build_rows(base_url, pairs)
        render_prompts(rows, tokenizer, enable_thinking)
        dataset = Dataset.from_list([{k: r[k] for k in _cols} for r in rows])
        print(f"dataset: {len(dataset)} task-instance prompts")

        eval_rows: list[dict] = []
        eval_dataset = None
        if eval_pairs:
            eval_rows = build_rows(base_url, eval_pairs)
            render_prompts(eval_rows, tokenizer, enable_thinking)
            eval_dataset = Dataset.from_list([{k: r[k] for k in _cols} for r in eval_rows])
            print(f"eval dataset: {len(eval_dataset)} held-out prompts")

        import torch

        dtype_name = str(model_cfg.get("torch_dtype", "bfloat16"))
        torch_dtype = getattr(torch, dtype_name, torch.bfloat16)
        model_kwargs = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if model_cfg.get("attn_implementation"):
            model_kwargs["attn_implementation"] = str(model_cfg["attn_implementation"])
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        peft_config = None
        peft_cfg = cfg.get("peft", {})
        if args.resume_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
            print(f"resumed adapter from {args.resume_adapter}")
        elif peft_cfg.get("enabled", True):
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
        if args.output_dir:
            grpo_cfg["output_dir"] = args.output_dir

        # Monitoring: must run before make_reward_functions so the reward fn records rollouts.
        wandb_callbacks, wandb_enabled = setup_wandb(cfg, grpo_cfg)

        reward_env, reward_format = make_reward_functions(
            base_url, concurrency=concurrency, max_steps=max_steps, record=wandb_enabled
        )
        reward_cfg = cfg.get("reward", {})
        success_weight, probe_shaping_weight = shaping_weights(reward_cfg)
        # Bounded, training-only probe shaping (P28) — multi-turn only. shaping_on gates
        # whether the extra reward func is attached at all, so it provably vanishes from
        # any config that doesn't ask for it (e.g. eval).
        shaping_on = multi_turn and probe_shaping_weight > 0.0
        if multi_turn:
            # Trusted trajectory reward only (the rollout enforces well-formed turns by
            # construction, so no format shaping), plus the bounded probe bonus if on.
            reward_weights = [success_weight] + ([probe_shaping_weight] if shaping_on else [])
        else:
            reward_weights = [success_weight, float(reward_cfg.get("format_weight", 0.1))]

        valid = {f.name for f in dataclasses.fields(GRPOConfig)}
        if eval_dataset is not None:
            grpo_cfg.setdefault("per_device_eval_batch_size", grpo_cfg.get("per_device_train_batch_size", 8))
            grpo_cfg.setdefault("eval_steps", grpo_cfg.get("save_steps", 50))
            strategy_key = "eval_strategy" if "eval_strategy" in valid else "evaluation_strategy"
            grpo_cfg.setdefault(strategy_key, "steps")
        # Only forward keys GRPOConfig actually defines, so a newer/older TRL doesn't
        # reject the file (fail loud on genuinely unknown keys instead).
        grpo_cfg.pop("reward_weights", None)  # derived from reward: above, not grpo:
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
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
        if multi_turn:
            # Fail closed if the installed TRL has no rollout_func hook — never silently
            # fall back to single-shot (that would train on a different, easier problem).
            if "rollout_func" not in inspect.signature(GRPOTrainer.__init__).parameters:
                raise SystemExit(
                    "rollout.multi_turn requires a TRL build exposing GRPOTrainer(rollout_func=...); "
                    f"installed trl {getattr(__import__('trl'), '__version__', '?')} does not. "
                    "Upgrade TRL (see IMPLEMENTATION_PLAN_V3 §2 open decision #1) or unset rollout.multi_turn."
                )
            # TRL calls rollout_func(prompts, trainer) with only prompt strings, so map
            # each pre-rendered prompt back to its (seed, task). Includes eval prompts so
            # eval rollouts resolve their episode too.
            prompt_to_meta = {
                r["prompt"]: (int(r["seed"]), json.loads(r["task_json"]) if r["task_json"] else None)
                for r in (rows + eval_rows)
            }
            rollout_func, reward_trusted, reward_probe_shaping = make_multiturn_rollout_func(
                base_url,
                tokenizer,
                model,
                prompt_to_meta,
                max_turns=max_turns,
                max_new_tokens=int(grpo_cfg.get("max_completion_length", 256)),
                enable_thinking=enable_thinking,
                temperature=float(grpo_cfg.get("temperature", 1.0)),
                max_turn_tokens=max_turn_tokens,
                max_prompt_tokens=max_prompt_tokens,
                concurrency=concurrency,
                emit_probe_shaping=shaping_on,
            )
            # Reward funcs line up positionally with reward_weights above: trusted
            # success first, the bounded probe bonus second iff shaping is on.
            reward_funcs = [reward_trusted] + ([reward_probe_shaping] if shaping_on else [])
            print(
                f"multi-turn GRPO rollout enabled (max_turns={max_turns}"
                + (f", probe_shaping_weight={probe_shaping_weight}" if shaping_on else "")
                + ")"
            )
            trainer = GRPOTrainer(
                reward_funcs=reward_funcs,
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
