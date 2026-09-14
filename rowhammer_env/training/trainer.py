"""Curriculum GRPO trainer: the one entry point the training scope asks for.

Order of operations (``run``):
  1. optional SFT warm start from reference-solver demonstrations (training seeds only),
     saved as its own checkpoint before any GRPO;
  2. for each curriculum stage, in order:
       a. a reference gate — the deterministic solver must clear the stage on held-out
          validation seeds before the stage receives training time;
       b. ``curriculum[stage]`` GRPO steps, each a group of same-seed rollouts scored by
          trusted reward plus bounded probe shaping, normalized into advantages;
       c. a stage checkpoint.
  3. a final checkpoint plus the resolved config and provenance.

Resume restores adapter, optimizer, scheduler, stage/step, and RNG. A group with no
reward variance yields no gradient; a run that produces only such groups for too long
stops with a diagnostic rather than spinning (the scope's zero-variance stop).

Torch is imported lazily inside :meth:`_build_model`, so constructing a ``Trainer`` and
running the reference gate needs no model stack; the GRPO/SFT paths do.
"""
from __future__ import annotations

import json
import pathlib
import random
from typing import Any

from rowhammer_env.poc import PoCEnv, load_task

from . import config as configlib
from . import grpo
from .artifacts import RunWriter, WandB
from .metrics import aggregate
from .reference import ReferenceSolver
from .rollout import run_episode

# Consecutive zero-variance groups tolerated before the run stops with a diagnostic.
# Generous, because early groups are legitimately all-zero until the policy (helped by
# probe shaping) first produces reward spread.
ZERO_VARIANCE_PATIENCE = 24
# Fraction of validation seeds the reference solver must clear to open a stage.
REFERENCE_GATE_THRESHOLD = 0.9
REFERENCE_GATE_SEEDS = 8


class ReferenceGateError(RuntimeError):
    """The deterministic reference solver could not clear a stage it must scaffold."""


class ZeroVarianceStop(RuntimeError):
    """Neither trusted success nor probe shaping produced any reward variance."""


class Trainer:
    def __init__(
        self, config: dict[str, Any], writer: RunWriter, *,
        resume: str | None = None, stages: list[str] | None = None,
    ) -> None:
        self.config = config
        self.writer = writer
        self.resume_dir = resume
        self.run_seed = int(config["run_seed"])
        self.stages = stages or list(configlib.STAGES)
        stage_paths = dict(zip(configlib.STAGES, configlib.TASK_PATHS))
        self.tasks = {stage: load_task(stage_paths[stage]) for stage in self.stages}
        self.max_turns = int(config["rollout"]["max_turns"])
        self.shaping = float(config["grpo"]["probe_shaping"])
        self.total_updates = sum(int(config["curriculum"][s]) for s in self.stages)
        self.rng = random.Random(configlib.sampling_seed(self.run_seed, "seed-order"))
        self.env = PoCEnv(task=self.tasks[self.stages[0]])
        self.wandb = WandB(config, provenance=None)
        # Resume bookkeeping, overwritten by _load_checkpoint.
        self.state = {"global_step": 0, "stage_index": 0, "step_in_stage": 0, "sft_done": False}
        self.zero_variance_streak = 0
        self._model_built = False

    # ---- model / optimizer (lazy: the reference gate needs none of this) -----
    def _build_model(self) -> None:
        if self._model_built:
            return
        import torch
        from transformers import get_linear_schedule_with_warmup

        from .policy import QwenPolicy

        torch.manual_seed(self.run_seed)
        self.torch = torch
        self.policy = QwenPolicy(self.config)
        opt = self.config["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.policy.trainable_parameters(), lr=float(opt["learning_rate"]),
            weight_decay=float(opt["weight_decay"]),
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, int(opt["warmup_steps"]), max(1, self.total_updates)
        )
        self._model_built = True
        if self.resume_dir:
            self._load_checkpoint(self.resume_dir)

    # ---- public entry point --------------------------------------------------
    def run(self) -> dict[str, Any]:
        provenance = configlib.provenance(self.config)
        self.writer.write_json("provenance.json", provenance)
        self.writer.write_yaml("config.resolved.yaml", self.config)
        self._build_model()

        if self.config["sft"]["enabled"] and not self.state["sft_done"]:
            self._sft_warm_start()
            self.state["sft_done"] = True
            self._save_checkpoint("checkpoints/sft")

        for index in range(self.state["stage_index"], len(self.stages)):
            stage = self.stages[index]
            task = self.tasks[stage]
            self._reference_gate(stage, task)
            self._train_stage(index, stage, task)
            self.state["stage_index"] = index + 1
            self.state["step_in_stage"] = 0
            self._save_checkpoint(f"checkpoints/stage_{index}_{stage}")

        final = self._save_checkpoint("checkpoints/final")
        self.wandb.finish()
        self.env.close()
        return {"final_checkpoint": str(final), "provenance": provenance, "global_step": self.state["global_step"]}

    # ---- reference gate ------------------------------------------------------
    def _reference_gate(self, stage: str, task: dict[str, Any]) -> None:
        seeds = configlib.seeds(self.config, "validation")[:REFERENCE_GATE_SEEDS]
        results = [
            run_episode(self.env, task=task, seed=s, stage=stage, policy=ReferenceSolver(), max_turns=self.max_turns)
            for s in seeds
        ]
        rate = aggregate(results)["success_rate"]
        self.writer.append_jsonl("reference_gate.jsonl", {"stage": stage, "seeds": seeds, "success_rate": rate})
        if rate < REFERENCE_GATE_THRESHOLD:
            raise ReferenceGateError(
                f"reference solver cleared only {rate:.0%} of {stage} validation seeds "
                f"(need {REFERENCE_GATE_THRESHOLD:.0%}); the stage is not a valid scaffold."
            )

    # ---- one curriculum stage ------------------------------------------------
    def _train_stage(self, index: int, stage: str, task: dict[str, Any]) -> None:
        num_steps = int(self.config["curriculum"][stage])
        group_size = int(self.config["grpo"]["group_size"])
        for step in range(self.state["step_in_stage"], num_steps):
            seed = self.rng.choice(configlib.seeds(self.config, "train"))
            episodes = [
                run_episode(
                    self.env, task=task, seed=seed, stage=stage, policy=self.policy,
                    max_turns=self.max_turns, shaping_weight=self.shaping,
                )
                for _ in range(group_size)
            ]
            rewards = [e.reward for e in episodes]
            variance = grpo.reward_variance(rewards)
            advantages = grpo.group_advantages(rewards)
            metrics = {"loss": 0.0, "kl": 0.0, "ratio": 1.0, "clip_frac": 0.0, "samples": 0}

            if variance < grpo.ADV_EPSILON:
                self.zero_variance_streak += 1
                if self.zero_variance_streak >= ZERO_VARIANCE_PATIENCE:
                    raise ZeroVarianceStop(
                        f"{self.zero_variance_streak} consecutive GRPO groups had zero reward "
                        f"variance at stage {stage!r}. Neither trusted success nor probe shaping "
                        f"(weight={self.shaping}) is producing spread — raise probe_shaping, "
                        f"revisit the stage difficulty, or check the policy's tool-call validity."
                    )
            else:
                self.zero_variance_streak = 0
                samples = grpo.samples_from_group(episodes, advantages)
                metrics = grpo.grpo_update(self.policy, samples, self.config, self.optimizer)
                self.scheduler.step()

            self.state["global_step"] += 1
            self.state["step_in_stage"] = step + 1
            self._log_step(stage, seed, episodes, rewards, variance, metrics)
            if self.state["global_step"] % int(self.config["checkpoint_every"]) == 0:
                self._save_checkpoint(f"checkpoints/step_{self.state['global_step']}")

    def _log_step(self, stage, seed, episodes, rewards, variance, metrics) -> None:
        agg = aggregate(episodes)
        record = {
            "global_step": self.state["global_step"], "stage": stage, "seed": seed,
            "reward_mean": sum(rewards) / len(rewards), "reward_variance": variance,
            "success_rate": agg["success_rate"], "mean_turns": agg["mean_turns"],
            "mean_unique_probes": agg["mean_unique_probes"], "valid_call_rate": agg["valid_call_rate"],
            "mean_acts_used": agg["mean_acts_used"], "budget_exhausted_rate": agg["budget_exhausted_rate"],
            "lr": self.scheduler.get_last_lr()[0], "zero_variance_streak": self.zero_variance_streak,
            **{f"grpo_{k}": v for k, v in metrics.items()},
        }
        if self._model_built:
            record["gpu_mem_mb"] = self._gpu_mem_mb()
        self.writer.append_jsonl("train_log.jsonl", record)
        for e in episodes:
            self.writer.log_episode(e, condition="train")
        self.wandb.log(record, step=self.state["global_step"])
        print(f"[{stage}] step {self.state['global_step']} "
              f"reward={record['reward_mean']:.3f} success={record['success_rate']:.2f} "
              f"var={variance:.3f} loss={metrics['loss']:.4f} kl={metrics['kl']:.4f}")

    def _gpu_mem_mb(self) -> float:
        if self.config["model"]["device"] == "cuda" and self.torch.cuda.is_available():
            return self.torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    # ---- SFT warm start ------------------------------------------------------
    def _sft_warm_start(self) -> None:
        torch = self.torch
        per_stage = int(self.config["sft"]["demonstrations_per_stage"])
        epochs = int(self.config["sft"]["epochs"])
        train_seeds = configlib.seeds(self.config, "train")
        examples: list[tuple[list[int], list[int]]] = []
        for stage, task in self.tasks.items():
            for seed in train_seeds[:per_stage]:
                capture: list[dict[str, Any]] = []
                result = run_episode(
                    self.env, task=task, seed=seed, stage=stage, policy=ReferenceSolver(),
                    max_turns=self.max_turns, sft_capture=capture,
                )
                if result.success < 1.0:
                    continue  # only imitate successful demonstrations
                for turn in capture:
                    examples.append(self._sft_tokenize(turn["prompt"], turn["assistant"]))
        self.writer.append_jsonl("sft.jsonl", {"demonstration_turns": len(examples)})
        if not examples:
            return
        self.policy.model.train()
        for epoch in range(epochs):
            self.rng.shuffle(examples)
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for prompt_ids, target_ids in examples:
                logp = self.policy.logprobs_for(prompt_ids, target_ids, grad=True)
                loss = -logp.mean() / len(examples)
                loss.backward()
                total += float(loss) * len(examples)
            torch.nn.utils.clip_grad_norm_(
                self.policy.trainable_parameters(), float(self.config["optimizer"]["max_grad_norm"])
            )
            self.optimizer.step()
            print(f"[sft] epoch {epoch} loss={total / len(examples):.4f} over {len(examples)} turns")
            self.wandb.log({"sft_loss": total / len(examples), "sft_epoch": epoch})

    def _sft_tokenize(self, prompt_messages: list[dict[str, str]], assistant_text: str) -> tuple[list[int], list[int]]:
        prompt_ids = self.policy._template(prompt_messages)
        target_ids = self.policy.tokenizer.encode(assistant_text, add_special_tokens=False)
        target_ids.append(self.policy.tokenizer.eos_token_id)
        return prompt_ids, target_ids

    # ---- checkpointing -------------------------------------------------------
    def _save_checkpoint(self, rel: str) -> pathlib.Path:
        torch = self.torch
        path = self.writer.path(rel)
        path.mkdir(parents=True, exist_ok=True)
        self.policy.save_adapter(str(path / "adapter"))
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), path / "scheduler.pt")
        torch.save({
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "sampler_rng": self.rng.getstate(),
        }, path / "rng.pt")
        self.writer.write_json(str((pathlib.Path(rel) / "trainer_state.json")), self.state)
        return path

    def _load_checkpoint(self, rel: str) -> None:
        torch = self.torch
        path = self.writer.dir / rel if not pathlib.Path(rel).is_absolute() else pathlib.Path(rel)
        self.policy.load_adapter(str(path / "adapter"))
        self.optimizer = torch.optim.AdamW(
            self.policy.trainable_parameters(),
            lr=float(self.config["optimizer"]["learning_rate"]),
            weight_decay=float(self.config["optimizer"]["weight_decay"]),
        )
        self.optimizer.load_state_dict(torch.load(path / "optimizer.pt"))
        self.scheduler.load_state_dict(torch.load(path / "scheduler.pt"))
        rng = torch.load(path / "rng.pt", weights_only=False)
        random.setstate(rng["python_rng"])
        torch.set_rng_state(rng["torch_rng"])
        if rng["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda_rng"])
        self.rng.setstate(rng["sampler_rng"])
        self.state = json.loads((path / "trainer_state.json").read_text())
        print(f"[resume] restored from {path}: {self.state}")
