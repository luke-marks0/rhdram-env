"""Optional Weights & Biases rollout logging for GRPO training.

TRL's ``GRPOTrainer`` already streams the scalar training metrics (loss, reward
means, clip ratios, ...) to whatever ``report_to`` names, and — with
``log_completions=True`` — a prompt/completion table. This module adds the pieces
TRL does *not* give you for this environment:

* a per-rollout table showing the **parsed tool calls**, the trusted env reward,
  and the *size* of the emitted command list (``n_commands`` / ``n_pairs``), and
* custom scalar metrics (``rollout/env_success_frac``, ``rollout/n_commands_mean``,
  ...) so you can watch, at a glance, whether completions are anywhere near the
  disclosed disturbance threshold or just being truncated.

The reward function fills a process-global buffer during scoring; a
``TrainerCallback`` drains it on every ``on_log`` and writes to the *same* wandb
run the trainer opened. Nothing here is imported unless wandb logging is enabled,
so ``scripts/train_grpo.py --dry-run`` stays free of torch/transformers/wandb.
"""

from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_BUFFER: list[dict[str, Any]] = []


def record_rollout(record: dict[str, Any]) -> None:
    """Append one scored rollout (called from the reward function)."""
    with _LOCK:
        _BUFFER.append(dict(record))


def drain_rollouts() -> list[dict[str, Any]]:
    """Atomically take and clear the buffered rollouts (called from the callback)."""
    with _LOCK:
        items = list(_BUFFER)
        _BUFFER.clear()
    return items


def make_rollout_logger_callback(*, max_table_rows: int = 32):
    """Return a ``TrainerCallback`` that logs sampled rollouts to the active wandb run.

    transformers + wandb are imported here (not at module load) so this module is
    importable without the training stack.
    """
    import wandb
    from transformers import TrainerCallback

    class WandbRolloutLogger(TrainerCallback):
        _COLS = [
            "step", "family", "seed", "task_id",
            "reward_env", "reward_format",
            "n_actions", "first_tool", "n_commands", "n_pairs",
            "completion",
        ]

        def on_log(self, args, state, control, **kwargs):  # noqa: ANN001
            records = drain_rollouts()
            if not records or wandb.run is None:
                return
            step = int(getattr(state, "global_step", 0))

            n_cmds = [int(r.get("n_commands", 0)) for r in records]
            n_pairs = [int(r.get("n_pairs", 0)) for r in records]
            successes = sum(1 for r in records if float(r.get("reward_env", 0.0)) >= 1.0)
            parseable = sum(1 for r in records if int(r.get("n_actions", 0)) > 0)
            total = len(records)
            wandb.log(
                {
                    "rollout/count": total,
                    "rollout/env_success_count": successes,
                    "rollout/env_success_frac": successes / total,
                    "rollout/parseable_frac": parseable / total,
                    "rollout/n_commands_mean": sum(n_cmds) / total,
                    "rollout/n_commands_max": max(n_cmds) if n_cmds else 0,
                    "rollout/n_pairs_mean": sum(n_pairs) / total,
                },
                step=step,
            )

            table = wandb.Table(columns=self._COLS)
            for r in records[:max_table_rows]:
                table.add_data(
                    step,
                    str(r.get("family", "")),
                    int(r.get("seed", -1)),
                    str(r.get("task_id", "")),
                    float(r.get("reward_env", 0.0)),
                    float(r.get("reward_format", 0.0)),
                    int(r.get("n_actions", 0)),
                    str(r.get("first_tool", "")),
                    int(r.get("n_commands", 0)),
                    int(r.get("n_pairs", 0)),
                    (str(r.get("completion", "")) or "")[:4000],
                )
            wandb.log({"rollouts": table}, step=step)

    return WandbRolloutLogger()


__all__ = ["record_rollout", "drain_rollouts", "make_rollout_logger_callback"]
