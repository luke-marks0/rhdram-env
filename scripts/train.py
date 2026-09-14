#!/usr/bin/env python3
"""Train one Qwen3 LoRA policy with multi-turn GRPO on the RowHammer PoC curriculum.

    python scripts/train.py --config configs/training.yaml --out runs/grpo1
    python scripts/train.py --config configs/training.yaml --out runs/smoke --smoke
    python scripts/train.py --config configs/training.yaml --out runs/grpo1 \
        --resume runs/grpo1/checkpoints/stage_1_bounded_sweep_easy

``--smoke`` runs a single GRPO step on the first stage (the short end-to-end GPU test
from the definition of done): it updates parameters and writes a reloadable checkpoint.
The model stack (torch/transformers/peft) is imported here, not at package import, so
this script needs ``requirements-train.txt`` while the rest of the package does not.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.training import config as configlib  # noqa: E402
from rowhammer_env.training.artifacts import RunWriter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training YAML (overrides configs/training.yaml defaults)")
    parser.add_argument("--out", required=True, help="run output directory")
    parser.add_argument("--resume", default=None, help="checkpoint directory to resume from")
    parser.add_argument("--smoke", action="store_true", help="one GRPO step on the first stage, then checkpoint")
    args = parser.parse_args()

    config = configlib.load_config(args.config)
    writer = RunWriter(args.out)

    stages = None
    if args.smoke:
        first = configlib.STAGES[0]
        config["curriculum"][first] = 1
        config["sft"]["enabled"] = False
        config["checkpoint_every"] = 1
        stages = [first]
        print(f"[smoke] one GRPO step on {first!r}, then a reloadable checkpoint")

    # Imported here so --help and config validation do not require torch.
    from rowhammer_env.training.trainer import Trainer

    trainer = Trainer(config, writer, resume=args.resume, stages=stages)
    result = trainer.run()
    print(f"done. final checkpoint: {result['final_checkpoint']}  global_step={result['global_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
