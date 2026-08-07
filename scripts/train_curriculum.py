#!/usr/bin/env python3
"""Run a staged curriculum: one GRPO run per stage, each resuming the previous adapter.

Trains the curriculum stages in order (easiest first). Each stage trains only its own
band, saves its LoRA adapter to ``<output_dir>/<stage_name>``, and the next stage loads
that adapter as its starting weights. This is the real "master easy before hard"
progression that a single flat dataset does not give.

    python3 -B scripts/train_curriculum.py --config configs/training/grpo_curriculum.yaml

Per-stage eval reward (held-out seeds) is logged by each run via the config's ``eval:``
block. Pass --from-stage to resume the ladder at a given stage.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.llm.curriculum import load_curriculum  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--output-root", default=None, help="base dir for per-stage runs (default from grpo.output_dir)")
    ap.add_argument("--base-url", default=None, help="reuse a running env server across all stages")
    ap.add_argument("--from-stage", default=None, help="start at this stage name (skip earlier stages)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    stages = load_curriculum(cfg)
    output_root = pathlib.Path(args.output_root or cfg.get("grpo", {}).get("output_dir", "runs/grpo_curriculum"))

    names = [s.name for s in stages]
    if args.from_stage:
        if args.from_stage not in names:
            raise SystemExit(f"--from-stage {args.from_stage!r} not in {names}")
        start = names.index(args.from_stage)
    else:
        start = 0

    prev_adapter = None
    if start > 0:
        prev_adapter = str(output_root / names[start - 1])

    for stage in stages[start:]:
        out = output_root / stage.name
        cmd = [
            sys.executable, "-B", str(ROOT / "scripts/train_grpo.py"),
            "--config", str(args.config),
            "--stage", stage.name,
            "--output-dir", str(out),
        ]
        if args.base_url:
            cmd += ["--base-url", args.base_url]
        if prev_adapter:
            cmd += ["--resume-adapter", prev_adapter]

        print(f"\n=== stage {stage.name} -> {out} " + (f"(resume {prev_adapter})" if prev_adapter else "(fresh)") + " ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"stage {stage.name} failed (exit {result.returncode}); stopping ladder")
            return result.returncode
        prev_adapter = str(out)

    print(f"\ncurriculum complete; final adapter at {prev_adapter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
