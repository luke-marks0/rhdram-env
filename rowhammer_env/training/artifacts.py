"""Local, writeup-ready run artifacts, plus an optional W&B mirror.

Everything a run reports lands under one directory as plain files: the resolved
config, provenance (git revision + source/profile hashes + seeds), per-episode JSONL,
disclosed trajectories, aggregate metrics, and the training log. The result bundle is
self-contained so a table can be reproduced without rerunning anything. Torch-free.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import yaml

from .metrics import EpisodeResult, aggregate


class RunWriter:
    """Owns a run directory and the files written into it."""

    def __init__(self, outdir: str | pathlib.Path) -> None:
        self.dir = pathlib.Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> pathlib.Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, name: str, obj: Any) -> pathlib.Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
        return p

    def write_yaml(self, name: str, obj: Any) -> pathlib.Path:
        p = self.path(name)
        p.write_text(yaml.safe_dump(obj, sort_keys=False))
        return p

    def append_jsonl(self, name: str, record: Any) -> None:
        with self.path(name).open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ---- episode + trajectory logs ------------------------------------------
    def log_episode(self, result: EpisodeResult, *, condition: str, jsonl: str = "episodes.jsonl") -> None:
        self.append_jsonl(jsonl, {"condition": condition, **result.summary()})

    def log_trajectory(self, result: EpisodeResult, *, condition: str, jsonl: str = "trajectories.jsonl") -> None:
        self.append_jsonl(jsonl, {
            "condition": condition, "stage": result.stage, "seed": result.seed,
            "success": result.success, "done_reason": result.done_reason,
            "turns": [t.as_public() for t in result.turns],
        })

    def write_condition_metrics(self, metrics_by_condition: dict[str, Any], name: str = "metrics.json") -> pathlib.Path:
        return self.write_json(name, metrics_by_condition)


def condition_report(results_by_condition: dict[str, list[EpisodeResult]]) -> dict[str, Any]:
    """Aggregate each condition (success rate + CI and the rest) for the writeup."""
    report: dict[str, Any] = {}
    for condition, results in results_by_condition.items():
        by_stage = {}
        stages = sorted({r.stage for r in results})
        for stage in stages:
            by_stage[stage] = aggregate([r for r in results if r.stage == stage])
        report[condition] = {"overall": aggregate(results), "by_stage": by_stage}
    return report


class WandB:
    """Thin, fail-open W&B wrapper. Any W&B failure is swallowed so it can never
    change rollout, reward, or training behaviour (a training-scope requirement)."""

    def __init__(self, config: dict[str, Any], *, provenance: dict[str, Any] | None = None) -> None:
        self.run = None
        wb = config.get("wandb", {})
        if not wb.get("enabled"):
            return
        try:
            import wandb

            self.run = wandb.init(
                project=wb.get("project"), entity=wb.get("entity"),
                config={"config": config, "provenance": provenance or {}},
            )
        except Exception as exc:  # noqa: BLE001 - W&B must never break a run
            print(f"[wandb] disabled (init failed: {exc})")
            self.run = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] log failed: {exc}")

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:  # noqa: BLE001
            pass
