"""Small, strict experiment configuration and provenance helpers."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
from typing import Any

import yaml

from rowhammer_env.poc import ROOT, TASK_PATHS, load_task

STAGES = tuple(pathlib.Path(p).stem for p in TASK_PATHS)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | pathlib.Path) -> dict:
    defaults = yaml.safe_load((ROOT / "configs/training.yaml").read_text())
    supplied = yaml.safe_load(pathlib.Path(path).read_text())

    def merge(base, overrides, prefix=""):
        if not isinstance(overrides, dict):
            raise ValueError(f"{prefix or 'config'} must be a mapping")
        for key, value in overrides.items():
            if key not in base:
                raise ValueError(f"unknown config field: {prefix}{key}")
            if isinstance(base[key], dict):
                merge(base[key], value, f"{prefix}{key}.")
            else:
                base[key] = value

    merge(defaults, supplied)
    validate_config(defaults)
    return defaults


def validate_config(c: dict) -> None:
    def positive(value, name, *, zero=False, integer=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (not zero and value == 0)
                or (integer and not isinstance(value, int))):
            raise ValueError(f"invalid {name}: {value!r}")

    positive(c["run_seed"], "run_seed", zero=True, integer=True)
    for group, fields in {
        "model": ("lora_rank", "lora_alpha"),
        "rollout": ("max_turns", "max_context_tokens", "max_new_tokens"),
        "grpo": ("group_size", "updates_per_group"),
        "sft": ("demonstrations_per_stage", "epochs"),
    }.items():
        for field in fields:
            positive(c[group][field], f"{group}.{field}", integer=True)
    for field in STAGES:
        positive(c["curriculum"][field], field, integer=True)
    positive(c["checkpoint_every"], "checkpoint_every", integer=True)
    positive(c["optimizer"]["warmup_steps"], "warmup_steps", zero=True, integer=True)
    for group, field, zero in (
        ("rollout", "temperature", False), ("optimizer", "learning_rate", False),
        ("optimizer", "max_grad_norm", False), ("optimizer", "weight_decay", True),
        ("grpo", "clip_epsilon", False), ("grpo", "kl_beta", True),
        ("grpo", "probe_shaping", True),
    ):
        positive(c[group][field], f"{group}.{field}", zero=zero)
    if c["grpo"]["group_size"] < 2:
        raise ValueError("GRPO needs at least two episodes per environment seed")
    if not 0 <= c["grpo"]["probe_shaping"] < 1 or c["grpo"]["clip_epsilon"] >= 1:
        raise ValueError("shaping must be < success reward (1); clip_epsilon must be < 1")
    if c["rollout"]["max_context_tokens"] <= c["rollout"]["max_new_tokens"]:
        raise ValueError("context must leave room for the prompt")
    if c["model"]["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("supported dtypes: bfloat16, float32")
    if c["model"]["device"] not in ("cuda", "cpu"):
        raise ValueError("supported devices: cuda (one GPU), cpu")
    for group, field in (("model", "gradient_checkpointing"), ("sft", "enabled"), ("wandb", "enabled")):
        if not isinstance(c[group][field], bool):
            raise ValueError(f"{group}.{field} must be a boolean")
    splits = []
    for name, bounds in c["seeds"].items():
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(v) is not int for v in bounds) or bounds[0] > bounds[1]):
            raise ValueError(f"seeds.{name} must be an inclusive integer range")
        splits.append(set(range(bounds[0], bounds[1] + 1)))
    if any(a & b for i, a in enumerate(splits) for b in splits[i + 1:]):
        raise ValueError("training, validation and benchmark seeds must be disjoint")
    if c["sft"]["demonstrations_per_stage"] > len(splits[0]):
        raise ValueError("SFT demonstrations must use distinct training seeds per stage")


def seeds(c: dict, split: str) -> list[int]:
    lo, hi = c["seeds"][split]
    return list(range(lo, hi + 1))


def tasks() -> list[dict]:
    return [load_task(p) for p in TASK_PATHS]


def provenance(c: dict) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    files = [ROOT / "SOURCE_MANIFEST.yaml", ROOT / "requirements.txt", ROOT / "requirements-train.txt"]
    for folder in ("rowhammer_env", "cpp", "configs", "profiles/ddr4_vts25_v1", "profile_builder/trust"):
        files.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files.extend(ROOT / p for p in ("scripts/train.py", "build/phase2/ramulator_worker"))
    # Hash only what is actually present: the pinned worker binary or an optional
    # profile-trust dir may be absent on a given machine, and provenance runs at the
    # very start of every run — a missing file should not abort it.
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(files)) if p.is_file()}
    return {
        "git_revision": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "source_profile_hashes": hashes, "source_fingerprint": digest(hashes),
        "config_hash": digest(c), "run_seed": c["run_seed"],
        "environment_seeds": {s: seeds(c, s) for s in c["seeds"]},
        "tasks": tasks(), "model_requested": c["model"],
    }


def sampling_seed(run_seed: int, *parts: Any) -> int:
    return int(digest([run_seed, *parts])[:8], 16)
