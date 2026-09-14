"""One local, multi-turn Qwen3 LoRA experiment on the real PoC environment.

The torch-free pieces (config, rollout driver, reference/controls, probe shaping,
metrics, artifacts) are exported here. The model stack (``policy``, ``grpo``,
``trainer``) imports torch and is imported directly from its module by the entry-point
scripts, so ``import rowhammer_env.training`` stays cheap and dependency-light.
"""
from __future__ import annotations

from . import artifacts, config, metrics, probe, prompt, reference, rollout

__all__ = ["artifacts", "config", "metrics", "probe", "prompt", "reference", "rollout"]
