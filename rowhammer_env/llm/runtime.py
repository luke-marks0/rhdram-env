"""Fail-closed training stack checks and checkpoint/parameter evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform

PINS = {"torch": "2.8.0", "transformers": "4.57.6", "trl": "1.8.0",
        "peft": "0.18.1", "accelerate": "1.12.0", "datasets": "4.7.0"}


def check_training_stack(*, require_cuda: bool = True) -> dict:
    versions = {name: importlib.metadata.version(name) for name in PINS}
    mismatch = {name: version for name, version in versions.items() if version.split("+")[0] != PINS[name]}
    if mismatch:
        raise RuntimeError(f"unsupported PoC training versions {mismatch}; install requirements-train.txt")
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError("supported PoC training requires Python 3.12")
    import torch
    import inspect
    from trl import GRPOTrainer

    if "rollout_func" not in inspect.signature(GRPOTrainer.__init__).parameters:
        raise RuntimeError("TRL is missing the required rollout_func hook")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("GPU training requires a working CUDA device; CPU tests do not certify this gate")
    return {"versions": versions, "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def trainable_fingerprint(model) -> str:
    """Hash actual trainable tensor bytes, rather than trusting loss/log output."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_snapshot(model):
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def parameter_delta(model, before) -> float:
    return max((float((p.detach().float().cpu() - before[name]).abs().max())
                for name, p in model.named_parameters() if name in before), default=0.0)


def evidence_callback(directory, *, require_update=False):
    import json
    import math
    import pathlib
    from transformers import TrainerCallback

    path = pathlib.Path(directory)

    class Evidence(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            # Trainer has loaded checkpoint model/optimizer state before this hook.
            self.before = parameter_snapshot(model)
            self.start_step = state.global_step
            self.fingerprint = trainable_fingerprint(model)

        def on_log(self, args, state, control, logs=None, **kwargs):
            with (path / "training_log.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state.global_step, **(logs or {})}, allow_nan=False) + "\n")

        def on_train_end(self, args, state, control, model=None, **kwargs):
            delta = parameter_delta(model, self.before)
            finite = all(bool(p.detach().isfinite().all()) for p in model.parameters() if p.requires_grad)
            report = {"start_step": self.start_step, "end_step": state.global_step,
                      "max_parameter_delta": delta, "parameters_changed": delta > 0 and math.isfinite(delta),
                      "finite_trainable_parameters": finite, "before_sha256": self.fingerprint,
                      "after_sha256": trainable_fingerprint(model)}
            (path / "parameter_update.json").write_text(json.dumps(report, indent=2) + "\n")
            del self.before
            if not finite or (require_update and not report["parameters_changed"]):
                raise RuntimeError("training did not produce a finite nonzero parameter update; see parameter_update.json and signal.jsonl")

    return Evidence()
