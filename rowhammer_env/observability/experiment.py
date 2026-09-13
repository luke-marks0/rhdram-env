"""Local, reproducible experiment bundles. Benchmark scores are always sparse."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import pathlib
import platform
import subprocess
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from .metrics import EpisodeResult, observation_dict

ROOT = pathlib.Path(__file__).resolve().parents[2]


def file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    paths = [ROOT / "SOURCE_MANIFEST.yaml", ROOT / "requirements.txt", ROOT / "requirements-train.txt"]
    for directory in ("rowhammer_env", "scripts", "configs", "cpp", "profiles/ddr4_vts25_v1"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.extend((ROOT / "profile_builder").rglob("*.py"))
    paths.extend((ROOT / "profile_builder/trust").glob("*.pub"))
    paths.extend(p for p in (ROOT / "build/phase2").glob("*") if p.is_file() and p.name in {"ramulator_worker", "p2_external_ddr4.yaml"})
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    return {
        "repository_revision": git("rev-parse", "HEAD"),
        "repository_status": git("status", "--short"),
        "files_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths)) if p.exists()},
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
    }


def snapshot_sources(directory: pathlib.Path, source_provenance: dict[str, Any]) -> None:
    """Retain the measured working source, even when it has uncommitted edits."""
    with zipfile.ZipFile(directory / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, expected in source_provenance["files_sha256"].items():
            path = ROOT / name
            if file_hash(path) != expected:
                raise RuntimeError(f"source changed while snapshotting the run: {name}")
            archive.write(path, name)


def wilson_interval(successes: int, n: int) -> list[float]:
    """Two-sided 95% Wilson binomial interval, including all-success/all-failure."""
    if n == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def episode_record(result: EpisodeResult, *, condition: str, messages: list | None = None) -> dict[str, Any]:
    from rowhammer_env.llm.shaping import count_decisive_probes, unique_probe_candidates

    initial = result.initial_observation.get("metadata", {})
    initial_budget = initial.get("budget_remaining", {})
    final_budget = result.budget_remaining
    errors = Counter((s.error or {}).get("code") for s in result.trajectory if s.error)
    reset_error = result.initial_observation.get("error")
    if reset_error:
        errors[reset_error["code"]] += 1
    # A well-formed hammer that exhausts budget is still a valid tool call. Keep
    # budget/worker errors separate from parser, schema, and admission failures.
    invalid_codes = {"BAD_SCHEMA", "UNSUPPORTED_TOOL", "ADDRESS_NOT_DISCLOSED", "ILLEGAL_COMMAND"}
    valid = sum(s.parse_valid and (s.error or {}).get("code") not in invalid_codes for s in result.trajectory)
    record = {
        **result.as_metrics_input(), "condition": condition,
        "trusted_reward": result.reward, "success": result.reward == 1.0,
        "candidate_count": len(initial.get("candidates", [])),
        "valid_tool_calls": valid, "tool_calls": len(result.trajectory),
        "valid_tool_call_rate": valid / len(result.trajectory) if result.trajectory else 0.0,
        "decisive_probes": count_decisive_probes(result.trajectory),
        "unique_candidate_probes": len(unique_probe_candidates(result)),
        "resource_usage": {k: initial_budget[k] - final_budget[k] for k in initial_budget if k in final_budget},
        "error_codes": dict(errors), "budget_exhausted": "BUDGET_EXCEEDED" in errors,
        "truncated": not result.done or any(s.driver_generated for s in result.trajectory),
        "termination_reason": ("context_limit" if any(s.driver_generated for s in result.trajectory) else
                               "turn_limit" if not result.done else "success" if result.reward == 1.0 else
                               "error" if errors else "finish"),
        "initial_observation": result.initial_observation,
        "final_observation": observation_dict(result.final_observation),
        "trajectory": [{**s.as_public(), "feedback": s.feedback, "observation": s.observation,
                        "assistant_text": s.assistant_text, "parse_valid": s.parse_valid,
                        "driver_generated": s.driver_generated} for s in result.trajectory],
    }
    if messages is not None:
        record["messages"] = messages
    return record


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        successes = sum(r.get("trusted_reward") == 1.0 for r in items)
        calls = sum(r.get("tool_calls", 0) for r in items)
        return {
            "episodes": n, "successes": successes,
            "success_rate": successes / n if n else 0.0,
            "success_ci95": wilson_interval(successes, n),
            "valid_tool_call_rate": sum(r.get("valid_tool_calls", 0) for r in items) / calls if calls else 0.0,
            "error_rate": sum(bool(r.get("error_codes")) for r in items) / n if n else 0.0,
            "budget_exhaustion_rate": sum(r.get("budget_exhausted", False) for r in items) / n if n else 0.0,
            "truncation_rate": sum(r.get("truncated", False) for r in items) / n if n else 0.0,
            **{f"mean_{key}": sum(r.get(key, 0) for r in items) / n if n else 0.0
               for key in ("steps", "decisive_probes", "unique_candidate_probes")},
            **{f"mean_{key}": sum(r.get("resource_usage", {}).get(key, 0) for r in items) / n if n else 0.0
               for key in ("acts", "cycles", "tool_calls")},
        }

    buckets: dict[tuple, list] = defaultdict(list)
    for record in records:
        buckets[(record["condition"], record["task_id"], record["family"], record["difficulty"], record.get("candidate_count", 0))].append(record)
    groups = [{"condition": key[0], "task_id": key[1], "family": key[2], "difficulty": key[3],
               "candidate_count": key[4], **aggregate(items)} for key, items in sorted(buckets.items())]
    # The grouped table is primary; the pooled interval mixes different tasks.
    return {"score": "trusted_reward == 1.0", "confidence_interval": "Wilson 95%", "groups": groups}


class RunArtifacts:
    def __init__(self, directory: str | pathlib.Path, *, config: dict, identity: dict) -> None:
        self.path = pathlib.Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / "run.json").exists():
            raise FileExistsError(f"existing run bundle: {self.path}; choose a new output directory")
        self.records: list[dict[str, Any]] = []
        self.run = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                    "identity": identity, "provenance": provenance()}
        snapshot_sources(self.path, self.run["provenance"])
        self.write_json("resolved_config.json", config)
        self.write_json("run.json", self.run)

    def write_json(self, name: str, value: Any) -> None:
        # Atomic replacement protects a readable manifest/summary on interruption.
        path = self.path / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(path)

    def append(self, record: dict[str, Any]) -> None:
        with (self.path / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        # Retain only summary fields in RAM; full trajectories are already on disk.
        self.records.append({k: v for k, v in record.items() if k not in
                             {"trajectory", "messages", "initial_observation", "final_observation"}})

    def finish(self, *, status: str = "complete", error: str | None = None) -> dict[str, Any]:
        summary = summarize_records(self.records)
        self.write_json("summary.json", summary)
        rows = summary["groups"]
        with (self.path / "summary.csv").open("w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        self.run.update(status=status, error=error, episodes=len(self.records),
                        finished_utc=datetime.now(timezone.utc).isoformat())
        self.write_json("run.json", self.run)
        return summary
