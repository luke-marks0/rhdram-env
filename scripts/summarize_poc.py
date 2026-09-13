#!/usr/bin/env python3
"""Combine PoC evaluation bundles and compute paired changes on identical seeds."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rowhammer_env.observability.experiment import RunArtifacts


def evaluation_contract(run, config):
    """Reject comparisons that changed the simulator, disclosure, or sampling."""
    hashes = run["provenance"]["files_sha256"]
    measured = {name: digest for name, digest in hashes.items() if name.startswith(
        ("rowhammer_env/", "configs/tasks/", "cpp/", "profiles/", "profile_builder/", "build/phase2/"))
        or name in {"SOURCE_MANIFEST.yaml", "scripts/eval_poc.py", "scripts/train_grpo.py"}}
    model = run["identity"].get("model") or {"model": config["model"]["name"],
                                              "revision": config["model"].get("revision", "main")}
    return {"tasks": config["resolved_tasks"], "split": run["identity"]["split"],
            "model": model, "prompt": config["prompt"], "rollout": config["rollout"],
            "sampling": {key: config["grpo"][key] for key in ("seed", "temperature", "top_p", "top_k")},
            "measured_files_sha256": measured}


def paired_changes(records):
    indexed = {}
    for r in records:
        key = r["condition"], r["task_id"], r["seed"]
        if key in indexed:
            raise ValueError(f"duplicate episode: {key}")
        indexed[key] = r["trusted_reward"] == 1.0
    out = []
    tasks = sorted({r["task_id"] for r in records})
    for before, after in (("untrained", "trained"), ("sft", "trained"), ("trained", "trained_timing_hidden")):
        for task in tasks:
            a = {seed: score for (c, t, seed), score in indexed.items() if c == before and t == task}
            b = {seed: score for (c, t, seed), score in indexed.items() if c == after and t == task}
            if not a or not b:
                continue
            if a.keys() != b.keys():
                raise ValueError(f"unpaired seed sets for {before}/{after} on {task}")
            out.append({"from": before, "to": after, "task_id": task, "episodes": len(a),
                        "success_rate_change": sum(int(b[s])-int(a[s]) for s in a) / len(a),
                        "improved": sum(b[s] and not a[s] for s in a),
                        "regressed": sum(a[s] and not b[s] for s in a),
                        "unchanged": sum(a[s] == b[s] for s in a)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=pathlib.Path, nargs="+", required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    records, sources = [], []
    reference_config = None
    for path in args.runs:
        run = json.loads((path / "run.json").read_text())
        config = json.loads((path / "resolved_config.json").read_text())
        contract = evaluation_contract(run, config)
        if run["status"] != "complete" or run["identity"]["evaluation_shaping_weight"] != 0.0:
            raise ValueError(f"incomplete or shaped evaluation bundle: {path}")
        if reference_config is not None and contract != reference_config:
            raise ValueError(f"inconsistent simulator/source, task, split, model, or sampling contract: {path}")
        reference_config = contract
        for line in (path / "episodes.jsonl").read_text().splitlines():
            records.append(json.loads(line))
        sources.append({"path": str(path), "run": run})
    changes = paired_changes(records)
    bundle = RunArtifacts(args.output, config={"evaluation_contract": reference_config}, identity={"source_runs": sources})
    for record in records:
        bundle.append(record)
    summary = bundle.finish()
    bundle.write_json("paired_changes.json", changes)
    lines = ["| Condition | Task | Success | 95% Wilson CI |", "|---|---|---:|---:|"]
    for row in summary["groups"]:
        low, high = row["success_ci95"]
        lines.append(f"| {row['condition']} | {row['task_id']} | {row['successes']}/{row['episodes']} | {low:.1%}–{high:.1%} |")
    (bundle.path / "table.md").write_text("\n".join(lines) + "\n")
    print(f"Saved result table: {bundle.path / 'table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
