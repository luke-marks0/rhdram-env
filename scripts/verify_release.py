#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import pathlib
import subprocess
import sys
import unittest
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


PHASE_GATE_NUMBERS = tuple(list(range(0, 10)) + list(range(11, 20)))
FORBIDDEN_SYMBOLS = (
    "mock_dram",
    "fake_dram",
    "placeholder_flip",
    "synthetic_calibration",
    "fallback_sandbox",
    "unsandboxed_script_runner",
    "canned_reward",
    "noop_mitigation",
)
EXECUTABLE_ROOTS = ("cpp", "rowhammer_env", "profile_builder", "sdk", "scripts", "tests")
SYMBOL_SCAN_EXEMPTIONS = {
    "scripts/verify_phase0.py",
    "scripts/verify_release.py",
    "scripts/verify_phase20.py",
}
RELEASE_DOCS = (
    "README.md",
    "SOURCE_MANIFEST.yaml",
    "spec/VALIDATION_REPORT.md",
    "spec/SBOM.md",
    "profiles/ddr4_vts25_v1/model_card.md",
)
PINNED_GIT_SOURCES = {
    "ramulator2": "third_party/ramulator2",
    "openenv": "third_party/openenv",
}


def run_python(*args: str) -> None:
    print("+ python -B " + " ".join(args), flush=True)
    subprocess.run([sys.executable, "-B", *args], cwd=ROOT, check=True)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"release verification failed: {message}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rebuild:
        run_python("scripts/build_phase2.py")
    check_release_artifacts_present()
    check_manifest_provenance()
    check_no_mock_symbols()
    run_phase_gate_matrix()
    run_unittest_matrix_without_skips()
    check_deterministic_replay()
    check_tracked_file_hygiene()
    print("release verification passed")
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P20 release re-qualification gate")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild Ramulator configs and worker before running the release matrix",
    )
    return parser.parse_args(argv)


def check_release_artifacts_present() -> None:
    missing = [name for name in RELEASE_DOCS if not (ROOT / name).is_file()]
    if missing:
        fail("missing release artifact(s): " + ", ".join(missing))


def check_manifest_provenance() -> None:
    manifest = yaml.safe_load((ROOT / "SOURCE_MANIFEST.yaml").read_text())
    if manifest.get("manifest_version") != 1:
        fail("SOURCE_MANIFEST.yaml manifest_version is not 1")
    if manifest.get("phase") != "P20":
        fail("SOURCE_MANIFEST.yaml phase is not P20")

    sources = manifest.get("sources") or []
    by_id = {source.get("id"): source for source in sources}
    for source in sources:
        if source.get("admission") != "admitted":
            continue
        unresolved = [field for field in ("commit", "sha256", "retrieved_utc", "license") if source.get(field) == "pending"]
        if unresolved:
            fail(f"admitted source has pending provenance: {source.get('id')} fields={unresolved}")

    for source_id, rel_path in PINNED_GIT_SOURCES.items():
        source = by_id.get(source_id)
        if not source:
            fail(f"manifest missing pinned source {source_id}")
        expected = source.get("commit")
        if not expected:
            fail(f"manifest source {source_id} has no commit")
        path = ROOT / rel_path
        if not (path / ".git").exists():
            fail(f"missing pinned checkout: {rel_path}")
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if actual != expected:
            fail(f"{rel_path} is {actual}, expected {expected}")

    ddr4 = by_id.get("ddr4_vts25")
    if not ddr4 or ddr4.get("admission") != "admitted":
        fail("ddr4_vts25 source is not admitted")
    hash_list = ROOT / str(ddr4.get("data_sha256sums", ""))
    if not hash_list.is_file():
        fail("ddr4_vts25 data hash list is missing")
    if sha256_file(hash_list) != ddr4.get("sha256"):
        fail("ddr4_vts25 manifest sha256 does not anchor the committed data hash list")


def check_no_mock_symbols() -> None:
    for root_name in EXECUTABLE_ROOTS:
        base = ROOT / root_name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if rel in SYMBOL_SCAN_EXEMPTIONS:
                continue
            text = path.read_text(errors="ignore").lower()
            for symbol in FORBIDDEN_SYMBOLS:
                if symbol in text:
                    fail(f"forbidden no-mock symbol {symbol!r} in {rel}")


def run_phase_gate_matrix() -> None:
    for number in PHASE_GATE_NUMBERS:
        script = ROOT / "scripts" / f"verify_phase{number}.py"
        if not script.is_file():
            fail(f"missing phase gate script: {script.relative_to(ROOT)}")
        run_python(str(script.relative_to(ROOT)))


def run_unittest_matrix_without_skips() -> None:
    print("+ python -B -m unittest discover -s tests (zero skips required)", flush=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    print(output, end="")
    if not result.wasSuccessful():
        fail("unit test matrix failed")
    if result.skipped:
        skipped = "; ".join(f"{test.id()}: {reason}" for test, reason in result.skipped)
        fail(f"unit test matrix has required skips: {skipped}")


def check_deterministic_replay() -> None:
    from rowhammer_env import Phase2Action, RowHammerTaskEnv

    manifest_digest = sha256_file(ROOT / "SOURCE_MANIFEST.yaml")
    for seed in (20, 21, 22):
        first = _replay_episode(RowHammerTaskEnv, Phase2Action, seed)
        second = _replay_episode(RowHammerTaskEnv, Phase2Action, seed)
        if first != second:
            fail(f"deterministic replay mismatch for seed {seed} under manifest {manifest_digest}")
    print(f"deterministic replay passed for seeds 20,21,22 under manifest {manifest_digest[:12]}")


def _replay_episode(RowHammerTaskEnv: Any, Phase2Action: Any, seed: int) -> list[dict[str, Any]]:
    env = RowHammerTaskEnv(task={"family": "known_target_anybit"})
    snapshots: list[dict[str, Any]] = []
    try:
        obs = env.reset(seed=seed, episode_id=f"release_replay_{seed}")
        if obs.error:
            fail(f"replay reset failed for seed {seed}: {obs.error}")
        snapshots.append(_snapshot(obs))
        assert env.disturbance is not None
        target = env.disturbance.target_addr
        row_bytes = env.disturbance.row_bytes
        threshold = env.disturbance.known_threshold
        left = target - row_bytes
        right = target + row_bytes
        snapshots.append(_snapshot(env.step(Phase2Action(tool="dram.info", args={}))))
        snapshots.append(
            _snapshot(
                env.step(
                    Phase2Action(
                        tool="dram.write",
                        args={"addr": {"kind": "logical", "addr": target}, "data_b64": base64.b64encode(b"\x00").decode()},
                    )
                )
            )
        )
        remaining_pairs = max(1, threshold // 2)
        while remaining_pairs > 0:
            pairs = min(remaining_pairs, 512)
            commands = []
            for _ in range(pairs):
                commands.append({"op": "RD", "addr": {"kind": "logical", "addr": left}})
                commands.append({"op": "RD", "addr": {"kind": "logical", "addr": right}})
            obs = env.step(Phase2Action(tool="dram.issue", args={"commands": commands}))
            snapshots.append(_snapshot(obs))
            remaining_pairs -= pairs
            if obs.done:
                break
        if not snapshots[-1]["done"] or snapshots[-1]["reward"] != 1.0:
            fail(f"replay did not reach trusted reward for seed {seed}")
        return snapshots
    finally:
        env.close()


def _snapshot(obs: Any) -> dict[str, Any]:
    data = obs.model_dump(mode="json") if hasattr(obs, "model_dump") else dict(obs)
    return {
        "reward": data.get("reward"),
        "done": data.get("done"),
        "cycle": data.get("cycle"),
        "data_b64": data.get("data_b64"),
        "last_action": data.get("last_action"),
        "public_counters": data.get("public_counters"),
        "feedback": data.get("feedback"),
        "error": data.get("error"),
        "metadata": data.get("metadata"),
    }


def check_tracked_file_hygiene() -> None:
    hidden = [p for p in git("ls-files").splitlines() if pathlib.PurePosixPath(p).name.startswith(".")]
    if hidden != [".gitignore"]:
        fail(f"unexpected tracked hidden files: {hidden}")


if __name__ == "__main__":
    raise SystemExit(main())
