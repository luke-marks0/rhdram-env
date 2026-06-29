#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = [
    (
        "third_party/ramulator2",
        "https://github.com/CMU-SAFARI/ramulator2.git",
        "v2.1",
        "278f1effc3838099a6ffe0ad5f9f572fea80c948",
    ),
    (
        "third_party/openenv",
        "https://github.com/huggingface/OpenEnv.git",
        "v0.3.1",
        "7449c5dfe375c4c6e6f0827826925a46efd9249f",
    ),
]


def run(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def rev_parse(path: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def fetch_commit(path: pathlib.Path, url: str, commit: str) -> None:
    # Pin by exact commit SHA: the manifest pins a commit, not a branch tip, and
    # upstream refs (e.g. ramulator2 has no "v2.1" ref, only the v2.1.0 tag) do
    # not necessarily expose the pinned commit as a branch. GitHub allows
    # fetching a reachable commit directly, which honors the pin exactly.
    path.mkdir(parents=True, exist_ok=True)
    run(["git", "-C", str(path), "init", "-q"])
    run(["git", "-C", str(path), "remote", "add", "origin", url])
    run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", commit])
    run(["git", "-C", str(path), "checkout", "-q", "FETCH_HEAD"])


def main() -> int:
    for rel, url, ref, commit in SOURCES:
        del ref  # human-readable tag for provenance; we pin by commit below
        path = ROOT / rel
        if not (path / ".git").exists():
            fetch_commit(path, url, commit)
        actual = rev_parse(path)
        if actual != commit:
            raise SystemExit(f"{rel} is {actual}, expected {commit}")
    print("phase1 sources present")
    return 0


if __name__ == "__main__":
    sys.exit(main())

