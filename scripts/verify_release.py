#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run([sys.executable, "-B", *args], cwd=ROOT, check=True)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    run("scripts/verify_phase9.py")
    run("scripts/verify_phase3.py")
    subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT, check=True)

    manifest = yaml.safe_load((ROOT / "SOURCE_MANIFEST.yaml").read_text())
    if manifest.get("phase") != "P10":
        raise SystemExit("SOURCE_MANIFEST.yaml phase is not P10")
    for src in manifest["sources"]:
        if src["admission"] == "admitted" and any(src.get(k) == "pending" for k in ("commit", "sha256", "retrieved_utc")):
            raise SystemExit(f"admitted source has pending provenance: {src['id']}")

    hidden = [p for p in git("ls-files").splitlines() if pathlib.PurePosixPath(p).name.startswith(".")]
    if hidden != [".gitignore"]:
        raise SystemExit(f"unexpected tracked hidden files: {hidden}")
    print("release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
