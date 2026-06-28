#!/usr/bin/env python3
"""Materialize the admitted DDR4 read-disturbance source data.

The upstream artifact is git-ignored (like the other ``third_party`` sources), so
this script reproduces the 48 vendored read-disturbance CSVs from the pinned commit
and verifies them against the committed hash list. It fails closed on any mismatch;
it never substitutes stand-in data.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile_builder.errors import ProfileError  # noqa: E402
from profile_builder.manifest import load_source_pin  # noqa: E402
from profile_builder.paths import DATA_DIR, SOURCE_DIR, SOURCE_HASHES  # noqa: E402

URL = "https://github.com/CMU-SAFARI/ReadDisturbanceVTS25.git"
COMMIT = "5d734309457cc8a4ea3b1ec36b93932925548bac"


def _needed_files() -> list[str]:
    names = []
    for line in SOURCE_HASHES.read_text().splitlines():
        _, _, name = line.strip().partition("  ")
        if name:
            names.append(name.strip())
    return names


def _already_present() -> bool:
    try:
        load_source_pin(verify_data=True)
        return True
    except ProfileError:
        return False


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True)


def _clone_and_copy() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cache = pathlib.Path(tmp) / "src"
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", URL, str(cache)])
        _run(["git", "-C", str(cache), "sparse-checkout", "init", "--cone"])
        _run(["git", "-C", str(cache), "sparse-checkout", "set", "data"])
        _run(["git", "-C", str(cache), "checkout", COMMIT])
        for name in _needed_files():
            shutil.copyfile(cache / "data" / name, DATA_DIR / name)
        readme = cache / "README.md"
        if readme.is_file():
            shutil.copyfile(readme, SOURCE_DIR / "UPSTREAM_README.md")


def main() -> int:
    if _already_present():
        print("phase3 source data present and verified")
        return 0
    _clone_and_copy()
    pin = load_source_pin(verify_data=True)  # fail closed on any hash mismatch
    print(f"phase3 source data fetched and verified: {len(pin.files)} files at {pin.commit}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProfileError as exc:
        raise SystemExit(f"{exc.code}: {exc.message}")
