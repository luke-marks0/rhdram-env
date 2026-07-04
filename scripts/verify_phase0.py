#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "SOURCE_MANIFEST.yaml",
    "docs/phase0.md",
    "docs/threat_model.md",
    "spec/SPEC.md",
    "spec/IMPLEMENTATION_PLAN.md",
]

REQUIRED_SOURCES = {
    "ramulator2",
    "openenv",
    "ddr4_vts25",
    "hbm2_read_disturbance",
}

SOURCE_FIELDS = {
    "id",
    "role",
    "url",
    "commit",
    "license",
    "sha256",
    "retrieved_utc",
    "admission",
}

POLICY_FLAGS = {
    "no_mocks",
    "no_mock_fallbacks",
    "fail_closed_for_unavailable_capabilities",
    "simulation_only",
    "executable_work_requires_admitted_pins",
}

DENYLIST_TERMS = {
    "/proc/pagemap",
    "/dev/mem",
    "/dev/kmem",
    "/dev/kvm",
    "huge-page discovery",
    "cache-control attack utilities",
    "RDMA",
    "PCIe memory handles",
    "GPU memory handles",
    "host device access",
    "host mounts",
    "network from policy sandbox",
    "unsandboxed script execution",
}

ERROR_CODES = {
    "BAD_SCHEMA",
    "UNSUPPORTED_TOOL",
    "UNAVAILABLE_CAPABILITY",
    "BUDGET_EXCEEDED",
    "ADDRESS_NOT_DISCLOSED",
    "ILLEGAL_COMMAND",
    "QUEUE_FULL",
    "SANDBOX_VIOLATION",
    "SCRIPT_TIMEOUT",
    "PROFILE_REJECTED",
    "INTERNAL_SIMULATOR_ERROR",
}

EXECUTABLE_ROOTS = ["cpp", "rowhammer_env", "profile_builder", "sdk", "scripts", "tests"]

FORBIDDEN_SYMBOLS = [
    "mock_dram",
    "fake_dram",
    "placeholder_flip",
    "synthetic_calibration",
    "fallback_sandbox",
    "unsandboxed_script_runner",
    "canned_reward",
    "noop_mitigation",
]

SYMBOL_SCAN_EXEMPTIONS = {
    "scripts/verify_phase0.py",
    "scripts/verify_release.py",
    "scripts/verify_phase20.py",
}


def fail(message: str) -> None:
    raise SystemExit(f"phase0 verification failed: {message}")


def scalar(line: str) -> tuple[str, str] | None:
    match = re.match(r"\s*([A-Za-z0-9_]+):\s*(.*?)\s*$", line)
    if not match:
        return None
    value = match.group(2).strip('"')
    return match.group(1), value


def read_manifest() -> tuple[list[dict[str, str]], dict[str, str]]:
    path = ROOT / "SOURCE_MANIFEST.yaml"
    sources: list[dict[str, str]] = []
    policy: dict[str, str] = {}
    current: dict[str, str] | None = None
    section = ""

    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith(" "):
            section = stripped.rstrip(":")
            current = None
            continue
        if section == "sources" and stripped.startswith("- "):
            current = {}
            sources.append(current)
            item = scalar(stripped[2:])
            if item:
                current[item[0]] = item[1]
            continue
        item = scalar(raw)
        if not item:
            continue
        if section == "sources" and current is not None:
            current[item[0]] = item[1]
        elif section == "policy":
            policy[item[0]] = item[1]

    return sources, policy


def check_required_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        fail("missing required files: " + ", ".join(missing))


def check_manifest() -> bool:
    sources, policy = read_manifest()
    by_id = {source.get("id", ""): source for source in sources}

    missing_sources = REQUIRED_SOURCES - set(by_id)
    if missing_sources:
        fail("manifest missing sources: " + ", ".join(sorted(missing_sources)))

    for source_id in sorted(REQUIRED_SOURCES):
        missing_fields = SOURCE_FIELDS - set(by_id[source_id])
        if missing_fields:
            fail(f"{source_id} missing fields: {', '.join(sorted(missing_fields))}")

    for flag in sorted(POLICY_FLAGS):
        if policy.get(flag) != "true":
            fail(f"manifest policy {flag} must be true")

    return any(source["admission"] == "pending_pin" for source in sources)


def check_pending_pin_boundary(has_pending_pins: bool) -> None:
    if not has_pending_pins:
        return
    present = [name for name in EXECUTABLE_ROOTS if (ROOT / name).exists()]
    if present:
        fail("pending pins block executable roots: " + ", ".join(present))


def check_denylist() -> None:
    text = (ROOT / "docs/threat_model.md").read_text()
    missing = [term for term in sorted(DENYLIST_TERMS) if term not in text]
    if missing:
        fail("denylist missing terms: " + ", ".join(missing))


def check_error_codes() -> None:
    text = (ROOT / "docs/phase0.md").read_text()
    missing = [code for code in sorted(ERROR_CODES) if f"`{code}`" not in text]
    if missing:
        fail("stable error code set missing: " + ", ".join(missing))


def check_no_mock_symbols() -> None:
    scanned_roots = [ROOT / name for name in EXECUTABLE_ROOTS if (ROOT / name).exists()]
    for base in scanned_roots:
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
                    fail(f"forbidden no-mock symbol {symbol!r} in {path.relative_to(ROOT)}")


def main() -> int:
    check_required_files()
    has_pending_pins = check_manifest()
    check_pending_pin_boundary(has_pending_pins)
    check_denylist()
    check_error_codes()
    check_no_mock_symbols()
    print("phase0 verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
