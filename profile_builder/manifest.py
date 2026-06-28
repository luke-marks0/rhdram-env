"""Source-manifest verifier for the admitted DDR4 read-disturbance source.

Implements the profile side of TEST_PLAN P1: the ``ddr4_vts25`` manifest entry
must carry URL, commit, license, hash, and retrieval date, must be ``admitted``,
and the committed per-file hash list must match both the manifest anchor and the
on-disk source data. Any mismatch fails closed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib

import yaml

from . import SOURCE_ID
from .errors import ProfileRejected, SourceUnavailable
from .paths import DATA_DIR, MANIFEST, ROOT, SOURCE_HASHES

REQUIRED_FIELDS = ("url", "commit", "license", "sha256", "retrieved_utc", "admission")


@dataclasses.dataclass(frozen=True)
class SourcePin:
    source_id: str
    url: str
    commit: str
    license: str
    sha256: str
    retrieved_utc: str
    files: tuple[tuple[str, str], ...]  # (filename, sha256), sorted by filename

    @property
    def file_map(self) -> dict[str, str]:
        return dict(self.files)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest_entry() -> dict[str, str]:
    if not MANIFEST.is_file():
        raise SourceUnavailable(f"missing manifest: {MANIFEST}")
    doc = yaml.safe_load(MANIFEST.read_text())
    for source in doc.get("sources", []):
        if source.get("id") == SOURCE_ID:
            return source
    raise SourceUnavailable(f"manifest has no source {SOURCE_ID!r}")


def _parse_hash_list(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if not digest or not name:
            raise ProfileRejected(f"malformed hash line: {line!r}")
        rows.append((name.strip(), digest.strip()))
    return sorted(rows)


def load_source_pin(*, verify_data: bool = True) -> SourcePin:
    """Verify and return the admitted ddr4_vts25 source pin.

    When ``verify_data`` is true (the default), every listed source file must be
    present on disk and hash-match the committed list. Set it false only for
    metadata-only checks that must run without the fetched data.
    """
    entry = _load_manifest_entry()

    for field in REQUIRED_FIELDS:
        value = str(entry.get(field, "")).strip()
        if not value or value.lower() == "pending":
            raise ProfileRejected(f"{SOURCE_ID} manifest field {field!r} is unresolved")
    if entry["admission"] != "admitted":
        raise SourceUnavailable(
            f"{SOURCE_ID} is {entry['admission']!r}, not admitted; profile capability is absent"
        )

    if not SOURCE_HASHES.is_file():
        raise ProfileRejected(f"missing committed hash list: {SOURCE_HASHES}")
    anchor = _sha256_file(SOURCE_HASHES)
    if anchor != entry["sha256"]:
        raise ProfileRejected(
            f"manifest sha256 {entry['sha256']} does not anchor hash list {anchor}"
        )

    files = tuple(_parse_hash_list(SOURCE_HASHES.read_text()))

    if verify_data:
        if not DATA_DIR.is_dir():
            raise SourceUnavailable(
                f"source data absent at {DATA_DIR.relative_to(ROOT)}; "
                "run scripts/fetch_phase3_sources.py"
            )
        for name, expected in files:
            path = DATA_DIR / name
            if not path.is_file():
                raise ProfileRejected(f"source file missing: {name}")
            actual = _sha256_file(path)
            if actual != expected:
                raise ProfileRejected(f"source file hash mismatch: {name}")

    return SourcePin(
        source_id=SOURCE_ID,
        url=entry["url"],
        commit=entry["commit"],
        license=entry["license"],
        sha256=entry["sha256"],
        retrieved_utc=entry["retrieved_utc"],
        files=files,
    )
