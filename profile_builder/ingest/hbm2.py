"""HBM2 read-disturbance source ingestion (deferred; fail-closed).

Wires the optional HBM2 profile source (``hbm2_read_disturbance`` in
``SOURCE_MANIFEST.yaml``, the CMU-SAFARI HBM-Read-Disturbance artifact, SPEC §3)
into the profile pipeline. The source is pinned but **not admitted** — its
manifest entry is ``deferred_pending_license_and_hash`` (no SPDX license and no
verified data hash yet), so this module must never fabricate calibration: every
entry point fails closed with ``UNAVAILABLE_CAPABILITY`` until the manifest
resolves the license + hashes and flips the entry to ``admitted``.

This is the HBM2 counterpart of :mod:`profile_builder.ingest.vts25`. The parser
shape (per-pseudo-channel RowHammer/RowPress records, on-die-ECC-aware) is stubbed
behind the admission gate so it can be filled in against the real tables the day
the artifact is admitted, without changing the fail-closed contract.
"""

from __future__ import annotations

import dataclasses

import yaml

from ..errors import ProfileRejected, SourceUnavailable
from ..paths import MANIFEST

HBM2_SOURCE_ID = "hbm2_read_disturbance"

# Fields the manifest entry must carry (resolved, non-"pending") before HBM2 can
# be ingested — the same provenance contract the admitted DDR4 source meets.
REQUIRED_FIELDS = ("url", "commit", "license", "sha256", "retrieved_utc", "admission")


@dataclasses.dataclass(frozen=True)
class Hbm2SourceStatus:
    """Admission status of the HBM2 source, read from the pinned manifest."""

    admission: str
    url: str
    commit: str

    @property
    def admitted(self) -> bool:
        return self.admission == "admitted"


def _manifest_entry() -> dict[str, str]:
    if not MANIFEST.is_file():
        raise SourceUnavailable(f"missing manifest: {MANIFEST}")
    doc = yaml.safe_load(MANIFEST.read_text())
    for source in doc.get("sources", []):
        if source.get("id") == HBM2_SOURCE_ID:
            return source
    raise SourceUnavailable(f"manifest has no source {HBM2_SOURCE_ID!r}")


def source_status() -> Hbm2SourceStatus:
    """Return the HBM2 source admission status without requiring the data."""
    entry = _manifest_entry()
    return Hbm2SourceStatus(
        admission=str(entry.get("admission", "")),
        url=str(entry.get("url", "")),
        commit=str(entry.get("commit", "")),
    )


def require_admitted() -> dict[str, str]:
    """Verify the HBM2 source is admitted with resolved provenance, or fail closed.

    Mirrors ``manifest.load_source_pin``: while any required field is unresolved
    (``pending``) or the entry is not ``admitted``, ingestion is unavailable — the
    HBM2 profile capability is simply absent, never a stand-in.
    """
    entry = _manifest_entry()
    for field in REQUIRED_FIELDS:
        value = str(entry.get(field, "")).strip()
        if not value or value.lower() == "pending":
            raise SourceUnavailable(
                f"{HBM2_SOURCE_ID} manifest field {field!r} is unresolved; "
                "HBM2 profile capability is deferred (see SPEC §3, SOURCE_MANIFEST.yaml)"
            )
    if entry["admission"] != "admitted":
        raise SourceUnavailable(
            f"{HBM2_SOURCE_ID} is {entry['admission']!r}, not admitted; "
            "HBM2 profile capability is absent (fail closed, no fabricated calibration)"
        )
    return entry


def load_rd():
    """Load HBM2 RowHammer/RowPress records — deferred until the source is admitted."""
    require_admitted()
    # Reached only once the artifact is admitted (license + hashes resolved). The
    # real per-pseudo-channel parser lands here then; failing closed until then is
    # deliberate (SPEC §2 no-mock: never substitute stand-in HBM2 data).
    raise ProfileRejected(
        f"{HBM2_SOURCE_ID} is admitted but its ingestion parser is not yet implemented"
    )
