"""Canonical normalized tables with deterministic digests.

Canonicalization here is structural: the typed records from :mod:`profile_builder.ingest`
are sorted into a stable order and serialized deterministically so that the same
pinned source always yields the same content digest. The record counts must reproduce
the raw source row counts exactly (TEST_PLAN P2).
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib

from ..ingest import DsBerRecord, RdRecord, chip_ids, load_ds_ber, load_rd
from ..paths import require_source_data

RD_KINDS = ("rd_hcf", "rd_ber", "rd_rp")


@dataclasses.dataclass(frozen=True)
class CanonicalTables:
    rd: dict[str, list[RdRecord]]  # kind -> records (sorted)
    ds_ber: list[DsBerRecord]


def _rd_key(r: RdRecord) -> tuple:
    return (r.chip, r.victim_row, r.pattern, r.aggr_type, r.hc, r.num_bitflips, r.itr)


def _ds_key(r: DsBerRecord) -> tuple:
    return (r.chip, r.temp, r.victim_row, r.hc, r.itr)


def build_tables() -> CanonicalTables:
    rd = {kind: sorted(load_rd(kind), key=_rd_key) for kind in RD_KINDS}
    ds_ber = sorted(load_ds_ber(), key=_ds_key)
    return CanonicalTables(rd=rd, ds_ber=ds_ber)


def canonical_counts(tables: CanonicalTables) -> dict[str, int]:
    """Count canonical records per ``<chip>_<kind>`` table key."""
    counts: dict[str, int] = {}
    for kind, records in tables.rd.items():
        for r in records:
            counts[f"{r.chip}_{kind}"] = counts.get(f"{r.chip}_{kind}", 0) + 1
    for r in tables.ds_ber:
        counts[f"{r.chip}_ds_ber_sweep"] = counts.get(f"{r.chip}_ds_ber_sweep", 0) + 1
    return counts


def source_row_counts() -> dict[str, int]:
    """Count raw CSV data rows per table key, independent of the parser."""
    data = require_source_data()
    counts: dict[str, int] = {}
    for chip in chip_ids():
        for kind in (*RD_KINDS, "ds_ber_sweep"):
            with (data / f"{chip}_{kind}.csv").open(newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)  # header
                counts[f"{chip}_{kind}"] = sum(1 for _ in reader)
    return counts


def table_digest(tables: CanonicalTables) -> str:
    """Deterministic sha256 over the full canonical content."""
    digest = hashlib.sha256()
    for kind in RD_KINDS:
        digest.update(f"#{kind}\n".encode())
        for r in tables.rd[kind]:
            digest.update(
                f"{r.chip},{r.victim_row},{r.pattern},{r.aggr_type},"
                f"{r.hc},{r.num_bitflips},{r.itr}\n".encode()
            )
    digest.update(b"#ds_ber_sweep\n")
    for d in tables.ds_ber:
        digest.update(f"{d.chip},{d.temp},{d.victim_row},{d.hc},{d.itr}\n".encode())
    return digest.hexdigest()
