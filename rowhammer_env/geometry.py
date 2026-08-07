from __future__ import annotations

from typing import Any


def _log2_exact(value: int) -> int:
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"expected a positive power of two, got {value}")
    return value.bit_length() - 1


class Geometry:
    """DDR4 / RoBaRaCoCh address geometry as reported by the simulator worker.

    Built from the worker's ``INFO`` response (see ``ramulator_worker`` / the
    ``IssuedEventRecorder`` plugin), so the mapping is derived from the real
    ``DRAMSpec`` rather than hardcoded constants. This is deliberately the narrow
    slice P11 needs to interpret the issued-event stream and to place flips at the
    right linear address; the full bidirectional address projection is P12.
    """

    def __init__(self, info: dict[str, Any]) -> None:
        self.standard = str(info["standard"])
        self.tx_bytes = int(info["tx_bytes"])
        self.prefetch = int(info["prefetch"])
        self.channel_width = int(info.get("channel_width", 0))
        names = [str(n) for n in info["level_names"]]
        sizes = [int(s) for s in info["level_sizes"]]
        self.level_names = [n.lower() for n in names]
        self.level_sizes = {n.lower(): s for n, s in zip(names, sizes)}
        self.row_stride = self._row_stride()

    def _row_stride(self) -> int:
        """Linear bytes between two physically adjacent rows in the same bank.

        RoBaRaCoCh lays out (from the LSB, above the transaction offset): the
        prefetch-adjusted Column field, then every other non-Channel level in
        ``DRAMSpec`` order up to and including Row, with Row as the most-significant
        mapped field (``addr_mapper_base.cpp``: Column is sliced first, then levels
        ``0..row_idx``). Advancing the Row index by one thus advances the linear
        address by ``2 ** (bits of every level below Row)``.

        The set of levels below Row is read straight from the reported
        ``level_names`` (``level_names[1:row_index]``, excluding the LSB Channel and
        the separately-placed Column), so the stride is derived from the real
        geometry for *any* standard — DDR (Rank/BankGroup/Bank) as well as HBM
        (PseudoChannel/BankGroup/Bank) — rather than a hardcoded DDR level set.
        """
        if "row" not in self.level_names:
            raise ValueError("geometry has no Row level")
        if self.level_names[-1] != "column":
            raise ValueError("RoBaRaCoCh geometry assumes Column is the last level")
        row_index = self.level_names.index("row")
        tx_offset = _log2_exact(self.tx_bytes)
        below_row = _log2_exact(self.level_sizes["column"]) - _log2_exact(self.prefetch)
        for name in self.level_names[1:row_index]:
            below_row += _log2_exact(self.level_sizes[name])
        return 1 << (tx_offset + below_row)

    def public_block(self) -> dict[str, Any]:
        """Architecture-level DRAM geometry disclosed unconditionally (P21).

        Standard-level public information — the row stride and the row/bank/
        bankgroup *counts* plus the standard name — identical across every episode
        of a profile, so it leaks nothing about the hidden target. Derived purely
        from the worker-reported level sizes (not the active address mapper), so it
        is byte-identical whatever mapping function an episode secretly uses; the
        bank-select *bit function* itself (P24's per-episode secret) is deliberately
        never exposed here — only sizes and the stride.
        """
        return {
            "row_bytes": self.row_stride,
            "row_count": self.level_sizes["row"],
            "bank_count": self.level_sizes["bank"],
            "bankgroup_count": self.level_sizes.get("bankgroup", 1),
            "standard": self.standard,
        }
