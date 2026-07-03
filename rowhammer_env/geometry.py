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
        prefetch-adjusted Column field, then Rank, BankGroup, Bank, and finally
        Row as the most-significant field. Advancing the Row index by one thus
        advances the linear address by ``2 ** (bits of every level below Row)``.
        """
        tx_offset = _log2_exact(self.tx_bytes)
        column_bits = _log2_exact(self.level_sizes["column"]) - _log2_exact(self.prefetch)
        below_row = column_bits
        for name in ("rank", "bankgroup", "bank"):
            if name in self.level_sizes:
                below_row += _log2_exact(self.level_sizes[name])
        return 1 << (tx_offset + below_row)
