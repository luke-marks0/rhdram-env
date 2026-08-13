from __future__ import annotations

from typing import Any

from ..geometry import Geometry, _log2_exact


# Coordinate keys of a physical address form, in the order they appear in a
# DRAMSpec level list (channel first, column last). Used by the mapper and by the
# disclosure leakage guard when stripping hidden coordinates from public output.
COORD_KEYS: tuple[str, ...] = ("channel", "rank", "bankgroup", "bank", "row", "column")


class AddressError(ValueError):
    """An address could not be resolved to a linear worker address.

    Carries a stable SPEC §8 error code (``ADDRESS_NOT_DISCLOSED`` for a form the
    task hides, ``BAD_SCHEMA`` for a malformed or out-of-range coordinate) so the
    environment can surface it verbatim instead of collapsing to ``BAD_SCHEMA``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AddressMapper:
    """Bidirectional map between physical coordinates and linear addresses.

    The linear address is exactly what the ``External`` frontend / worker consume
    (``req.addr``); the physical coordinates are exactly what Ramulator's
    ``RoBaRaCoCh`` mapper decodes into ``req.addr_vec``. The bit layout is derived
    from the geometry the worker publishes (``DRAMSpec`` org ``count`` + prefetch),
    not hardcoded, so a round-trip reproduces Ramulator's own decode.

    ``RoBaRaCoCh`` (see ``addr_mapper/impl/ro_ba_ra_co_ch.cpp``) places, above the
    transaction offset and from the LSB: the prefetch-adjusted Column field, then
    the remaining non-channel levels in DRAMSpec order up to and including Row.
    Channel is handled by the channel mapper; the admitted config is single-channel
    (``CacheLineInterleave`` collapses to channel 0), which is the only geometry
    this projection accepts — anything else fails closed.
    """

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry
        names = geometry.level_names  # lower-cased, DRAMSpec order incl. Channel
        sizes = geometry.level_sizes
        if not names or names[0] != "channel":
            raise ValueError("expected Channel as the first DRAMSpec level")
        if names[-1] != "column":
            raise ValueError("RoBaRaCoCh projection assumes Column is the last level")
        if "row" not in names:
            raise ValueError("geometry has no Row level")

        self.tx_offset = _log2_exact(geometry.tx_bytes)
        self.channel_count = sizes["channel"]
        if self.channel_count != 1:
            raise ValueError("address projection supports single-channel geometry only")

        # Bit width of each non-channel level; Column loses the prefetch bits that
        # index the burst within a transaction (they live in the tx offset).
        widths: dict[str, int] = {name: _log2_exact(sizes[name]) for name in names[1:]}
        widths["column"] -= _log2_exact(geometry.prefetch)
        if any(w < 0 for w in widths.values()):
            raise ValueError("negative field width from geometry")
        self.widths = widths

        # Placement order above the tx offset: Column first (LSB), then Rank..Row.
        row_pos = names.index("row")
        self.placement = ["column"] + names[1:row_pos + 1]

        shift = self.tx_offset
        self.shifts: dict[str, int] = {}
        for field in self.placement:
            self.shifts[field] = shift
            shift += widths[field]
        self.total_bits = shift  # first bit above the whole mapped address

    @property
    def capacity(self) -> int:
        """Mapped byte capacity of the device: the exclusive bound on a linear address.

        The mapper is a bijection only inside ``[0, capacity)``. Above it every field
        is masked to its width (as Ramulator's own ``slice_lower_bits`` truncation
        does), so an out-of-domain address silently aliases an in-domain one — two
        different functional-memory cells sharing one physical DRAM location. Callers
        enforce this bound so oversized input fails closed instead of aliasing.
        """
        return 1 << self.total_bits

    def decode(self, linear: int) -> dict[str, int]:
        """Decode a linear address into physical coordinates (matches ``addr_vec``).

        Every coordinate key is present; single-channel geometry always yields
        ``channel == 0``. High bits above the device alias the way Ramulator's own
        ``slice_lower_bits`` truncation does (each field is masked to its width).
        """
        coords: dict[str, int] = {"channel": 0}
        for field in self.placement:
            coords[field] = (linear >> self.shifts[field]) & ((1 << self.widths[field]) - 1)
        return coords

    def encode(self, coords: dict[str, Any]) -> int:
        """Encode physical coordinates into the canonical linear address.

        Canonical = byte offset 0 within the transaction. Missing coordinates
        default to 0; any coordinate outside its field range (including a non-zero
        value on a zero-width level such as Rank on a single-rank part) fails closed
        with ``BAD_SCHEMA``.
        """
        linear = 0
        for field in self.placement:
            value = self._coord_int(coords, field)
            limit = 1 << self.widths[field]
            if value < 0 or value >= limit:
                raise AddressError("BAD_SCHEMA", f"{field}={value} is outside [0,{limit}) for this device")
            linear |= value << self.shifts[field]
        if self._coord_int(coords, "channel") != 0:
            raise AddressError("BAD_SCHEMA", "channel must be 0 for single-channel geometry")
        return linear

    @staticmethod
    def _coord_int(coords: dict[str, Any], field: str) -> int:
        """A physical coordinate, which must be a JSON integer (not a coercible value).

        ``spec/schemas/action.schema.json`` types every coordinate as an integer, and
        the broad runtime action model does not apply that nested schema, so this is
        the effective validator. ``int(value)`` would silently redirect the request to
        a *different* physical row — ``row: 1.9`` and ``row: true`` both truncate to
        row 1 — making behaviour depend on Python coercion rules rather than on the
        action contract. ``bool`` is excluded explicitly because it is an ``int``.
        """
        value = coords.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise AddressError(
                "BAD_SCHEMA", f"coordinate '{field}' must be a JSON integer, got {value!r}"
            )
        return value
