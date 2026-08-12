from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .geometry import Geometry
from .mitigations import require_admitted_mitigation
from .profiles import load_profile
from .standards import StandardModel, UnsupportedStandard


# A same-row activation that is precharged again after longer than this many
# controller cycles is treated as an open-row *dwell* (RowPress regime, SPEC
# §5.3). A normal back-to-back access closes its row in far fewer cycles, so
# ordinary traffic accrues no RowPress bonus and the RowHammer thresholds used by
# the phase-4..9 fixtures are unchanged.
ROWPRESS_DWELL_NOMINAL = 512
# Dwell (controller cycles) at which the RowPress hcfirst reduction saturates —
# i.e. one long-open activation is worth the profile's full
# ``rowhammer_to_rowpress_hc_reduction`` factor of ordinary hammers.
ROWPRESS_DWELL_SATURATION = 100_000

# Upper bound on how many bits a single victim row may flip (bounded multiplicity).
MAX_FLIPPED_BITS_PER_ROW = 64

# Accessing commands that precharge the bank themselves instead of emitting a
# separate PRE. Ramulator's RDA/WRA run ``PREpb::action`` as their own action
# (``dram/commands/RDA.h``), so no PRE event marks the close.
AUTO_PRECHARGE_OPS = ("RDA", "WRA")

# Commands with no direct effect on disturbance state: the data accesses, whose
# exposure is credited to the ACT that opened the row, and VRR, whose victim-row
# refresh is modelled from the ACT counter in ``_oracle_on_act`` rather than folded
# from the event (it is targeted, not part of the JEDEC auto-refresh window).
INERT_OPS = ("RD", "WR", "VRR")

# Resolves a row's own column-0 linear address from its decoded coordinates
# (``channel``/``rank``/``bankgroup``/``bank``/``row``/``column``, the keys the
# worker's address ops use). This is the *only* way the engine turns a decoded
# victim key into an address: the arithmetic form (aggressor ± d * row_stride)
# holds only for the public RoBaRaCoCh mapper and lands in the wrong bank under a
# secret row->bank mapping. Production passes the worker's ENCODE op;
# ``AddressMapper.encode`` is the equivalent for the public mapper.
# @spec:sim-exposure-flip
RowEncoder = Callable[[Mapping[str, int]], int]


def classify_command(op: str) -> str | None:
    """Classify an issued DRAM command into the disturbance model's event classes.

    Returns ``"hammer"``, ``"close"``, ``"refresh"``, ``"inert"``, or ``None`` for a
    command the model does not know how to account for.
    """
    if op == "ACT":
        return "hammer"
    if op.startswith("PRE") or op in AUTO_PRECHARGE_OPS:
        return "close"
    if op.startswith("REF") or op.startswith("RFM"):
        return "refresh"
    if op in INERT_OPS:
        return "inert"
    return None


def unclassified_commands(command_names: Iterable[str]) -> list[str]:
    """The command names in ``command_names`` the disturbance model cannot classify."""
    return [op for op in command_names if classify_command(op) is None]


@dataclass
class Victim:
    """Latent per-row vulnerability state (SPEC §5.4).

    ``threshold`` is the double-sided hcfirst (kept under this name for
    backward compatibility with the P11 exposure tests); ``single_threshold`` is
    the higher single-sided hcfirst. Both are sampled once, at episode start,
    from the admitted profile and never resampled per access.
    """

    threshold: int  # double-sided hcfirst
    single_threshold: int  # single-sided hcfirst
    direction: str  # "0->1" or "1->0", from the stored data pattern
    addr: int  # linear (logical) byte address of this victim row at column 0
    data_pattern: str = "all_zeros"  # written pattern of the victim region
    first_bit: int = 0  # bit position of the first cell to flip
    left: int = 0  # ACTs on the physical row immediately to the left (row-1)
    right: int = 0  # ACTs on the physical row immediately to the right (row+1)
    bonus: float = 0.0  # extra effective hammers accrued from RowPress dwell
    flipped: bool = False
    flipped_bits: int = 0


@dataclass
class DisturbanceResult:
    new_flips: int = 0
    oracle_refreshes: int = 0
    public_flips: list[dict[str, int | str]] = field(default_factory=list)


class DisturbanceEngine:
    """Profile-driven read-disturbance overlay driven by issued DRAM commands.

    Exposure is accumulated from the *actual post-schedule* commands the
    Ramulator worker emits (SPEC §4/§5), keyed by decoded physical
    ``(channel, rank, bankgroup, bank, row)``. Compared with the P11 model this
    engine makes the fidelity concrete (P14, defects B/C/E):

    * **Stratum selection** — single- vs double-sided is inferred from the issued
      ACT neighbourhood of each victim, and the data pattern (``all_ones`` /
      ``all_zeros``) from the victim region's written bytes; the matching profile
      stratum drives the threshold, not a constructor default.
    * **Hierarchical thresholds** — thresholds are drawn from the profile's
      ``mu + module_offset(σ_between) + row_eps(σ_within)`` model, so a whole
      module shares a latent offset and rows vary within it.
    * **Direction & multiplicity** — flip direction follows the stored value and
      the profile's dominant bias; the number of flipped bits grows with
      exposure per the profile ``multiplicity`` curve.
    * **RowPress** — long open-row dwell (ACT→PRE spacing) reduces the effective
      hcfirst for RowPress-supporting profiles only.
    * **Blast radius** — the victim set is the standard's physical topology
      (``±1``, optionally ``±2`` half-double) over decoded rows, and each victim's
      linear anchor is its *own* column-0 address under the active mapper,
      resolved through ``row_encoder``.
    * **Refresh/decay + oracle** — auto-refresh events decay exposure over a
      rolling refresh window; the ``oracle`` mitigation is a faithful
      target-row-refresh model (Ramulator ``OracleRH``: per-aggressor ACT
      counter, victim-row refresh at ``tRH``, counters cleared on all-bank
      refresh).

    The *known target* row keeps a fixed, calibrated threshold and a single
    bit-0 / ``0->1`` first flip so the deterministic phase-4..9 fixtures stay
    reproducible; every other row is fully profile-sampled.
    """

    def __init__(
        self,
        *,
        geometry: Geometry,
        row_encoder: RowEncoder,
        seed: int = 0,
        family: str = "hisasa",
        stratum: str = "double|all_zeros",
        known_target_row: int = 10,
        known_target_channel: int = 0,
        known_target_rank: int = 0,
        known_target_bank: int = 0,
        known_target_bankgroup: int = 0,
        known_first_bit: int = 0,
        mitigation: str = "none",
        mitigation_params: dict[str, Any] | None = None,
        temperature: int = 50,
        profile_id: str = "ddr4_vts25_v1",
    ) -> None:
        self.profile = load_profile(profile_id)
        self.profile_id = profile_id
        if not self.profile["validation"]["passed"]:
            raise ValueError("profile validation did not pass")
        # Standard is profile-driven, not DDR4-gated (P15, defect F). The profile's
        # standard must have an admitted read-disturbance adapter, and the geometry
        # the worker reports must be of that same standard — a profile fitted on one
        # standard is never run against another's geometry (no parameter pooling).
        self.standard = str(self.profile["standard"])
        if geometry.standard != self.standard:
            raise ValueError(
                f"PROFILE_REJECTED:profile standard {self.standard} does not match "
                f"geometry standard {geometry.standard}"
            )
        # Half-double ±2 coupling is added only when the profile characterises it
        # (schema-v2 ``topology`` block; absent/False for the DDR4 VTS25 profile).
        topology = self.profile.get("topology") or {}
        half_double = bool(topology.get("half_double_supported", False))
        try:
            self.standard_model = StandardModel.from_geometry(geometry, half_double=half_double)
        except UnsupportedStandard as exc:
            raise ValueError(str(exc))
        domain = self.profile["domain"]
        if domain["temperature_extrapolation"] or domain["timing_extrapolation"]:
            raise ValueError("profile extrapolation is not admitted")
        # Temperature/domain bounds (SPEC §5.6, test D9): out-of-domain
        # temperatures fail closed unless the profile advertises extrapolation.
        supported_temps = set(domain["supported_temperatures_celsius"])
        if temperature not in supported_temps and not domain["temperature_extrapolation"]:
            raise ValueError(
                f"PROFILE_REJECTED:temperature {temperature}C outside admitted domain "
                f"{sorted(supported_temps)}"
            )
        self.temperature = temperature

        self.geometry = geometry
        # Fail closed on a standard whose command vocabulary this model does not
        # fully account for, rather than silently dropping the unknown commands.
        unclassified = unclassified_commands(geometry.command_names)
        if unclassified:
            raise ValueError(
                f"disturbance model does not classify {geometry.standard} command(s) {unclassified}; "
                f"extend classify_command before admitting this standard"
            )
        # Standard-specific parameters, all derived from the real geometry + the
        # standard adapter (not DDR4 constants): the linear stride between adjacent
        # physical rows, the blast neighbourhood, and the refresh-window length.
        self.row_bytes = geometry.row_stride
        # Bytes one physical row occupies inside its own bank. The stride above is
        # *not* that size: it steps over every bank at a row index, so it is the
        # right unit for "the next row along" and the wrong one for "still inside
        # this row". Cell offsets and the victim anchor are bounded by this.
        self.row_span = geometry.row_span
        self.row_count = int(geometry.level_sizes["row"])
        self.tx_bytes = geometry.tx_bytes
        self.blast = list(self.standard_model.blast)
        self.refresh_window = self.standard_model.refresh_window

        self.seed = seed
        self.family = family
        self.stratum = stratum
        capability = require_admitted_mitigation(mitigation)
        self.mitigation = capability.name
        params = mitigation_params or {}
        # OracleRH's activation-count threshold. Default to a conservative
        # fraction of the calibrated double-sided hcfirst so the victim-row
        # refresh always fires before the row's exposure can cross (matches the
        # phase-7 protection fixture); overridable per task.
        self.tRH = int(params.get("tRH", max(1, self.known_threshold * 2 // 5)))

        self.known_target_row = known_target_row
        # Decoded (channel, rank, bankgroup, bank) of the known-target victim. All
        # zero by default (the RoBaRaCoCh victim's row-aligned linear address decodes
        # to channel/rank/bankgroup/bank 0); a secret mapper (P24) scrambles the
        # victim into another bank and a multi-rank part can place it off rank 0, so
        # the fixed calibrated threshold is pinned to the decoded coordinates the
        # compiler resolved instead of an assumed bank 0.
        self.known_target_channel = int(known_target_channel)
        self.known_target_rank = int(known_target_rank)
        self.known_target_bank = int(known_target_bank)
        self.known_target_bankgroup = int(known_target_bankgroup)
        # First bit the known target flips (bit 0 by default); the task compiler
        # sets it so target-cell / pattern objectives can pin a specific bit.
        self.known_first_bit = int(known_first_bit) & 0x7
        self._row_encoder = row_encoder
        # Column-0 linear address per decoded row key, memoized: a row's address is
        # needed only the first time the row is seen, so an episode makes one
        # encoder call per touched row rather than one per ACT.
        self._row_addrs: dict[tuple[int, ...], int] = {}
        self.victims: dict[tuple[int, ...], Victim] = {}
        self.flips: dict[int, int] = {}
        # Decoded (channel, rank, bankgroup, bank, row) keys of every victim that has
        # flipped. Mapper-agnostic (keys come from decoded issued events), so the
        # discovery-family success predicate can read the trusted decoded victim key
        # instead of ``linear // row_bytes``, which only holds under RoBaRaCoCh (P24).
        self.flipped_row_keys: set[tuple[int, ...]] = set()
        # Written data pattern per decoded row key, used for stratum + direction.
        self._row_pattern: dict[tuple[int, ...], str] = {}
        # Per-aggressor-row ACT counter for the oracle target-row-refresh model.
        self._oracle_acts: dict[tuple[int, ...], int] = {}
        # Open-row bookkeeping for RowPress dwell (per bank).
        self._open: dict[tuple[int, ...], tuple[int, int, list[tuple[int, ...]]]] = {}
        # All-bank auto-refresh command count per rank (drives window decay).
        self._refresh_count: dict[tuple[int, ...], int] = {}
        # Latent per-(family,stratum) module offset, drawn once (σ_between).
        self._module_offsets: dict[tuple[str, str], float] = {}

    # ---- calibration helpers -------------------------------------------------

    @property
    def target_addr(self) -> int:
        return self._row_addr(self._known_target_key())

    @property
    def known_threshold(self) -> int:
        return int(self._stratum()["quantiles"]["min"])

    @property
    def known_single_threshold(self) -> int:
        aggr, pattern = self._split_stratum(self.stratum)
        return int(self._stratum_dict("single", pattern)["quantiles"]["min"])

    def _known_target_key(self) -> tuple[int, ...]:
        # The known target's decoded key, in the same (channel, rank, bankgroup,
        # bank, row) shape the issued events decode to.
        return (
            self.known_target_channel,
            self.known_target_rank,
            self.known_target_bankgroup,
            self.known_target_bank,
            self.known_target_row,
        )

    # ---- address resolution --------------------------------------------------

    def _row_addr(self, key: tuple[int, ...]) -> int:
        """The column-0 linear address of a decoded row key under the active mapper.

        Every linear address the overlay records is anchored here, so a flip
        recorded for a victim key always reads back at that key's real address —
        including under the per-episode secret row->bank mapping, where the
        neighbouring row *in the same bank* is not the aggressor's address plus a
        row stride. # @spec:sim-exposure-flip
        """
        addr = self._row_addrs.get(key)
        if addr is None:
            channel, rank, bankgroup, bank, row = key
            addr = int(
                self._row_encoder(
                    {
                        "channel": channel,
                        "rank": rank,
                        "bankgroup": bankgroup,
                        "bank": bank,
                        "row": row,
                        "column": 0,
                    }
                )
            )
            self._row_addrs[key] = addr
        return addr

    def _row_exists(self, row: int) -> bool:
        """Whether ``row`` is a row of this device (the blast radius can run off the end)."""
        return 0 <= row < self.row_count

    # ---- public overlay API --------------------------------------------------

    def apply(self, addr: int, data: bytes) -> bytes:
        out = bytearray(data)
        for i in range(len(out)):
            bit = self.flips.get(addr + i)
            if bit is not None:
                out[i] ^= 1 << bit
        return bytes(out)

    def restore(self, addr: int, size: int) -> None:
        for i in range(size):
            self.flips.pop(addr + i, None)

    def note_write(self, row_key: tuple[int, ...], data: bytes) -> None:
        """Record the data pattern a write leaves in a physical row.

        Stratum selection (SPEC §5.3, test D6) keys off the ``all_ones`` /
        ``all_zeros`` pattern of the victim region; fresh seeded memory is all
        zeros, and a write of uniform bytes updates the row's pattern.

        Takes the write's *decoded* ``(channel, rank, bankgroup, bank, row)`` key
        rather than its linear address: which row an address belongs to is a
        property of the active mapper, which only the worker knows.
        """
        if not data:
            return
        if all(b == 0x00 for b in data):
            self._row_pattern[row_key] = "all_zeros"
        elif all(b == 0xFF for b in data):
            self._row_pattern[row_key] = "all_ones"
        else:
            self._row_pattern[row_key] = "mixed"

    # ---- event consumption ---------------------------------------------------

    def consume(
        self, events: list[dict[str, Any]], request: dict[str, Any] | None = None
    ) -> DisturbanceResult:
        """Fold one worker response's issued events into disturbance state.

        ACT commands drive exposure and the oracle counter; precharges (any
        ``PRE*``) and the auto-precharge accesses ``RDA``/``WRA`` close the rows in
        their scope and settle the RowPress dwell; REF/RFM commands decay exposure
        over the refresh window; a WRITE restores the cells it overwrote.

        ``request`` is used only for that restore: every address the overlay records
        comes from the events' own decoded coordinates, never from the address the
        request happened to carry.
        """
        result = DisturbanceResult()
        for event in events:
            command_class = classify_command(str(event.get("op", "")))
            if command_class == "hammer":
                self._hammer(event, result)
            elif command_class == "close":
                self._close_bank(event, result)
            elif command_class == "refresh":
                self._refresh(event, result)

        if request is not None and request.get("op") == "WR":
            addr = int(request.get("addr", 0))
            size = int(request.get("size", 0))
            self.restore(addr, size)
        return result

    # ---- exposure accounting -------------------------------------------------

    def _hammer(self, act_event: dict[str, Any], result: DisturbanceResult) -> None:
        row = int(act_event.get("row", -1))
        if row < 0:
            return
        channel = int(act_event.get("channel", 0))
        rank = int(act_event.get("rank", 0))
        bankgroup = int(act_event.get("bankgroup", 0))
        bank = int(act_event.get("bank", 0))
        bank_key = (channel, rank, bankgroup, bank)
        clk = int(act_event.get("clk", 0))

        # A new activation in this bank precharges whatever row was open — settle
        # that row's RowPress dwell before opening the new one.
        if bank_key in self._open and self._open[bank_key][0] != row:
            self._settle_dwell(bank_key, clk, result)

        credited: list[tuple[int, ...]] = []
        # An ACT on `row` exposes the standard's physical neighbourhood: the rows at
        # ±d *in this same bank*. Each victim is identified by its decoded key, and
        # its linear anchor is resolved from that key through the active mapper — so
        # the anchor is independent of the address the activating access carried, and
        # correct under a mapper that scatters consecutive rows across banks.
        # @spec:sim-exposure-flip
        for distance, weight in self.blast:
            for victim_row, side in (
                (row - distance, "right"),
                (row + distance, "left"),
            ):
                if not self._row_exists(victim_row):
                    continue
                key = (channel, rank, bankgroup, bank, victim_row)
                victim = self._victim(key)
                if side == "left":
                    victim.left += weight
                else:
                    victim.right += weight
                credited.append(key)
                self._maybe_flip(key, victim, result)

        # Track the newly-opened row for RowPress dwell accounting.
        self._open[bank_key] = (row, clk, credited)

        # Oracle target-row-refresh (Ramulator OracleRH): count this aggressor's
        # activations and, on crossing tRH, refresh its victim rows.
        if self.mitigation == "oracle":
            self._oracle_on_act(channel, rank, bankgroup, bank, row, result)

    def _maybe_flip(self, key: tuple[int, ...], victim: Victim, result: DisturbanceResult) -> None:
        if victim.flipped and victim.flipped_bits >= MAX_FLIPPED_BITS_PER_ROW:
            return
        double = victim.left > 0 and victim.right > 0
        if double:
            exposure = min(victim.left, victim.right) * 2 + victim.bonus
            threshold = victim.threshold
        else:
            exposure = max(victim.left, victim.right) + victim.bonus
            threshold = victim.single_threshold
        if exposure < threshold:
            return
        self._flip(key, victim, exposure / threshold, result)

    def _flip(self, key: tuple[int, ...], victim: Victim, ratio: float, result: DisturbanceResult) -> None:
        n_bits = self._multiplicity_bits(victim, ratio)
        if n_bits <= victim.flipped_bits:
            return
        for byte_off, bit in self._flip_positions(key, victim, n_bits):
            self.flips[victim.addr + byte_off] = bit
            victim.flipped_bits += 1
            result.new_flips += 1
            result.public_flips.append(
                {
                    "row": key[-1],
                    "addr": victim.addr + byte_off,
                    "bit": bit,
                    "direction": victim.direction,
                }
            )
        victim.flipped = True
        self.flipped_row_keys.add(key)

    # ---- RowPress dwell ------------------------------------------------------
    # @spec:sim-rowpress

    def _close_bank(self, event: dict[str, Any], result: DisturbanceResult) -> None:
        """Settle the RowPress dwell of every open row this precharge closes.

        A precharge's scope is its decoded coordinates, where -1 means "every
        node at this level" (Ramulator's ``BankTarget::All`` addressing): DDR4's
        per-bank ``PREpb`` pins bankgroup and bank, while the all-bank ``PREab``
        that precedes each auto-refresh arrives rank-scoped with
        ``bankgroup``/``bank`` of -1 and closes every open bank in that rank.
        """
        clk = int(event.get("clk", 0))
        scope = (
            int(event.get("channel", 0)),
            int(event.get("rank", 0)),
            int(event.get("bankgroup", 0)),
            int(event.get("bank", 0)),
        )
        closed = [
            bank_key
            for bank_key in self._open
            if all(s < 0 or s == coord for s, coord in zip(scope, bank_key))
        ]
        for bank_key in closed:
            self._settle_dwell(bank_key, clk, result)

    def _settle_dwell(self, bank_key: tuple[int, ...], close_clk: int, result: DisturbanceResult) -> None:
        row, act_clk, credited = self._open.pop(bank_key)
        dwell = close_clk - act_clk
        factor = self._rowpress_factor(dwell)
        if factor <= 1.0:
            return
        # A long-open activation is worth ``factor`` ordinary hammers; credit the
        # extra ``factor - 1`` to every victim it exposed and re-check for flips.
        for key in credited:
            victim = self.victims.get(key)
            if victim is None or (victim.flipped and victim.flipped_bits >= MAX_FLIPPED_BITS_PER_ROW):
                continue
            victim.bonus += factor - 1.0
            self._maybe_flip(key, victim, result)

    def _rowpress_factor(self, dwell: int) -> float:
        """Effective hammer multiplier for an activation held open ``dwell`` cycles."""
        if dwell <= ROWPRESS_DWELL_NOMINAL:
            return 1.0
        rp = self._family_fit()["rowpress"]
        if not rp.get("supported", False):
            return 1.0
        reduction = float(rp["rowhammer_to_rowpress_hc_reduction"])
        span = ROWPRESS_DWELL_SATURATION - ROWPRESS_DWELL_NOMINAL
        frac = min(1.0, (dwell - ROWPRESS_DWELL_NOMINAL) / span)
        return 1.0 + (reduction - 1.0) * frac

    # ---- refresh / decay -----------------------------------------------------

    def _refresh(self, event: dict[str, Any], result: DisturbanceResult) -> None:
        """Decay accumulated exposure over the auto-refresh window (SPEC §5.7).

        One all-bank refresh restores ~1/8192 of the rows; a full window of
        ``self.refresh_window`` refreshes (the standard's JEDEC refresh divisor)
        restores every row exactly once. At each window boundary the rank's
        accumulated activations are
        cleared: a hammer slow enough to straddle the boundary is refreshed away
        and must re-accumulate, while a fast burst that crosses its threshold
        inside a single window still flips. Already-flipped cells are *not*
        corrected (test D10): a refresh rewrites the disturbed value it reads.
        """
        channel = int(event.get("channel", 0))
        rank = int(event.get("rank", 0))
        rank_key = (channel, rank)
        count = self._refresh_count.get(rank_key, 0) + 1
        self._refresh_count[rank_key] = count
        if count % self.refresh_window != 0:
            return
        for key, victim in self.victims.items():
            if key[0] == channel and key[1] == rank:
                victim.left = 0
                victim.right = 0
                victim.bonus = 0.0
        # OracleRH clears its per-row activation counters when a refresh restores
        # the rows (see oracle_rh.cpp); at a window boundary every row is covered.
        if self.mitigation == "oracle":
            for agg_key in list(self._oracle_acts):
                if agg_key[0] == channel and agg_key[1] == rank:
                    self._oracle_acts[agg_key] = 0

    # ---- oracle target-row refresh (Ramulator OracleRH) ----------------------

    def _oracle_on_act(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        row: int,
        result: DisturbanceResult,
    ) -> None:
        """Faithful port of Ramulator ``OracleRH::on_issue`` (ACT branch).

        Per aggressor (bank, row) activation counter; on reaching ``tRH`` issue a
        victim-row refresh — restoring the aggressor's physical neighbours (its
        RowHammer victims) and resetting the counter. See
        ``third_party/ramulator2/src/ramulator/controller/plugin/impl/oracle_rh.cpp``.
        """
        agg_key = (channel, rank, bankgroup, bank, row)
        count = self._oracle_acts.get(agg_key, 0) + 1
        if count < self.tRH:
            self._oracle_acts[agg_key] = count
            return
        self._oracle_acts[agg_key] = 0
        # VRR restores the aggressor's neighbouring victim rows.
        for distance, _weight in self.blast:
            for victim_row in (row - distance, row + distance):
                if not self._row_exists(victim_row):
                    continue
                key = (channel, rank, bankgroup, bank, victim_row)
                victim = self.victims.get(key)
                if victim is None:
                    continue
                victim.left = 0
                victim.right = 0
                victim.bonus = 0.0
        result.oracle_refreshes += 1

    # ---- latent state sampling ----------------------------------------------

    def _victim(self, key: tuple[int, ...]) -> Victim:
        victim = self.victims.get(key)
        if victim is not None:
            return victim
        victim_addr = self._row_addr(key)
        pattern = self._victim_pattern(key)
        aggr_pattern = pattern if pattern in ("all_ones", "all_zeros") else "all_zeros"
        if key == self._known_target_key():
            double_threshold = self.known_threshold
            single_threshold = self.known_single_threshold
            first_bit = self.known_first_bit
        else:
            double_threshold = self._sample_threshold(key, "double", aggr_pattern)
            single_threshold = self._sample_threshold(key, "single", aggr_pattern)
            first_bit = self._sample_first_bit(key)
        direction = "1->0" if aggr_pattern == "all_ones" else "0->1"
        victim = Victim(
            threshold=double_threshold,
            single_threshold=single_threshold,
            direction=direction,
            addr=victim_addr,
            data_pattern=aggr_pattern,
            first_bit=first_bit,
        )
        self.victims[key] = victim
        return victim

    def _victim_pattern(self, key: tuple[int, ...]) -> str:
        return self._row_pattern.get(key, "all_zeros")

    def _module_offset(self, family: str, stratum: str) -> float:
        cached = self._module_offsets.get((family, stratum))
        if cached is not None:
            return cached
        params = self.profile["fit"]["families"][family]["strata"][stratum]["hcfirst_lognormal"]
        sigma_between = float(params.get("sigma_between_chip", 0.0))
        digest = hashlib.sha256(f"{self.seed}:{family}:{stratum}:module".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        offset = rng.gauss(0.0, sigma_between)
        self._module_offsets[(family, stratum)] = offset
        return offset

    def _sample_threshold(self, key: tuple[int, ...], aggr: str, pattern: str) -> int:
        """Draw a row threshold from the hierarchical profile model (SPEC §5.4).

        ``log N = mu + module_offset + row_eps`` with
        ``module_offset ~ Normal(0, sigma_between_chip)`` shared across the module
        and ``row_eps ~ Normal(0, sigma_within_chip)`` per row.
        """
        stratum = f"{aggr}|{pattern}"
        params = self.profile["fit"]["families"][self.family]["strata"][stratum]["hcfirst_lognormal"]
        mu = float(params["mu"])
        sigma_within = float(params.get("sigma_within_chip", params["sigma"]))
        offset = self._module_offset(self.family, stratum)
        digest = hashlib.sha256(f"{self.seed}:{self.family}:{stratum}:{key}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        row_eps = rng.gauss(0.0, sigma_within)
        return max(1, round(math.exp(mu + offset + row_eps)))

    def _sample_first_bit(self, key: tuple[int, ...]) -> int:
        digest = hashlib.sha256(f"{self.seed}:{self.family}:bit:{key}".encode()).digest()
        return int.from_bytes(digest[:2], "big") % 8

    def _multiplicity_bits(self, victim: Victim, ratio: float) -> int:
        """Number of bits flipped in a victim row at exposure/threshold ``ratio``.

        Exactly one bit flips as the row crosses its hcfirst; further hammering
        recruits more cells at a rate set by the profile ``multiplicity`` curve
        for the row's stratum (denser-flipping strata recruit faster).
        """
        double = victim.left > 0 and victim.right > 0
        aggr = "double" if double else "single"
        mult = self._family_fit()["multiplicity"].get(f"{aggr}|{victim.data_pattern}")
        if not mult:
            return 1
        nrh, mean_flips = mult[0]
        gain = 1.0 + (float(mean_flips) / float(nrh)) * 200.0
        extra = int(max(0.0, ratio - 1.0) * gain)
        return min(1 + extra, MAX_FLIPPED_BITS_PER_ROW)

    def _flip_positions(self, key: tuple[int, ...], victim: Victim, n_bits: int) -> list[tuple[int, int]]:
        """Deterministic (byte_offset, bit) positions for the first ``n_bits`` flips.

        Cell 0 is always column 0 / ``first_bit`` (bit 0 for the known target);
        further cells are drawn pseudo-randomly and distinctly across the victim
        row. Offsets are bounded by ``row_span`` — the row's own size — and not by
        the row *stride*, which spans every bank at this row index and would place
        the extra cells in other banks' rows entirely. The full sequence is
        regenerated from the seed each call, so it is stable as multiplicity grows;
        only the cells beyond the ones already flipped are returned.
        """
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{self.seed}:mult:{key}".encode()).digest()[:8], "big"))
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        idx = 0
        while len(cells) < n_bits:
            cell = (0, victim.first_bit) if idx == 0 else (rng.randrange(self.row_span), rng.randrange(8))
            idx += 1
            if cell in seen:
                continue
            seen.add(cell)
            cells.append(cell)
        return cells[victim.flipped_bits:n_bits]

    # ---- profile access helpers ---------------------------------------------

    def _family_fit(self) -> dict[str, Any]:
        return self.profile["fit"]["families"][self.family]

    def _stratum(self) -> dict[str, Any]:
        return self._family_fit()["strata"][self.stratum]

    def _stratum_dict(self, aggr: str, pattern: str) -> dict[str, Any]:
        return self._family_fit()["strata"][f"{aggr}|{pattern}"]

    @staticmethod
    def _split_stratum(stratum: str) -> tuple[str, str]:
        aggr, pattern = stratum.split("|", 1)
        return aggr, pattern
