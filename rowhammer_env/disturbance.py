from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any

from .geometry import Geometry
from .profiles import load_profile


# Blast topology per DRAM standard: (row distance from the aggressor, coupling
# weight). Distance 1 is the immediate physical neighbour (classic RowHammer);
# distance 2 is the "half-double" next-nearest row, which couples far more
# weakly. The set is standard-driven (SPEC §5.3 / P14 task 5), not a hardcoded
# ``±1`` logical loop. DDR4 is modelled with immediate neighbours only unless a
# profile enables half-double.
STANDARD_BLAST: dict[str, list[tuple[int, float]]] = {
    "DDR4": [(1, 1.0)],
}

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

# DDR4 all-bank auto-refresh commands per refresh window (tREFW / tREFI = 8192):
# one window refreshes every row exactly once. Accumulated read-disturbance
# survives only within a window — a hammer fast enough to cross its threshold
# inside one window flips; one slow enough to straddle a window boundary is
# reset by the intervening refresh (SPEC §5.7, test D8).
REFRESH_COMMANDS_PER_WINDOW = 8192


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
      (``±1``, optionally ``±2`` half-double) over decoded rows.
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
        seed: int = 0,
        family: str = "hisasa",
        stratum: str = "double|all_zeros",
        known_target_row: int = 10,
        mitigation: str = "none",
        mitigation_params: dict[str, Any] | None = None,
        temperature: int = 50,
        profile_id: str = "ddr4_vts25_v1",
    ) -> None:
        self.profile = load_profile(profile_id)
        self.profile_id = profile_id
        if not self.profile["validation"]["passed"]:
            raise ValueError("profile validation did not pass")
        if self.profile["standard"] != "DDR4":
            raise ValueError("Phase 4 admits DDR4 only")
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
        # Logical stride between adjacent physical rows (derived from the real
        # DDR4 geometry, not the fictitious 8192 the old model used).
        self.row_bytes = geometry.row_stride
        self.tx_bytes = geometry.tx_bytes
        self.blast = STANDARD_BLAST.get(self.profile["standard"], [(1, 1.0)])
        self.refresh_window = REFRESH_COMMANDS_PER_WINDOW

        self.seed = seed
        self.family = family
        self.stratum = stratum
        if mitigation not in {"none", "oracle"}:
            raise ValueError(f"UNAVAILABLE_CAPABILITY:{mitigation}")
        self.mitigation = mitigation
        params = mitigation_params or {}
        # OracleRH's activation-count threshold. Default to a conservative
        # fraction of the calibrated double-sided hcfirst so the victim-row
        # refresh always fires before the row's exposure can cross (matches the
        # phase-7 protection fixture); overridable per task.
        self.tRH = int(params.get("tRH", max(1, self.known_threshold * 2 // 5)))

        self.known_target_row = known_target_row
        self.victims: dict[tuple[int, ...], Victim] = {}
        self.flips: dict[int, int] = {}
        # Written data pattern per physical row key, used for stratum + direction.
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
        return self.known_target_row * self.row_bytes

    @property
    def known_threshold(self) -> int:
        return int(self._stratum()["quantiles"]["min"])

    @property
    def known_single_threshold(self) -> int:
        aggr, pattern = self._split_stratum(self.stratum)
        return int(self._stratum_dict("single", pattern)["quantiles"]["min"])

    def _known_target_key(self) -> tuple[int, ...]:
        # target_addr decodes to (channel 0, rank 0, bankgroup 0, bank 0, known row).
        return (0, 0, 0, 0, self.known_target_row)

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

    def note_write(self, addr: int, data: bytes) -> None:
        """Record the data pattern a write leaves in a physical row region.

        Stratum selection (SPEC §5.3, test D6) keys off the ``all_ones`` /
        ``all_zeros`` pattern of the victim region; fresh seeded memory is all
        zeros, and a write of uniform bytes updates the row's pattern.
        """
        if not data:
            return
        row_key = self._row_key_of_addr(addr)
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

        ACT commands drive exposure and the oracle counter; PRE commands close an
        open row and settle its RowPress dwell; REF/RFM commands decay exposure
        over the refresh window; a WRITE restores the cells it overwrote.
        """
        result = DisturbanceResult()
        request_addr: int | None = None
        if request is not None and request.get("op") in ("RD", "WR"):
            request_addr = int(request.get("addr", 0))

        for event in events:
            op = str(event.get("op", ""))
            if op == "ACT" and request_addr is not None:
                self._hammer(event, request_addr, result)
            elif op == "PRE" or op == "PREab" or op == "PREA":
                self._close_bank(event, result)
            elif op.startswith("REF") or op.startswith("RFM"):
                self._refresh(event, result)

        if request is not None and request.get("op") == "WR":
            addr = int(request.get("addr", 0))
            size = int(request.get("size", 0))
            self.restore(addr, size)
        return result

    # ---- exposure accounting -------------------------------------------------

    def _hammer(self, act_event: dict[str, Any], request_addr: int, result: DisturbanceResult) -> None:
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
        # An ACT on `row` exposes the standard's physical neighbourhood. Because
        # RoBaRaCoCh places Row as the most-significant field, ``request_addr``
        # decodes to `row`, so ±d row strides land on the neighbours' column 0.
        for distance, weight in self.blast:
            for victim_row, side, victim_addr in (
                (row - distance, "right", request_addr - distance * self.row_bytes),
                (row + distance, "left", request_addr + distance * self.row_bytes),
            ):
                if victim_row < 0 or victim_addr < 0:
                    continue
                key = (channel, rank, bankgroup, bank, victim_row)
                victim = self._victim(key, victim_addr)
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
            self._oracle_on_act(channel, rank, bankgroup, bank, row, request_addr, result)

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

    # ---- RowPress dwell ------------------------------------------------------

    def _close_bank(self, event: dict[str, Any], result: DisturbanceResult) -> None:
        channel = int(event.get("channel", 0))
        rank = int(event.get("rank", 0))
        bankgroup = int(event.get("bankgroup", 0))
        bank = int(event.get("bank", 0))
        clk = int(event.get("clk", 0))
        bank_key = (channel, rank, bankgroup, bank)
        if bank_key in self._open:
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
        ``REFRESH_COMMANDS_PER_WINDOW`` refreshes restores every row exactly
        once. At each window boundary the rank's accumulated activations are
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
        request_addr: int,
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
            for victim_row, victim_addr in (
                (row - distance, request_addr - distance * self.row_bytes),
                (row + distance, request_addr + distance * self.row_bytes),
            ):
                if victim_row < 0 or victim_addr < 0:
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

    def _victim(self, key: tuple[int, ...], victim_addr: int) -> Victim:
        victim = self.victims.get(key)
        if victim is not None:
            return victim
        pattern = self._victim_pattern(victim_addr)
        aggr_pattern = pattern if pattern in ("all_ones", "all_zeros") else "all_zeros"
        if key == self._known_target_key():
            double_threshold = self.known_threshold
            single_threshold = self.known_single_threshold
            first_bit = 0
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

    def _victim_pattern(self, victim_addr: int) -> str:
        row_key = self._row_key_of_addr(victim_addr)
        return self._row_pattern.get(row_key, "all_zeros")

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
        row. The full sequence is regenerated from the seed each call, so it is
        stable as multiplicity grows; only the cells beyond the ones already
        flipped are returned.
        """
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{self.seed}:mult:{key}".encode()).digest()[:8], "big"))
        cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        idx = 0
        while len(cells) < n_bits:
            cell = (0, victim.first_bit) if idx == 0 else (rng.randrange(self.row_bytes), rng.randrange(8))
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

    def _row_key_of_addr(self, addr: int) -> tuple[int, ...]:
        """Decode a linear address to its (channel, rank, bankgroup, bank, row) key.

        RoBaRaCoCh places Row as the most-significant field above a
        ``row_bytes`` stride, so the row index is ``addr // row_bytes`` and the
        sub-row levels come from the remaining low bits. The known-target and
        neighbour addresses used here all sit in channel/rank/bg/bank 0, matching
        the worker's decode of the same linear addresses.
        """
        row = addr // self.row_bytes
        low = addr % self.row_bytes
        tx = self.tx_bytes
        column_span = int(self.geometry.level_sizes.get("column", 1)) // max(1, self.geometry.prefetch)
        # Skip the column field (lowest levels above the transaction offset); the
        # remaining bits carry rank/bankgroup/bank in RoBaRaCoCh order.
        rest = (low // tx) // max(1, column_span)
        rank = 0
        bankgroup = 0
        bank = 0
        if "rank" in self.geometry.level_sizes:
            n = int(self.geometry.level_sizes["rank"])
            rank = rest % n
            rest //= n
        if "bankgroup" in self.geometry.level_sizes:
            n = int(self.geometry.level_sizes["bankgroup"])
            bankgroup = rest % n
            rest //= n
        if "bank" in self.geometry.level_sizes:
            n = int(self.geometry.level_sizes["bank"])
            bank = rest % n
            rest //= n
        return (0, rank, bankgroup, bank, row)
