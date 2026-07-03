from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from .geometry import Geometry
from .profiles import load_profile


@dataclass
class Victim:
    threshold: int
    direction: str
    addr: int  # linear (logical) byte address of this victim row at column 0
    left: int = 0  # ACT count on the row immediately to this victim's left (row-1)
    right: int = 0  # ACT count on the row immediately to this victim's right (row+1)
    flipped: bool = False


@dataclass
class DisturbanceResult:
    new_flips: int = 0
    oracle_refreshes: int = 0
    public_flips: list[dict[str, int | str]] = field(default_factory=list)


class DisturbanceEngine:
    """Profile-backed read-disturbance overlay driven by issued ACT commands.

    Exposure is accumulated from the *actual post-schedule* ACT commands the
    Ramulator worker emits (SPEC §4/§5), keyed by decoded physical
    ``(channel, rank, bankgroup, bank, row)``. Reads that hit an already-open row
    issue no ACT and therefore contribute no disturbance — the row-buffer
    locality the previous frontend-completion model ignored. A victim flips once
    it has been activated from both physical neighbours enough times to cross its
    sampled threshold.

    Flips are stored keyed by the victim's linear (logical) address so the
    functional read/write overlay and the reward predicates stay address-based.
    Because RoBaRaCoCh places Row as the most-significant field, the victim's
    linear address is simply the aggressor's request address offset by one row
    stride, which decodes back to the same bank and the adjacent physical row.
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

        self.geometry = geometry
        # Logical stride between adjacent physical rows (derived from the real
        # DDR4 geometry, not the fictitious 8192 the old model used).
        self.row_bytes = geometry.row_stride
        self.tx_bytes = geometry.tx_bytes

        self.seed = seed
        self.family = family
        self.stratum = stratum
        self.known_target_row = known_target_row
        if mitigation not in {"none", "oracle"}:
            raise ValueError(f"UNAVAILABLE_CAPABILITY:{mitigation}")
        self.mitigation = mitigation
        self.victims: dict[tuple[int, ...], Victim] = {}
        self.flips: dict[int, int] = {}

    @property
    def target_addr(self) -> int:
        return self.known_target_row * self.row_bytes

    @property
    def known_threshold(self) -> int:
        return int(self._stratum()["quantiles"]["min"])

    def consume(self, events: list[dict[str, Any]], request: dict[str, Any] | None = None) -> DisturbanceResult:
        """Fold one worker response's issued events into disturbance state.

        ``events`` is the issued-command stream for a single frontend request;
        ``request`` describes that frontend request (``op``/``addr``/``size``).
        ACT commands drive exposure; a WRITE restores the cells it overwrote.
        REF/RFM restoration and decay are modeled in P14 and are no-ops here.
        """
        result = DisturbanceResult()
        request_addr: int | None = None
        if request is not None and request.get("op") in ("RD", "WR"):
            request_addr = int(request.get("addr", 0))

        for event in events:
            if event.get("op") == "ACT" and request_addr is not None:
                self._hammer(event, request_addr, result)

        if request is not None and request.get("op") == "WR":
            self.restore(int(request.get("addr", 0)), int(request.get("size", 0)))
        return result

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

    def _hammer(self, act_event: dict[str, Any], request_addr: int, result: DisturbanceResult) -> None:
        row = int(act_event.get("row", -1))
        if row < 0:
            return
        channel = int(act_event.get("channel", 0))
        rank = int(act_event.get("rank", 0))
        bankgroup = int(act_event.get("bankgroup", 0))
        bank = int(act_event.get("bank", 0))

        # An ACT that opens `row` exposes its two physically-adjacent neighbours;
        # `request_addr` decodes to `row`, so ±one row stride lands on them.
        for victim_row, side, victim_addr in (
            (row - 1, "right", request_addr - self.row_bytes),
            (row + 1, "left", request_addr + self.row_bytes),
        ):
            if victim_row < 0 or victim_addr < 0:
                continue
            key = (channel, rank, bankgroup, bank, victim_row)
            victim = self._victim(key, victim_addr)
            if side == "left":
                victim.left += 1
            else:
                victim.right += 1
            exposure = min(victim.left, victim.right) * 2
            if not victim.flipped and exposure >= victim.threshold:
                if self.mitigation == "oracle":
                    victim.left = 0
                    victim.right = 0
                    result.oracle_refreshes += 1
                    continue
                bit = 0
                self.flips[victim_addr] = bit
                victim.flipped = True
                result.new_flips += 1
                result.public_flips.append(
                    {"row": victim_row, "addr": victim_addr, "bit": bit, "direction": victim.direction}
                )

    def _victim(self, key: tuple[int, ...], victim_addr: int) -> Victim:
        victim = self.victims.get(key)
        if victim is not None:
            return victim
        threshold = self.known_threshold if key == self._known_target_key() else self._sample_threshold(key)
        victim = Victim(threshold=threshold, direction="0->1", addr=victim_addr)
        self.victims[key] = victim
        return victim

    def _known_target_key(self) -> tuple[int, ...]:
        # target_addr decodes to (channel 0, rank 0, bankgroup 0, bank 0, known row).
        return (0, 0, 0, 0, self.known_target_row)

    def _sample_threshold(self, key: tuple[int, ...]) -> int:
        params = self._stratum()["hcfirst_lognormal"]
        digest = hashlib.sha256(f"{self.seed}:{self.family}:{self.stratum}:{key}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return max(1, round(rng.lognormvariate(params["mu"], params["sigma"])))

    def _stratum(self) -> dict[str, Any]:
        return self.profile["fit"]["families"][self.family]["strata"][self.stratum]
