from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from profile_builder.package.build import verify_package


@dataclass
class Victim:
    threshold: int
    direction: str
    left: int = 0
    right: int = 0
    flipped: bool = False


@dataclass
class DisturbanceResult:
    new_flips: int = 0
    oracle_refreshes: int = 0
    public_flips: list[dict[str, int | str]] = field(default_factory=list)


class DisturbanceEngine:
    """Profile-backed read-disturbance flip overlay for Phase 4."""

    def __init__(
        self,
        *,
        seed: int = 0,
        row_bytes: int = 8192,
        family: str = "hisasa",
        stratum: str = "double|all_zeros",
        known_target_row: int = 10,
        mitigation: str = "none",
    ) -> None:
        self.profile = verify_package()
        if not self.profile["validation"]["passed"]:
            raise ValueError("profile validation did not pass")
        if self.profile["standard"] != "DDR4":
            raise ValueError("Phase 4 admits DDR4 only")
        domain = self.profile["domain"]
        if domain["temperature_extrapolation"] or domain["timing_extrapolation"]:
            raise ValueError("profile extrapolation is not admitted")

        self.seed = seed
        self.row_bytes = row_bytes
        self.family = family
        self.stratum = stratum
        self.known_target_row = known_target_row
        if mitigation not in {"none", "oracle"}:
            raise ValueError(f"UNAVAILABLE_CAPABILITY:{mitigation}")
        self.mitigation = mitigation
        self.victims: dict[int, Victim] = {}
        self.flips: dict[int, int] = {}

    @property
    def target_addr(self) -> int:
        return self.known_target_row * self.row_bytes

    @property
    def known_threshold(self) -> int:
        return int(self._stratum()["quantiles"]["min"])

    def consume(self, events: list[dict[str, Any]]) -> DisturbanceResult:
        result = DisturbanceResult()
        for event in events:
            op = event.get("op")
            addr = int(event.get("addr", 0))
            size = int(event.get("size", 0))
            if op == "RD":
                result = self._hammer(addr // self.row_bytes, result)
            elif op == "WR":
                self.restore(addr, size)
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

    def _hammer(self, row: int, result: DisturbanceResult) -> DisturbanceResult:
        for victim_row, side in ((row - 1, "right"), (row + 1, "left")):
            if victim_row < 0:
                continue
            victim = self._victim(victim_row)
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
                addr = victim_row * self.row_bytes
                self.flips[addr] = 0
                victim.flipped = True
                result.new_flips += 1
                result.public_flips.append(
                    {"row": victim_row, "addr": addr, "bit": 0, "direction": victim.direction}
                )
        return result

    def _victim(self, row: int) -> Victim:
        victim = self.victims.get(row)
        if victim is not None:
            return victim
        threshold = self.known_threshold if row == self.known_target_row else self._sample_threshold(row)
        victim = Victim(threshold=threshold, direction="0->1")
        self.victims[row] = victim
        return victim

    def _sample_threshold(self, row: int) -> int:
        params = self._stratum()["hcfirst_lognormal"]
        digest = hashlib.sha256(f"{self.seed}:{self.family}:{self.stratum}:{row}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return max(1, round(rng.lognormvariate(params["mu"], params["sigma"])))

    def _stratum(self) -> dict[str, Any]:
        return self.profile["fit"]["families"][self.family]["strata"][self.stratum]
