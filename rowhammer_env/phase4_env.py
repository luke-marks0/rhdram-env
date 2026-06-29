from __future__ import annotations

import base64
from typing import Any

from .disturbance import DisturbanceEngine
from .phase2_env import Phase2Observation, RowHammerEnv


class RowHammerDisturbanceEnv(RowHammerEnv):
    def __init__(
        self,
        *args: Any,
        disturbance: DisturbanceEngine | None = None,
        mitigation: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.disturbance = disturbance
        self.mitigation = mitigation or {"name": "none", "params": {}}

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> Phase2Observation:
        obs = super().reset(seed=seed, episode_id=episode_id, **kwargs)
        if obs.error:
            return obs
        try:
            self.disturbance = DisturbanceEngine(seed=seed or 0, mitigation=self.mitigation["name"])
        except ValueError as exc:
            if str(exc).startswith("UNAVAILABLE_CAPABILITY:"):
                self.close()
                return self._error("UNAVAILABLE_CAPABILITY", str(exc).split(":", 1)[1])
            raise
        obs.metadata["profile"] = "ddr4_vts25_v1"
        obs.metadata["mitigation"] = self.mitigation
        obs.metadata["disturbance"] = {
            "family": self.disturbance.family,
            "stratum": self.disturbance.stratum,
            "known_target_row": self.disturbance.known_target_row,
            "known_threshold": self.disturbance.known_threshold,
        }
        return obs

    def _from_worker(self, payload: dict[str, Any]) -> Phase2Observation:
        obs = super()._from_worker(payload)
        if obs.error or self.disturbance is None:
            return obs

        result = self.disturbance.consume(payload.get("events", []))
        if obs.data_b64 and payload.get("events"):
            event = payload["events"][-1]
            if event.get("op") == "RD":
                raw = base64.b64decode(obs.data_b64)
                flipped = self.disturbance.apply(int(event["addr"]), raw)
                obs.data_b64 = base64.b64encode(flipped).decode()

        obs.feedback["new_public_flips"] = result.new_flips
        obs.feedback["oracle_refreshes"] = result.oracle_refreshes
        if result.public_flips:
            obs.feedback["public_flips"] = result.public_flips
        return obs
