from __future__ import annotations

import base64
from typing import Any

from .disturbance import DisturbanceEngine
from .geometry import Geometry
from .phase2_env import Phase2Observation, RowHammerEnv
from .worker_protocol import WorkerRequest


class RowHammerDisturbanceEnv(RowHammerEnv):
    def __init__(
        self,
        *args: Any,
        disturbance: DisturbanceEngine | None = None,
        mitigation: dict[str, Any] | None = None,
        profile_id: str = "ddr4_vts25_v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.disturbance = disturbance
        self.mitigation = mitigation or {"name": "none", "params": {}}
        self.profile_id = profile_id

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> Phase2Observation:
        obs = super().reset(seed=seed, episode_id=episode_id, **kwargs)
        if obs.error:
            return obs
        try:
            geometry = self._fetch_geometry()
        except RuntimeError as exc:
            self.close()
            return self._error("INTERNAL_SIMULATOR_ERROR", str(exc))
        try:
            self.disturbance = DisturbanceEngine(
                geometry=geometry,
                seed=seed or 0,
                mitigation=self.mitigation["name"],
                profile_id=self.profile_id,
            )
        except ValueError as exc:
            if str(exc).startswith("UNAVAILABLE_CAPABILITY:"):
                self.close()
                return self._error("UNAVAILABLE_CAPABILITY", str(exc).split(":", 1)[1])
            if str(exc).startswith("PROFILE_REJECTED:"):
                self.close()
                return self._error("PROFILE_REJECTED", str(exc).split(":", 1)[1])
            raise
        obs.metadata["profile"] = self.disturbance.profile_id
        obs.metadata["mitigation"] = self.mitigation
        obs.metadata["disturbance"] = {
            "family": self.disturbance.family,
            "stratum": self.disturbance.stratum,
            "known_target_row": self.disturbance.known_target_row,
            "known_threshold": self.disturbance.known_threshold,
        }
        return obs

    def _fetch_geometry(self) -> Geometry:
        if self._worker is None:
            raise RuntimeError("worker is unavailable")
        payload = self._worker.call(WorkerRequest("INFO", "info", ()))
        if not payload.get("ok") or "geometry" not in payload:
            err = payload.get("error") or {}
            raise RuntimeError(err.get("message", "worker did not report geometry"))
        return Geometry(payload["geometry"])

    def _from_worker(self, payload: dict[str, Any]) -> Phase2Observation:
        obs = super()._from_worker(payload)
        if obs.error or self.disturbance is None:
            return obs

        request = payload.get("request", {})
        result = self.disturbance.consume(payload.get("events", []), request)
        if obs.data_b64 and request.get("op") == "RD":
            raw = base64.b64decode(obs.data_b64)
            flipped = self.disturbance.apply(int(request["addr"]), raw)
            obs.data_b64 = base64.b64encode(flipped).decode()

        obs.feedback["new_public_flips"] = result.new_flips
        obs.feedback["oracle_refreshes"] = result.oracle_refreshes
        if result.public_flips:
            obs.feedback["public_flips"] = result.public_flips
        return obs
