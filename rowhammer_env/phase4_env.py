from __future__ import annotations

import base64
from typing import Any

from .disturbance import DisturbanceEngine
from .geometry import Geometry
from .phase2_env import Phase2Observation, RowHammerEnv
from .tasks import AddressResolver, Disclosure, HandleTable
from .tools.addressing import AddressMapper
from .worker_protocol import WorkerRequest


# Default disclosure for the bare disturbance env (no task compiler): everything
# is visible. This keeps the physical address form and full issued-event trace
# available for the P11 event-stream checks and the P12 differential decode test.
FULL_DISCLOSURE = Disclosure(
    mapping="physical",
    adjacency="exact",
    victim="exact",
    profile="public_profile_id",
    feedback="full_trace",
)


class RowHammerDisturbanceEnv(RowHammerEnv):
    def __init__(
        self,
        *args: Any,
        disturbance: DisturbanceEngine | None = None,
        mitigation: dict[str, Any] | None = None,
        profile_id: str = "ddr4_vts25_v1",
        temperature: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.disturbance = disturbance
        self.mitigation = mitigation or {"name": "none", "params": {}}
        self.profile_id = profile_id
        self.temperature = temperature
        self.disclosure = FULL_DISCLOSURE
        self.address_mapper: AddressMapper | None = None

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
            self.address_mapper = AddressMapper(geometry)
        except ValueError as exc:
            self.close()
            return self._error("UNAVAILABLE_CAPABILITY", f"geometry not projectable: {exc}")
        # Build the resolver first so any address form the disclosure permits is
        # usable as soon as the episode is live; the task layer (P13) registers
        # handles into this same table after reset.
        self._resolver = AddressResolver(self.address_mapper, self.disclosure, HandleTable(seed or 0))
        try:
            self.disturbance = DisturbanceEngine(
                geometry=geometry,
                seed=seed or 0,
                mitigation=self.mitigation["name"],
                mitigation_params=self.mitigation.get("params"),
                temperature=self.temperature,
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
        obs.metadata["address_forms"] = sorted(self.disclosure.allowed_forms())
        obs.metadata["disclosure"] = self.disclosure.as_public()
        obs.metadata["disturbance"] = self._disturbance_metadata()
        return obs

    def _disturbance_metadata(self) -> dict[str, Any]:
        assert self.disturbance is not None
        meta: dict[str, Any] = {"family": self.disturbance.family, "stratum": self.disturbance.stratum}
        if self.disclosure.expose_victim():
            meta["known_target_row"] = self.disturbance.known_target_row
            meta["known_threshold"] = self.disturbance.known_threshold
        return meta

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
        # The disturbance engine always consumes the *raw* decoded events above;
        # only the policy-facing feedback is projected to the disclosure level, so
        # hidden coordinates never reach the trace, flip list, or counts.
        obs.feedback = self.disclosure.project_feedback(obs.feedback)
        return obs
