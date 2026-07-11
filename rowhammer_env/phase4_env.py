from __future__ import annotations

import base64
from typing import Any

from .disturbance import DisturbanceEngine
from .geometry import Geometry
from .mitigations import normalize_mitigation, worker_config_for_mitigation
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
DEFAULT_PROFILE_ID = "ddr4_vts25_v1"


class RowHammerDisturbanceEnv(RowHammerEnv):
    def __init__(
        self,
        *args: Any,
        disturbance: DisturbanceEngine | None = None,
        mitigation: dict[str, Any] | None = None,
        profile_id: str = DEFAULT_PROFILE_ID,
        temperature: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.disturbance = disturbance
        self.mitigation = normalize_mitigation(mitigation)
        self.profile_id = profile_id
        self.temperature = temperature
        self.disclosure = FULL_DISCLOSURE
        self.address_mapper: AddressMapper | None = None

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> Phase2Observation:
        try:
            self.config_path = worker_config_for_mitigation(self._base_config_path, self.mitigation)
        except ValueError as exc:
            if str(exc).startswith("UNAVAILABLE_CAPABILITY:"):
                return self._error("UNAVAILABLE_CAPABILITY", str(exc).split(":", 1)[1])
            if str(exc).startswith("BAD_SCHEMA:"):
                return self._error("BAD_SCHEMA", str(exc).split(":", 1)[1])
            raise
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
        # Let the task layer (P13) compile its task against the real geometry —
        # sampling the target row, choosing the disclosure, and picking the engine
        # overrides (known target row / first bit / disturbance family) — before
        # the resolver and engine are built.
        overrides = self._disturbance_overrides(geometry, seed or 0)
        # Build the resolver next so any address form the disclosure permits is
        # usable as soon as the episode is live; the task layer registers handles
        # into this same table after reset.
        self._resolver = AddressResolver(self.address_mapper, self.disclosure, HandleTable(seed or 0))
        engine_kwargs: dict[str, Any] = dict(
            geometry=geometry,
            seed=seed or 0,
            mitigation=self.mitigation["name"],
            mitigation_params=self.mitigation.get("params"),
            temperature=self.temperature,
            profile_id=self.profile_id,
        )
        engine_kwargs.update(overrides)
        try:
            self.disturbance = DisturbanceEngine(**engine_kwargs)
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
        obs.metadata["geometry"] = geometry.public_block()
        return obs

    def _geometry_block(self) -> dict[str, Any] | None:
        # Available once the worker has reported geometry and the mapper is built.
        if self.address_mapper is None:
            return None
        return self.address_mapper.geometry.public_block()

    def _disturbance_overrides(self, geometry: Geometry, seed: int) -> dict[str, Any]:
        """Engine constructor overrides for this episode (hook for the task layer).

        The bare disturbance env keeps the default fixed known target (row 10);
        ``RowHammerTaskEnv`` overrides this to compile the task and inject the
        sampled target row, first-flip bit, and disturbance family.
        """
        del geometry, seed
        return {}

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
