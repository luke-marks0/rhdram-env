from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

from .disturbance import DisturbanceEngine
from .geometry import Geometry
from .mappers import DEFAULT_MAPPER, worker_config_for_mapper
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
        self.geometry: Geometry | None = None
        self.address_mapper: AddressMapper | None = None
        # The per-episode active mapper (impl name + params). The bare disturbance
        # env always uses the public default; the task layer overrides
        # ``_active_mapper`` to pick a per-episode secret for discovery families.
        # Server-internal only — never disclosed to the policy (P24 leakage guard).
        self._active_mapper_impl = DEFAULT_MAPPER
        self._active_mapper_params: dict[str, int] = {}

    def _active_mapper(self, seed: int) -> tuple[str, dict[str, int]]:
        """The address mapper for this episode: default public unless overridden."""
        return (DEFAULT_MAPPER, {})

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> Phase2Observation:
        # Select the (possibly per-episode-secret) address mapper *before* the
        # worker is built, then compose the mitigation plugins on top. The default
        # mapper reuses the base config verbatim, so the non-discovery path is
        # byte-identical to before.
        self._active_mapper_impl, self._active_mapper_params = self._active_mapper(seed or 0)
        try:
            mapped_config = worker_config_for_mapper(
                self._base_config_path, self._active_mapper_impl, self._active_mapper_params
            )
            self.config_path = worker_config_for_mitigation(mapped_config, self.mitigation)
        except ValueError as exc:
            # Every failure below is a *reset* failure: the episode never starts, so it
            # is terminal regardless of the code. @spec:rl-episode-termination
            if str(exc).startswith("UNAVAILABLE_CAPABILITY:"):
                return self._terminal_error("UNAVAILABLE_CAPABILITY", str(exc).split(":", 1)[1])
            if str(exc).startswith("BAD_SCHEMA:"):
                return self._terminal_error("BAD_SCHEMA", str(exc).split(":", 1)[1])
            raise
        obs = super().reset(seed=seed, episode_id=episode_id, **kwargs)
        if obs.error:
            return obs
        try:
            geometry = self._fetch_geometry()
        except RuntimeError as exc:
            self.close()
            return self._terminal_error("INTERNAL_SIMULATOR_ERROR", str(exc))
        self.geometry = geometry
        try:
            self.address_mapper = AddressMapper(geometry)
        except ValueError as exc:
            self.close()
            return self._terminal_error("UNAVAILABLE_CAPABILITY", f"geometry not projectable: {exc}")
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
            # Victim anchors come from the worker's own mapper, for every episode —
            # the public mapper included, so there is one address path rather than a
            # Python projection that silently diverges under a secret mapping.
            row_encoder=self._encode,
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
                return self._terminal_error("UNAVAILABLE_CAPABILITY", str(exc).split(":", 1)[1])
            if str(exc).startswith("PROFILE_REJECTED:"):
                self.close()
                return self._terminal_error("PROFILE_REJECTED", str(exc).split(":", 1)[1])
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

    def _decode(self, linear: int) -> dict[str, int]:
        """Decode a linear address under the *active* worker mapper (P24).

        Server-internal only: the ``DECODE`` op is never in ``ALLOWED_TOOLS`` and is
        never reachable through ``step``. The task compiler uses it to build
        candidate sets against the true (per-episode-secret) mapping without
        re-deriving the bit function in Python. Raises if the worker is down.
        """
        if self._worker is None:
            raise RuntimeError("worker unavailable for decode")
        resp = self._worker.call(WorkerRequest("DECODE", self._next_id(), (str(int(linear)),)))
        if not resp.get("ok"):
            raise RuntimeError((resp.get("error") or {}).get("message", "decode failed"))
        return resp["addr_vec"]

    def _encode(self, coords: Mapping[str, int]) -> int:
        """Encode physical coordinates to a linear address under the *active* mapper.

        The exact inverse of :meth:`_decode`, and server-internal in the same way:
        ``ENCODE`` is never in ``ALLOWED_TOOLS``. The disturbance engine uses it to
        resolve each victim row's own column-0 address, which under a secret
        row->bank mapping is not the aggressor's address plus a row stride
        (# @spec:sim-exposure-flip). Raises if the worker is down or the active
        mapper has no linear address for these coordinates.
        """
        if self._worker is None or self.geometry is None:
            raise RuntimeError("worker unavailable for encode")
        args = tuple(str(int(coords.get(level, 0))) for level in self.geometry.level_names)
        resp = self._worker.call(WorkerRequest("ENCODE", self._next_id(), args))
        if not resp.get("ok"):
            raise RuntimeError((resp.get("error") or {}).get("message", "encode failed"))
        return int(resp["linear"])

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
        geometry = payload["geometry"]
        # The command vocabulary is what lets the disturbance engine check its event
        # classification against a closed set, so a worker that does not publish it
        # is rejected rather than silently skipping that check.
        if not geometry.get("command_names"):
            raise RuntimeError("worker reported geometry without a command vocabulary; rebuild the worker")
        try:
            return Geometry(geometry)
        except (ValueError, KeyError, TypeError) as exc:
            # A malformed INFO payload is a simulator failure, not a policy error, and
            # must end the episode with a stable code rather than letting the geometry
            # validation exception escape reset.
            raise RuntimeError(f"worker reported an invalid geometry: {exc}") from exc

    def _from_worker(
        self, payload: dict[str, Any], *, written_data: bytes | None = None
    ) -> Phase2Observation:
        obs = super()._from_worker(payload, written_data=written_data)
        if obs.error or self.disturbance is None:
            return obs

        request = payload.get("request", {})
        try:
            events = payload.get("events", [])
            if request.get("op") == "WR":
                if written_data is None:
                    raise RuntimeError("worker write response has no corresponding write data")
                write_events = [event for event in events if event.get("op") == "WR"]
                if len(write_events) != 1:
                    raise RuntimeError(
                        f"worker write response reported {len(write_events)} issued WR events; expected 1"
                    )
                write_event = write_events[0]
                row_key = tuple(
                    int(write_event.get(level, 0))
                    for level in ("channel", "rank", "bankgroup", "bank", "row")
                )
                if row_key[-1] < 0:
                    raise RuntimeError("worker write event has no decoded row")
                # Record the written stratum from the worker-decoded physical row
                # before consume restores overwritten flip cells. The decoded key
                # is mapper-correct even for secret row->bank mappings.
                # @spec:tool-dram-write @spec:sim-latent-vulnerability
                self.disturbance.note_write(row_key, written_data)
            result = self.disturbance.consume(events, request)
        except RuntimeError as exc:
            # Folding events needs the worker to resolve victim anchors (``ENCODE``).
            # If that fails the overlay would be silently incomplete, so the episode
            # ends rather than continuing on partial disturbance state.
            return self._terminal_error("INTERNAL_SIMULATOR_ERROR", str(exc))
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
