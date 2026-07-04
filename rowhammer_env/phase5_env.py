from __future__ import annotations

from typing import Any

from .phase2_env import Phase2Action, Phase2Observation
from .phase4_env import RowHammerDisturbanceEnv
from .script_sandbox import RestrictedScriptBroker, ScriptViolation
from .tasks import Disclosure


# Per-family default disclosure when the task config does not pin one explicitly
# (SPEC §7). Known-target families disclose physical coordinates; the hidden and
# unknown-adjacency families withhold the mapping and hand out opaque handles.
FAMILY_DISCLOSURE: dict[str, Disclosure] = {
    "known_target_anybit": Disclosure("physical", "exact", "exact", "public_profile_id", "summarized_counts"),
    "target_cell": Disclosure("physical", "exact", "exact", "public_profile_id", "summarized_counts"),
    "pattern_target": Disclosure("physical", "exact", "exact", "public_profile_id", "summarized_counts"),
    "any_flip": Disclosure("physical", "exact", "hidden_until_finish", "public_profile_id", "summarized_counts"),
    "hidden_target": Disclosure("logical_only", "hidden", "row_handle", "public_profile_id", "summarized_counts"),
    "unknown_adjacency": Disclosure("logical_only", "candidate_set", "row_handle", "public_profile_id", "summarized_counts"),
}
DEFAULT_FAMILY = "known_target_anybit"


class RowHammerTaskEnv(RowHammerDisturbanceEnv):
    def __init__(
        self,
        *args: Any,
        budgets: dict[str, int] | None = None,
        task: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.initial_budgets = budgets or {"tool_calls": 20000, "cycles": 5_000_000}
        self.budget_remaining = dict(self.initial_budgets)
        self.task_config = task or {"family": DEFAULT_FAMILY}
        self.task_family = self.task_config["family"]
        self.target_row = 10
        self.success = False
        self._target_handle: str | None = None
        self._candidate_handles: list[str] = []

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> Phase2Observation:
        self.budget_remaining = dict(self.initial_budgets)
        self.success = False
        self.task_family = self.task_config.get("family", DEFAULT_FAMILY)
        # Pin the disclosure before super().reset() so phase 4 builds the resolver
        # (and its address-form gate) against this task's level.
        self.disclosure = self._resolve_disclosure()
        obs = super().reset(seed=seed, episode_id=episode_id, **kwargs)
        if obs.error:
            return obs
        self.target_row = self.disturbance.known_target_row  # type: ignore[union-attr]
        self._register_handles()
        obs.metadata.update(self._task_metadata(seed))
        return obs

    def step(self, action: Phase2Action, timeout_s: float | None = None, **kwargs: Any) -> Phase2Observation:
        if action.tool == "script.run":
            return self._script(action)
        if action.tool == "episode.finish":
            obs = Phase2Observation(reward=1.0 if self._trusted_success() else 0.0, done=True, cycle=self._state.cycle)
            self.close()
            return obs

        before = self._state.cycle
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        self._charge(obs, before)
        self.success = self._trusted_success()
        if self.success:
            obs.reward = 1.0
            obs.done = True
        obs.metadata["budget_remaining"] = dict(self.budget_remaining)
        return obs

    def _resolve_disclosure(self) -> Disclosure:
        if "disclosure" in self.task_config:
            return Disclosure.from_config(self.task_config["disclosure"])
        return FAMILY_DISCLOSURE.get(self.task_family, FAMILY_DISCLOSURE[DEFAULT_FAMILY])

    def _register_handles(self) -> None:
        """Mint per-episode opaque handles the disclosure calls for.

        A ``row_handle``/``cell_handle`` victim gets a ``target`` handle resolving
        to the target row's linear address; a ``candidate_set`` adjacency gets
        handles for the neighbouring aggressor rows. All resolve server-side only.
        """
        assert self.disturbance is not None and self._resolver is not None
        handles = self._resolver.handles
        self._target_handle = None
        self._candidate_handles = []
        target_addr = self.disturbance.target_addr

        if self.disclosure.victim in ("row_handle", "cell_handle"):
            self._target_handle = handles.register("target", target_addr)
        if self.disclosure.adjacency == "candidate_set":
            row_bytes = self.disturbance.row_bytes
            for i, offset in enumerate((-row_bytes, row_bytes, 2 * row_bytes)):
                self._candidate_handles.append(handles.register(f"candidate:{i}", target_addr + offset))

    def _script(self, action: Phase2Action) -> Phase2Observation:
        code = str(action.args.get("code", ""))
        try:
            result = RestrictedScriptBroker(self, max_calls=min(10_000, self.budget_remaining["tool_calls"])).run(code)
        except ScriptViolation as exc:
            return self._error(exc.code, str(exc))
        obs = Phase2Observation(reward=1.0 if self.success else 0.0, done=self.success, cycle=self._state.cycle)
        obs.feedback["script"] = result
        obs.metadata["budget_remaining"] = dict(self.budget_remaining)
        return obs

    def _charge(self, obs: Phase2Observation, before_cycle: int) -> None:
        self.budget_remaining["tool_calls"] -= 1
        self.budget_remaining["cycles"] -= max(0, obs.cycle - before_cycle)
        if self.budget_remaining["tool_calls"] < 0 or self.budget_remaining["cycles"] < 0:
            obs.error = {"code": "BUDGET_EXCEEDED", "message": "episode budget exhausted"}
            obs.done = True

    def _trusted_success(self) -> bool:
        if self.disturbance is None:
            return False
        if self.task_family == "any_flip":
            return bool(self.disturbance.flips)
        target_addr = self.target_row * self.disturbance.row_bytes
        if self.task_family == "target_cell":
            return self.disturbance.flips.get(target_addr) == int(self.task_config.get("bit", 0))
        if self.task_family == "pattern_target":
            desired = int(self.task_config.get("value", 1))
            return self.disturbance.flips.get(target_addr) == 0 and (desired & 1) == 1
        return any(addr // self.disturbance.row_bytes == self.target_row for addr in self.disturbance.flips)

    def _task_metadata(self, seed: int | None) -> dict[str, Any]:
        assert self.disturbance is not None
        meta = {
            "task_id": f"ddr4_{self.task_family}_v1",
            "task_family": self.task_family,
            "difficulty": {"band": "smoke", "seed": seed},
            "budget_remaining": dict(self.budget_remaining),
            "allowed_tools": [
                "dram.info",
                "dram.read",
                "dram.write",
                "dram.issue",
                "script.run",
                "episode.finish",
            ],
            "address_forms": sorted(self.disclosure.allowed_forms()),
            "disclosure": self.disclosure.as_public(),
            "disturbance": self._public_disturbance(),
        }
        meta.update(self._objective_and_target())
        return meta

    def _public_disturbance(self) -> dict[str, Any]:
        assert self.disturbance is not None
        pub: dict[str, Any] = {"family": self.disturbance.family, "stratum": self.disturbance.stratum}
        if self.disclosure.expose_victim():
            pub["known_target_row"] = self.disturbance.known_target_row
            pub["known_threshold"] = self.disturbance.known_threshold
        return pub

    def _objective_and_target(self) -> dict[str, Any]:
        assert self.disturbance is not None
        out: dict[str, Any] = {}
        fam = self.task_family
        target_addr = self.disturbance.target_addr

        if fam == "any_flip":
            out["objective"] = {"type": "any_flip"}
        elif fam == "target_cell":
            out["objective"] = {"type": "target_cell_flip", "bit": int(self.task_config.get("bit", 0))}
        elif fam == "pattern_target":
            out["objective"] = {
                "type": "pattern_target",
                "mask": int(self.task_config.get("mask", 1)),
                "value": int(self.task_config.get("value", 1)),
            }
        elif fam in ("hidden_target", "unknown_adjacency"):
            out["objective"] = {"type": "target_row_flip", "target": {"kind": "handle", "id": self._target_handle}}
        else:
            out["objective"] = {"type": "target_row_flip"}
            if self.disclosure.expose_victim():
                out["objective"]["target_row"] = self.disturbance.known_target_row

        # Target disclosure: exact -> physical coordinates + linear address;
        # handle -> opaque id only; hidden_until_finish -> nothing.
        if self.disclosure.victim == "exact":
            out["target"] = self._physical_target(target_addr)
        elif self.disclosure.victim in ("row_handle", "cell_handle") and self._target_handle is not None:
            out["target"] = {"kind": "handle", "id": self._target_handle}

        if self.disclosure.adjacency == "candidate_set":
            out["candidates"] = [{"kind": "handle", "id": h} for h in self._candidate_handles]
        return out

    def _physical_target(self, addr: int) -> dict[str, Any]:
        assert self.address_mapper is not None
        coords = self.address_mapper.decode(addr)
        return {"kind": "physical", **coords, "addr": addr}
