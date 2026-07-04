from __future__ import annotations

from typing import Any

from . import rewards
from .geometry import Geometry
from .phase2_env import Phase2Action, Phase2Observation
from .phase4_env import RowHammerDisturbanceEnv
from .script_sandbox import RestrictedScriptBroker, ScriptViolation
from .tasks.compiler import CompiledTask, TaskSpec

ALLOWED_TOOLS = ["dram.info", "dram.read", "dram.write", "dram.issue", "script.run", "episode.finish"]


class RowHammerTaskEnv(RowHammerDisturbanceEnv):
    """Task-compiled RowHammer env (SPEC §7/§10).

    A task config — either the SPEC §10 YAML shape or the ``{"family": ...}``
    shorthand — is parsed into a :class:`TaskSpec` at construction, then *compiled*
    against the real worker geometry at ``reset`` (P13): the target row/bit is
    sampled per ``(task_id, seed)``, budgets and difficulty band are resolved, the
    disclosure level is pinned, and one trusted per-family success predicate
    (`rowhammer_env.rewards`) decides reward — replacing the old hardcoded
    ``target_row=10`` / monolithic ``_trusted_success``.
    """

    def __init__(
        self,
        *args: Any,
        budgets: dict[str, int] | None = None,
        task: dict[str, Any] | None = None,
        mitigation: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.spec = TaskSpec.from_config(task)
        # Task config carries the mitigation and profile; an explicit kwarg wins so
        # existing callers (e.g. RowHammerTaskEnv(mitigation=..., profile_id=...))
        # keep their meaning.
        effective_mitigation = mitigation or self.spec.mitigation
        if "profile_id" not in kwargs and self.spec.profile_id:
            kwargs["profile_id"] = self.spec.profile_id
        super().__init__(*args, mitigation=effective_mitigation, **kwargs)
        self._budgets_override = budgets
        self.initial_budgets = budgets or self.spec.resolved_budgets()
        self.budget_remaining = dict(self.initial_budgets)
        self.task_family = self.spec.family
        self.success = False
        self._compiled: CompiledTask | None = None
        self.target_row = 0
        self._acts_prev = 0
        self._target_handle: str | None = None
        self._candidate_handles: list[str] = []

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> Phase2Observation:
        self.budget_remaining = dict(self.initial_budgets)
        self.success = False
        self._acts_prev = 0
        self._compiled = None
        # super().reset() calls _disturbance_overrides, which compiles the task
        # against the reported geometry and pins the disclosure + engine target.
        obs = super().reset(seed=seed, episode_id=episode_id, **kwargs)
        if obs.error:
            return obs
        assert self._compiled is not None
        self.target_row = self._compiled.target_row
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

    # ---- task compilation hook ----------------------------------------------
    def _disturbance_overrides(self, geometry: Geometry, seed: int) -> dict[str, Any]:
        self._compiled = self.spec.compile(seed, geometry)
        self.disclosure = self._compiled.disclosure
        self.task_family = self._compiled.family
        return self._compiled.disturbance_overrides()

    def _register_handles(self) -> None:
        """Mint per-episode opaque handles the disclosure calls for (SPEC §8).

        A ``row_handle``/``cell_handle`` victim gets a ``target`` handle resolving
        to the target row's linear address; a ``candidate_set`` adjacency gets
        handles for the neighbouring aggressor rows. All resolve server-side only.
        """
        assert self._compiled is not None and self._resolver is not None
        handles = self._resolver.handles
        self._target_handle = None
        self._candidate_handles = []
        target_addr = self._compiled.target_addr
        row_bytes = self._compiled.row_bytes

        if self.disclosure.victim in ("row_handle", "cell_handle"):
            self._target_handle = handles.register("target", target_addr)
        if self.disclosure.adjacency == "candidate_set":
            for i, offset in enumerate((-row_bytes, row_bytes, 2 * row_bytes)):
                self._candidate_handles.append(handles.register(f"candidate:{i}", target_addr + offset))

    def _script(self, action: Phase2Action) -> Phase2Observation:
        code = str(action.args.get("code", ""))
        try:
            result = RestrictedScriptBroker(self, max_calls=min(10_000, self.budget_remaining["tool_calls"])).run(code)
        except ScriptViolation as exc:
            return self._error(exc.code, str(exc))
        self.success = self._trusted_success()
        obs = Phase2Observation(reward=1.0 if self.success else 0.0, done=self.success, cycle=self._state.cycle)
        obs.feedback["script"] = result
        obs.metadata["budget_remaining"] = dict(self.budget_remaining)
        return obs

    def _charge(self, obs: Phase2Observation, before_cycle: int) -> None:
        self.budget_remaining["tool_calls"] -= 1
        self.budget_remaining["cycles"] -= max(0, obs.cycle - before_cycle)
        if "acts" in self.budget_remaining:
            acts_now = int(obs.public_counters.get("acts", self._acts_prev))
            self.budget_remaining["acts"] -= max(0, acts_now - self._acts_prev)
            self._acts_prev = acts_now
        exceeded = self.budget_remaining["tool_calls"] < 0 or self.budget_remaining["cycles"] < 0
        if "acts" in self.budget_remaining:
            exceeded = exceeded or self.budget_remaining["acts"] < 0
        if exceeded:
            obs.error = {"code": "BUDGET_EXCEEDED", "message": "episode budget exhausted"}
            obs.done = True

    def _trusted_success(self) -> bool:
        return rewards.success_for(self._compiled, self.disturbance)

    # ---- observation metadata (disclosure-projected) ------------------------
    def _task_metadata(self, seed: int | None) -> dict[str, Any]:
        assert self._compiled is not None
        ct = self._compiled
        meta = {
            "task_id": ct.task_id,
            "task_family": ct.family,
            "difficulty": {
                "band": ct.difficulty,
                "seed": seed,
                "expected_success": list(ct.expected_success_window()),
            },
            "reward": ct.reward_kind,
            "budget_remaining": dict(self.budget_remaining),
            "allowed_tools": list(ALLOWED_TOOLS),
            "address_forms": sorted(self.disclosure.allowed_forms()),
            "disclosure": self.disclosure.as_public(),
            "disturbance": self._public_disturbance(),
        }
        meta.update(self._objective_and_target())
        return meta

    def _public_disturbance(self) -> dict[str, Any]:
        assert self.disturbance is not None and self._compiled is not None
        pub: dict[str, Any] = {"family": self.disturbance.family, "stratum": self.disturbance.stratum}
        # Exact-victim tasks disclose the target row; the fixed calibrated hcfirst
        # is disclosed only for the deterministic known-target families (a sampled
        # target's threshold is intentionally variable and stays hidden).
        if self.disclosure.expose_victim():
            pub["known_target_row"] = self._compiled.target_row
            if self._compiled.target_kind == "known":
                pub["known_threshold"] = self.disturbance.known_threshold
        return pub

    def _objective_and_target(self) -> dict[str, Any]:
        assert self._compiled is not None
        ct = self._compiled
        out: dict[str, Any] = {}
        fam = ct.family

        if ct.objective_type == "any_flip":
            out["objective"] = {"type": "any_flip"}
        elif ct.objective_type == "target_cell_flip":
            out["objective"] = {"type": "target_cell_flip", "bit": ct.target_bit}
        elif ct.objective_type == "pattern_target":
            out["objective"] = {"type": "pattern_target", "mask": ct.target_mask, "value": ct.target_value}
        elif fam in ("hidden_target", "unknown_adjacency"):
            out["objective"] = {"type": "target_row_flip", "target": {"kind": "handle", "id": self._target_handle}}
        else:
            out["objective"] = {"type": ct.objective_type}
            if self.disclosure.expose_victim():
                out["objective"]["target_row"] = ct.target_row

        # Target disclosure: exact -> physical coordinates + linear address;
        # handle -> opaque id only; hidden_until_finish -> nothing.
        if self.disclosure.victim == "exact":
            out["target"] = self._physical_target(ct.target_addr)
        elif self.disclosure.victim in ("row_handle", "cell_handle") and self._target_handle is not None:
            out["target"] = {"kind": "handle", "id": self._target_handle}

        if self.disclosure.adjacency == "candidate_set":
            out["candidates"] = [{"kind": "handle", "id": h} for h in self._candidate_handles]
        return out

    def _physical_target(self, addr: int) -> dict[str, Any]:
        assert self.address_mapper is not None
        coords = self.address_mapper.decode(addr)
        return {"kind": "physical", **coords, "addr": addr}
