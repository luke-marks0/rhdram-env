from __future__ import annotations

from typing import Any

from . import rewards
from .geometry import Geometry
from .mappers import DEFAULT_MAPPER, is_python_projectable, select_secret_mapper
from .mitigations import normalize_mitigation
from .phase2_env import Phase2Action, Phase2Observation
from .phase4_env import DEFAULT_PROFILE_ID, RowHammerDisturbanceEnv
from .script_sandbox import RestrictedScriptBroker, ScriptError
from .tasks.compiler import FAMILIES, CompiledTask, TaskConfigError, TaskSpec

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

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        *args: Any,
        budgets: dict[str, int] | None = None,
        task: dict[str, Any] | None = None,
        mitigation: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.spec = TaskSpec.from_config(task)
        self._explicit_mitigation = mitigation
        explicit_profile_id = kwargs.get("profile_id") if "profile_id" in kwargs else None
        # Task config carries the mitigation and profile; an explicit kwarg wins so
        # existing callers (e.g. RowHammerTaskEnv(mitigation=..., profile_id=...))
        # keep their meaning.
        effective_mitigation = mitigation or self.spec.mitigation
        if "profile_id" not in kwargs and self.spec.profile_id:
            kwargs["profile_id"] = self.spec.profile_id
        super().__init__(*args, mitigation=effective_mitigation, **kwargs)
        self._explicit_profile_id = explicit_profile_id
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

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: dict[str, Any] | None = None,
        budgets: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> Phase2Observation:
        # Per-episode task selection (P17): an orchestrator can pass a task config
        # (SPEC §10 shape or the ``{"family": ...}`` shorthand) at reset to drive a
        # curriculum without restarting the server. Task selection is a training-
        # orchestration control (it flows through the WS ``reset``), not a policy
        # action. Profile/mitigation are honoured too — the disturbance engine is
        # rebuilt from ``self.profile_id``/``self.mitigation`` on every reset.
        if task is not None:
            try:
                self._configure_task(task, budgets=budgets)
            except TaskConfigError as exc:
                self.close()
                return self._error("BAD_SCHEMA", f"invalid task config: {exc}")
        elif budgets is not None:
            self._budgets_override = budgets
            self.initial_budgets = self._budgets_override or self.spec.resolved_budgets()
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

    def _configure_task(self, task: dict[str, Any], budgets: dict[str, int] | None = None) -> None:
        self.spec = TaskSpec.from_config(task)
        if self._explicit_mitigation is None:
            self.mitigation = normalize_mitigation(self.spec.mitigation)
        if self._explicit_profile_id is None:
            self.profile_id = self.spec.profile_id or DEFAULT_PROFILE_ID
        if budgets is not None:
            self._budgets_override = budgets
        self.initial_budgets = self._budgets_override or self.spec.resolved_budgets()
        self.budget_remaining = dict(self.initial_budgets)
        self.task_family = self.spec.family

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

    # ---- secret address mapping (P24) ---------------------------------------
    def _active_mapper(self, seed: int) -> tuple[str, dict[str, int]]:
        """Discovery families get a per-episode secret mapper; others stay public.

        Secret-mapping families must be ``logical_only`` (the policy never gets a
        physical decoder, so a secret bank function cannot leak through
        ``physical`` addressing); fail closed otherwise.
        """
        fam = FAMILIES.get(self.spec.family)
        if fam is not None and fam.secret_mapping:
            if fam.disclosure.mapping != "logical_only":
                raise TaskConfigError(
                    f"secret-mapping family {self.spec.family!r} must be logical_only"
                )
            return select_secret_mapper(self.spec.task_id, seed)
        return (DEFAULT_MAPPER, {})

    # ---- task compilation hook ----------------------------------------------
    def _disturbance_overrides(self, geometry: Geometry, seed: int) -> dict[str, Any]:
        # Discovery families build their candidate window against the true (secret)
        # mapping via the worker DECODE op; RoBaRaCoCh families need no decoder.
        decode = self._decode if not is_python_projectable(self._active_mapper_impl) else None
        self._compiled = self.spec.compile(seed, geometry, decode=decode)
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
        # Candidate *handles* are minted only for handle-victim discovery families
        # (Tier 2a ``bounded_sweep`` / legacy ``unknown_adjacency``). The Tier 2b
        # ``hidden_adjacency`` family (``victim: logical_addr``) discloses its
        # candidates as numeric logical addresses instead (emitted directly in
        # ``_objective_and_target``), so it registers no handles.
        if self.disclosure.adjacency == "candidate_set" and self.disclosure.victim in ("row_handle", "cell_handle"):
            # ``bounded_sweep`` (P23) compiles an explicit N-candidate window; the
            # legacy ``unknown_adjacency`` keeps its fixed 3 same-bank offsets.
            if self._compiled.candidates:
                offsets = [c.offset for c in self._compiled.candidates]
            else:
                offsets = [-row_bytes, row_bytes, 2 * row_bytes]
            for i, offset in enumerate(offsets):
                self._candidate_handles.append(handles.register(f"candidate:{i}", target_addr + offset))

    def _script(self, action: Phase2Action) -> Phase2Observation:
        code = str(action.args.get("code", ""))
        language = str(action.args.get("language", "python-rh-sdk"))
        if language != "python-rh-sdk":
            return self._error("BAD_SCHEMA", "script.run requires language=python-rh-sdk")
        try:
            timeout_ms = int(action.args.get("timeout_ms", 5000))
        except (TypeError, ValueError):
            return self._error("BAD_SCHEMA", "script.run timeout_ms must be an integer")
        try:
            result = RestrictedScriptBroker(
                self,
                max_calls=min(10_000, self.budget_remaining["tool_calls"]),
                timeout_ms=timeout_ms,
            ).run(code)
        except ScriptError as exc:
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
        elif fam == "hidden_adjacency":
            # Tier 2b (P25): disclose the victim's own numeric address (real-attacker
            # knowledge). Bank membership is not computable from it (secret mapper),
            # so this leaks no adjacency — only the row, which is public geometry.
            out["objective"] = {"type": "target_row_flip", "target": {"kind": "logical", "addr": ct.target_addr}}
        elif fam in ("hidden_target", "unknown_adjacency", "bounded_sweep"):
            out["objective"] = {"type": "target_row_flip", "target": {"kind": "handle", "id": self._target_handle}}
        else:
            out["objective"] = {"type": ct.objective_type}
            if self.disclosure.expose_victim():
                out["objective"]["target_row"] = ct.target_row

        # Target disclosure: exact -> physical coordinates + linear address;
        # logical_addr -> the victim's numeric linear address only (no coords, P25);
        # handle -> opaque id only; hidden_until_finish -> nothing.
        if self.disclosure.victim == "exact":
            out["target"] = self._physical_target(ct.target_addr)
        elif self.disclosure.victim == "logical_addr":
            out["target"] = {"kind": "logical", "addr": ct.target_addr}
        elif self.disclosure.victim in ("row_handle", "cell_handle") and self._target_handle is not None:
            out["target"] = {"kind": "handle", "id": self._target_handle}

        # Candidate disclosure: opaque handles for handle-victim families (Tier 2a),
        # numeric logical addresses for ``hidden_adjacency`` (Tier 2b). The ordering
        # matches ``self._compiled.candidates`` in both cases so a fixed seed is
        # reproducible; the list itself is already role-shuffled by the compiler.
        if self.disclosure.adjacency == "candidate_set":
            if self._candidate_handles:
                out["candidates"] = [{"kind": "handle", "id": h} for h in self._candidate_handles]
            elif self._compiled.candidates:
                out["candidates"] = [
                    {"kind": "logical", "addr": ct.target_addr + c.offset} for c in self._compiled.candidates
                ]
        return out

    def _physical_target(self, addr: int) -> dict[str, Any]:
        assert self.address_mapper is not None
        # The Python projection only reproduces the public RoBaRaCoCh mapper; under
        # a secret mapper it would emit the wrong (and bank-secret-leaking)
        # coordinates, so physical disclosure fails closed (P24 task 4). Only
        # ``victim: exact`` families reach here, and those never use a secret
        # mapper — this is a defensive guard, not a live path.
        if not is_python_projectable(self._active_mapper_impl):
            raise TaskConfigError("physical target disclosure is unavailable under a secret mapper")
        coords = self.address_mapper.decode(addr)
        return {"kind": "physical", **coords, "addr": addr}
