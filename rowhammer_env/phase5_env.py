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
from .worker_protocol import WorkerRequest

ALLOWED_TOOLS = ["dram.info", "dram.read", "dram.write", "dram.issue", "script.run", "episode.finish"]


class TrustedRewardReadError(RuntimeError):
    """The worker could not provide authoritative memory for reward evaluation."""


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
        self._episode_done = False
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
        self._episode_done = False
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
                self._episode_done = True
                # The episode never starts, so this is terminal even though BAD_SCHEMA
                # only rejects an action mid-episode. @spec:rl-episode-termination
                return self._terminal_error("BAD_SCHEMA", f"invalid task config: {exc}")
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
            self._episode_done = True
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
        """Admit one action, or refuse it because the episode can no longer pay for it.

        Both guards run *before* dispatch, so a refused action never reaches the
        worker and never commits disturbance. This is what makes budget honesty a
        property of the env rather than of whichever driver happens to stop on
        ``done`` (``script_sandbox`` forwards ``done`` to the script without acting
        on it, and would otherwise keep hammering an episode that is already over).
        @spec:invariant-budget-honesty @spec:rl-episode-termination
        """
        if self._episode_done:
            return self._refuse("UNAVAILABLE_CAPABILITY", "episode is over; reset to start a new one")
        # A dispatched action needs a tool call to spend: ``dram.*`` charges one
        # directly, ``script.run`` charges one per brokered inner call. At zero
        # remaining there is nothing left to pay with, so nothing is dispatched —
        # ``episode.finish`` included, since its reward was already granted by the
        # step that achieved success, and the refusal tears the episode down anyway.
        if self.budget_remaining["tool_calls"] <= 0:
            return self._refuse("BUDGET_EXCEEDED", "tool-call budget exhausted")
        try:
            obs = self._dispatch(action, timeout_s=timeout_s, **kwargs)
        except TrustedRewardReadError as exc:
            return self._refuse("INTERNAL_SIMULATOR_ERROR", str(exc))
        # ``obs.done`` is the single authority on termination: ``_error`` sets it only
        # for TERMINAL_ERROR_CODES, so a rejected action (a malformed command, a
        # script the sandbox refused) leaves the episode running rather than ending it
        # over a typo, while success, ``episode.finish`` and a terminal error latch.
        if obs.done:
            self._episode_done = True
        return obs

    def _dispatch(self, action: Phase2Action, timeout_s: float | None = None, **kwargs: Any) -> Phase2Observation:
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

    def _refuse(self, code: str, message: str) -> Phase2Observation:
        """Terminal refusal: the action is not dispatched and no budget is charged.

        The episode is over either way, so the worker is torn down here — otherwise
        a policy that spends its last tool call and *then* calls ``episode.finish``
        (the usual shape) would leave the simulator process running, since the
        refusal replaces the ``finish`` that used to close it.
        """
        self._episode_done = True
        self.close()
        # Terminal whatever the code: BUDGET_EXCEEDED ends the episode by itself, and
        # the post-termination refusal (UNAVAILABLE_CAPABILITY) reports an episode
        # that is already over rather than rejecting an action within a live one.
        obs = self._terminal_error(code, message)
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
        # A brokered inner call may have ended the episode (budget exhaustion, a
        # simulator error); the script keeps running to completion either way, but
        # the observation must not report the episode as live once it is over.
        done = self.success or self._episode_done
        obs = Phase2Observation(reward=1.0 if self.success else 0.0, done=done, cycle=self._state.cycle)
        obs.feedback["script"] = result
        obs.metadata["budget_remaining"] = dict(self.budget_remaining)
        return obs

    def _issue_acts_ceiling(self) -> int | None:
        # Cap a single dram.issue at the episode's remaining ACT budget so a
        # monolithic HAMMER is truncated at the budget rather than running to
        # completion and crediting an over-budget flip. ``_acts_prev`` is the
        # cumulative ACT counter as of the last charge; adding the remaining budget
        # gives the absolute counter value this issue may reach — the point at which
        # the budget is exactly spent, which is what ``_issue`` compares its live
        # counter against. Anchoring on ``_acts_prev`` rather than the live counter
        # keeps the ceiling tied to the budget that was actually debited, so any
        # drift between the two can only make the guard stricter, never looser.
        # Only families that budget activations are constrained (others hammer freely).
        if "acts" not in self.budget_remaining:
            return None
        return self._acts_prev + max(0, self.budget_remaining["acts"])

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
        return rewards.success_for(self._compiled, self.disturbance, self._trusted_read_byte)

    # @spec:rl-reward @spec:invariant-trusted-reward
    def _trusted_read_byte(self, addr: int) -> int:
        """Read one authoritative post-disturbance byte without issuing DRAM traffic.

        Reward evaluation must use the worker's actual functional-memory byte, not
        reconstruct it from ``flips`` or the disturbance model's ``data_pattern``.
        The worker's internal ``READ ... STORED`` form reads that byte without
        ticking Ramulator or draining events; applying the committed disturbance
        overlay here produces exactly the byte a policy-facing ``dram.read`` would
        return, without letting reward evaluation create unbudgeted disturbance.
        """
        if self._worker is None or self.disturbance is None:
            raise TrustedRewardReadError("worker unavailable for trusted reward read")
        request_id = f"{self._next_id()}:reward"
        payload = self._worker.call(WorkerRequest("READ", request_id, (str(int(addr)), "1", "STORED")))
        if not payload.get("ok"):
            error = payload.get("error") or {}
            raise TrustedRewardReadError(error.get("message", "trusted reward read failed"))
        data_hex = payload.get("data_hex")
        if not isinstance(data_hex, str):
            raise TrustedRewardReadError("trusted reward read returned no data")
        try:
            stored = bytes.fromhex(data_hex)
        except ValueError as exc:
            raise TrustedRewardReadError("trusted reward read returned invalid data") from exc
        if len(stored) != 1:
            raise TrustedRewardReadError(f"trusted reward read returned {len(stored)} bytes, expected 1")
        return self.disturbance.apply(int(addr), stored)[0]

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

        # Candidate disclosure: opaque handles for handle-victim families
        # (``bounded_sweep``), numeric logical addresses for ``hidden_adjacency``.
        # The ordering matches ``self._compiled.candidates`` in both cases so a fixed seed is
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
