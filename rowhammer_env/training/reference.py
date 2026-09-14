"""Deterministic reference solver and the negative controls, as scripted policies.

Every one of these drives the environment through the same :func:`rollout.run_episode`
loop the model uses, so they share its trace, scoring, and artifacts. They use only
the *disclosed* observation — candidate list, the returned ``timing_digest``, and the
disclosed remaining budget — never hidden state (the secret mapper, the sampled
threshold, decoded coordinates), which is what lets the reference run demonstrate that
the disclosed signal is sufficient and the controls demonstrate what is load-bearing.

Roles (training scope / PoC scope "experimental controls"):
- ``ReferenceSolver``     timing probe -> same-bank set -> hammer to threshold. Should win.
- ``FinishOnlyControl``   finish immediately. Cannot fabricate success.
- ``BelowThresholdControl`` correct rows, too few sweeps. Requests alone don't flip.
- ``TimingBlindControl``  ignore timing, hammer numerically-adjacent rows. The hidden
                          mapping sends them to another bank, so it fails/exhausts budget.
"""
from __future__ import annotations

from typing import Any

from .rollout import ActResult

PROBE_PAIRS = 8            # sweeps per bank-conflict probe (matches the probe-signal tests)
BELOW_THRESHOLD_PAIRS = 200  # deliberately far below any admitted flip threshold
_ACTS_SAFETY = 0.98        # leave a sliver of the acts budget so the final issue fits


def _rd(addr: Any) -> dict[str, Any]:
    return {"op": "RD", "addr": addr}


def _hammer(rows: list[Any], pairs: int) -> dict[str, Any]:
    return {"tool": "dram.issue", "args": {"commands": [{"op": "HAMMER", "rows": rows, "pairs": int(pairs)}]}}


def _finish() -> dict[str, Any]:
    return {"tool": "episode.finish", "args": {}}


def _victim(metadata: dict[str, Any]) -> Any:
    objective = metadata.get("objective") or {}
    return metadata.get("target") or objective.get("target")


def _adjacent_physical(victim: Any) -> list[dict[str, Any]] | None:
    """The two same-bank rows physically adjacent to an exact-disclosed victim.

    Only ``known_target`` discloses the victim as physical coordinates (public
    mapper, no secret bank scramble), so row +/- 1 in the same bank are exactly its
    double-sided aggressors.
    """
    if not isinstance(victim, dict) or victim.get("kind") != "physical":
        return None
    base = {k: victim[k] for k in ("channel", "rank", "bankgroup", "bank") if k in victim}
    row = int(victim["row"])
    return [
        {"kind": "physical", **base, "row": row - 1, "column": 0},
        {"kind": "physical", **base, "row": row + 1, "column": 0},
    ]


def _remaining_acts(obs: Any) -> int:
    budget = obs.metadata.get("budget_remaining") or {}
    return int(budget.get("acts", 0) or 0)


class ReferenceSolver:
    """Probe each candidate against the victim, keep the same-bank ones, hammer them.

    The hammer count is chosen from the disclosed remaining activation budget (spread
    across the identified rows), not the hidden threshold — so it crosses the flip
    threshold whenever the budget can pay for it, without reading hidden state.
    """

    use_timing = True

    def begin_episode(self, *, obs: Any, metadata: dict[str, Any]) -> None:
        self.victim = _victim(metadata)
        self.candidates = list(metadata.get("candidates") or [])
        # ``known_threshold`` is disclosed only for the exact known-target family; when
        # present, hammer just past it (bounded by budget) instead of spending the whole
        # activation budget. For discovery families it is hidden and stays None.
        self.known_threshold = (metadata.get("disturbance") or {}).get("known_threshold")
        self.i = 0
        self.pending_probe = False
        self.await_result_for: Any = None
        self.same_bank: list[Any] = []
        # No candidate window means an exact (known_target) victim: skip probing and
        # hammer its disclosed physical neighbours directly.
        self._known_rows = _adjacent_physical(self.victim) if not self.candidates else None
        if self._known_rows is not None:
            self.phase = "hammer"
        else:
            self.phase = "probe" if self.use_timing else "hammer"

    def _classify(self, obs: Any) -> None:
        """A same-bank alternation forces new ACTs after both rows were warmed open;
        a different-bank one leaves them open (acts_delta == 0)."""
        digest = obs.feedback.get("timing_digest") or {}
        if int(digest.get("acts_delta", 0) or 0) > 0:
            self.same_bank.append(self.await_result_for)
        self.await_result_for = None

    def _hammer_rows(self) -> list[Any]:
        if self._known_rows is not None:
            return self._known_rows
        return self.same_bank or list(self.candidates)

    def _hammer_pairs(self, obs: Any, rows: list[Any]) -> int:
        budget = max(1, int(_remaining_acts(obs) * _ACTS_SAFETY) // max(1, len(rows)))
        if self.known_threshold:  # disclosed exact threshold: a small margin over it
            return min(budget, int(self.known_threshold * 1.1) + 1)
        return budget

    def act(self, *, obs: Any, messages: list[dict[str, str]]) -> ActResult:
        if self.await_result_for is not None:
            self._classify(obs)

        if self.phase == "probe":
            if self.pending_probe:
                cand = self.candidates[self.i]
                self.pending_probe = False
                self.await_result_for = cand
                self.i += 1
                return ActResult(_hammer([self.victim, cand], PROBE_PAIRS))
            if self.i < len(self.candidates):
                cand = self.candidates[self.i]
                self.pending_probe = True
                return ActResult({"tool": "dram.issue", "args": {"commands": [_rd(self.victim), _rd(cand)]}})
            self.phase = "hammer"

        if self.phase == "hammer":
            self.phase = "finish"
            rows = self._hammer_rows()
            return ActResult(_hammer(rows, self._hammer_pairs(obs, rows)))

        return ActResult(_finish())


class BelowThresholdControl(ReferenceSolver):
    """Finds the right rows but hammers far too few times: requests alone don't flip."""

    def _hammer_pairs(self, obs: Any, rows: list[Any]) -> int:
        return BELOW_THRESHOLD_PAIRS


class TimingBlindControl(ReferenceSolver):
    """Ignores the timing channel. For numeric tasks it hammers the arithmetically
    adjacent rows (victim +/- one row stride); the hidden mapping puts those in a
    different bank, so no flip. For handle tasks (no numeric address) it hammers the
    first two candidates unprobed. Either way the load-bearing timing signal is unused.
    """

    use_timing = False

    def begin_episode(self, *, obs: Any, metadata: dict[str, Any]) -> None:
        super().begin_episode(obs=obs, metadata=metadata)
        self.saw_info = False
        self.row_bytes: int | None = None

    def _numeric_victim(self) -> int | None:
        if isinstance(self.victim, dict) and self.victim.get("kind") == "logical":
            return int(self.victim["addr"])
        return None

    def act(self, *, obs: Any, messages: list[dict[str, str]]) -> ActResult:
        victim_addr = self._numeric_victim()
        # Numeric task: learn the row stride once, then hammer the arithmetic neighbours.
        if victim_addr is not None:
            if not self.saw_info:
                self.saw_info = True
                return ActResult({"tool": "dram.info", "args": {}})
            if self.row_bytes is None:
                geometry = obs.metadata.get("geometry") or {}
                self.row_bytes = int(geometry.get("row_bytes", 0) or 0)
            if self.phase != "finish" and self.row_bytes:
                self.phase = "finish"
                rows = [
                    {"kind": "logical", "addr": victim_addr - self.row_bytes},
                    {"kind": "logical", "addr": victim_addr + self.row_bytes},
                ]
                return ActResult(_hammer(rows, self._hammer_pairs(obs, rows)))
            if self.phase == "finish":
                return ActResult(_finish())
        # Handle task: no arithmetic and no timing to narrow with, so the budget must
        # be spread across every candidate at once. With enough candidates that leaves
        # too few sweeps per row to cross the threshold on the true aggressor pair —
        # which is exactly the point (timing lets the reference concentrate budget).
        if self.phase != "finish":
            self.phase = "finish"
            rows = list(self.candidates) or [self.victim]
            return ActResult(_hammer(rows, self._hammer_pairs(obs, rows)))
        return ActResult(_finish())


class FinishOnlyControl:
    """Declares completion without hammering: success can never be self-asserted."""

    def act(self, *, obs: Any, messages: list[dict[str, str]]) -> ActResult:
        return ActResult(_finish())


CONTROLS = {
    "reference": ReferenceSolver,
    "finish_only": FinishOnlyControl,
    "below_threshold": BelowThresholdControl,
    "timing_blind": TimingBlindControl,
}
