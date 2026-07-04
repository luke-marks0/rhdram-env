from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..tools.addressing import COORD_KEYS, AddressError, AddressMapper


@dataclass(frozen=True)
class Disclosure:
    """A task's topology-disclosure level (SPEC §7).

    Each axis independently narrows what the policy may see and address:
    ``mapping`` decides the accepted direct address forms, ``victim``/``adjacency``
    decide whether target/candidate coordinates are handed out (and whether opaque
    handles exist), and ``feedback`` decides how much of the issued-event trace is
    echoed back. The projection helpers below are the single enforcement point for
    the leakage guard.
    """

    mapping: str = "logical_only"
    adjacency: str = "hidden"
    victim: str = "hidden_until_finish"
    profile: str = "public_profile_id"
    feedback: str = "summarized_counts"

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "Disclosure":
        cfg = cfg or {}
        return cls(
            mapping=cfg.get("mapping", "logical_only"),
            adjacency=cfg.get("adjacency", "hidden"),
            victim=cfg.get("victim", "hidden_until_finish"),
            profile=cfg.get("profile", "public_profile_id"),
            feedback=cfg.get("feedback", "summarized_counts"),
        )

    # --- address forms -----------------------------------------------------
    def direct_forms(self) -> set[str]:
        """Direct (non-handle) address forms the mapping level permits."""
        if self.mapping == "physical":
            return {"logical", "physical"}
        if self.mapping == "logical_only":
            return {"logical"}
        return set()  # opaque_handles: no direct addressing at all

    def handles_allowed(self) -> bool:
        return self.mapping == "opaque_handles" or self.victim in {"row_handle", "cell_handle"}

    def allowed_forms(self) -> set[str]:
        forms = set(self.direct_forms())
        if self.handles_allowed():
            forms.add("handle")
        return forms

    # --- leakage predicates ------------------------------------------------
    def expose_coords(self) -> bool:
        """Physical coordinates may appear in public output only under physical mapping."""
        return self.mapping == "physical"

    def expose_victim(self) -> bool:
        """Exact target coordinates are disclosed only at ``victim: exact``."""
        return self.victim == "exact"

    def as_public(self) -> dict[str, str]:
        return {
            "mapping": self.mapping,
            "adjacency": self.adjacency,
            "victim": self.victim,
            "profile": self.profile,
            "feedback": self.feedback,
        }

    # --- observation projection (the leakage guard) ------------------------
    def project_trace(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Project the issued-event trace to the disclosed feedback level."""
        if self.feedback in ("summarized_counts", "reward_only"):
            return []
        if self.expose_coords():
            return list(events)
        return [{k: v for k, v in e.items() if k not in COORD_KEYS} for e in events]

    def project_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        """Strip hidden physical state from a step's public feedback block."""
        out = dict(feedback)
        out["trace_tail"] = self.project_trace(out.get("trace_tail", []))
        # Per-flip coordinates (row/addr) are only shown when the victim is fully
        # disclosed *and* physical mapping is public; otherwise the count survives
        # (it is the reward signal) but never the location.
        if not (self.expose_victim() and self.expose_coords()):
            out.pop("public_flips", None)
        if self.feedback == "reward_only":
            out.pop("trace_tail", None)
            out.pop("new_public_flips", None)
            out.pop("oracle_refreshes", None)
        return out


class HandleTable:
    """Per-episode opaque-handle registry, resolved server-side only (SPEC §8).

    Handle ids are a hash of ``(seed, role)`` — deterministic for replay but
    non-invertible: they encode no coordinate, address, or threshold, so a policy
    cannot recover hidden physical state from the handle name.
    """

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._linear: dict[str, int] = {}

    def register(self, role: str, linear: int) -> str:
        handle_id = self._make_id(role)
        self._linear[handle_id] = int(linear)
        return handle_id

    def _make_id(self, role: str) -> str:
        digest = hashlib.sha256(f"{self._seed}:{role}".encode()).hexdigest()[:16]
        return f"h_{digest}"

    def resolve(self, handle_id: Any) -> int:
        if not isinstance(handle_id, str) or handle_id not in self._linear:
            raise AddressError("ADDRESS_NOT_DISCLOSED", "unknown or undisclosed handle")
        return self._linear[handle_id]

    def ids(self) -> set[str]:
        return set(self._linear)


class AddressResolver:
    """Resolve a policy-supplied address form to a linear worker address.

    Enforces the disclosure mapping level (fail closed with
    ``ADDRESS_NOT_DISCLOSED`` for a hidden form), then delegates: ``logical`` is
    the linear address itself, ``physical`` goes through the geometry-derived
    mapper, and ``handle`` is resolved by the per-episode table.
    """

    def __init__(self, mapper: AddressMapper, disclosure: Disclosure, handles: HandleTable) -> None:
        self.mapper = mapper
        self.disclosure = disclosure
        self.handles = handles

    def to_linear(self, form: Any) -> int:
        if not isinstance(form, dict):
            raise AddressError("BAD_SCHEMA", "address must be an object")
        kind = form.get("kind")
        if kind is None:
            raise AddressError("BAD_SCHEMA", "address is missing 'kind'")
        if kind not in self.disclosure.allowed_forms():
            raise AddressError("ADDRESS_NOT_DISCLOSED", f"address form '{kind}' is not disclosed for this task")
        if kind == "logical":
            try:
                return int(form["addr"])
            except (KeyError, TypeError, ValueError):
                raise AddressError("BAD_SCHEMA", "logical address requires an integer 'addr'")
        if kind == "physical":
            return self.mapper.encode(form)
        if kind == "handle":
            return self.handles.resolve(form.get("id"))
        raise AddressError("BAD_SCHEMA", f"unknown address kind '{kind}'")
