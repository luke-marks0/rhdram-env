from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..tools.addressing import AddressError, AddressMapper


class DisclosureConfigError(ValueError):
    """A disclosure axis carried a value outside its closed enum (fail closed)."""


# The closed enum of each disclosure axis (SPEC §7). Validated at parse time so an
# unknown value is rejected rather than silently selecting a code path — the
# ``feedback`` axis in particular used to *fail open*, because every value other than
# the two hidden modes was treated as disclosed. @spec:disclosure-levels
DISCLOSURE_LEVELS: dict[str, frozenset[str]] = {
    "mapping": frozenset({"physical", "logical_only", "opaque_handles"}),
    "adjacency": frozenset({"exact", "candidate_set", "hidden"}),
    "victim": frozenset({"exact", "logical_addr", "row_handle", "cell_handle", "hidden_until_finish"}),
    "profile": frozenset({"public_profile_id", "family_only", "hidden"}),
    "feedback": frozenset({"full_trace", "summarized_counts", "reward_only"}),
}

# Event fields of the issued-command stream that carry no physical coordinate and are
# therefore safe to echo under a hidden mapping. This is deliberately an *allowlist*:
# the previous denylist enumerated the six DDR coordinate names, so any standard-
# specific level the worker publishes (HBM2's ``pseudochannel``, say) silently became
# public. @spec:disclosure-leakage-guard @spec:invariant-no-leakage
PUBLIC_EVENT_KEYS: frozenset[str] = frozenset({"op", "clk", "type_id", "row_hit"})


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
        """Parse a disclosure block, rejecting any value outside its closed enum.

        Every axis is validated here because an unvalidated value does not fail
        uniformly: an unknown ``mapping`` happens to fail closed (no direct address
        form is admitted) while an unknown ``feedback`` failed *open*, publishing the
        trace and the load-bearing timing digest for a task that asked for neither.
        A misspelt level is a config error, not a disclosure choice.
        """
        cfg = cfg or {}
        unknown_axes = set(cfg) - set(DISCLOSURE_LEVELS)
        if unknown_axes:
            raise DisclosureConfigError(f"unknown disclosure axis/axes {sorted(unknown_axes)}")
        for axis, allowed in DISCLOSURE_LEVELS.items():
            if axis in cfg and cfg[axis] not in allowed:
                raise DisclosureConfigError(
                    f"disclosure.{axis}={cfg[axis]!r} is not one of {sorted(allowed)}"
                )
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

    def expose_victim_address(self) -> bool:
        """Whether the victim's own *linear* address is disclosed to the policy.

        ``exact`` hands physical coordinates (and the linear address); the
        ``logical_addr`` level used by ``hidden_adjacency`` (Tier 2b, P25) hands the
        victim's numeric logical address *without* physical coordinates — the
        real-attacker-knowledge model, where
        the attacker knows its own allocation's address but the address->bank
        mapping is a per-episode secret (P24), so bank membership is not computable
        from that number and must be reverse-engineered by timing (DRAMA).
        """
        return self.victim in ("exact", "logical_addr")

    def as_public(self) -> dict[str, str]:
        return {
            "mapping": self.mapping,
            "adjacency": self.adjacency,
            "victim": self.victim,
            "profile": self.profile,
            "feedback": self.feedback,
        }

    # --- observation projection (the leakage guard) ------------------------
    def trace_disclosed(self) -> bool:
        """Whether the issued-event trace (and its timing digest) is echoed at all.

        Stated positively against the one level that discloses it, so a feedback value
        this build does not know about can never select the most informative path.
        @spec:timing-channel
        """
        return self.feedback == "full_trace"

    def project_trace(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Project the issued-event trace to the disclosed feedback level."""
        if not self.trace_disclosed():
            return []
        if self.expose_coords():
            return list(events)
        # Allowlist, not denylist: under a hidden mapping only the coordinate-free
        # fields survive, so a coordinate name this build has never seen (a future or
        # standard-specific DRAMSpec level) cannot leak through unrecognised.
        return [{k: v for k, v in e.items() if k in PUBLIC_EVENT_KEYS} for e in events]

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


def check_logical_addr(value: Any, capacity: int | None) -> int:
    """Validate a policy-supplied logical address against the schema and the device.

    Two separate fail-closed checks the environment used to skip. ``int(value)`` was
    accepting non-integers (``"2"``, ``1.9``, ``True``) that
    ``spec/schemas/action.schema.json`` types as integers, and no range check bounded
    the address to the mapped device, so an oversized value aliased an in-domain one
    (on the admitted DDR4 geometry ``0`` and ``8589934592`` decode identically) —
    splitting the functional-memory overlay, which keys on the raw linear value, from
    the DRAM location Ramulator and the disturbance model actually see. ``capacity``
    is ``None`` where no geometry is known yet, which only skips the range check.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise AddressError(
            "BAD_SCHEMA", f"logical address must be a JSON integer, got {value!r}"
        )
    if value < 0:
        raise AddressError("BAD_SCHEMA", f"logical address must be non-negative, got {value}")
    if capacity is not None and value >= capacity:
        raise AddressError(
            "BAD_SCHEMA", f"logical address {value} is outside [0,{capacity}) for this device"
        )
    return value


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
        # A bare (non-bool) int is shorthand for {"kind":"logical","addr":N} — the
        # compact HAMMER `rows`/`addrs` list (SPEC §8) is documented as a plain
        # address list, so this is the one place that needs to accept it; it still
        # goes through the normal `logical`-form disclosure/allowed_forms check
        # below, so a task that hides logical addressing still fails closed.
        if isinstance(form, int) and not isinstance(form, bool):
            form = {"kind": "logical", "addr": form}
        if not isinstance(form, dict):
            raise AddressError("BAD_SCHEMA", "address must be an object")
        kind = form.get("kind")
        if kind is None:
            raise AddressError("BAD_SCHEMA", "address is missing 'kind'")
        if kind not in self.disclosure.allowed_forms():
            raise AddressError("ADDRESS_NOT_DISCLOSED", f"address form '{kind}' is not disclosed for this task")
        if kind == "logical":
            if "addr" not in form:
                raise AddressError("BAD_SCHEMA", "logical address requires an integer 'addr'")
            return check_logical_addr(form["addr"], self.mapper.capacity)
        if kind == "physical":
            return self.mapper.encode(form)
        if kind == "handle":
            return self.handles.resolve(form.get("id"))
        raise AddressError("BAD_SCHEMA", f"unknown address kind '{kind}'")
