from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from ..geometry import Geometry
from ..tools.addressing import AddressMapper
from .disclosure import Disclosure


# --- difficulty bands ---------------------------------------------------------
#
# A difficulty band is a *budget* the reference policy is given, calibrated
# against the profile-sampled activation thresholds a graded (sampled-target)
# family faces. Because the sampled hcfirst is drawn from the profile's
# hierarchical lognormal, the fraction of seeds a fixed activation budget can
# flip is a smooth function of that budget — so three budgets give three
# genuinely graded success-rate windows (SPEC §7 family 10 difficulty; P13
# task 3). ``BAND_ACTS`` is the calibrated activation budget; ``BAND_WINDOW`` is
# the expected reference-policy success-rate window used by
# ``scripts/verify_phase13.py``. The known-target (deterministic) families keep
# the fixed calibrated hcfirst, so their bands are near-0 / near-1.
BAND_ACTS: dict[str, int] = {"easy": 150_000, "medium": 20_000, "hard": 5_000}
BAND_WINDOW: dict[str, tuple[float, float]] = {
    "easy": (0.85, 1.0),
    "medium": (0.25, 0.75),
    "hard": (0.0, 0.20),
}
DEFAULT_BAND = "medium"

# Candidate-window width for the Tier 2a ``bounded_sweep`` discovery family (P23).
# Difficulty scales with how many candidate handles the policy must narrow by
# bank-conflict timing before hammering the survivors — ``N`` per band. Only the
# two true aggressors (the victim's immediate same-bank neighbours) actually flip;
# the rest are same-bank-far / different-bank decoys.
BAND_CANDIDATES: dict[str, int] = {"easy": 4, "medium": 16, "hard": 64}

# A same-bank decoy must sit well outside the physical blast neighbourhood
# (DDR4 blast is distance 1) so it is a genuine non-aggressor — no exposure ever
# reaches the victim from it, whatever the profile.
FAR_ROW_MARGIN = 64

# Legacy budgets for the ``{"family": ...}`` shorthand (no acts budget so the
# deterministic phase-4..9 fixtures, which hammer freely, stay unbounded).
LEGACY_BUDGETS: dict[str, int] = {"tool_calls": 20_000, "cycles": 5_000_000}

# Reserved known-target row for *sampled*-target families. The engine still keeps
# one fixed-threshold "known" row (SPEC §5.4 reproducibility fixture); for graded
# families nothing hammers it, and the task target is a separately sampled row.
RESERVED_KNOWN_ROW = 10

# Disturbance sub-families in the admitted DDR4 profile, split for the
# profile-generalization family (SPEC §7 family 10): a task drawn on the eval
# split uses a chip family held out of the train split.
TRAIN_FAMILIES = ("hisasa", "hyhy", "sasa")
EVAL_FAMILIES = ("axmicr",)


@dataclass(frozen=True)
class FamilyDef:
    """Static shape of a task family (SPEC §7): objective + default disclosure."""

    objective_type: str
    disclosure: Disclosure
    target_kind: str  # "known" (fixed calibrated hcfirst) | "sampled" (graded)
    graded: bool = False  # difficulty band drives an acts budget
    secret_mapping: bool = False  # per-episode secret address mapper (P24 discovery)


@dataclass(frozen=True)
class Candidate:
    """One entry in a ``bounded_sweep`` candidate window (P23, Tier 2a).

    ``offset`` is the linear byte distance from the victim's address (so the
    handle registers at ``target_addr + offset``); ``role`` is the trusted ground
    truth — used only server-side for calibration/tests, never disclosed. The
    policy sees an opaque handle whose *position* in the candidate list carries no
    role information (the list is shuffled per episode), so which handle is the
    real aggressor is derivable only from the trusted final flip.
    """

    offset: int
    role: str  # "aggressor" | "same_bank_far" | "different_bank"

    @property
    def is_aggressor(self) -> bool:
        return self.role == "aggressor"


_P = "physical"
_L = "logical_only"


# The ten SPEC §7 families. ``known_target_anybit`` keeps its historical name
# (aliased below to ``known_target``); ``target_row`` is the graded, profile-
# sampled variant used for difficulty calibration.
FAMILIES: dict[str, FamilyDef] = {
    "known_target_anybit": FamilyDef(
        "target_row_flip", Disclosure(_P, "exact", "exact", "public_profile_id", "summarized_counts"), "known"
    ),
    "target_row": FamilyDef(
        "target_row_flip", Disclosure(_P, "exact", "exact", "public_profile_id", "summarized_counts"),
        "sampled", graded=True,
    ),
    "target_cell": FamilyDef(
        "target_cell_flip", Disclosure(_P, "exact", "exact", "public_profile_id", "summarized_counts"), "known"
    ),
    "pattern_target": FamilyDef(
        "pattern_target", Disclosure(_P, "exact", "exact", "public_profile_id", "summarized_counts"), "known"
    ),
    "any_flip": FamilyDef(
        "any_flip", Disclosure(_P, "exact", "hidden_until_finish", "public_profile_id", "summarized_counts"),
        "sampled", graded=True,
    ),
    # DEPRECATED (V3 §5): the fully-opaque, zero-addressability victim (a handle
    # with no numeric address and no addressable neighbours) asks for stricter-than-
    # real-attacker knowledge and is dropped in favour of ``hidden_adjacency`` (the
    # numeric real-attacker redefinition, P25). Kept working for the existing low-
    # disclosure regression/leakage tests; do not build new tasks on it.
    "hidden_target": FamilyDef(
        "target_row_flip", Disclosure(_L, "hidden", "row_handle", "public_profile_id", "summarized_counts"), "known"
    ),
    "hidden_adjacency": FamilyDef(
        # Tier 2b (P25): the real-attacker-knowledge model. The victim's own numeric
        # (logical) address and a set of *numeric* candidate addresses are disclosed
        # (``victim: logical_addr``), but the address->bank mapping is a per-episode
        # secret (``secret_mapping``), so which candidates are same-bank neighbours
        # must be reverse-engineered from the bank-conflict timing channel (§0.2),
        # not computed from the address. ``full_trace`` so that timing reaches the
        # policy; ``known`` target so a found aggressor flips reliably.
        "target_row_flip", Disclosure(_L, "candidate_set", "logical_addr", "public_profile_id", "full_trace"),
        "known", secret_mapping=True,
    ),
    "unknown_adjacency": FamilyDef(
        "target_row_flip", Disclosure(_L, "candidate_set", "row_handle", "public_profile_id", "summarized_counts"),
        "known",
    ),
    "bounded_sweep": FamilyDef(
        # Tier 2a (P23): an ``N``-handle candidate window at a controlled mix of
        # same-bank/different-bank, adjacent/far offsets, only the two immediate
        # same-bank neighbours of which are true aggressors. ``full_trace`` so the
        # bank-conflict timing side channel (§0.2) actually reaches the policy;
        # ``known`` target so a found aggressor flips reliably (the discovery
        # difficulty is the candidate window, not a variable threshold).
        # ``secret_mapping`` (P24): a per-episode secret row->bank mapper makes bank
        # membership non-computable from the linear address, so the candidate
        # window is built against the true mapping via the worker DECODE op and the
        # success predicate reads the *decoded* victim key, not linear arithmetic.
        "target_row_flip", Disclosure(_L, "candidate_set", "row_handle", "public_profile_id", "full_trace"),
        "known", secret_mapping=True,
    ),
    "mitigation_aware": FamilyDef(
        "mitigation_aware", Disclosure(_P, "exact", "exact", "public_profile_id", "summarized_counts"), "known"
    ),
    "low_disclosure": FamilyDef(
        "target_row_flip", Disclosure(_L, "hidden", "hidden_until_finish", "family_only", "reward_only"), "known"
    ),
    "profile_generalization": FamilyDef(
        "any_flip", Disclosure(_P, "exact", "hidden_until_finish", "family_only", "summarized_counts"),
        "sampled", graded=True,
    ),
}

# Backward-compatible alias.
FAMILY_ALIASES = {"known_target": "known_target_anybit"}

# objective.type -> a canonical family for full SPEC §10 configs that do not name
# a family explicitly. Ambiguous ``target_row_flip`` is disambiguated by the
# disclosure level in ``_derive_family``.
_OBJECTIVE_FAMILY = {
    "any_flip": "any_flip",
    "target_cell_flip": "target_cell",
    "pattern_target": "pattern_target",
    "mitigation_aware": "mitigation_aware",
}


class TaskConfigError(ValueError):
    """Raised for a task config the compiler cannot admit (fail closed)."""


def _canonical_family(name: str) -> str:
    name = FAMILY_ALIASES.get(name, name)
    if name not in FAMILIES:
        raise TaskConfigError(f"unknown task family {name!r}")
    return name


def _derive_family(config: dict[str, Any]) -> str:
    """Infer the family of a full SPEC §10 config that does not name one."""
    objective = config.get("objective") or {}
    otype = objective.get("type", "target_row_flip")
    if otype in _OBJECTIVE_FAMILY:
        return _OBJECTIVE_FAMILY[otype]
    # target_row_flip: disambiguate by disclosure.
    disc = Disclosure.from_config(config.get("disclosure"))
    if disc.victim == "logical_addr":
        return "hidden_adjacency"
    if disc.adjacency == "candidate_set":
        return "unknown_adjacency"
    if disc.mapping == "physical" and disc.victim == "exact":
        return "known_target_anybit"
    if disc.victim in ("row_handle", "cell_handle"):
        return "hidden_target"
    return "low_disclosure"


@dataclass(frozen=True)
class TaskSpec:
    """Geometry-independent parse of a task config (SPEC §10).

    Holds everything decided before the worker reports geometry: the family, the
    disclosure level, the objective, the mitigation, budgets, the difficulty
    band, and target *hints* (explicit bit/mask/value). ``compile()`` then samples
    the concrete target from the episode seed once geometry is known.
    """

    task_id: str
    family: str
    objective_type: str
    disclosure: Disclosure
    mitigation: dict[str, Any]
    reward_kind: str
    difficulty: str
    budgets: dict[str, int] | None
    disturbance_family: str | None
    profile_id: str | None
    standard: str
    # explicit target hints (None -> sampled)
    bit_hint: int | None = None
    mask_hint: int | None = None
    value_hint: int | None = None
    split: str | None = None

    @property
    def graded(self) -> bool:
        return FAMILIES[self.family].graded

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "TaskSpec":
        config = dict(config or {})
        family = config.get("family")
        if family:
            family = _canonical_family(family)
        elif not config:
            # The bare task env (no config) keeps the historical default family.
            family = "known_target_anybit"
        else:
            family = _derive_family(config)
        fam = FAMILIES[family]

        disclosure = Disclosure.from_config(config["disclosure"]) if "disclosure" in config else fam.disclosure

        objective = dict(config.get("objective") or {})
        objective_type = objective.get("type", fam.objective_type)

        mitigation = config.get("mitigation") or {"name": "none", "params": {}}
        if "name" not in mitigation:
            raise TaskConfigError("mitigation requires a name")
        mitigation = {"name": mitigation["name"], "params": dict(mitigation.get("params") or {})}

        budgets_cfg = config.get("budgets")
        budgets = {str(k): int(v) for k, v in budgets_cfg.items()} if budgets_cfg else None

        # Graded families default to the middle band; the deterministic
        # known-target families flip reliably under an unbounded budget, so they
        # report ``easy`` unless the config pins a band.
        difficulty = config.get("difficulty")
        if difficulty is None:
            difficulty = DEFAULT_BAND if FAMILIES[family].graded else "easy"
        difficulty = str(difficulty)
        if difficulty not in BAND_ACTS:
            raise TaskConfigError(f"unknown difficulty band {difficulty!r}")

        split = objective.get("split") or config.get("split")
        disturbance_family = config.get("disturbance_family")

        task_id = str(config.get("id") or f"ddr4_{family}_v1")
        reward_kind = str(config.get("reward", "sparse_success"))
        standard = str(config.get("standard", "DDR4"))
        profile_id = config.get("profile")

        def _int_or_none(*keys: str) -> int | None:
            for src in (config, objective):
                for k in keys:
                    if k in src:
                        return int(src[k])
            return None

        return cls(
            task_id=task_id,
            family=family,
            objective_type=objective_type,
            disclosure=disclosure,
            mitigation=mitigation,
            reward_kind=reward_kind,
            difficulty=difficulty,
            budgets=budgets,
            disturbance_family=disturbance_family,
            profile_id=profile_id,
            standard=standard,
            bit_hint=_int_or_none("bit"),
            mask_hint=_int_or_none("mask"),
            value_hint=_int_or_none("value"),
            split=split,
        )

    # --- compilation ------------------------------------------------------
    def resolved_budgets(self) -> dict[str, int]:
        """Episode budgets: explicit config budgets, else a band/legacy default."""
        if self.budgets is not None:
            budgets = dict(self.budgets)
        elif self.graded:
            budgets = {"tool_calls": BAND_ACTS[self.difficulty] + 1000, "acts": BAND_ACTS[self.difficulty],
                       "cycles": 5_000_000}
        else:
            budgets = dict(LEGACY_BUDGETS)
        return budgets

    def compile(
        self,
        seed: int,
        geometry: Geometry,
        decode: "Callable[[int], dict[str, int]] | None" = None,
    ) -> "CompiledTask":
        """Compile the episode target + candidate window against the real geometry.

        ``decode`` is the worker ``DECODE`` op (P24). Whether it is *required* is a
        property of the family, not of the caller's intent: a ``secret_mapping``
        family is exactly the family the env runs under a per-episode secret mapper
        (``phase5_env._active_mapper``), and under that mapper neither the victim's
        bank nor the candidate window is computable in Python — so compiling one
        without a decoder fails closed here rather than yielding a bank-0 victim key
        no flip can ever match and an empty candidate window. Every other family runs
        under the public ``RoBaRaCoCh`` mapper, whose decode the Python projection
        reproduces exactly, so ``decode`` is optional for them. Either way
        :meth:`_decode_victim` resolves the victim's full decoded key, so no
        coordinate of it is assumed.
        """
        fam = FAMILIES[self.family]
        if fam.secret_mapping and decode is None:
            raise TaskConfigError(
                f"task family {self.family!r} runs under a per-episode secret address "
                "mapper; compiling it requires the worker DECODE op"
            )
        row_count = int(geometry.level_sizes.get("row", 1 << 16))
        row_bytes = geometry.row_stride
        rng = self._rng(seed, "target")

        if fam.target_kind == "known":
            # Sample the *row index* of the known target (no longer hardcoded 10);
            # it still routes through the engine's fixed-threshold known row so the
            # deterministic fixtures remain reproducible.
            target_row = rng.randrange(RESERVED_KNOWN_ROW + 1, max(RESERVED_KNOWN_ROW + 2, row_count - 2))
            engine_known_row = target_row
        else:
            # Graded family: the target is a sampled row that gets a profile-sampled
            # threshold. Keep it clear of the reserved known row + neighbours.
            target_row = rng.randrange(1024, max(1025, row_count - 1024))
            engine_known_row = RESERVED_KNOWN_ROW

        bit = self.bit_hint if self.bit_hint is not None else self._rng(seed, "bit").randrange(8)
        bit &= 0x7
        if self.family == "pattern_target":
            value = self.value_hint if self.value_hint is not None else (1 << bit)
            mask = self.mask_hint if self.mask_hint is not None else (1 << bit)
        else:
            value = self.value_hint if self.value_hint is not None else (1 << bit)
            mask = self.mask_hint if self.mask_hint is not None else 0xFF

        disturbance_family = self._resolve_disturbance_family(seed)

        target_addr = target_row * row_bytes
        victim = self._decode_victim(geometry, target_addr, decode)
        target_channel = int(victim.get("channel", 0))
        target_rank = int(victim.get("rank", 0))
        target_bankgroup = int(victim.get("bankgroup", 0))
        target_bank = int(victim.get("bank", 0))
        candidates: tuple[Candidate, ...] = ()
        if fam.secret_mapping:
            # A candidate window exists because the family is a discovery family —
            # not because a decoder happened to be supplied. The guard at the top of
            # ``compile`` has already established that ``decode`` is the worker's, so
            # the window is built against the true secret mapping.
            if self.family == "bounded_sweep":
                candidates = self._build_candidates(
                    seed, geometry, target_row, row_count, (target_bankgroup, target_bank), decode
                )
            elif self.family == "hidden_adjacency":
                candidates = self._build_numeric_candidates(
                    seed, geometry, target_row, row_count, (target_bankgroup, target_bank), decode
                )

        return CompiledTask(
            task_id=self.task_id,
            family=self.family,
            objective_type=self.objective_type,
            disclosure=self.disclosure,
            mitigation=self.mitigation,
            reward_kind=self.reward_kind,
            difficulty=self.difficulty,
            budgets=self.resolved_budgets(),
            target_kind=fam.target_kind,
            target_row=target_row,
            target_column=0,
            target_bit=bit,
            target_value=int(value) & 0xFF,
            target_mask=int(mask) & 0xFF,
            target_addr=target_addr,
            target_channel=target_channel,
            target_rank=target_rank,
            target_bank=target_bank,
            target_bankgroup=target_bankgroup,
            engine_known_row=engine_known_row,
            disturbance_family=disturbance_family,
            seed=seed,
            row_bytes=row_bytes,
            candidates=candidates,
        )

    # @spec:task-compiler
    @staticmethod
    def _decode_victim(
        geometry: Geometry,
        target_addr: int,
        decode: "Callable[[int], dict[str, int]] | None",
    ) -> dict[str, int]:
        """Trusted decoded coordinates of the victim's linear address.

        Every level the success predicate's row key needs — channel and rank as
        well as bankgroup and bank — comes from a decode of the victim address, so
        none of them is assumed. Under a per-episode secret mapper the worker
        ``DECODE`` op is the only authority (and :meth:`compile` refuses to compile
        such a family without it); under the public ``RoBaRaCoCh`` mapper
        the Python projection in ``tools.addressing`` reproduces the worker's decode
        exactly (``mappers.is_python_projectable``), so it answers the same question
        without a worker round trip. A geometry that projection cannot represent
        (multi-channel, per ``@spec:sim-not-modeled``) fails closed here rather than
        producing a key that silently never matches a flip.
        """
        if decode is not None:
            return decode(target_addr)
        try:
            return AddressMapper(geometry).decode(target_addr)
        except ValueError as exc:
            raise TaskConfigError(f"cannot decode the victim address for this geometry: {exc}") from exc

    @staticmethod
    def _bank_slot_addrs(
        mapper: AddressMapper,
        row_bytes: int,
        r: int,
        decode: "Callable[[int], dict[str, int]]",
    ) -> dict[tuple[int, int], int]:
        """Decoded ``(bankgroup, bank)`` -> a linear address at row ``r``, column 0.

        Bank membership under the per-episode secret mapper is not computable from
        the linear address, so it is read from the worker ``DECODE`` op — no mapper
        logic is duplicated in Python. The *raw* RoBaRaCoCh bit positions (where the
        Bank/BankGroup/Row fields sit before the secret XOR) are public geometry;
        enumerating the raw bank slots at a fixed row and decoding each yields the
        full decoded-bank → linear-address map for that row (a bijection over the
        bank space), which both candidate builders use to select true same-bank
        neighbours (aggressors) and same/different-bank decoys against the real
        mapping.
        """
        sh_bank = mapper.shifts["bank"]
        bank_w = mapper.widths["bank"]
        sh_bg = mapper.shifts.get("bankgroup", 0)
        bg_w = mapper.widths.get("bankgroup", 0)
        out: dict[tuple[int, int], int] = {}
        for bg in range(1 << bg_w):
            for bank in range(1 << bank_w):
                addr = r * row_bytes + (bg << sh_bg) + (bank << sh_bank)
                d = decode(addr)
                out[(int(d["bankgroup"]), int(d["bank"]))] = addr
        return out

    def _build_candidates(
        self,
        seed: int,
        geometry: Geometry,
        target_row: int,
        row_count: int,
        victim_bank: tuple[int, int],
        decode: "Callable[[int], dict[str, int]]",
    ) -> tuple[Candidate, ...]:
        """Build the ``bounded_sweep`` candidate window against the *secret* mapper (P24).

        Tier 2a candidates are opaque **handles** whose linear offset is hidden, so
        decoy placement need only control the bank/adjacency *mix*: two same-bank
        neighbours (aggressors), same-bank-far decoys, and different-bank decoys. The
        list is shuffled per episode so position leaks no role. (``hidden_adjacency``
        — :meth:`_build_numeric_candidates` — additionally hides proximity, because
        there the address itself is disclosed.)
        """
        mapper = AddressMapper(geometry)  # public raw RoBaRaCoCh field positions
        row_bytes = geometry.row_stride
        target_addr = target_row * row_bytes
        bank_w = mapper.widths["bank"]
        bg_w = mapper.widths.get("bankgroup", 0)
        rng = self._rng(seed, "candidates")

        def bank_map_at_row(r: int) -> dict[tuple[int, int], int]:
            return self._bank_slot_addrs(mapper, row_bytes, r, decode)

        cands: list[Candidate] = []

        # Two true aggressors: the victim's immediate physical neighbours — the
        # linear addresses that *decode* to (victim bank, row ± 1) under the secret
        # mapping (not victim ± row_bytes, which the row->bank XOR sends elsewhere).
        for drow in (-1, 1):
            addr = bank_map_at_row(target_row + drow)[victim_bank]
            cands.append(Candidate(addr - target_addr, "aggressor"))

        n_decoy = BAND_CANDIDATES[self.difficulty] - len(cands)
        n_banks = (1 << bg_w) * (1 << bank_w)
        n_diff = n_decoy // 2 if n_banks > 1 else 0
        n_far = n_decoy - n_diff

        # Same-bank, far-from-victim decoys: the victim's bank at a row well outside
        # the blast neighbourhood, so no exposure reaches the victim.
        far_pool = [r for r in range(row_count) if abs(r - target_row) > FAR_ROW_MARGIN]
        for r in rng.sample(far_pool, min(n_far, len(far_pool))):
            cands.append(Candidate(bank_map_at_row(r)[victim_bank] - target_addr, "same_bank_far"))

        # Different-bank decoys: any decoded bank other than the victim's, spread
        # over rows so decoys reusing a bank still yield distinct addresses.
        made = 0
        r = (target_row + FAR_ROW_MARGIN + 1) % row_count
        while made < n_diff:
            others = [(b, a) for b, a in bank_map_at_row(r).items() if b != victim_bank]
            rng.shuffle(others)
            for _bank, addr in others:
                if made >= n_diff:
                    break
                cands.append(Candidate(addr - target_addr, "different_bank"))
                made += 1
            r = (r + FAR_ROW_MARGIN + 1) % row_count
            if abs(r - target_row) <= FAR_ROW_MARGIN:
                r = (r + FAR_ROW_MARGIN + 1) % row_count

        # Shuffle so a candidate's list position never encodes its role.
        rng.shuffle(cands)
        return tuple(cands)

    def _build_numeric_candidates(
        self,
        seed: int,
        geometry: Geometry,
        target_row: int,
        row_count: int,
        victim_bank: tuple[int, int],
        decode: "Callable[[int], dict[str, int]]",
    ) -> tuple[Candidate, ...]:
        """Build the Tier 2b (``hidden_adjacency``) numeric candidate window (P25).

        Same real-decode search as :meth:`_build_candidates`, but Tier 2b discloses
        each candidate as its numeric *logical address*, so placement must defeat a
        second shortcut on top of the secret bank function: **proximity/arithmetic**.
        The two aggressors sit at the victim's immediate physical neighbours (same
        bank, rows ± 1); they are camouflaged by *different-bank* decoys placed at
        the **same two adjacent rows**, balanced across both sides, so the aggressors
        are not the numerically-closest candidates and no disclosed address reveals
        which candidate is the real one — only the bank-conflict timing channel and
        the trusted final flip can (SPEC §8/§9). ``victim ± row_stride`` is itself one
        of those different-bank near decoys (the row->bank XOR sends it to another
        bank), so a numeric-arithmetic control hammers a different bank and fails
        (P25 done-when). Any decoys beyond the adjacent shell (only needed for the
        wider bands) are same-bank-far / different-bank far distractors.
        """
        mapper = AddressMapper(geometry)
        row_bytes = geometry.row_stride
        target_addr = target_row * row_bytes
        bank_w = mapper.widths["bank"]
        bg_w = mapper.widths.get("bankgroup", 0)
        n_banks = (1 << bg_w) * (1 << bank_w)
        rng = self._rng(seed, "candidates")

        def bank_map_at_row(r: int) -> dict[tuple[int, int], int]:
            return self._bank_slot_addrs(mapper, row_bytes, r, decode)

        cands: list[Candidate] = []

        # Two true aggressors: same bank, immediate physical neighbours (rows ± 1).
        for drow in (-1, 1):
            addr = bank_map_at_row(target_row + drow)[victim_bank]
            cands.append(Candidate(addr - target_addr, "aggressor"))

        n_decoy = BAND_CANDIDATES[self.difficulty] - len(cands)

        # Adjacent shell: the *other* bank slots at rows ± 1 — same numeric proximity
        # as the aggressors, so proximity/arithmetic cannot isolate them. Drawn
        # round-robin from both adjacent rows so each aggressor is hidden among
        # same-row, same-proximity different-bank decoys.
        near_by_row: dict[int, list[int]] = {}
        for drow in (-1, 1):
            slots = bank_map_at_row(target_row + drow)
            others = [addr for bank, addr in slots.items() if bank != victim_bank]
            rng.shuffle(others)
            near_by_row[drow] = others
        while (near_by_row[-1] or near_by_row[1]) and len(cands) - 2 < n_decoy:
            for drow in (-1, 1):
                if near_by_row[drow] and len(cands) - 2 < n_decoy:
                    cands.append(Candidate(near_by_row[drow].pop() - target_addr, "different_bank"))

        # Outer decoys: only reached when the adjacent shell cannot fill the band
        # (the widest bands). Same-bank-far survives the timing probe but never
        # flips the victim; different-bank far adds volume. Proximity camouflage is
        # already provided by the adjacent shell, so these may sit at far rows.
        remaining = n_decoy - (len(cands) - 2)
        if remaining > 0:
            n_far_same = remaining // 2 if n_banks > 1 else remaining
            far_pool = [r for r in range(row_count) if abs(r - target_row) > FAR_ROW_MARGIN]
            for r in rng.sample(far_pool, min(n_far_same, len(far_pool))):
                cands.append(Candidate(bank_map_at_row(r)[victim_bank] - target_addr, "same_bank_far"))
            made = 0
            n_far_diff = remaining - n_far_same
            r = (target_row + FAR_ROW_MARGIN + 1) % row_count
            while made < n_far_diff and n_banks > 1:
                others = [(b, a) for b, a in bank_map_at_row(r).items() if b != victim_bank]
                rng.shuffle(others)
                for _bank, addr in others:
                    if made >= n_far_diff:
                        break
                    cands.append(Candidate(addr - target_addr, "different_bank"))
                    made += 1
                r = (r + FAR_ROW_MARGIN + 1) % row_count
                if abs(r - target_row) <= FAR_ROW_MARGIN:
                    r = (r + FAR_ROW_MARGIN + 1) % row_count

        # Shuffle so a candidate's list position never encodes its role.
        rng.shuffle(cands)
        return tuple(cands)

    def _resolve_disturbance_family(self, seed: int) -> str | None:
        if self.disturbance_family:
            return self.disturbance_family
        if self.family != "profile_generalization":
            return None
        pool = EVAL_FAMILIES if self.split == "eval" else TRAIN_FAMILIES
        return pool[self._rng(seed, "chip").randrange(len(pool))]

    def _rng(self, seed: int, salt: str) -> random.Random:
        digest = hashlib.sha256(f"{self.task_id}:{seed}:{salt}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class CompiledTask:
    """A concrete episode: the sampled target + resolved disclosure/budget/reward."""

    task_id: str
    family: str
    objective_type: str
    disclosure: Disclosure
    mitigation: dict[str, Any]
    reward_kind: str
    difficulty: str
    budgets: dict[str, int]
    target_kind: str
    target_row: int
    target_column: int
    target_bit: int
    target_value: int
    target_mask: int
    target_addr: int
    engine_known_row: int
    disturbance_family: str | None
    seed: int
    row_bytes: int
    # The victim's *decoded* physical coordinates, from the worker DECODE op under a
    # secret mapper (P24) and from the equivalent public projection otherwise — never
    # assumed. All zero under the public RoBaRaCoCh mapper, whose row-aligned victim
    # address has every sub-row field at 0; a secret mapper scrambles the bank, and a
    # multi-rank part can place the victim off rank 0. The fixed known threshold and
    # the success predicate both key on these, so they must be the real ones.
    target_channel: int = 0
    target_rank: int = 0
    target_bank: int = 0
    target_bankgroup: int = 0
    # Tier 2a candidate window (P23); empty for every non-``bounded_sweep`` family.
    candidates: tuple[Candidate, ...] = ()

    @property
    def target_row_key(self) -> tuple[int, ...]:
        """The victim's decoded ``(channel, rank, bankgroup, bank, row)`` key.

        Same shape and origin as the keys ``DisturbanceEngine.flipped_row_keys``
        holds (both decode the same address space), so the target-row success
        predicates compare the two directly.
        """
        return (
            self.target_channel,
            self.target_rank,
            self.target_bankgroup,
            self.target_bank,
            self.target_row,
        )

    def disturbance_overrides(self) -> dict[str, Any]:
        """Engine constructor overrides implied by this compiled task."""
        overrides: dict[str, Any] = {"known_target_row": self.engine_known_row}
        if self.target_kind == "known":
            # The known target's first flipped bit is the (sampled or explicit)
            # target bit, so target-cell / pattern objectives land on it.
            overrides["known_first_bit"] = self.target_bit
            # Pin the known threshold to the victim's *decoded* coordinates, so the
            # fixed calibrated threshold lands on the same key the success predicate
            # reads rather than on an assumed channel 0 / rank 0 / bank 0.
            overrides["known_target_channel"] = self.target_channel
            overrides["known_target_rank"] = self.target_rank
            overrides["known_target_bank"] = self.target_bank
            overrides["known_target_bankgroup"] = self.target_bankgroup
        if self.disturbance_family:
            overrides["family"] = self.disturbance_family
        return overrides

    def expected_success_window(self) -> tuple[float, float]:
        return BAND_WINDOW[self.difficulty]
