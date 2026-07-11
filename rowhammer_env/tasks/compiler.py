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
    "hidden_target": FamilyDef(
        "target_row_flip", Disclosure(_L, "hidden", "row_handle", "public_profile_id", "summarized_counts"), "known"
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

        ``decode`` (the worker ``DECODE`` op, P24) is provided by the env for
        discovery families running under a per-episode *secret* address mapper: the
        candidate window is built against the true mapping, and the victim's decoded
        ``(bankgroup, bank)`` is recorded so the success predicate reads the trusted
        decoded key instead of assuming the public RoBaRaCoCh linear layout. For
        RoBaRaCoCh families ``decode`` is ``None`` and the victim sits in bank 0.
        """
        fam = FAMILIES[self.family]
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
        target_bankgroup = 0
        target_bank = 0
        candidates: tuple[Candidate, ...] = ()
        if decode is not None:
            dv = decode(target_addr)
            target_bankgroup = int(dv.get("bankgroup", 0))
            target_bank = int(dv.get("bank", 0))
            if self.family == "bounded_sweep":
                candidates = self._build_candidates(
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
            target_bank=target_bank,
            target_bankgroup=target_bankgroup,
            engine_known_row=engine_known_row,
            disturbance_family=disturbance_family,
            seed=seed,
            row_bytes=row_bytes,
            candidates=candidates,
        )

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

        Bank membership under the per-episode secret mapper is not computable from
        the linear address, so candidates are found by the worker ``DECODE`` op — no
        mapper logic is duplicated in Python. The *raw* RoBaRaCoCh bit positions
        (where the Bank/BankGroup/Row fields sit before the secret XOR) are public
        geometry; enumerating the raw bank slots at a fixed row and decoding each
        yields the full decoded-bank → linear-address map for that row (a bijection
        over the bank space), from which the true same-bank neighbours (aggressors),
        same-bank-far decoys, and different-bank decoys are selected against the
        real mapping. The list is shuffled per episode so position leaks no role.
        """
        mapper = AddressMapper(geometry)  # public raw RoBaRaCoCh field positions
        row_bytes = geometry.row_stride
        target_addr = target_row * row_bytes
        sh_bank = mapper.shifts["bank"]
        bank_w = mapper.widths["bank"]
        sh_bg = mapper.shifts.get("bankgroup", 0)
        bg_w = mapper.widths.get("bankgroup", 0)
        rng = self._rng(seed, "candidates")

        def bank_map_at_row(r: int) -> dict[tuple[int, int], int]:
            """Decoded (bankgroup, bank) -> a linear address at row ``r``, column 0."""
            out: dict[tuple[int, int], int] = {}
            for bg in range(1 << bg_w):
                for bank in range(1 << bank_w):
                    addr = r * row_bytes + (bg << sh_bg) + (bank << sh_bank)
                    d = decode(addr)
                    out[(int(d["bankgroup"]), int(d["bank"]))] = addr
            return out

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
    # Decoded victim (bankgroup, bank). 0/0 under the public RoBaRaCoCh mapper (the
    # victim's linear address sits in bank 0); under a secret mapper (P24) these are
    # the true decoded coordinates, so the fixed known threshold and the success
    # predicate land on the real victim key rather than an assumed bank 0.
    target_bank: int = 0
    target_bankgroup: int = 0
    # Tier 2a candidate window (P23); empty for every non-``bounded_sweep`` family.
    candidates: tuple[Candidate, ...] = ()

    def disturbance_overrides(self) -> dict[str, Any]:
        """Engine constructor overrides implied by this compiled task."""
        overrides: dict[str, Any] = {"known_target_row": self.engine_known_row}
        if self.target_kind == "known":
            # The known target's first flipped bit is the (sampled or explicit)
            # target bit, so target-cell / pattern objectives land on it.
            overrides["known_first_bit"] = self.target_bit
            # Pin the known threshold to the victim's *decoded* bank. Defaults to
            # bank 0 (RoBaRaCoCh victim), so non-secret families are unchanged;
            # under a secret mapper the victim's scrambled bank is used instead.
            overrides["known_target_bank"] = self.target_bank
            overrides["known_target_bankgroup"] = self.target_bankgroup
        if self.disturbance_family:
            overrides["family"] = self.disturbance_family
        return overrides

    def expected_success_window(self) -> tuple[float, float]:
        return BAND_WINDOW[self.difficulty]
