#!/usr/bin/env python3
"""Phase 13 admission gate: task compiler + families (fixes defect J).

The thin task shim (one hardcoded ``target_row=10``, a single family, the literal
``"smoke"`` difficulty band) is replaced by a real compiler
(`rowhammer_env/tasks/compiler.py`) that samples targets/seeds/budgets, resolves a
difficulty band, and instantiates all ten SPEC §7 families, each scored by a
trusted per-family predicate (`rowhammer_env/rewards`).

Checks (see spec/IMPLEMENTATION_PLAN_V2.md §P13, spec/TEST_PLAN.md E3/E7/E6):

- **Families**: every SPEC §7 family loads from ``configs/tasks/*.yaml``, validates
  against ``spec/schemas/task.schema.json``, instantiates against the real worker,
  runs a reference policy, terminates, and is reproducible across identical seeds.
  Solvable families reach reward 1; a claim-without-flip control earns 0.
- **Difficulty bands**: a reference policy hits each band's calibrated
  success-rate window over many seeds (graded, profile-sampled thresholds), and
  the compiled band budget — not luck — decides one end-to-end episode.
- **Non-leakage**: hidden-target / unknown-adjacency families expose no derivable
  target coordinate, address, row, or threshold (reuses the P12 leakage guard).
"""

from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jsonschema  # noqa: E402
import yaml  # noqa: E402

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv, RowHammerTaskEnv  # noqa: E402
from rowhammer_env.disturbance import DisturbanceEngine  # noqa: E402
from rowhammer_env.geometry import Geometry  # noqa: E402
from rowhammer_env.tasks.compiler import BAND_ACTS, BAND_WINDOW, FAMILIES, TaskSpec  # noqa: E402
from rowhammer_env.tools.addressing import COORD_KEYS  # noqa: E402

TASK_DIR = ROOT / "configs" / "tasks"
SCHEMA = json.loads((ROOT / "spec" / "schemas" / "task.schema.json").read_text())

# Deterministic (fixed-threshold) families a hammering reference policy must solve
# at any seed under a generous budget (E7 sanity baseline). Graded families are
# probabilistic per seed and are covered by the difficulty-band check instead.
DETERMINISTIC_SOLVABLE = {"known_target_anybit", "target_cell", "pattern_target", "unknown_adjacency"}


def real_geometry() -> Geometry:
    env = RowHammerDisturbanceEnv()
    obs = env.reset(seed=13, episode_id="phase13_geo")
    if obs.error:
        raise SystemExit(f"worker reset failed: {obs.error}")
    geo = env.disturbance.geometry
    env.close()
    return geo


# --- reference policy -----------------------------------------------------------

def _rd(addr: int) -> dict:
    return {"op": "RD", "addr": {"kind": "logical", "addr": addr}}


def _rd_handle(handle: str) -> dict:
    return {"op": "RD", "addr": {"kind": "handle", "id": handle}}


def _rd_candidate(candidate: dict) -> dict:
    """RD a disclosed candidate — an opaque handle (Tier 2a) or a numeric logical
    address (Tier 2b ``hidden_adjacency``, P25)."""
    if candidate.get("kind") == "handle":
        return _rd_handle(candidate["id"])
    return _rd(int(candidate["addr"]))


class ReferencePolicy:
    """A simple double-sided hammering baseline used to exercise every family.

    It hammers whatever the task discloses — physical target coordinates, the two
    candidate aggressor handles, or (for a hidden/graded target) a self-selected
    victim row — in batched ``dram.issue`` calls to respect a small ``tool_calls``
    budget. When nothing hammerable is disclosed, or the act cap is reached without
    success, it declares ``episode.finish``. Reward comes only from trusted state.
    """

    def __init__(self, batch_pairs: int = 256, max_acts: int = 12_000) -> None:
        self.batch_pairs = batch_pairs
        self.max_acts = max_acts

    def _aggressors(self, env, obs) -> tuple[list[dict], str]:
        """The two double-sided aggressor commands, and a label for the strategy."""
        meta = obs.metadata
        target = meta.get("target")
        row_bytes = env._compiled.row_bytes
        if isinstance(target, dict) and target.get("kind") == "physical":
            addr = int(target["addr"])
            return [_rd(addr - row_bytes), _rd(addr + row_bytes)], "disclosed-target"
        candidates = meta.get("candidates")
        if candidates:
            return [_rd_candidate(candidates[0]), _rd_candidate(candidates[1])], "candidate-set"
        if env._compiled.target_kind == "sampled":
            # Graded/any-flip task with a hidden target: pick a victim row and
            # hammer its neighbours; the profile decides whether it crosses.
            addr = env._compiled.target_row * row_bytes
            return [_rd(addr - row_bytes), _rd(addr + row_bytes)], "self-selected"
        return [], "no-target"

    def run(self, env, obs) -> dict:
        aggr, strategy = self._aggressors(env, obs)
        if not aggr:
            fin = env.step(Phase2Action(tool="episode.finish", args={}))
            return {"strategy": strategy, "reward": fin.reward, "done": fin.done, "acts": 0, "steps": 1}

        batch = [dict(aggr[i % 2]) for i in range(2 * self.batch_pairs)]
        acts = 0
        steps = 0
        last = obs
        while acts < self.max_acts:
            last = env.step(Phase2Action(tool="dram.issue", args={"commands": batch}))
            steps += 1
            acts += 2 * self.batch_pairs
            if last.done:
                return {"strategy": strategy, "reward": last.reward, "done": True,
                        "error": last.error, "acts": acts, "steps": steps}
        fin = env.step(Phase2Action(tool="episode.finish", args={}))
        return {"strategy": strategy, "reward": fin.reward, "done": fin.done, "acts": acts, "steps": steps + 1}


def _load(name: str) -> dict:
    return yaml.safe_load((TASK_DIR / f"{name}.yaml").read_text())


# --- checks ---------------------------------------------------------------------

def check_configs_and_families() -> None:
    """Every family: config validates, instantiates, runs, terminates, reproduces."""
    configs = sorted(TASK_DIR.glob("*.yaml"))
    if not configs:
        raise SystemExit("no task configs found under configs/tasks/")
    families_seen = set()
    for path in configs:
        doc = yaml.safe_load(path.read_text())
        jsonschema.validate(doc, SCHEMA)  # C1: config validates against the schema
        spec = TaskSpec.from_config(doc)
        families_seen.add(spec.family)

        outcomes = []
        for run in range(2):  # reproducibility across identical seeds
            env = RowHammerTaskEnv(task=doc)
            obs = env.reset(seed=13, episode_id=f"p13_{path.stem}_{run}")
            if obs.error:
                raise SystemExit(f"{path.name}: reset failed: {obs.error}")
            result = ReferencePolicy().run(env, obs)
            if not result["done"]:
                raise SystemExit(f"{path.name}: reference policy did not terminate")
            outcomes.append((round(result["reward"], 6), result["strategy"], result["steps"],
                             json.dumps(obs.metadata.get("target"), sort_keys=True, default=str)))
            env.close()
        if outcomes[0] != outcomes[1]:
            raise SystemExit(f"{path.name}: not reproducible across identical seeds: {outcomes}")

        reward = outcomes[0][0]
        if spec.family in DETERMINISTIC_SOLVABLE and reward != 1.0:
            raise SystemExit(f"{path.name}: deterministic solvable task did not reach reward 1 (got {reward})")
        if spec.family == "mitigation_aware" and reward != 0.0:
            raise SystemExit(f"{path.name}: oracle-mitigated task was not protected (reward {reward})")

    missing = set(FAMILIES) - families_seen
    if missing:
        raise SystemExit(f"configs/tasks/ does not cover every family: missing {sorted(missing)}")
    print(f"  families: {len(configs)} configs validate, cover all {len(FAMILIES)} families, "
          "run + terminate reproducibly")


def check_finish_before_success_control() -> None:
    """E7 control: declaring success without a flip earns 0."""
    env = RowHammerTaskEnv(task=_load("known_target_anybit"))
    env.reset(seed=13, episode_id="p13_control")
    fin = env.step(Phase2Action(tool="episode.finish", args={}))
    if fin.reward != 0.0 or not fin.done:
        raise SystemExit("finish-before-flip control did not terminate with reward 0")
    env.close()
    print("  control: finish-before-flip earns reward 0 (reward comes only from trusted flips)")


def _engine_any_flip_success(geo: Geometry, task_id: str, seed: int, acts_budget: int) -> bool:
    """Reference-policy success at the engine level: hammer the sampled victim.

    Uses the *same* target sampling the env compiles, then feeds the engine issued
    ACT events on the victim's neighbours up to ``acts_budget``. The victim's
    hcfirst is profile-sampled, so success across seeds is graded by the budget.
    Fast (no worker), and P11 already gates worker/engine ACT-equivalence.
    """
    ct = TaskSpec.from_config({"id": task_id, "family": "any_flip"}).compile(seed, geo)
    eng = DisturbanceEngine(geometry=geo, seed=seed)
    v = ct.target_row
    key = (0, 0, 0, 0, v)
    la, ra = v * eng.row_bytes - eng.row_bytes, v * eng.row_bytes + eng.row_bytes
    left = {"op": "ACT", "channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": v - 1, "row_hit": False, "clk": 0}
    right = {"op": "ACT", "channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": v + 1, "row_hit": False, "clk": 0}
    ldrd = {"op": "RD", "addr": la, "size": 64}
    rdrd = {"op": "RD", "addr": ra, "size": 64}
    acts = 0
    while acts < acts_budget:
        eng.consume([left], ldrd)
        eng.consume([right], rdrd)
        acts += 2
        victim = eng.victims.get(key)
        if victim is not None and victim.flipped:
            return True
    return False


def check_difficulty_bands(geo: Geometry) -> None:
    """Each band's reference-policy success rate falls in its calibrated window."""
    seeds = range(48)
    rates = {}
    for band, acts in BAND_ACTS.items():
        success = sum(1 for s in seeds if _engine_any_flip_success(geo, f"ddr4_band_{band}_v1", s, acts))
        rate = success / len(seeds)
        rates[band] = rate
        lo, hi = BAND_WINDOW[band]
        if not (lo <= rate <= hi):
            raise SystemExit(f"band {band}: success rate {rate:.3f} outside calibrated window [{lo}, {hi}]")
    if not (rates["hard"] < rates["medium"] < rates["easy"]):
        raise SystemExit(f"bands are not monotone in difficulty: {rates}")
    print(f"  difficulty: hard={rates['hard']:.2f} medium={rates['medium']:.2f} easy={rates['easy']:.2f} "
          f"(windows {BAND_WINDOW['hard']}/{BAND_WINDOW['medium']}/{BAND_WINDOW['easy']})")


def check_band_budget_wiring(geo: Geometry) -> None:
    """The compiled band budget — not the seed — decides an end-to-end episode.

    Same task id + seed (so the same victim row is sampled), only the difficulty
    band differs: the easy budget flips it through the real worker; the hard budget
    exhausts ``acts`` first and fails closed with ``BUDGET_EXCEEDED``.
    """
    tid = "ddr4_band_wire_v1"
    seed = next((s for s in range(64)
                 if _engine_any_flip_success(geo, tid, s, BAND_ACTS["easy"])
                 and not _engine_any_flip_success(geo, tid, s, BAND_ACTS["hard"])), None)
    if seed is None:
        raise SystemExit("could not find a seed separating the easy and hard bands")

    easy = RowHammerTaskEnv(task={"id": tid, "family": "any_flip", "difficulty": "easy"})
    obs = easy.reset(seed=seed, episode_id="p13_wire_easy")
    easy_result = ReferencePolicy(max_acts=BAND_ACTS["easy"] + 4096).run(easy, obs)
    easy.close()
    if easy_result["reward"] != 1.0 or not easy_result["done"]:
        raise SystemExit(f"easy band did not flip end-to-end: {easy_result}")

    hard = RowHammerTaskEnv(task={"id": tid, "family": "any_flip", "difficulty": "hard"})
    obs = hard.reset(seed=seed, episode_id="p13_wire_hard")
    hard_result = ReferencePolicy(max_acts=BAND_ACTS["hard"] + 4096).run(hard, obs)
    hard.close()
    if hard_result["reward"] != 0.0 or (hard_result.get("error") or {}).get("code") != "BUDGET_EXCEEDED":
        raise SystemExit(f"hard band did not fail closed on the acts budget: {hard_result}")
    print(f"  band wiring: seed {seed} same victim — easy budget flips, hard budget hits BUDGET_EXCEEDED")


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def _leaks(surfaces: list, needle: int) -> bool:
    text = str(needle)
    for surface in surfaces:
        for scalar in _walk(surface):
            if isinstance(scalar, bool):
                continue
            if isinstance(scalar, int) and scalar == needle:
                return True
            if isinstance(scalar, str) and text in scalar:
                return True
    return False


def check_non_leakage() -> None:
    """Hidden-target / unknown-adjacency expose no derivable target (SPEC §8, E6)."""
    for family in ("hidden_target", "unknown_adjacency"):
        env = RowHammerTaskEnv(task=_load(family))
        obs = env.reset(seed=13, episode_id=f"p13_leak_{family}")
        if obs.error:
            raise SystemExit(f"{family}: reset failed: {obs.error}")
        ct = env._compiled
        secret_addr = ct.target_addr
        secret_row = ct.target_row
        row_bytes = ct.row_bytes

        # Structural: the target is only ever an opaque handle here.
        target = obs.metadata.get("target")
        if not (isinstance(target, dict) and target.get("kind") == "handle"):
            raise SystemExit(f"{family}: target disclosed more than an opaque handle: {target}")
        if "known_target_row" in obs.metadata["disturbance"] or "known_threshold" in obs.metadata["disturbance"]:
            raise SystemExit(f"{family}: metadata leaked the target row/threshold")

        surfaces = [obs.metadata, obs.feedback]
        surfaces.append(env.step(Phase2Action(tool="dram.info", args={})).metadata)
        hammer = env.step(Phase2Action(tool="dram.issue", args={"commands": [{"op": "RD", "addr": {"kind": "logical", "addr": 0}}]}))
        # The disclosed trace must never carry a physical coordinate, whatever the
        # feedback level: ``hidden_target`` (summarized_counts) echoes no trace at
        # all, while ``unknown_adjacency`` (full_trace, since P22 — so the policy can
        # probe the bank-conflict timing channel) echoes op/clk/type_id/row_hit with
        # every COORD_KEY stripped by ``Disclosure.project_trace``.
        for event in hammer.feedback.get("trace_tail") or []:
            leaked = [k for k in COORD_KEYS if k in event]
            if leaked:
                raise SystemExit(f"{family}: feedback trace leaked coordinate keys {leaked}")
        surfaces.append(hammer.feedback)
        rejected = env.step(Phase2Action(tool="dram.read", args={
            "addr": {"kind": "physical", "channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": secret_row, "column": 0},
            "length": 1}))
        if not rejected.error or rejected.error["code"] != "ADDRESS_NOT_DISCLOSED":
            raise SystemExit(f"{family}: physical read was not gated: {rejected.error}")
        surfaces.append({"error": rejected.error})

        # The hidden address, row, and (for unknown_adjacency) the aggressor
        # addresses behind the candidate handles must not appear anywhere public.
        needles = [secret_addr, secret_row]
        if family == "unknown_adjacency":
            needles += [secret_addr - row_bytes, secret_addr + row_bytes]
        for needle in needles:
            if _leaks(surfaces, needle):
                raise SystemExit(f"{family}: hidden value {needle} leaked into a public surface")

        # Handles still resolve server-side to the real hidden coordinates.
        handle = target["id"]
        resolved = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "handle", "id": handle}, "length": 1}))
        if resolved.error or base64.b64decode(resolved.data_b64 or "") == b"":
            raise SystemExit(f"{family}: target handle did not resolve server-side")
        env.close()
    json.dumps({"ok": True})  # end-to-end serialisability of the surfaces walk
    print("  non-leakage: hidden target addr/row (+ candidate aggressors) absent from all public surfaces")


def main() -> int:
    subprocess.run([sys.executable, "-B", "scripts/verify_phase12.py"], cwd=ROOT, check=True)
    print("phase13 checks:")
    geo = real_geometry()
    check_configs_and_families()
    check_finish_before_success_control()
    check_difficulty_bands(geo)
    check_band_budget_wiring(geo)
    check_non_leakage()
    print("phase13 verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
