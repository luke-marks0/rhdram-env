"""Public geometry disclosure (IMPLEMENTATION_PLAN_V3 P21).

Architecture-level DRAM geometry — the row stride plus the row/bank/bankgroup
*counts* and the standard name — is disclosed unconditionally in ``dram.info`` and
in reset metadata. It is standard public information, identical across every
episode of a profile, so it leaks nothing about the hidden target: these tests
prove the block is present at every disclosure level, byte-identical across seeds
and targets, and carries no bank-select bit function or level ordering.

The ``dram.info``/reset-metadata surface is exercised through the *real*
``RowHammerTaskEnv`` (worker-gated), not just the ``Geometry`` unit, so the wiring
is covered end-to-end.
"""

from __future__ import annotations

import pathlib
import unittest

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv, RowHammerTaskEnv
from rowhammer_env.geometry import Geometry

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"

# The DDR4_8Gb_x8 geometry the worker publishes (see IssuedEventRecorder); the
# same fixture used by test_phase12/test_phase13.
DDR4_INFO = {
    "standard": "DDR4",
    "tx_bytes": 64,
    "prefetch": 8,
    "channel_width": 64,
    "level_names": ["Channel", "Rank", "BankGroup", "Bank", "Row", "Column"],
    "level_sizes": [1, 1, 4, 4, 65536, 1024],
}

# The block that geometry must publish for the admitted DDR4 geometry.
EXPECTED_BLOCK = {
    "row_bytes": 131072,
    "row_count": 65536,
    "bank_count": 4,
    "bankgroup_count": 4,
    "standard": "DDR4",
}

# Only sizes + stride + standard are public; the bank-select bit function and the
# level *order* are P24's per-episode secret and must never appear here.
GEOMETRY_KEYS = {"row_bytes", "row_count", "bank_count", "bankgroup_count", "standard"}


class GeometryBlockUnitTests(unittest.TestCase):
    def test_public_block_shape_and_values(self) -> None:
        self.assertEqual(Geometry(DDR4_INFO).public_block(), EXPECTED_BLOCK)

    def test_no_bit_function_or_level_order_leaks(self) -> None:
        block = Geometry(DDR4_INFO).public_block()
        # Exactly the size/stride/standard fields — no level_names, no shifts/
        # offsets, no bank-XOR function.
        self.assertEqual(set(block), GEOMETRY_KEYS)
        for banned in ("level_names", "level_order", "shifts", "offsets", "mapper", "bank_bits"):
            self.assertNotIn(banned, block)

    def test_block_is_mapper_independent_by_construction(self) -> None:
        # P21's "identical across RoBaRaCoCh vs MOP4CLXOR" cross-check needs the
        # secret-mapping engine change (P24), which does not exist yet. What P21
        # can guarantee now is structural: ``public_block`` is a pure function of
        # the worker-reported level sizes and takes no mapper input, so whatever
        # mapping an episode secretly runs, the disclosed block is unchanged.
        again = Geometry(dict(DDR4_INFO)).public_block()
        self.assertEqual(again, EXPECTED_BLOCK)


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class GeometryDisclosureIntegrationTests(unittest.TestCase):
    # A spread of disclosure levels: physical/exact, logical/hidden-handle,
    # candidate-set, and the lowest (reward_only) disclosure.
    FAMILIES = ("known_target_anybit", "hidden_target", "unknown_adjacency", "low_disclosure")

    def test_geometry_in_reset_metadata_every_disclosure_level(self) -> None:
        for family in self.FAMILIES:
            env = RowHammerTaskEnv(task={"family": family})
            try:
                obs = env.reset(seed=7)
                self.assertIsNone(obs.error, family)
                self.assertEqual(obs.metadata.get("geometry"), EXPECTED_BLOCK, family)
            finally:
                env.close()

    def test_geometry_in_dram_info(self) -> None:
        env = RowHammerTaskEnv(task={"family": "hidden_target"})
        try:
            env.reset(seed=7)
            info = env.step(Phase2Action(tool="dram.info", args={}))
            self.assertEqual(info.metadata.get("geometry"), EXPECTED_BLOCK)
        finally:
            env.close()

    def test_geometry_identical_across_seeds_and_targets(self) -> None:
        # The block must carry no per-episode information: identical across many
        # seeds (different sampled targets) and across families.
        seen = []
        for family in self.FAMILIES:
            for seed in range(6):
                env = RowHammerTaskEnv(task={"family": family})
                try:
                    obs = env.reset(seed=seed)
                    seen.append(obs.metadata.get("geometry"))
                finally:
                    env.close()
        self.assertEqual(len(set(map(repr, seen))), 1, seen)
        self.assertEqual(seen[0], EXPECTED_BLOCK)

    def test_no_coordinate_or_target_bits_in_geometry(self) -> None:
        # Even at physical/exact disclosure (which does expose the victim row
        # elsewhere), the geometry block itself must be only the public sizes.
        env = RowHammerTaskEnv(task={"family": "known_target_anybit"})
        try:
            obs = env.reset(seed=7)
            self.assertEqual(set(obs.metadata["geometry"]), GEOMETRY_KEYS)
        finally:
            env.close()

    def test_bare_disturbance_env_also_discloses_geometry(self) -> None:
        # The disclosure lives at the disturbance layer, so the bare (non-task)
        # env exposes it too — through both reset metadata and dram.info.
        env = RowHammerDisturbanceEnv()
        try:
            obs = env.reset(seed=7)
            self.assertEqual(obs.metadata.get("geometry"), EXPECTED_BLOCK)
            info = env.step(Phase2Action(tool="dram.info", args={}))
            self.assertEqual(info.metadata.get("geometry"), EXPECTED_BLOCK)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
