from __future__ import annotations

import base64
import pathlib
import unittest

from rowhammer_env import Phase2Action, RowHammerDisturbanceEnv, RowHammerTaskEnv
from rowhammer_env.geometry import Geometry
from rowhammer_env.tasks import AddressResolver, Disclosure, HandleTable
from rowhammer_env.tools.addressing import COORD_KEYS, AddressError, AddressMapper


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"

# The DDR4_8Gb_x8 geometry the worker publishes (see IssuedEventRecorder).
DDR4_INFO = {
    "standard": "DDR4",
    "tx_bytes": 64,
    "prefetch": 8,
    "channel_width": 64,
    "level_names": ["Channel", "Rank", "BankGroup", "Bank", "Row", "Column"],
    "level_sizes": [1, 1, 4, 4, 65536, 1024],
}


def mapper() -> AddressMapper:
    return AddressMapper(Geometry(DDR4_INFO))


class AddressMapperTests(unittest.TestCase):
    def test_layout_matches_robaracoch(self) -> None:
        m = mapper()
        # Col[6:13) Rank[13:13) BankGroup[13:15) Bank[15:17) Row[17:33).
        self.assertEqual(m.shifts, {"column": 6, "rank": 13, "bankgroup": 13, "bank": 15, "row": 17})
        self.assertEqual(m.widths["column"], 7)
        self.assertEqual(m.widths["row"], 16)

    def test_known_worker_decodes(self) -> None:
        m = mapper()
        # Values verified against the live worker's addr_vec.
        cases = {
            771136: {"channel": 0, "rank": 0, "bankgroup": 2, "bank": 3, "row": 5, "column": 17},
            8128: {"channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": 0, "column": 127},
            65535 << 17: {"channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": 65535, "column": 0},
        }
        for linear, coords in cases.items():
            self.assertEqual(m.decode(linear), coords)
            self.assertEqual(m.encode(coords), linear)

    def test_round_trip(self) -> None:
        m = mapper()
        for row in (0, 1, 137, 65535):
            for bank in range(4):
                for column in (0, 63, 127):
                    coords = {"channel": 0, "rank": 0, "bankgroup": 1, "bank": bank, "row": row, "column": column}
                    self.assertEqual(m.decode(m.encode(coords)), coords)

    def test_row_stride(self) -> None:
        m = mapper()
        self.assertEqual(m.encode({"row": 1}) - m.encode({"row": 0}), 131072)

    def test_out_of_range_fails_closed(self) -> None:
        m = mapper()
        for bad in ({"column": 128}, {"bank": 4}, {"bankgroup": 4}, {"rank": 1}, {"row": 65536}, {"channel": 1}):
            with self.assertRaises(AddressError) as ctx:
                m.encode(bad)
            self.assertEqual(ctx.exception.code, "BAD_SCHEMA")

    def test_multi_channel_rejected(self) -> None:
        info = dict(DDR4_INFO, level_sizes=[2, 1, 4, 4, 65536, 1024])
        with self.assertRaises(ValueError):
            AddressMapper(Geometry(info))


class DisclosureTests(unittest.TestCase):
    def test_allowed_forms_per_mapping(self) -> None:
        self.assertEqual(Disclosure(mapping="physical", victim="exact").allowed_forms(), {"logical", "physical"})
        self.assertEqual(Disclosure(mapping="logical_only", victim="exact").allowed_forms(), {"logical"})
        self.assertEqual(Disclosure(mapping="opaque_handles", victim="row_handle").allowed_forms(), {"handle"})

    def test_victim_handle_adds_handle_form(self) -> None:
        self.assertEqual(
            Disclosure(mapping="logical_only", victim="row_handle").allowed_forms(),
            {"logical", "handle"},
        )

    def test_trace_projection_strips_hidden_coords(self) -> None:
        events = [{"op": "ACT", "clk": 1, "row": 10, "bank": 0, "row_hit": False}]
        # summarized/reward-only feedback exposes no per-command trace at all.
        self.assertEqual(Disclosure(feedback="summarized_counts").project_trace(events), [])
        # full trace without physical mapping keeps the command but drops coordinates.
        stripped = Disclosure(mapping="logical_only", feedback="full_trace").project_trace(events)
        self.assertNotIn("row", stripped[0])
        self.assertNotIn("bank", stripped[0])
        self.assertEqual(stripped[0]["op"], "ACT")
        # full trace under physical mapping keeps everything.
        self.assertEqual(Disclosure(mapping="physical", feedback="full_trace").project_trace(events), events)

    def test_public_flips_hidden_unless_exact_physical(self) -> None:
        feedback = {"new_public_flips": 1, "public_flips": [{"row": 10, "addr": 1310720}], "trace_tail": []}
        hidden = Disclosure(mapping="logical_only", victim="row_handle").project_feedback(feedback)
        self.assertNotIn("public_flips", hidden)
        self.assertEqual(hidden["new_public_flips"], 1)  # count survives; location does not
        shown = Disclosure(mapping="physical", victim="exact", feedback="full_trace").project_feedback(feedback)
        self.assertIn("public_flips", shown)


class HandleTableTests(unittest.TestCase):
    def test_non_invertible_and_deterministic(self) -> None:
        a = HandleTable(7)
        b = HandleTable(7)
        hid = a.register("target", 1310720)
        self.assertEqual(hid, b.register("target", 1310720))  # deterministic per seed+role
        for coordinate in ("1310720", "10", "5000"):
            self.assertNotIn(coordinate, hid)  # encodes no coordinate

    def test_seed_changes_handle(self) -> None:
        self.assertNotEqual(HandleTable(1).register("target", 0), HandleTable(2).register("target", 0))

    def test_resolve_unknown_fails_closed(self) -> None:
        table = HandleTable(0)
        with self.assertRaises(AddressError) as ctx:
            table.resolve("h_does_not_exist")
        self.assertEqual(ctx.exception.code, "ADDRESS_NOT_DISCLOSED")


class AddressResolverTests(unittest.TestCase):
    def resolver(self, disclosure: Disclosure) -> AddressResolver:
        handles = HandleTable(0)
        self.target_handle = handles.register("target", 1310720)
        return AddressResolver(mapper(), disclosure, handles)

    def test_logical_only_rejects_physical_and_handle(self) -> None:
        r = self.resolver(Disclosure(mapping="logical_only", victim="exact"))
        self.assertEqual(r.to_linear({"kind": "logical", "addr": 4096}), 4096)
        for form in ({"kind": "physical", "row": 1}, {"kind": "handle", "id": "x"}):
            with self.assertRaises(AddressError) as ctx:
                r.to_linear(form)
            self.assertEqual(ctx.exception.code, "ADDRESS_NOT_DISCLOSED")

    def test_physical_form_encodes(self) -> None:
        r = self.resolver(Disclosure(mapping="physical", victim="exact"))
        self.assertEqual(r.to_linear({"kind": "physical", "row": 10}), 1310720)

    def test_handle_resolves_only_when_disclosed(self) -> None:
        r = self.resolver(Disclosure(mapping="opaque_handles", victim="row_handle"))
        self.assertEqual(r.to_linear({"kind": "handle", "id": self.target_handle}), 1310720)
        with self.assertRaises(AddressError) as ctx:
            r.to_linear({"kind": "logical", "addr": 0})
        self.assertEqual(ctx.exception.code, "ADDRESS_NOT_DISCLOSED")

    def test_missing_kind_is_bad_schema(self) -> None:
        r = self.resolver(Disclosure(mapping="physical", victim="exact"))
        with self.assertRaises(AddressError) as ctx:
            r.to_linear({"addr": 0})
        self.assertEqual(ctx.exception.code, "BAD_SCHEMA")

    def test_bare_int_is_logical_shorthand(self) -> None:
        # The compact HAMMER `rows` list (SPEC §8) is a plain address list, so a
        # bare int must resolve exactly like {"kind":"logical","addr":N}.
        r = self.resolver(Disclosure(mapping="physical", victim="exact"))
        self.assertEqual(r.to_linear(4096), r.to_linear({"kind": "logical", "addr": 4096}))

    def test_bare_int_shorthand_still_fails_closed_when_undisclosed(self) -> None:
        r = self.resolver(Disclosure(mapping="opaque_handles", victim="row_handle"))
        with self.assertRaises(AddressError) as ctx:
            r.to_linear(4096)
        self.assertEqual(ctx.exception.code, "ADDRESS_NOT_DISCLOSED")

    def test_bool_is_not_treated_as_int_shorthand(self) -> None:
        r = self.resolver(Disclosure(mapping="physical", victim="exact"))
        with self.assertRaises(AddressError) as ctx:
            r.to_linear(True)
        self.assertEqual(ctx.exception.code, "BAD_SCHEMA")


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class WorkerDifferentialTests(unittest.TestCase):
    """The Python projection must agree with Ramulator's own decode."""

    def _coords_of(self, obs) -> dict:
        rd = [e for e in obs.feedback.get("trace_tail", []) if e.get("op") == "RD"]
        self.assertTrue(rd, "expected an RD event in the trace")
        return {k: rd[-1][k] for k in COORD_KEYS}

    def test_logical_and_physical_match_worker(self) -> None:
        env = RowHammerDisturbanceEnv()
        obs = env.reset(seed=12, episode_id="t12_diff")
        self.assertIsNone(obs.error)
        m = env.address_mapper
        for coords in (
            {"channel": 0, "rank": 0, "bankgroup": 2, "bank": 3, "row": 5, "column": 17},
            {"channel": 0, "rank": 0, "bankgroup": 1, "bank": 2, "row": 4096, "column": 63},
            {"channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": 65535, "column": 0},
        ):
            linear = m.encode(coords)
            logical = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "logical", "addr": linear}, "length": 1}))
            self.assertEqual(self._coords_of(logical), coords)
            physical = env.step(Phase2Action(tool="dram.read", args={"addr": {"kind": "physical", **coords}, "length": 1}))
            self.assertEqual(self._coords_of(physical), coords)
        env.close()


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class DisclosureIntegrationTests(unittest.TestCase):
    def read(self, env, addr):
        return env.step(Phase2Action(tool="dram.read", args={"addr": addr, "length": 1}))

    def test_physical_task_gates_forms(self) -> None:
        env = RowHammerTaskEnv(task={"family": "known_target_anybit"})
        obs = env.reset(seed=12)
        self.assertEqual(set(obs.metadata["address_forms"]), {"logical", "physical"})
        coords = {k: obs.metadata["target"][k] for k in COORD_KEYS}
        self.assertIsNone(self.read(env, {"kind": "logical", "addr": obs.metadata["target"]["addr"]}).error)
        self.assertIsNone(self.read(env, {"kind": "physical", **coords}).error)
        self.assertEqual(self.read(env, {"kind": "handle", "id": "h_x"}).error["code"], "ADDRESS_NOT_DISCLOSED")
        env.close()

    def test_hidden_target_hides_coords_and_gates_forms(self) -> None:
        env = RowHammerTaskEnv(task={"family": "hidden_target"})
        obs = env.reset(seed=12)
        self.assertEqual(set(obs.metadata["address_forms"]), {"logical", "handle"})
        self.assertEqual(obs.metadata["target"]["kind"], "handle")
        self.assertNotIn("known_target_row", obs.metadata["disturbance"])
        handle = obs.metadata["target"]["id"]
        self.assertIsNone(self.read(env, {"kind": "handle", "id": handle}).error)
        physical = {"kind": "physical", "channel": 0, "rank": 0, "bankgroup": 0, "bank": 0, "row": 10, "column": 0}
        self.assertEqual(self.read(env, physical).error["code"], "ADDRESS_NOT_DISCLOSED")
        env.close()

    def test_hidden_target_summarized_trace_has_no_coords(self) -> None:
        env = RowHammerTaskEnv(task={"family": "hidden_target"})
        env.reset(seed=12)
        obs = env.step(Phase2Action(tool="dram.issue", args={"commands": [{"op": "RD", "addr": {"kind": "logical", "addr": 0}}]}))
        self.assertEqual(obs.feedback.get("trace_tail"), [])
        env.close()

    def test_handle_resolves_to_hidden_target(self) -> None:
        env = RowHammerTaskEnv(task={"family": "hidden_target"})
        obs = env.reset(seed=12)
        handle = obs.metadata["target"]["id"]
        # The handle resolves to the same linear address the engine tracks.
        via_handle = self.read(env, {"kind": "handle", "id": handle})
        self.assertIsNone(via_handle.error)
        self.assertEqual(env._resolver.handles.resolve(handle), env.disturbance.target_addr)
        env.close()


if __name__ == "__main__":
    unittest.main()
