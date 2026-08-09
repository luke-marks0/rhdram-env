"""Unit tests for compact dram.issue expansion (repeat / HAMMER).

These exercise the pure ``expand_commands`` helper — no worker/server needed — and
lock in that a compact command expands to *exactly* the explicit primitive stream
it stands in for, so budget/disturbance accounting (which runs on the expanded
stream) and the trusted reward are unchanged.
"""

from __future__ import annotations

import unittest

from rowhammer_env.phase2_env import (
    MAX_ISSUE_ACTIVATIONS,
    IssueExpansionError,
    expand_commands,
)

A = {"kind": "logical", "addr": 100}
B = {"kind": "logical", "addr": 200}


class ExpandCommandsTest(unittest.TestCase):
    def test_plain_primitives_pass_through_unchanged(self):
        cmds = [{"op": "RD", "addr": A}, {"op": "WAIT", "cycles": 8}, {"op": "WR", "addr": B, "data_b64": "AA=="}]
        self.assertEqual(expand_commands(cmds), cmds)

    def test_repeat_expands_primitive(self):
        out = expand_commands([{"op": "RD", "addr": A, "repeat": 3}])
        self.assertEqual(out, [{"op": "RD", "addr": A, "repeat": 3}] * 3)

    def test_count_is_a_repeat_alias(self):
        self.assertEqual(len(expand_commands([{"op": "RD", "addr": A, "count": 5}])), 5)

    def test_hammer_equals_explicit_alternating_pairs(self):
        hammer = expand_commands([{"op": "HAMMER", "rows": [A, B], "pairs": 4}])
        explicit = []
        for _ in range(4):
            explicit.append({"op": "RD", "addr": A})
            explicit.append({"op": "RD", "addr": B})
        self.assertEqual(hammer, explicit)

    def test_hammer_count_alias_and_multi_row(self):
        out = expand_commands([{"op": "HAMMER", "addrs": [A, B, A], "count": 2}])
        self.assertEqual([c["addr"] for c in out], [A, B, A, A, B, A])
        self.assertTrue(all(c["op"] == "RD" for c in out))

    def test_zero_repeat_emits_nothing(self):
        self.assertEqual(expand_commands([{"op": "RD", "addr": A, "repeat": 0}]), [])
        self.assertEqual(expand_commands([{"op": "HAMMER", "rows": [A, B], "pairs": 0}]), [])

    def test_realistic_threshold_sized_hammer(self):
        # ~25k double-sided pairs (the known-target hcfirst min) is a few tokens as
        # a HAMMER but expands to the full 50k-activation stream the worker runs.
        out = expand_commands([{"op": "HAMMER", "rows": [A, B], "pairs": 25000}])
        self.assertEqual(len(out), 50000)

    def test_controller_generated_ops_are_illegal_command(self):
        # @spec:tool-dram-issue — ACT/PRE/REF/RFM are controller-generated, so they
        # are not policy-issuable ops.
        for op in ("ACT", "PRE", "REF", "RFM"):
            with self.subTest(op=op), self.assertRaises(IssueExpansionError) as ctx:
                expand_commands([{"op": op, "addr": A}])
            self.assertEqual(ctx.exception.code, "ILLEGAL_COMMAND")

    def test_unknown_op_is_illegal_command(self):
        with self.assertRaises(IssueExpansionError) as ctx:
            expand_commands([{"op": "NOPE"}])
        self.assertEqual(ctx.exception.code, "ILLEGAL_COMMAND")

    def test_hammer_without_rows_is_bad_schema(self):
        with self.assertRaises(IssueExpansionError) as ctx:
            expand_commands([{"op": "HAMMER", "pairs": 4}])
        self.assertEqual(ctx.exception.code, "BAD_SCHEMA")

    def test_non_integer_repeat_is_bad_schema(self):
        with self.assertRaises(IssueExpansionError) as ctx:
            expand_commands([{"op": "RD", "addr": A, "repeat": "lots"}])
        self.assertEqual(ctx.exception.code, "BAD_SCHEMA")

    def test_negative_repeat_is_bad_schema(self):
        with self.assertRaises(IssueExpansionError) as ctx:
            expand_commands([{"op": "RD", "addr": A, "repeat": -1}])
        self.assertEqual(ctx.exception.code, "BAD_SCHEMA")

    def test_oversized_expansion_is_capped(self):
        with self.assertRaises(IssueExpansionError) as ctx:
            expand_commands([{"op": "RD", "addr": A, "repeat": MAX_ISSUE_ACTIVATIONS + 1}])
        self.assertEqual(ctx.exception.code, "ILLEGAL_COMMAND")


if __name__ == "__main__":
    unittest.main()
