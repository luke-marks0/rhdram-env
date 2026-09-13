"""Scoped admission, artifacts, and learning-signal regressions on real Ramulator."""
from __future__ import annotations

import asyncio
import copy
import json
import pathlib
import tempfile
import unittest

import yaml

from rowhammer_env import Phase2Action
from rowhammer_env.poc import POC_TOOLS, ROOT, TASK_PATHS, PoCEnv, load_task, validate_config
from rowhammer_env.llm.grpo_env import build_messages, launch_server
from rowhammer_env.llm.multiturn_rollout import (
    BatchToolPolicyGenerator, GeneratedTurn, ToolPolicyGenerator, run_batched_training_episodes,
    run_training_episode_local, to_grpo_example,
)
from rowhammer_env.llm.poc_policy import KnownTargetControl, hide_timing, timing_hidden_messages
from rowhammer_env.llm.policies import ClaimSuccessFixturePolicy, ReferenceProbePolicy
from rowhammer_env.llm.shaping import shaping_weights, unique_probe_shaping_reward
from rowhammer_env.llm.tools import poc_tool_schemas
from rowhammer_env.observability.experiment import RunArtifacts, episode_record, summarize_records, wilson_interval


class PoCContractTests(unittest.TestCase):
    def config(self):
        return yaml.safe_load((ROOT / "configs/training/poc.yaml").read_text())

    def test_frozen_tasks_and_disjoint_splits(self):
        cfg = self.config()
        validate_config(cfg)
        cfg["benchmark"]["seed_range"] = [1, 32]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_config(cfg)

    def test_invalid_reward_weights_fail_closed(self):
        for weight in (-0.1, float("nan"), float("inf"), 1.0):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                shaping_weights({"success_weight": 1.0, "probe_shaping_weight": weight})

    def test_tools_are_exactly_the_direct_surface(self):
        self.assertEqual({s["function"]["name"] for s in poc_tool_schemas()}, set(POC_TOOLS))

    def test_timing_ablation_removes_redundant_channels_and_history(self):
        observed = {"cycle": 42, "last_action": {"accepted": 4, "cycle_delta": 9},
                    "public_counters": {"acts": 7}, "budget_remaining": {"acts": 100, "cycles": 200, "tool_calls": 3},
                    "feedback": {"timing_digest": {"acts_delta": 7}, "trace_tail": [{"clk": 42}], "new_public_flips": 0}}
        hidden = hide_timing(observed)
        self.assertNotIn("cycle", hidden)
        self.assertNotIn("public_counters", hidden)
        self.assertEqual(hidden["budget_remaining"], {"tool_calls": 3})
        self.assertEqual(hidden["feedback"], {"new_public_flips": 0})
        self.assertEqual(hidden["last_action"], {"accepted": 4})
        messages = [{"role": "user", "content": "initial static budget"}, {"role": "tool", "content": json.dumps(observed)}]
        self.assertEqual(json.loads(timing_hidden_messages(messages)[1]["content"]), hidden)
        self.assertIn("cycle", observed)
        generated = GeneratedTurn("sample", prompt_ids=[1], token_ids=[2])
        self.assertEqual(timing_hidden_messages([{"role": "assistant", "content": generated}]),
                         [{"role": "assistant", "content": "sample"}])

    def test_confidence_intervals_cover_extreme_samples(self):
        self.assertAlmostEqual(wilson_interval(0, 32)[1], 0.1071791982550706)
        self.assertAlmostEqual(wilson_interval(32, 32)[0], 0.8928208017449293)
        summary = summarize_records([{"condition": "trained", "task_id": "x", "family": "bounded_sweep",
                                     "difficulty": "easy", "trusted_reward": 0.1, "success": True}])
        self.assertEqual(summary["groups"][0]["successes"], 0)

    def test_exact_sampled_tokens_exclude_template_and_driver_text(self):
        from rowhammer_env.llm.multiturn_rollout import MultiTurnRollout

        a = GeneratedTurn("action", prompt_ids=[1, 2], token_ids=[3, 4])
        b = GeneratedTurn("action2", prompt_ids=[1, 2, 3, 4, 8, 9], token_ids=[5])
        stop = GeneratedTurn("auto finish", prompt_ids=[1, 2, 3, 4, 8, 9, 5, 10], token_ids=[], sampled=False)
        rollout = MultiTurnRollout([], [{"role": "assistant", "content": t} for t in (a, b, stop)], [], None, 0.0)
        example = to_grpo_example(rollout, None)
        self.assertEqual(example["completion_ids"], [3, 4, 8, 9, 5])
        self.assertEqual(example["completion_mask"], [1, 1, 0, 0, 1])
        b.prompt_ids = [1, 2, 77]
        with self.assertRaisesRegex(ValueError, "rewrote"):
            to_grpo_example(rollout, None)

    def test_paired_summary_uses_sparse_rewards_and_identical_seeds(self):
        from scripts.summarize_poc import paired_changes

        records = [{"condition": condition, "task_id": "task", "seed": seed, "trusted_reward": reward}
                   for condition, values in (("sft", [0.1, 1.0, 0.0]), ("trained", [1.0, 0.0, 1.0]))
                   for seed, reward in enumerate(values)]
        change, = paired_changes(records)
        self.assertEqual((change["improved"], change["regressed"], change["unchanged"]), (2, 1, 0))
        self.assertAlmostEqual(change["success_rate_change"], 1 / 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_changes(records + [records[0]])
        with self.assertRaisesRegex(ValueError, "unpaired"):
            paired_changes(records[:-1])

    def test_summary_contract_includes_physics_and_sampling(self):
        from scripts.summarize_poc import evaluation_contract
        from rowhammer_env.poc import resolved_config

        cfg = resolved_config(self.config())
        run = {"identity": {"split": "benchmark", "model": {}}, "provenance": {"files_sha256": {
            "build/phase2/ramulator_worker": "first", "rowhammer_env/phase5_env.py": "source"}}}
        original = evaluation_contract(run, cfg)
        cfg["grpo"]["temperature"] = 0.5
        self.assertNotEqual(original, evaluation_contract(run, cfg))
        cfg["grpo"]["temperature"] = 0.8
        self.assertEqual(original, evaluation_contract(run, cfg))
        run["provenance"]["files_sha256"]["build/phase2/ramulator_worker"] = "different"
        self.assertNotEqual(original, evaluation_contract(run, cfg))


class PoCNativeTests(unittest.TestCase):
    def rollout(self, index, policy, seed=17):
        task = load_task(TASK_PATHS[index])
        env = PoCEnv(task=task)
        try:
            return run_training_episode_local(env, ToolPolicyGenerator(policy), seed=seed, task=task, max_turns=40)
        finally:
            env.close()

    def test_positive_negative_controls_and_resource_records(self):
        for policy, expected in [(KnownTargetControl(), 1.0), (KnownTargetControl(below_threshold=True), 0.0),
                                  (ClaimSuccessFixturePolicy(), 0.0)]:
            rollout = self.rollout(0, policy)
            self.assertEqual(rollout.reward, expected)
            record = episode_record(rollout.result, condition="control")
            self.assertEqual(record["task_id"], "ddr4_known_target_anybit_v1")
            self.assertEqual(record["resource_usage"]["tool_calls"], len(rollout.trajectory))
            self.assertGreaterEqual(record["resource_usage"]["acts"], 0)
            self.assertTrue(record["trajectory"][-1]["observation"])

    def test_poc_rejects_script_before_dispatch_and_can_continue(self):
        env = PoCEnv(task=load_task(TASK_PATHS[0]))
        try:
            reset = env.reset(seed=1)
            self.assertEqual(reset.metadata["allowed_tools"], list(POC_TOOLS))
            messages = json.dumps(build_messages(reset.metadata))
            for name in ("script.run", "dram.read", "dram.write"):
                self.assertNotIn(name, messages)
                obs = env.step(Phase2Action(tool=name, args={}))
                self.assertEqual(obs.error["code"], "UNSUPPORTED_TOOL")
                self.assertEqual(obs.cycle, 0)
                self.assertFalse(obs.done)
            self.assertIsNone(env.step(Phase2Action(tool="dram.info", args={})).error)
        finally:
            env.close()

    def test_poc_rejects_out_of_scope_task_reset(self):
        env = PoCEnv()
        try:
            for task in ({"family": "hidden_adjacency", "difficulty": "hard"}, {"standard": "DDR5"},
                         {"mitigation": {"name": "oracle"}}):
                obs = env.reset(seed=1, task=task)
                self.assertTrue(obs.done)
                self.assertEqual(obs.error["code"], "BAD_SCHEMA")
        finally:
            env.close()

    def test_unique_probe_shaping_cannot_be_farmed(self):
        rollout = self.rollout(1, ReferenceProbePolicy())
        self.assertEqual(rollout.reward, 1.0)
        self.assertEqual(unique_probe_shaping_reward(rollout), 1.0)
        probe = rollout.trajectory[1]
        rollout.trajectory = [probe] * 20
        self.assertEqual(unique_probe_shaping_reward(rollout), 0.25)
        grouped = copy.deepcopy(probe)
        a, b = grouped.action["args"]["commands"][0]["rows"]
        grouped.action["args"]["commands"] = [{"op": "RD", "addr": a, "repeat": 8}, {"op": "RD", "addr": b, "repeat": 8}]
        rollout.trajectory = [grouped]
        self.assertEqual(unique_probe_shaping_reward(rollout), 0.0)

    def test_invalid_command_suffix_has_no_issued_prefix(self):
        env = PoCEnv(task=load_task(TASK_PATHS[3]))
        try:
            reset = env.reset(seed=1)
            before = dict(env._public_counters)
            rejected = env.step(Phase2Action(tool="dram.issue", args={"commands": [
                {"op": "RD", "addr": reset.metadata["candidates"][0]}, {"op": "WAIT", "cycles": -1}]}))
            self.assertEqual(rejected.error["code"], "BAD_SCHEMA")
            self.assertEqual(env._public_counters, before)
            self.assertEqual(rejected.cycle, 0)
            self.assertEqual(rejected.reward, 0.0)
            self.assertFalse(rejected.done)
        finally:
            env.close()

    def test_medium_discovery_surfaces_hide_mapper_roles_and_coordinates(self):
        for index in (2, 4):
            env = PoCEnv(task=load_task(TASK_PATHS[index]))
            try:
                reset = env.reset(seed=7)
                observations = [reset, env.step(Phase2Action(tool="dram.info", args={})),
                    env.step(Phase2Action(tool="dram.issue", args={"commands": [
                        {"op": "HAMMER", "rows": [reset.metadata["target"], reset.metadata["candidates"][0]], "pairs": 8}]})),
                    env.step(Phase2Action(tool="dram.issue", args={"commands": [
                        {"op": "RD", "addr": {"kind": "physical", "bank": 0, "row": 0}}]}))]
                for obs in observations:
                    blob = json.dumps({"metadata": obs.metadata, "feedback": obs.feedback, "error": obs.error})
                    for token in ("RoBaRaCoCh", "xor_offset", "addr_mapper", "is_aggressor", "same_bank_far", "different_bank", "known_threshold"):
                        self.assertNotIn(token, blob)
                    for key in ("channel", "rank", "bankgroup", "bank", "row", "column"):
                        self.assertNotIn(f'"{key}":', blob)
                self.assertEqual(len(reset.metadata["candidates"]), 16)
            finally:
                env.close()

    def test_bundle_roundtrip_and_refuse_overwrite(self):
        rollout = self.rollout(0, ClaimSuccessFixturePolicy())
        with tempfile.TemporaryDirectory() as directory:
            bundle = RunArtifacts(directory, config={"test": True}, identity={"condition": "finish"})
            bundle.append(episode_record(rollout.result, condition="finish", messages=rollout.messages))
            bundle.finish()
            records = [json.loads(line) for line in (pathlib.Path(directory) / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["trusted_reward"], 0.0)
            self.assertEqual(json.loads((pathlib.Path(directory) / "run.json").read_text())["status"], "complete")
            with self.assertRaises(FileExistsError):
                RunArtifacts(directory, config={}, identity={})


class PoCWebSocketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No skip: a missing required transport is a failed PoC gate.
        cls.server = launch_server(root=str(ROOT), max_concurrent_envs=4, env_overrides={"RH_POC": "1"})

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_multiturn_ws_matches_local_and_isolates_same_seed_sessions(self):
        task = load_task(TASK_PATHS[3])
        remote = asyncio.run(run_batched_training_episodes(self.server.base_url,
            [(17, task, "ws_a"), (17, task, "ws_b"), (18, task, "ws_c")],
            BatchToolPolicyGenerator(ReferenceProbePolicy), max_turns=40))
        local = PoCNativeTests().rollout(3, ReferenceProbePolicy(), seed=17)
        for result in remote[:2]:
            self.assertEqual(result.reward, 1.0)
            self.assertEqual(result.messages, local.messages)
            self.assertEqual([s.as_public() for s in result.trajectory], [s.as_public() for s in local.trajectory])
        self.assertEqual(remote[2].reward, 1.0)
        self.assertNotEqual(remote[0].prompt_messages, remote[2].prompt_messages)
