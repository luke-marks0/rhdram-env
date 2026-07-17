"""Multi-turn training rollout + completion mask (P27).

Two host-runnable checks:

* **Trace-equivalence**: the P26 ``ReferenceProbePolicy`` driven through the training
  loop (``run_training_episode_local`` + ``ToolPolicyGenerator``) produces the same
  trajectory + reward as driving it through the eval loop (``run_episode_local``).
  Checked against a fake timing/flip env, and against the real ``RowHammerTaskEnv``
  discovery configs when the worker is built.
* **Completion mask**: ``build_masked_completion`` marks ``1`` on assistant-authored
  spans and ``0`` on tool-result turns + template scaffolding, using a fake
  ``apply_chat_template`` tokenizer (no ``transformers``).

A full GRPO run showing a non-trivial gradient signal needs ``trl``/``torch``/GPU and
is instance-only; not exercised here.
"""

from __future__ import annotations

import pathlib
import unittest

import yaml

from rowhammer_env import RowHammerTaskEnv
from rowhammer_env.llm import (
    BatchToolPolicyGenerator,
    ReferenceProbePolicy,
    ToolPolicyGenerator,
    build_masked_completion,
    render_tool_call,
    run_batched_training_episodes_local,
    run_episode_local,
    run_training_episode_local,
    to_grpo_example,
)
from rowhammer_env.llm.grpo_env import parse_actions
from rowhammer_env.llm.policies import ToolCall

# The fake timing/flip env is defined once in the reference-policy tests; reuse it so
# both suites drive the identical contract. ``tests`` is on sys.path under
# ``unittest discover``.
from test_reference_policy import _FakeDiscoveryEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"


def _config(family: str, band: str) -> dict:
    return yaml.safe_load((ROOT / f"configs/tasks/{family}_{band}.yaml").read_text())


def _window():
    victim = {"kind": "logical", "addr": 1000}
    cands = [{"kind": "logical", "addr": a} for a in (10, 20, 30, 40)]
    banks = {10: 0, 20: 1, 30: 0, 40: 0}   # 20 different-bank decoy
    aggressors = {10, 40}                    # 30 same-bank-far decoy
    return victim, cands, banks, aggressors


# --------------------------------------------------------------------------- #
# render/parse round-trip (the lossless property trace-equivalence relies on)
# --------------------------------------------------------------------------- #
class RenderRoundTripTests(unittest.TestCase):
    def test_render_then_parse_recovers_the_tool_call(self) -> None:
        calls = [
            ToolCall("episode.finish", {}),
            ToolCall("dram.issue", {"commands": [{"op": "RD", "addr": {"kind": "logical", "addr": 10}}]}),
            ToolCall(
                "dram.issue",
                {"commands": [{"op": "HAMMER", "rows": [{"kind": "handle", "id": "c0"}, {"kind": "logical", "addr": 7}], "pairs": 8}]},
            ),
        ]
        for call in calls:
            recovered = parse_actions(render_tool_call(call))
            self.assertEqual(len(recovered), 1, call)
            self.assertEqual(recovered[0].name, call.name, call)
            self.assertEqual(recovered[0].args, call.args, call)


# --------------------------------------------------------------------------- #
# Trace-equivalence: eval loop vs training loop (fake env, no worker)
# --------------------------------------------------------------------------- #
def _traj_tuple(step) -> tuple:
    return (step.action["tool"], step.action["args"], step.reward, step.done, step.error, step.feedback)


class TraceEquivalenceUnitTests(unittest.TestCase):
    def _run_both(self, use_timing: bool):
        victim, cands, banks, aggressors = _window()
        eval_env = _FakeDiscoveryEnv(victim, cands, banks, aggressors)
        eval_res = run_episode_local(eval_env, ReferenceProbePolicy(use_timing=use_timing), seed=0)

        train_env = _FakeDiscoveryEnv(victim, cands, banks, aggressors)
        train_roll = run_training_episode_local(
            train_env, ToolPolicyGenerator(ReferenceProbePolicy(use_timing=use_timing)), seed=0, max_turns=64
        )
        return eval_res, train_roll, eval_env, train_env

    def test_training_loop_matches_eval_loop_trajectory_and_reward(self) -> None:
        eval_res, train_roll, eval_env, train_env = self._run_both(use_timing=True)
        self.assertEqual(
            [_traj_tuple(s) for s in train_roll.trajectory],
            [_traj_tuple(s) for s in eval_res.trajectory],
        )
        self.assertEqual(train_roll.reward, eval_res.reward)
        self.assertEqual(train_roll.reward, 1.0)
        # Same real env interactions: identical probe order + identical hammer set.
        self.assertEqual(train_env.probed, eval_env.probed)
        self.assertEqual(train_env.hammered_rows, eval_env.hammered_rows)

    def test_timing_blind_control_also_matches(self) -> None:
        eval_res, train_roll, _, _ = self._run_both(use_timing=False)
        self.assertEqual(
            [_traj_tuple(s) for s in train_roll.trajectory],
            [_traj_tuple(s) for s in eval_res.trajectory],
        )
        self.assertEqual(train_roll.reward, eval_res.reward)

    def test_batched_driver_matches_sequential_per_episode(self) -> None:
        # The batched driver (many episodes, one generation per tick) must produce, for
        # each episode, the SAME trajectory + reward as running that episode through the
        # single-episode loop. Two episodes with distinct windows so batching genuinely
        # interleaves two different rollouts (not one duplicated).
        windows = [_window(), (
            {"kind": "logical", "addr": 2000},
            [{"kind": "logical", "addr": a} for a in (11, 21, 31, 41)],
            {11: 0, 21: 1, 31: 0, 41: 0},
            {11, 41},
        )]

        seq_rolls = []
        for victim, cands, banks, aggressors in windows:
            env = _FakeDiscoveryEnv(victim, cands, banks, aggressors)
            seq_rolls.append(run_training_episode_local(env, ToolPolicyGenerator(ReferenceProbePolicy()), seed=0, max_turns=64))

        envs = [_FakeDiscoveryEnv(*w) for w in windows]
        episodes = [(envs[i], 0, None, f"batch_ep_{i}") for i in range(len(windows))]
        batch_rolls = run_batched_training_episodes_local(
            episodes, BatchToolPolicyGenerator(lambda: ReferenceProbePolicy()), max_turns=64
        )

        self.assertEqual(len(batch_rolls), len(seq_rolls))
        for seq, bat in zip(seq_rolls, batch_rolls):
            self.assertEqual(
                [_traj_tuple(s) for s in bat.trajectory],
                [_traj_tuple(s) for s in seq.trajectory],
            )
            self.assertEqual(bat.reward, seq.reward)
            self.assertEqual(bat.messages, seq.messages)

    def test_transcript_is_prompt_plus_alternating_turns(self) -> None:
        _, train_roll, _, _ = self._run_both(use_timing=True)
        # Prompt is the fixed system+task prefix; the completion is alternating
        # assistant tool-call turns and tool-result turns.
        self.assertEqual([m["role"] for m in train_roll.prompt_messages], ["system", "user"])
        completion_roles = [m["role"] for m in train_roll.completion_messages]
        self.assertTrue(completion_roles, "completion should have at least one assistant turn")
        self.assertEqual(completion_roles[0], "assistant")
        # Every assistant turn is a parseable tool call; every tool turn is a result.
        for msg in train_roll.completion_messages:
            self.assertIn(msg["role"], ("assistant", "tool"))
            if msg["role"] == "assistant":
                self.assertTrue(parse_actions(msg["content"]), msg["content"])


# --------------------------------------------------------------------------- #
# Trace-equivalence against the real worker-gated env
# --------------------------------------------------------------------------- #
@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class TraceEquivalenceIntegrationTests(unittest.TestCase):
    def test_real_env_training_loop_matches_eval_loop(self) -> None:
        for family, band in (("bounded_sweep", "easy"), ("hidden_adjacency", "easy"), ("hidden_adjacency", "medium")):
            task = _config(family, band)
            for seed in range(3):
                eval_env = RowHammerTaskEnv(task=task)
                try:
                    eval_res = run_episode_local(eval_env, ReferenceProbePolicy(), seed=seed, task=task)
                finally:
                    eval_env.close()
                train_env = RowHammerTaskEnv(task=task)
                try:
                    train_roll = run_training_episode_local(
                        train_env, ToolPolicyGenerator(ReferenceProbePolicy()), seed=seed, task=task, max_turns=512
                    )
                finally:
                    train_env.close()
                self.assertEqual(
                    [_traj_tuple(s) for s in train_roll.trajectory],
                    [_traj_tuple(s) for s in eval_res.trajectory],
                    (family, band, seed),
                )
                self.assertEqual(train_roll.reward, eval_res.reward, (family, band, seed))
                self.assertEqual(train_roll.reward, 1.0, (family, band, seed))


# --------------------------------------------------------------------------- #
# Completion mask: only assistant spans are trainable
# --------------------------------------------------------------------------- #
class _FakeChatTokenizer:
    """Deterministic ChatML-style tokenizer for host-side mask tests (no transformers).

    Renders ``<|role|> tokens... <|end|>`` per message plus a trailing ``<|assistant|>``
    for ``add_generation_prompt`` — prefix-consistent, unlike the Qwen3 fakes below.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _id(self, token: str) -> int:
        return self._vocab.setdefault(token, len(self._vocab) + 1)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        toks: list[str] = []
        for m in messages:
            toks.append(f"<|{m['role']}|>")
            toks.extend(str(m["content"]).split())
            toks.append("<|end|>")
        if add_generation_prompt:
            toks.append("<|assistant|>")
        if not tokenize:
            return " ".join(toks)
        return [self._id(t) for t in toks]

    def decode_id(self, token_id: int) -> str:
        return {v: k for k, v in self._vocab.items()}[token_id]


class _ThinkInjectingTokenizer(_FakeChatTokenizer):
    """Reproduces Qwen3 under ``enable_thinking=False``: the generation prompt injects an
    empty ``<think></think>`` block that stored assistant messages do NOT carry.
    """

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=True):
        toks: list[str] = []
        for m in messages:
            toks.append(f"<|{m['role']}|>")
            toks.extend(str(m["content"]).split())
            toks.append("<|end|>")
        if add_generation_prompt:
            toks.append("<|assistant|>")
            if not enable_thinking:  # ephemeral empty think block, generation-prompt only
                toks += ["<think>", "</think>"]
        if not tokenize:
            return " ".join(toks)
        return [self._id(t) for t in toks]


class _Qwen3StyleTokenizer(_FakeChatTokenizer):
    """The real Qwen3 quirk: under ``enable_thinking=False`` the empty ``<think></think>``
    is attached to the *last* assistant turn of a render (and the generation prompt) but
    stripped from earlier ones, so the stored transcript is not prefix-consistent across
    turns. Unlike ``_ThinkInjectingTokenizer`` (generation-prompt only), this also injects
    into stored history.
    """

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=True):
        toks: list[str] = []
        n = len(messages)
        for idx, m in enumerate(messages):
            toks.append(f"<|{m['role']}|>")
            # Only the LAST message, when an assistant turn, carries the empty think
            # block in history; earlier assistant turns have it stripped.
            if m["role"] == "assistant" and idx == n - 1:
                toks += ["<think>", "</think>"]
            toks.extend(str(m["content"]).split())
            toks.append("<|end|>")
        if add_generation_prompt:
            toks.append("<|assistant|>")
            if not enable_thinking:  # ephemeral empty think block, generation-prompt only
                toks += ["<think>", "</think>"]
        if not tokenize:
            return " ".join(toks)
        return [self._id(t) for t in toks]


class _Qwen3ThinkingTokenizer(_FakeChatTokenizer):
    """Qwen3 with ``enable_thinking=True``: the template preserves authored
    ``<think>...</think>`` content on every turn (only stripped when empty) and injects
    no empty think block into the generation prompt, so the whole authored span
    (reasoning + tool call) is trainable.
    """

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, enable_thinking=True):
        toks: list[str] = []
        for m in messages:
            toks.append(f"<|{m['role']}|>")
            toks.extend(str(m["content"]).split())  # content already carries <think>...</think>
            toks.append("<|end|>")
        if add_generation_prompt:
            toks.append("<|assistant|>")  # enable_thinking=True => no empty think injected
        if not tokenize:
            return " ".join(toks)
        return [self._id(t) for t in toks]


class _NonPrefixTokenizer(_FakeChatTokenizer):
    """A template that rewrites earlier tokens as the transcript grows (pathological)."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        toks = [f"len{len(messages)}"]  # a leading token that changes every turn
        for m in messages:
            toks.append(f"<|{m['role']}|>")
            toks.extend(str(m["content"]).split())
            toks.append("<|end|>")
        if add_generation_prompt:
            toks.append("<|assistant|>")
        if not tokenize:
            return " ".join(toks)
        return [self._id(t) for t in toks]


class CompletionMaskTests(unittest.TestCase):
    def _fixed_transcript(self) -> list[dict[str, str]]:
        # Words are role-prefixed so the assertion can identify authorship per token:
        # P_ = prompt (system/user), A_ = assistant, T_ = tool.
        return [
            {"role": "system", "content": "P_sys0 P_sys1"},
            {"role": "user", "content": "P_task0 P_task1 P_task2"},
            {"role": "assistant", "content": "A_probe0 A_probe1"},
            {"role": "tool", "content": "T_result0 T_result1 T_result2"},
            {"role": "assistant", "content": "A_hammer0"},
            {"role": "tool", "content": "T_flip0"},
            {"role": "assistant", "content": "A_finish0"},
        ]

    def test_mask_covers_only_assistant_spans(self) -> None:
        messages = self._fixed_transcript()
        tok = _FakeChatTokenizer()
        prompt_ids, completion_ids, mask = build_masked_completion(messages, tok, n_prompt_messages=2)

        self.assertEqual(len(completion_ids), len(mask))
        self.assertGreater(len(prompt_ids), 0)
        # Both classes present: some trainable, some masked.
        self.assertIn(1, mask)
        self.assertIn(0, mask)

        words = [tok.decode_id(t) for t in completion_ids]
        trainable_words = {w for w, m in zip(words, mask) if m == 1}
        masked_words = {w for w, m in zip(words, mask) if m == 0}

        # Every assistant CONTENT token is trainable; no prompt/tool content token is.
        for word, m in zip(words, mask):
            if word.startswith("A_"):
                self.assertEqual(m, 1, word)
            if word.startswith(("P_", "T_")):
                self.assertEqual(m, 0, word)

        # Exactly the assistant content words are trainable (aside from turn markers).
        self.assertEqual({w for w in trainable_words if w.startswith(("A_", "P_", "T_"))},
                         {"A_probe0", "A_probe1", "A_hammer0", "A_finish0"})
        self.assertTrue(any(w.startswith("T_") for w in masked_words))
        # Prompt content lives only in prompt_ids, never in the completion region.
        self.assertFalse(any(w.startswith("P_") for w in words))

    def test_prompt_ids_are_the_completion_prefix(self) -> None:
        messages = self._fixed_transcript()
        tok = _FakeChatTokenizer()
        prompt_ids, completion_ids, _ = build_masked_completion(messages, tok, n_prompt_messages=2)
        full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        self.assertEqual(prompt_ids + completion_ids, full)

    def test_non_prefix_template_fails_closed(self) -> None:
        messages = self._fixed_transcript()
        with self.assertRaises(ValueError):
            build_masked_completion(messages, _NonPrefixTokenizer(), n_prompt_messages=2)

    def test_generation_prompt_think_injection_does_not_break_mask(self) -> None:
        # The injected empty <think></think> must land in the prompt, not the completion.
        messages = self._fixed_transcript()
        tok = _ThinkInjectingTokenizer()
        prompt_ids, completion_ids, mask = build_masked_completion(
            messages, tok, n_prompt_messages=2, enable_thinking=False
        )
        words = [tok.decode_id(t) for t in completion_ids]
        # The injected think scaffolding is part of the prompt the model saw, not the
        # completion it authored.
        self.assertIn("<think>", [tok.decode_id(t) for t in prompt_ids])
        self.assertNotIn("<think>", words)
        # Assistant content trainable; prompt/tool content masked; boundaries intact.
        for word, m in zip(words, mask):
            if word.startswith("A_"):
                self.assertEqual(m, 1, word)
            if word.startswith(("P_", "T_")):
                self.assertEqual(m, 0, word)
        trainable = {w for w, m in zip(words, mask) if m == 1 and w.startswith(("A_", "P_", "T_"))}
        self.assertEqual(trainable, {"A_probe0", "A_probe1", "A_hammer0", "A_finish0"})

    def test_qwen3_last_turn_think_injection_does_not_break_mask(self) -> None:
        # The stored transcript is not prefix-consistent turn to turn (see
        # _Qwen3StyleTokenizer); per-turn anchoring must still mask correctly.
        messages = self._fixed_transcript()
        tok = _Qwen3StyleTokenizer()
        prompt_ids, completion_ids, mask = build_masked_completion(
            messages, tok, n_prompt_messages=2, enable_thinking=False
        )
        words = [tok.decode_id(t) for t in completion_ids]
        for word, m in zip(words, mask):
            if word.startswith("A_"):
                self.assertEqual(m, 1, word)
            if word.startswith(("P_", "T_")):
                self.assertEqual(m, 0, word)
            # The empty think scaffolding on the last stored turn is never trainable —
            # the model did not author it (it is generation-prompt template output).
            if word in ("<think>", "</think>"):
                self.assertEqual(m, 0, word)
        trainable = {w for w, m in zip(words, mask) if m == 1 and w.startswith(("A_", "P_", "T_"))}
        self.assertEqual(trainable, {"A_probe0", "A_probe1", "A_hammer0", "A_finish0"})
        # Turn-0 think scaffolding rode into the prompt, matching the baked dataset prompt.
        self.assertIn("<think>", [tok.decode_id(t) for t in prompt_ids])

    def test_reasoning_mode_masks_think_and_content_as_trainable(self) -> None:
        # enable_thinking=True: no empty-think scaffolding, so the whole authored span
        # (reasoning + tool call) is trainable.
        messages = [
            {"role": "system", "content": "P_sys0"},
            {"role": "user", "content": "P_task0"},
            {"role": "assistant", "content": "<think> R_r0 R_r1 </think> A_probe0"},
            {"role": "tool", "content": "T_res0"},
            {"role": "assistant", "content": "<think> R_r2 </think> A_finish0"},
        ]
        tok = _Qwen3ThinkingTokenizer()
        prompt_ids, completion_ids, mask = build_masked_completion(
            messages, tok, n_prompt_messages=2, enable_thinking=True
        )
        words = [tok.decode_id(t) for t in completion_ids]
        for word, m in zip(words, mask):
            if word.startswith(("A_", "R_")) or word in ("<think>", "</think>"):
                self.assertEqual(m, 1, word)  # model-authored reasoning + content
            if word.startswith(("P_", "T_")):
                self.assertEqual(m, 0, word)
        trainable = {
            w for w, m in zip(words, mask)
            if m == 1 and (w.startswith(("A_", "R_", "P_", "T_")) or w in ("<think>", "</think>"))
        }
        self.assertEqual(trainable, {"<think>", "</think>", "R_r0", "R_r1", "R_r2", "A_probe0", "A_finish0"})


# --------------------------------------------------------------------------- #
# to_grpo_example: rollout + tokenizer -> training triple
# --------------------------------------------------------------------------- #
class ToGrpoExampleTests(unittest.TestCase):
    def test_example_carries_masked_completion_and_trusted_reward(self) -> None:
        victim, cands, banks, aggressors = _window()
        env = _FakeDiscoveryEnv(victim, cands, banks, aggressors)
        roll = run_training_episode_local(env, ToolPolicyGenerator(ReferenceProbePolicy()), seed=0, max_turns=64)
        example = to_grpo_example(roll, _FakeChatTokenizer())
        self.assertEqual(len(example["completion_ids"]), len(example["completion_mask"]))
        self.assertIn(1, example["completion_mask"])
        self.assertIn(0, example["completion_mask"])
        # Reward is the trusted trajectory reward, not fabricated from text.
        self.assertEqual(example["reward"], roll.reward)
        self.assertEqual(example["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
