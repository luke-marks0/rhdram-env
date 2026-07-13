"""Multi-turn training rollout + completion mask (P27).

Two host-runnable checks, matching P27's two host-side "Done when" bullets:

* **Trace-equivalence.** The P26 ``ReferenceProbePolicy`` driven through the new
  training loop (``run_training_episode_local`` + ``ToolPolicyGenerator``) produces the
  *exact same* trajectory + reward as driving it through the established eval loop
  (``run_episode_local``). The training loop serializes each tool call to a fenced-JSON
  completion and re-parses it through the single-shot trainer's parse path, so proving
  equivalence proves that render/parse round-trip is lossless and the loop steps the
  env identically. Checked against a fake timing/flip env (no worker) and, when the
  worker is built, against the real ``RowHammerTaskEnv`` discovery configs.

  (P27 says "manually via ``RowHammerClient``"; the host has no websocket server, so —
  as with the P26 reference checks — the in-process ``run_episode_local`` is the host
  analog, and the async HTTP twin shares the same loop body.)

* **Completion mask.** ``build_masked_completion`` marks ``1`` exactly on
  assistant-authored spans and ``0`` on tool-result turns + template scaffolding, on a
  fixed transcript, with a fake ``apply_chat_template`` tokenizer (no ``transformers``).

The third "Done when" bullet — a tiny end-to-end GRPO run showing a non-trivial
gradient/advantage signal — needs ``trl``/``torch``/GPU and is **instance-only**; it is
not exercised here.
"""

from __future__ import annotations

import pathlib
import unittest

import yaml

from rowhammer_env import RowHammerTaskEnv
from rowhammer_env.llm import (
    ReferenceProbePolicy,
    ToolPolicyGenerator,
    build_masked_completion,
    render_tool_call,
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

    Renders ``<|role|> tokens... <|end|>`` per message and a trailing ``<|assistant|>``
    for ``add_generation_prompt`` — the prefix-consistent shape real ChatML templates
    have. ``tokenize=True`` returns stable integer ids; ``decode_id`` reverses them.
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

    This is exactly what tripped the naive incremental diff — ``render(prompt,
    add_generation_prompt=True)`` is not a token-prefix of the fuller stored transcript.
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
        # Regression: Qwen3 enable_thinking=False injects an empty <think></think> into
        # the generation prompt only. The mask builder must not fail-closed on that, must
        # keep the think tokens in the PROMPT (not the completion), and must still mask
        # only assistant content as trainable.
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
