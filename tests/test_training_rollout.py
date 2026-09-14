"""Rollout driver contracts: GRPO advantages, assistant-only token masking, probe
shaping, and rollout trace equivalence between the scripted (eval) and rendered-text
(training) paths. The first three groups are torch-free and worker-free; the last drives
the real simulator.
"""
from __future__ import annotations

import pathlib
import unittest

from rowhammer_env.training import grpo, prompt
from rowhammer_env.training.metrics import EpisodeResult, TurnRecord
from rowhammer_env.training.probe import is_decisive_probe, unique_probe_candidates

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "build/phase2/ramulator_worker"


class AdvantageTests(unittest.TestCase):
    def test_group_advantages_are_zero_mean_and_normalized(self) -> None:
        adv = grpo.group_advantages([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(sum(adv), 0.0, places=6)
        self.assertGreater(adv[2], 0.0)
        self.assertLess(adv[0], 0.0)

    def test_zero_variance_group_yields_no_signal(self) -> None:
        self.assertEqual(grpo.group_advantages([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])
        self.assertEqual(grpo.reward_variance([0.3, 0.3]), 0.0)
        self.assertGreater(grpo.reward_variance([0.0, 1.0]), 0.0)


class FakeTokenizer:
    """Whitespace vocab: id 0 is eos, every other word gets a stable id."""

    def __init__(self) -> None:
        self.vocab = {"<eos>": 0}
        self.eos_token_id = 0

    def _id(self, token: str) -> int:
        return self.vocab.setdefault(token, len(self.vocab))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._id(t) for t in text.split()]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids if not (skip_special_tokens and i == 0))


class MaskingTests(unittest.TestCase):
    """Only the sampled assistant tokens are trained on — prompts/tool results are not."""

    def _turn(self, prompt_text: str, assistant_text: str, tok: FakeTokenizer) -> TurnRecord:
        prompt_ids = tok.encode(prompt_text)
        gen_ids = tok.encode(assistant_text)
        return TurnRecord(
            tool="dram.issue", args={}, text=assistant_text, accepted=True, error=None,
            prompt_ids=prompt_ids, gen_ids=gen_ids, gen_logprobs=[0.0] * len(gen_ids),
        )

    def test_samples_train_exactly_the_assistant_tokens(self) -> None:
        tok = FakeTokenizer()
        turn = self._turn("system and tool context here", "issue the hammer now", tok)
        episode = EpisodeResult(
            task="t", stage="s", seed=1, success=1.0, shaping=0.0, turns=[turn],
            decisive_probes=0, unique_probes=0, valid_calls=1, total_calls=1,
            budget_used={}, error=None, done_reason="success",
        )
        [sample] = grpo.samples_from_group([episode], [0.7])
        # The trained targets are precisely the assistant tokens, and the full sequence
        # is context followed by exactly those tokens (what logprobs_for scores).
        self.assertEqual(sample.gen_ids, tok.encode("issue the hammer now"))
        full = sample.prompt_ids + sample.gen_ids
        self.assertEqual(full[len(sample.prompt_ids):], sample.gen_ids)
        self.assertEqual(tok.decode(sample.gen_ids), "issue the hammer now")
        # No prompt/context token position is in the trained region.
        self.assertEqual(len(sample.gen_ids), len(tok.encode("issue the hammer now")))
        self.assertEqual(sample.advantage, 0.7)

    def test_scripted_turns_without_tokens_are_never_trained(self) -> None:
        scripted = TurnRecord(tool="episode.finish", args={}, text="", accepted=True, error=None)
        episode = EpisodeResult(
            task="t", stage="s", seed=1, success=0.0, shaping=0.0, turns=[scripted],
            decisive_probes=0, unique_probes=0, valid_calls=1, total_calls=1,
            budget_used={}, error=None, done_reason="finish",
        )
        self.assertEqual(grpo.samples_from_group([episode], [0.0]), [])


class ProbeShapingTests(unittest.TestCase):
    def _issue(self, rows: list, acts_delta: int, hits: int = 8) -> TurnRecord:
        per_addr = {}
        for r in rows:
            key = f"logical:{r['addr']}" if r.get("kind") == "logical" else f"handle:{r['id']}"
            per_addr[key] = {"acts": hits, "hits": hits, "misses": 0}
        return TurnRecord(
            tool="dram.issue",
            args={"commands": [{"op": "HAMMER", "rows": rows, "pairs": 8}]},
            text="", accepted=True, error=None,
            feedback={"timing_digest": {"acts_delta": acts_delta, "per_addr_hits": per_addr}},
        )

    def test_decisive_probe_requires_pairwise_repeated_clean_reading(self) -> None:
        victim = {"kind": "logical", "addr": 100}
        cand = {"kind": "logical", "addr": 200}
        self.assertTrue(is_decisive_probe(self._issue([victim, cand], acts_delta=0)))    # different bank
        self.assertTrue(is_decisive_probe(self._issue([victim, cand], acts_delta=16)))   # same bank
        self.assertFalse(is_decisive_probe(self._issue([victim, cand], acts_delta=1)))   # ambiguous
        self.assertFalse(is_decisive_probe(self._issue([victim, cand], acts_delta=0, hits=1)))  # one-shot warm

    def test_each_candidate_credited_once(self) -> None:
        victim = {"kind": "logical", "addr": 100}
        c1 = {"kind": "logical", "addr": 200}
        c2 = {"kind": "logical", "addr": 300}
        meta = {"objective": {"target": victim}, "candidates": [c1, c2]}
        turns = [
            self._issue([victim, c1], 0),   # decisive, c1
            self._issue([victim, c1], 16),  # re-probe c1 -> no new credit
            self._issue([victim, c2], 0),   # decisive, c2
            self._issue([c1, c2], 0),       # victim not involved -> not credited
        ]
        self.assertEqual(unique_probe_candidates(turns, meta), {"logical:200", "logical:300"})


@unittest.skipUnless(WORKER.is_file(), "Phase 2 worker not built")
class TraceEquivalenceTests(unittest.TestCase):
    """The scripted (eval) path and the rendered-then-parsed text (training) path drive
    the environment identically — the training-scope rollout-trace-equivalence claim."""

    def _run(self, policy):
        from rowhammer_env.poc import PoCEnv, load_task
        from rowhammer_env.training.reference import ReferenceSolver
        from rowhammer_env.training.rollout import ActResult, run_episode

        task = load_task("configs/tasks/hidden_adjacency_easy.yaml")
        env = PoCEnv(task=task)
        try:
            scripted = run_episode(env, task=task, seed=7, stage="hidden_adjacency_easy",
                                   policy=ReferenceSolver(), max_turns=40)
            texts = [prompt.render_action({"tool": t.tool, "args": t.args}) for t in scripted.turns]

            class ReplayText:
                def __init__(self) -> None:
                    self.i = 0

                def act(self, *, obs, messages):
                    text = texts[self.i]
                    self.i += 1
                    return ActResult(action=prompt.parse_action(text), text=text)

            replay = run_episode(env, task=task, seed=7, stage="hidden_adjacency_easy",
                                 policy=ReplayText(), max_turns=40)
            return scripted, replay
        finally:
            env.close()

    def test_scripted_and_text_paths_match(self) -> None:
        scripted, replay = self._run(None)
        self.assertEqual(scripted.success, replay.success)
        self.assertEqual(len(scripted.turns), len(replay.turns))
        for a, b in zip(scripted.turns, replay.turns):
            self.assertEqual(a.tool, b.tool)
            self.assertEqual(a.error, b.error)
            self.assertEqual(a.public_counters, b.public_counters)


if __name__ == "__main__":
    unittest.main()
