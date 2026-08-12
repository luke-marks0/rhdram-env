# Training loop overview

A catch-up for someone who knows the **environment** (reset/step/reward, task
compiler, disclosure, disturbance, worker) but not the **RL training** side built on
top of it. It maps the moving parts and where they live.

## What training is trying to do

Teach a language-model policy to **discover** RowHammer adjacency — probe the DRAM
timing side channel, narrow a candidate set to the real same-bank neighbors, then
hammer them — rather than recite a disclosed hint. The environment you built already
supplies the only thing training trusts: a **sparse, trusted reward** of `1.0` when a
real simulated flip satisfies the objective, `0.0` otherwise (`_trusted_success()`).
Nothing on the training side ever invents reward or reads it from model-declared
success.

## The algorithm in one paragraph

We use **GRPO** (Group Relative Policy Optimization) via TRL. For each prompt the
trainer samples `num_generations` completions, scores each with the trusted reward, and
the per-completion advantage is *its reward minus the group mean*. Consequence worth
internalizing: if all completions in a group get the same reward (all `0.0` or all
`1.0`), the advantage is zero and **no gradient flows**. So training only makes progress
on prompts where the policy *sometimes* succeeds — which is what the curriculum and the
shaping term below are for.

## The data model

One training example = one **task instance** = `(task_config, seed)`. Everything about an
instance (target row, candidate layout, secret address mapping, disturbance family) is a
deterministic hash of `task_id:seed` (`compiler.py:_rng`). So:

- The same seed always yields the same puzzle with the same answer.
- More seeds = more distinct puzzles. Seeds are cheap; a held-out seed range is a clean
  generalization test.

Configs express seeds as `seeds: [...]` or the shorthand `seed_range: [start, stop]`
(inclusive).

## Tiers and difficulty

| Tier | Family | What's disclosed | What's hard |
|---|---|---|---|
| Tier 0 | `known_target` | exact physical target + aggressor hint | nothing — forms a hammer command |
| Tier 2a | `bounded_sweep` | N opaque candidate handles | narrow by timing (handles aren't computable) |
| Tier 2b | `hidden_adjacency` | N numeric addrs + secret mapping | narrow by timing (mapping is a per-episode secret) |

Within the discovery tiers, difficulty = candidate-window width `N` (`easy=4`,
`medium=16`, `hard=64`, `BAND_CANDIDATES` in `compiler.py`). More candidates = longer
probe→narrow→hammer horizon.

## Components (by file)

**Reward / rollout paths — `rowhammer_env/llm/`**

- `grpo_env.py` — the single-shot reward path: parses one completion into an action
  list, replays it through the real server, returns the trusted reward. Also owns the
  prompt (`build_messages`), the `hint_level` toggle, and `public_hints`.
- `multiturn_rollout.py` — the turn-by-turn rollout the discovery families need (probe,
  read timing digest, narrow, hammer). Torch-free; builds the token-level completion
  mask (assistant spans trainable, tool-result spans masked). This is the loop GRPO
  actually trains on.
- `rollout.py` / `policies.py` — the interactive driver (`run_episode`) and policy
  adapters: `OpenAICompatibleToolPolicy` (any chat endpoint), `ReferenceProbePolicy`
  (the deterministic DRAMA solver), `CIHammerFixturePolicy` (disclosed-hint hammer).
- `curriculum.py` — parses the `curriculum:` config block into ordered stages.
- `shaping.py` — the bounded, training-only probe-shaping bonus (see below).
- `wandb_logging.py` — rollout/metric logging.

**Entry points — `scripts/`**

- `train_grpo.py` — the trainer. Builds the dataset from `(task, seed)` pairs, loads the
  model (LoRA), wires the reward funcs + multi-turn `rollout_func`, runs GRPO. Flags:
  `--dry-run` (data + reference-hammer reward, no model), `--stage NAME` (train one
  curriculum stage), `--resume-adapter PATH` (load a prior adapter as init),
  `--output-dir PATH`.
- `train_curriculum.py` — the staged-curriculum driver: runs `train_grpo.py` once per
  stage in order, chaining each stage's saved LoRA adapter into the next. This is the
  real "master easy before hard" progression.
- `eval_llm.py` — evaluate any chat model against the env with **no training**. Run this
  first to get a baseline before spending GPU time.

## How the curriculum works

The `curriculum:` block lists stages in increasing difficulty
(`Tier 0 → 2a easy/med/hard → 2b easy/med/hard`), each with a `reference_min_success`
gate. `curriculum.py` validates ordering; `tests/test_curriculum.py` runs the
deterministic reference policy on the host to prove each band is solvable *before* GPU
time.

Two ways to consume it:

1. **Flat** (`train_grpo.py` alone): all stages become one dataset (easiest first). Note
   TRL shuffles by default, so this is really "all tiers in one pot," not staged.
2. **Staged** (`train_curriculum.py`): one run per stage, adapters chained. Preferred —
   it keeps each run in a band where the policy sometimes succeeds (nonzero, non-
   saturated reward), which is the only regime GRPO learns from.

## Reward shaping (why the sparse reward becomes learnable)

`shaping.py` adds one auxiliary term: a small bonus for each turn on which the policy
made a **decisive** bank-conflict measurement (a clean same-bank/different-bank reading
in the trusted `timing_digest`). It is outcome-neutral (same/different bank score
equally), bounded in `[0,1]`, weighted strictly below `success_weight`, and training-
only. It gives gradient *before* the first real flip, bootstrapping the probe behavior.

## Prompt hint levels

`prompt.hint_level` (in `grpo_env.py`):

- `full` — discloses aggressor addresses. On Tier 0 this hands over the answer → all
  completions win → zero variance → no gradient. Avoid for real training.
- `geometry` — physics constants only.
- `none` — the discovery setting; forces the policy to actually probe.

## Running it

```bash
# 1) Baseline — how far does an untrained model get? (no GPU training)
export RHD_LLM_CHAT_COMPLETIONS_URL=http://127.0.0.1:8000/v1/chat/completions
export RHD_LLM_MODEL=Qwen/Qwen3-8B
python3 -B scripts/eval_llm.py --task configs/tasks/bounded_sweep_easy.yaml --seeds 1-16

# 2) Sanity — data + reward pipeline, no model
python3 -B scripts/train_grpo.py --config configs/training/grpo_curriculum.yaml --dry-run

# 3) Host gate — reference policy must clear every band (no torch)
python3 -m unittest tests.test_curriculum

# 4) Train (staged, on the instance)
python3 -B scripts/train_curriculum.py --config configs/training/grpo_curriculum.yaml
```

## Gotchas worth knowing up front

- **Zero-variance = no learning.** All-win (full hints) and all-lose (too hard) both
  kill the gradient. The curriculum + shaping exist to keep groups mixed.
- **Overfitting.** With few seeds the policy can memorize per-seed answers. Use many
  training seeds and a disjoint held-out `eval:` seed range (eval-during-training is
  wired in `train_grpo.py`).
- **Solvable ≠ learnable.** The reference-policy gate proves the signal is sufficient; it
  does not prove GRPO will discover the strategy. It's a floor, not a guarantee.
- **`torch`/`trl` are instance-only.** Everything else (env, rollout, curriculum,
  shaping, `eval_llm`, dry-run) runs on the host without them.
