# Training the RowHammer PoC policy

One Qwen3-8B LoRA policy, trained with multi-turn GRPO on the real Ramulator-backed
PoC environment, then compared against the base model on held-out seeds. This
implements `TRAINING_SCOPE.md`; a negative result is a valid outcome. The whole thing
is one config, one trainer, one in-process rollout path (`PoCEnv.reset()/step()`).

## Layout

```
rowhammer_env/training/
  config.py      strict config load/validate + provenance + seed splits
  prompt.py      observation -> chat turns, and fenced-JSON tool-call parsing
  rollout.py     the ONE multi-turn driver (Policy protocol, run_episode)   [torch-free]
  reference.py   deterministic reference solver + the negative controls      [torch-free]
  probe.py       decisive-probe detection + bounded training-only shaping    [torch-free]
  metrics.py     episode records + aggregation with Wilson CIs               [torch-free]
  artifacts.py   run-directory writer + optional W&B mirror                  [torch-free]
  policy.py      Qwen3 + LoRA: generate, capture tokens, recompute logprobs  [torch]
  grpo.py        group advantages + PPO-clip + KL update                     [torch]
  trainer.py     curriculum, reference gate, SFT, checkpoint/resume, logging [torch]
scripts/train.py      entry point
scripts/evaluate.py   held-out evaluation of every condition
```

Only `policy.py`, `grpo.py`, `trainer.py` import torch, and only lazily — the rollout
driver, reference/controls, and their tests run without the model stack. That is why
`scripts/evaluate.py --scripted-only` (reference + controls) needs no GPU.

## Install

```bash
pip install -r requirements.txt          # environment (already needed to build the sim)
# Install a CUDA build of torch for your machine first, then:
pip install -r requirements-train.txt    # torch, transformers, peft, accelerate
pip install wandb                         # optional, only if wandb.enabled: true
```

Build the native worker first (see `README.md`) so `build/phase2/ramulator_worker`
exists; every episode spawns a fresh worker.

## Run

```bash
# 1. Smoke: one GRPO step on the first stage, updates params, writes a reloadable checkpoint.
python scripts/train.py --config configs/training.yaml --out runs/smoke --smoke

# 2. Full curriculum run.
python scripts/train.py --config configs/training.yaml --out runs/grpo1

# 3. Resume from any checkpoint (adapter + optimizer + scheduler + stage/step + RNG).
python scripts/train.py --config configs/training.yaml --out runs/grpo1 \
    --resume runs/grpo1/checkpoints/stage_1_bounded_sweep_easy

# 4. Evaluate. Controls need no GPU; base/trained pull in torch.
python scripts/evaluate.py --config configs/training.yaml --out runs/grpo1/eval --scripted-only
python scripts/evaluate.py --config configs/training.yaml --out runs/grpo1/eval \
    --base --adapter runs/grpo1/checkpoints/final/adapter --split benchmark
```

## Curriculum

Trained in order, one adapter throughout (`configs/training.yaml: curriculum`):

1. `known_target_anybit`  — positive control: a valid hammer makes trusted reward.
2. `bounded_sweep_easy`   — discover aggressors from opaque candidate handles.
3. `bounded_sweep_medium`
4. `hidden_adjacency_easy` — numeric addresses, hidden address→bank mapping.
5. `hidden_adjacency_medium` — the research task.

Before a stage gets training time, the deterministic reference solver must clear it on
the validation seeds (`_reference_gate`, threshold 90%). The reference uses only the
*disclosed* signal — the candidate list, the returned `timing_digest`, and the disclosed
remaining budget — so clearing a stage proves the disclosed signal is sufficient.

## How reward and learning work

- **Trusted sparse reward.** `1.0` only when committed simulator state satisfies the
  task's success predicate; `0.0` otherwise. Never derivable from claims or
  `episode.finish`. This is the only reward used for validation/benchmark scoring.
- **Probe shaping (training only).** The discovery reward is otherwise all-or-nothing,
  so a bounded auxiliary term (`grpo.probe_shaping`, default 0.2) credits each *unique*
  decisive bank-conflict probe once — computed only from the trusted `timing_digest`,
  outcome-neutral, and strictly below a real success so it can never substitute for a
  flip. It is removed from all evaluation scoring.
- **GRPO.** For each step, `group_size` rollouts of one training seed are scored, their
  rewards normalized into group-relative advantages, and a PPO-clipped update with a KL
  penalty to the frozen base model (LoRA disabled) is applied. Only the exact sampled
  assistant tokens are trained on; prompts, tool results, and templates are masked.

### The one thing to watch: reward variance on the harder stages

GRPO learns from *spread* within a group. If every rollout in a group gets the same
reward, the advantage is zero and there is no gradient. Early in a stage — and
especially on `hidden_adjacency_medium` — an untrained policy may get zero trusted
reward on every rollout. Probe shaping is what keeps variance alive there: a policy that
stumbles onto one clean timing probe already scores above one that does not, so the
group has spread to climb before any full flip happens.

If shaping still produces no spread for `ZERO_VARIANCE_PATIENCE` (24) consecutive
groups, the run stops with a diagnostic (`ZeroVarianceStop`) rather than spinning. If
you hit it, the usual levers are: raise `grpo.probe_shaping` (kept < 1), add an SFT warm
start (`sft.enabled: true`, imitates successful reference demonstrations), or confirm
the policy is emitting valid tool calls at all (`valid_call_rate` in `train_log.jsonl`).

## Config knobs (`configs/training.yaml`)

Merged over the committed defaults; unknown keys are rejected. Highlights:

- `model.*` — checkpoint, LoRA rank/alpha, dtype, device, gradient checkpointing.
- `rollout.*` — `max_turns`, context/new-token caps, sampling temperature.
- `grpo.*` — `group_size`, PPO `clip_epsilon`, `kl_beta`, `probe_shaping`.
- `sft.*` — optional warm start from reference demonstrations (training seeds only).
- `curriculum.*` — GRPO steps per stage.
- `seeds.*` — inclusive `train` / `validation` / `benchmark` ranges; must be disjoint.
- `wandb.*` — optional mirror; any W&B failure is swallowed and never affects training.

## Artifacts (per run directory)

- `config.resolved.yaml`, `provenance.json` — resolved config; git revision, source and
  profile hashes, model identity, and all seeds.
- `train_log.jsonl` — per-step loss, KL, reward mean/variance, success, valid-call rate,
  probes, budget use, LR, GPU memory.
- `episodes.jsonl`, `trajectories.jsonl` — per-episode summaries and disclosed traces.
- `reference_gate.jsonl`, `sft.jsonl` — gate outcomes and SFT dataset size.
- `checkpoints/{sft,stage_*,step_*,final}/` — adapter + optimizer + scheduler + RNG +
  `trainer_state.json`, each reloadable and resumable.
- `metrics.json` (evaluation) — every condition, overall and by stage, with 95% Wilson
  confidence intervals.

## Evaluation conditions

`scripts/evaluate.py` runs, on identical held-out seeds with sparse reward only:
reference solver, finish-only, below-threshold, timing-blind (arithmetic adjacency),
the untrained base model, the trained policy, and the trained policy with the timing
channel hidden (`feedback: summarized_counts`). The primary metric is held-out
target-flip success rate; the timing-blind and timing-hidden conditions test whether
success depends on the intended timing feedback. Note that the RD-addressable
`bounded_sweep` scaffold is brute-forceable within budget, so the load-bearing
timing-channel result lives on the `hidden_adjacency` task, where arithmetic adjacency
lands in the wrong bank and fails regardless of budget.
```
