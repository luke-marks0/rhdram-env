# PoC training and evaluation

Local status: the native gate passed 166 tests with no skips. Tiny-model CPU
optimizer/reload/resume checks passed; the full 4B GPU smoke and main experiment
have **not** run here. A working CUDA driver is required on the compute instance.
See [CURRENT_WORK.md](../CURRENT_WORK.md) for the evidence and remaining checks.

Use **one H100 80 GB**, about 16 CPU cores, 64 GB host RAM, and 100 GB free local
storage. An A100 80 GB is a reasonable alternative; throughput and peak CUDA memory
have not been measured here. Measure the first ten optimizer steps before
estimating the full job's duration. No vLLM, quantization, or multi-GPU setup is
required by this recipe.

The supported configuration is [`poc.yaml`](../configs/training/poc.yaml). It
freezes the five scoped tasks, DDR4, `ddr4_vts25_v1`, 50 C, baseline refresh, and
three policy tools. Reference controls and training use the same native simulator.

## Recommended method

| Choice | Concrete setting |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Responses | Action-only JSON; no explicit chain-of-thought |
| Adaptation | BF16 LoRA, rank 32, alpha 64, dropout 0, attention and MLP projections |
| Cold start | 64 reference episodes per task, 320 total; one SFT epoch, LR `1e-4`, effective batch 16 |
| RL | Multi-turn GRPO with TRL's token-normalized `dapo` loss; no critic or learned reward model |
| Sampling | 8 trajectories per task instance, 4 task instances / 32 trajectories per optimizer update |
| Memory | Microbatch 1, accumulation 32, checkpointed gradients, HF generation in waves of 4 |
| RL optimizer | LR `5e-6`, cosine decay, 5% warmup, grad norm cap 1, KL weight 0 |
| Sampling distribution | Temperature 0.8, top-p 1, top-k 0; recorded log probabilities use the same temperature |
| Interaction limits | 40 turns, 512 generated tokens per turn, 32,768 total context tokens |
| Shaping | Unique-candidate probe coverage at weight 0.1 in intermediate stages; 0 for known target and final hidden-medium stage |
| Seed splits | Train 1–256; development validation 1000–1031; final test 10000–10031; training RNG seed 42 |

This model is a dense 4B instruction model whose published variant is explicitly
non-thinking. Its size leaves room for interactive trajectories on one GPU. This
choice is an engineering judgment, not a measured win over other models.
[Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

LoRA reduces trainable parameters and optimizer storage; it does not eliminate
context/activation/KV-cache costs. Here the interaction budget and model size
matter as much as the adapter rank. [LoRA paper](https://arxiv.org/abs/2106.09685).

The `dapo` setting only selects token-level loss normalization. This repository
does not implement the full DAPO dynamic-sampling system. The pinned TRL custom
rollout hook accepts the external-token mask; unsupported configuration keys and
missing reward columns fail rather than being silently ignored.
[TRL GRPO documentation](https://huggingface.co/docs/trl/v1.8.0/en/grpo_trainer),
[DAPO paper](https://arxiv.org/abs/2503.14476).

## Do you need reward shaping?

Not mathematically. Policy gradients differentiate the probability of the sampled
actions, so the simulator does not need to be differentiable. Sparse trusted
success is a valid objective. The practical problem is observing enough
reward variation to learn the long probe/classify/hammer sequence.

GRPO centers rewards within each group. With eight identical rewards, every
reward advantage is zero, whether all eight episodes failed or all eight
succeeded. KL regularization would not provide a task-solving signal for an
all-zero group. Under independent Bernoulli success probability `p`, the chance
that a group of eight contains both success and failure is:

`1 - (1-p)^8 - p^8`

At `p=1%`, only about 7.7% of groups have this sparse learning signal. Increasing
group size helps moderately; it cannot rescue a policy with essentially zero
success probability. The CPU integration test observed the actual TRL rollout
hook produce exactly zero loss, gradient norm, and parameter change when all real
episode rewards were zero. [GRPO formulation in DeepSeekMath](https://arxiv.org/abs/2402.03300).

Use the small SFT cold start first. The optional shaping term then rewards the
fraction of distinct disclosed candidates measured against the victim using
short, decisive, truly alternating probes. Same-bank and different-bank readings
earn equal credit. Repeating one measurement, grouped reads, huge hammers,
invalid actions, and hidden aggressor labels do not add credit. Trusted success
is always 1. With shaping enabled, training totals are 1–1.1 for success and
0–0.1 for failure; reported evaluation rewards remain exactly 0 or 1.

That bound does **not** guarantee a useful gradient or avoid a probing-only local
optimum. Reward normalization can amplify a small auxiliary term in batches with
no successes. This is not potential-based shaping, and optimal-policy preservation
is not claimed. The final stage removes it, and both validation and benchmark
success always use `trusted_reward == 1.0`.

Inspect `signal.jsonl`: sparse success, zero-variance fraction, and groups with a
nonzero advantage. The trainer stops after 20 consecutive batches without reward
variation. If all rewards are zero, inspect parsing, context truncation, and the
SFT warm start before paying for more RL. If rewards are uniformly one, evaluate
that stage on development seeds and move to the next unsolved stage. An already
solved stage does not need forced optimizer updates.

SFT teaches the reference strategy, so compare **SFT-only** with **SFT+GRPO** as
well as the original instruction model. Otherwise gains from demonstrations could
be misattributed to RL. “Untrained” in the evaluator means no environment-specific
adaptation, not a model without general instruction tuning.

## Install and verify

On a fresh Linux GPU checkout, use Python 3.12 and a working NVIDIA CUDA driver.
The exact primary library pins are in `requirements-train.txt`; each run also
records every installed package version. Keep a separate training environment if
the existing `.venv` uses another Python version:

```sh
python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install -r requirements.txt -r requirements-train.txt
PYTHON=.venv-train/bin/python ./setup.sh --skip-deps --poc-verify
source .venv-train/bin/activate
python -B scripts/train_grpo.py --config configs/training/poc.yaml --dry-run
```

The PoC gate requires native worker/profile availability and working local
WebSockets. Missing capabilities or skipped required tests are failures. It does
not run the optional script sandbox, OracleRH, other-standard, or restructure gates.
`RH_POC=1 python -m rowhammer_env.server.app` starts the scoped server manually;
the trainer normally launches it automatically. A reused `--base-url` must serve
that same PoC surface.

Before the GPU smoke gate, run the CPU library check. It downloads only the tokenizer, uses a tiny random
Qwen3, and tests real TRL integration, full-task token masks, LoRA updates, exact
reload, and optimizer/RNG resume. It does not certify GPU training or model skill:

```sh
python -B scripts/verify_training_stack.py --output runs/stack_cpu
```

This also verifies the final two-episode training-time validation batch setting;
that extension was added after the retained local CPU-library run. Transfer the
current source changes to the compute instance, not just an older repository
revision. `runs/` is git-ignored: copy any local evidence or exported demonstrations
separately if you want to reuse them.

## Run the experiment

First preserve the deterministic controls and the untouched instruction-model
baseline on the final test split. All output directories must be new:

```sh
python -B scripts/eval_poc.py --output runs/controls
python -B scripts/eval_poc.py --conditions untrained --output runs/untrained

python -B scripts/eval_poc.py --conditions reference --split train --seed-limit 64 \
  --export-sft --output runs/sft_data
python -B scripts/train_sft.py --data runs/sft_data/sft.jsonl --output runs/poc_sft
python -B scripts/eval_poc.py --conditions sft --adapter runs/poc_sft --output runs/sft_baseline
```

Use **development validation** for tuning/stage selection; final test outputs are
for reporting. Do not retune budgets or prompts against the final test results.

Run the actual GPU smoke gate before the main RL job:

```sh
python -B scripts/smoke_train_poc.py --adapter runs/poc_sft --output runs/gpu_smoke
```

This gate requires actual sampled GRPO rewards to produce a finite nonzero LoRA
update, fresh WS sessions, masked multi-turn tool feedback, exact adapter reload,
optimizer/scheduler/RNG resume from checkpoint 1 to step 2, and sparse evaluation
of the reloaded adapter. It fails if a cold or fully solved policy has no reward
variation; that is a useful diagnosis, not a reason to inject artificial rewards.
Choose `--stage` using development results if the default stage is already solved.

Then train the remaining ladder, starting after the known-target positive control:

```sh
python -B scripts/train_curriculum.py --config configs/training/poc.yaml \
  --from-stage bounded_easy --initial-adapter runs/poc_sft

python -B scripts/eval_poc.py --conditions trained,trained_timing_hidden \
  --adapter runs/poc_training/hidden_medium --output runs/trained

python -B scripts/summarize_poc.py --runs runs/controls runs/untrained runs/sft_baseline runs/trained \
  --output runs/writeup
```

The default remaining stage caps are 100, 150, 100, and 200 optimizer steps.
Intermediate stages use 0.1 shaping; the final stage uses zero. These are a
starting allocation, not evidence that 550 updates are necessary or sufficient.
Use `--from-stage` and `--initial-adapter` to move past a development-validated
solved stage. Stage chaining loads adapter weights into a fresh optimizer. To
resume an interrupted stage with optimizer, scheduler, and RNG state instead:

```sh
python -B scripts/train_grpo.py --config configs/training/poc.yaml --stage hidden_medium \
  --output-dir runs/poc_training/hidden_medium \
  --resume-from-checkpoint runs/poc_training/hidden_medium/checkpoint-50
```

## Interpret and retain the artifacts

Each evaluation saves `run.json`, `resolved_config.json`, `source_snapshot.zip`,
`episodes.jsonl`, `summary.json`, and `summary.csv`. Source snapshots preserve the
measured working code even when it has uncommitted changes. Records contain the
disclosed reset and per-step observations, actions, model text, validity/errors,
probe counts, resource use, remaining budgets, termination reason, and sparse
reward. Validity counts parsed/admitted calls; a syntactically valid hammer that
exhausts budget is recorded separately as a budget failure. Model sampling uses
`42 XOR environment_seed` per episode in every evaluation condition.

Training saves configuration/provenance/source snapshots, `rollouts.jsonl`,
`signal.jsonl`, `training_log.jsonl`, `parameter_update.json`, periodic full-state
checkpoints, and the final adapter/tokenizer. Tensor updates are measured from
actual parameter bytes. A missing final report/checkpoint means the run did not
complete. Keep failed runs and inspect their last records.

The combined table reports per-task success with 95% Wilson intervals. Paired
changes compare identical task seeds (and reject missing/duplicate pairs).
Combining runs also requires matching measured simulator/disclosure source hashes,
model revision, context, and sampling settings.
Timing-hidden evaluation removes timing digests, traces, public counters, clocks,
cycle deltas, and remaining ACT/cycle budgets from **all** model-visible history.
Initial budgets remain visible. Scoring and logged simulator resource costs still
come from the original trusted observations; `messages` records what the model saw.

The existing control matrix gives reference 32/32 on all tasks. Timing-blind is
also 32/32 on the easier tasks and bounded-sweep medium; only hidden-adjacency
medium separates strongly (1/32). Thus the main timing-dependence claim belongs
to that medium task. With 32 seeds, 32/32 has a Wilson interval of about
89.3%–100%; it is not proof of universal success. These are simulation results
under one calibrated profile, not a real-DIMM exploit or cross-hardware prediction.
