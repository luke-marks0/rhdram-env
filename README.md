# RowHammer OpenEnv

Repository for the RowHammer-OpenEnv environment described in `spec/`.

Current scope:

- provenance manifest format and source admission rules;
- simulation-only threat model;
- host-interface denylist;
- no-mock and fail-closed policy;
- local phase 0 verification.
- phase 1 bootstrap from OpenEnv-style `reset()`/`step()` into a real
  Ramulator 2.1 `External` frontend request path.
- phase 2 worker protocol with fresh episode lifecycle, sparse memory overlay,
  logical reads/writes, `RD`/`WR`/`WAIT` command issue, and public event traces.
- phase 3 empirical DDR4 read-disturbance profile fitted from the admitted
  VTS25 real-chip data, held-out validated, and signed.
- phase 4 disturbance engine consuming accepted worker events and the signed
  DDR4 VTS25 profile to produce persistent simulated flips.
- phase 5 known-target task wrapper with trusted reward and budgets.
- phase 6 restricted `script.run` path for a small `rh_sdk` broker subset.
- phase 7 oracle mitigation; other mitigations fail closed.
- phase 8 lean advanced task families: any-flip, hidden-target,
  unknown-adjacency, target-cell, pattern-target, and mitigation-aware variants.
- phase 9 profile registry: admitted DDR4 profile loads, non-admitted profiles
  fail closed.
- phase 10 release gate for admitted phase checks, provenance, unit tests, and
  tracked-file hygiene.
- phase 16 mitigation capability discovery: admitted mitigations are listed in
  `dram.info`, while unvalidated Ramulator mitigations fail closed.
- phase 17 OpenEnv HTTP/WebSocket serving with a policy-side client.
- phase 18 OS-isolated `script.run` using `unshare` + `bwrap` and JSON-line
  IPC back to the normal tool surface.
- phase 19 HTTP policy adapter, rollout/eval metrics, and a reward-updated
  training example for the LLM-testable milestone.
- phase 20 release re-qualification: full admitted gate matrix, zero-skip unit
  suite, deterministic replay, no-mock executable scan, and release provenance.

Ramulator and OpenEnv are admitted for the Phase 1 bootstrap; the `ddr4_vts25`
source and the `ddr4_vts25_v1` profile are admitted. HBM2 remains non-admitted
until its license, hashes, and validation package are resolved.

## Quickstart

One command bootstraps a fresh machine: check the toolchain, install the Python
deps, fetch the pinned upstream sources, build the native Ramulator worker,
rebuild the signed DDR4 profile, and run a verification gate. Every stage is
idempotent and safe to re-run.

```sh
./setup.sh                 # full setup + fast verify (P0/P1/P2/P3 gates)
./setup.sh --train         # also install the GRPO training extras
./setup.sh --venv          # install deps into a local ./.venv
./setup.sh --clean         # force a clean native rebuild (after a source re-pin)
./setup.sh --full-verify   # run the complete release re-qualification gate
./setup.sh --help          # all options
```

Host prerequisites: `python3` (3.10+), `git`, `cmake`, a C++20 `g++`, and
optionally `ninja` for faster native builds. The remaining sections document the
individual phase gates that `setup.sh` orchestrates; run them directly when
working on a single phase.

Run the phase 0 gate:

```sh
python3 -B scripts/verify_phase0.py
python3 -B -m unittest discover -s tests
```

Phase 1 needs fetched upstream sources and a local Ramulator build:

```sh
python3 -B scripts/fetch_phase1_sources.py
python3 -B scripts/build_phase1.py
python3 -B scripts/verify_phase1.py
```

Phase 2 builds the persistent simulator worker and runs the command/memory gate:

```sh
python3 -B scripts/build_phase2.py
python3 -B scripts/verify_phase2.py
```

Phase 3 fetches the pinned read-disturbance data, builds the signed profile, and
runs the admission gate:

```sh
python3 -B scripts/fetch_phase3_sources.py
python3 -B -m profile_builder.package.build
python3 -B scripts/verify_phase3.py
```

Phase 4 runs a known-vulnerable DDR4 fixture and no-flip controls:

```sh
python3 -B scripts/verify_phase4.py
```

Phase 5-7 gates:

```sh
python3 -B scripts/verify_phase5.py
python3 -B scripts/verify_phase6.py
python3 -B scripts/verify_phase7.py
```

Phase 8-10 gates:

```sh
python3 -B scripts/verify_phase8.py
python3 -B scripts/verify_phase9.py
python3 -B scripts/verify_release.py
```

Phase 16 checks admitted mitigation discovery and fail-closed behavior:

```sh
python3 -B scripts/verify_phase16.py
```

Phase 17 serves the task environment through the vendored OpenEnv HTTP transport:

```sh
python3 -m pip install -r requirements.txt
python3 -B scripts/verify_phase17.py
python3 -m rowhammer_env.server.app
```

Episodes are stateful only over the WebSocket `/ws` transport — use
`rowhammer_env.client.RowHammerClient` (or the policy-side SDK on
`PYTHONPATH=sdk`: `from rh_sdk import connect`). The HTTP `POST /reset` and
`/step` endpoints are stateless (a fresh env per request) and cannot carry an
episode; set `RH_SERVER_MODE=production` to drop them and expose only
`/ws`, `/health`, `/schema`, `/metadata`, `/mcp`.

Server env vars: `MAX_CONCURRENT_ENVS` (default 8), `RH_SERVER_MODE`
(`simulation`/`production`), and `RH_TASK` (a JSON task config that pins the
default served family). An orchestrator can also override the task per episode by
passing `task=` to `reset` over the WebSocket transport, e.g.
`await client.reset(seed=17, task={"family": "any_flip"})`.

Phase 18 verifies the OS-level sandbox and trace-equivalent IPC broker:

```sh
python3 -B scripts/verify_phase18.py
```

Phase 19 verifies the HTTP policy adapter, held-out eval metrics, and training
example:

```sh
python3 -B scripts/verify_phase19.py
```

Phase 20 runs the release re-qualification gate:

```sh
python3 -B scripts/verify_release.py
python3 -B scripts/verify_phase20.py
```

## GRPO training (TRL)

`scripts/train_grpo.py` trains a small (default `Qwen/Qwen3-4B`) model against the
served environment with TRL's `GRPOTrainer`. Each dataset row is one task instance
(task config + seed); the environment discloses its objective/target at reset,
which is baked into the prompt. GRPO samples several completions per prompt, each
is parsed into a tool-call sequence and **replayed through the real OpenEnv
server**, and the reward is the trusted sparse episode reward (`1.0` only on a real
flip). Hyperparameters — including the `enable_thinking` toggle that disables the
Qwen3 `<think>` block — live in `configs/training/grpo_qwen4b.yaml`.

```sh
python3 -m pip install -r requirements.txt -r requirements-train.txt

# Validate the data + reward pipeline first (no GPU/model load; needs a built
# Phase-2 worker and the P17 HTTP deps):
python3 -B scripts/train_grpo.py --config configs/training/grpo_qwen4b.yaml --dry-run

# Train (launches its own server unless env.base_url is set):
python3 -B scripts/train_grpo.py --config configs/training/grpo_qwen4b.yaml
```

### Monitoring (Weights & Biases)

Set `wandb.enabled: true` in the config (on by default) and training autologs to
wandb: the TRL scalar metrics, TRL's own prompt/completion table
(`log_completions`), and a custom per-rollout table + metrics from
`rowhammer_env.llm.wandb_logging` — the parsed tool calls, the trusted env reward,
and the emitted command-list size (`rollout/n_commands_mean|max`, `rollout/n_pairs_mean`).
The command-list size is the key signal: a real flip needs on the order of the
disclosed `known_threshold` activations, so it makes plain whether completions are
anywhere near the threshold or just truncated at `max_completion_length`. Needs
`pip install wandb` and a `wandb login` (or set `wandb.mode: offline`).
