# Operations: P19 Rollouts And Evaluation

Start the environment server:

```sh
python3 -m rowhammer_env.server.app
```

Run the P19 gate:

```sh
python3 -B scripts/verify_phase19.py
```

Run the reward-updated training example against an already-running server:

```sh
python3 -B scripts/train_phase19_policy.py --base-url http://127.0.0.1:8000
```

The evaluation harness lives in `rowhammer_env.llm.rollout`; metrics are
summarized by `rowhammer_env.observability.summarize_episodes`. The held-out
profile-generalization split is loaded from
`configs/tasks/profile_generalization_eval.yaml` and reported under the `eval`
split in metrics.

## Release Re-Qualification

Run the P20 release gate after all admitted phase gates are green:

```sh
python3 -B scripts/verify_release.py
```

Use `--rebuild` when the local Ramulator worker/config artifacts need to be
rebuilt before qualification:

```sh
python3 -B scripts/verify_release.py --rebuild
```

The release gate runs every admitted phase verifier, the complete unit suite
with zero required skips, deterministic replay for fixed seeds, executable
no-mock scanning, manifest pin checks, and release bundle hygiene.
