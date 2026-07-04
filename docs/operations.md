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
