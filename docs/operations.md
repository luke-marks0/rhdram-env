# Operations

Start the environment server:

```sh
python3 -m rowhammer_env.server.app
```

The previous policy rollout, evaluation, and training harness has been removed.
See `TRAINING_SCOPE.md` for the replacement boundary.

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
