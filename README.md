# RowHammer OpenEnv

Phase 0 repository for the RowHammer-OpenEnv environment described in
`spec/`.

Current scope:

- provenance manifest format and source admission rules;
- simulation-only threat model;
- host-interface denylist;
- no-mock and fail-closed policy;
- local phase 0 verification.
- phase 1 bootstrap from OpenEnv-style `reset()`/`step()` into a real
  Ramulator 2.1 `External` frontend request path.

Ramulator and OpenEnv are admitted only for the Phase 1 bootstrap. Disturbance,
profile, sandbox, task, reward, SDK, and mitigation implementation remain
unavailable.

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
