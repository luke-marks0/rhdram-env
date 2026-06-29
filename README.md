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
- phase 2 worker protocol with fresh episode lifecycle, sparse memory overlay,
  logical reads/writes, `RD`/`WR`/`WAIT` command issue, and public event traces.
- phase 3 empirical DDR4 read-disturbance profile fitted from the admitted
  VTS25 real-chip data, held-out validated, and signed.

Ramulator and OpenEnv are admitted for the Phase 1 bootstrap; the `ddr4_vts25`
source and the `ddr4_vts25_v1` profile are admitted for Phase 3. The disturbance
engine, sandbox, tasks, reward, SDK, and mitigations remain unavailable until
their own phase gates pass, so the profile is not yet wired into a running flip
model.

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
