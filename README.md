# RowHammer OpenEnv

Phase 0 repository for the RowHammer-OpenEnv environment described in
`spec/`.

Current scope:

- provenance manifest format and source admission rules;
- simulation-only threat model;
- host-interface denylist;
- no-mock and fail-closed policy;
- local phase 0 verification.

No Ramulator, OpenEnv, disturbance, profile, sandbox, or task runtime code is
admitted yet. Executable implementation starts only after the external source
pins in `SOURCE_MANIFEST.yaml` are resolved and admitted.

Run the phase 0 gate:

```sh
python3 -B scripts/verify_phase0.py
python3 -B -m unittest discover -s tests
```
