# Phase 0 Policy Gate

Phase 0 admits policy and provenance only. It does not admit simulator,
profile, sandbox, task, reward, SDK, or mitigation implementation.

## Source Admission

Every external source must have a manifest entry with:

- `id`
- `role`
- `url`
- `commit`
- `license`
- `sha256`
- `retrieved_utc`
- `admission`

`admission: pending_pin` is allowed only while the repository contains no
executable feature work. Later phases require exact commits, licenses, archive
hashes, retrieval dates, and `admission: admitted`.

## No-Mock Policy

Production, test, training, and release paths must not use mock DRAM,
placeholder flip generators, synthetic stand-in calibration data, no-op
mitigations, permissive sandbox fallbacks, local unsandboxed script runners,
canned rewards, or compatibility shims that pretend missing features exist.

Unavailable features must be absent from capability discovery. Explicit use of
an unavailable feature must fail closed with `UNAVAILABLE_CAPABILITY`.

## Stable Error Codes

- `BAD_SCHEMA`
- `UNSUPPORTED_TOOL`
- `UNAVAILABLE_CAPABILITY`
- `BUDGET_EXCEEDED`
- `ADDRESS_NOT_DISCLOSED`
- `ILLEGAL_COMMAND`
- `QUEUE_FULL`
- `SANDBOX_VIOLATION`
- `SCRIPT_TIMEOUT`
- `PROFILE_REJECTED`
- `INTERNAL_SIMULATOR_ERROR`

## Local Gate

The local phase 0 gate is:

```sh
python3 -B scripts/verify_phase0.py
python3 -B -m unittest discover -s tests
```
