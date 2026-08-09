# Test plan

The suite should be thorough but organized around release gates. Prefer semantic tests over giant schemas.

## A. Contract and schema tests

| ID | Test | Required result |
|---|---|---|
| C1 | Validate action, observation, and task examples against compact schemas. | All examples pass; malformed examples fail. |
| C2 | Unknown tool or malformed action. | `BAD_SCHEMA` or `UNSUPPORTED_TOOL`; no state mutation. |
| C3 | Capability discovery for unavailable mitigation/profile. | Capability absent; explicit request returns `UNAVAILABLE_CAPABILITY`. |
| C4 | Stable error-code compatibility. | Error code set is versioned and documented. |
| C5 | Observation projection per disclosure level. | Hidden physical fields never appear. |

## B. Ramulator and command semantics

| ID | Test | Required result |
|---|---|---|
| R1 | Build pinned Ramulator 2.1 from source. | Reproducible build; version and commit recorded. |
| R2 | External/frontend read/write completion path. | Requests complete through real Ramulator. |
| R3 | Direct command legality. | Illegal timings/addresses rejected without disturbance updates. |
| R4 | RD/WR/WAIT sequence. | Issued event trace (incl. the controller's ACT/PRE) matches Ramulator state and cycles. |
| R5 | Refresh events. | Refresh timing and decoded coverage visible to extension. |
| R6 | Queue full/backpressure. | Stable `QUEUE_FULL`; no hidden info leak. |
| R7 | Replay determinism. | Same manifest+seed yields identical public trace and reward. |

## C. Functional memory and disturbance

| ID | Test | Required result |
|---|---|---|
| D1 | Sparse memory initialization. | Reads match seeded initial state. |
| D2 | Writes and masks. | Reads reflect writes; write restores affected modeled cells. |
| D3 | Exposure accounting uses issued events only. | Rejected commands do not change exposure. |
| D4 | Known no-flip control. | Below-threshold exposure produces no flip across fixed seeds. |
| D5 | Known flip fixture. | Admitted profile can produce a real simulated flip under controlled exposure. |
| D6 | Single/double-sided distinction. | Exposure differs according to profile and adjacency. |
| D7 | RowPress dwell. | Open-row time affects only profiles that support RowPress. |
| D8 | Refresh/decay interaction. | Refresh/RFM changes exposure/restoration as specified. |
| D9 | Temperature/domain bounds. | Unsupported temperatures rejected; supported temperatures alter model as validated. |
| D10 | Persistence. | Flips persist until overwritten or restored by modeled event. |

## D. Empirical profile validation

| ID | Test | Required result |
|---|---|---|
| P1 | Source manifest verifier. | URLs, commits, hashes, and licenses present. |
| P2 | VTS25 ingestion. | Canonical tables reproduce source counts within parser tolerances. |
| P3 | Train/held-out split. | Held-out units are never used in fitting. |
| P4 | Distribution fit checks. | Activation thresholds, directions, multiplicity, and variability pass gates. |
| P5 | Model card completeness. | Supported domain, limitations, validation stats, and source trace included. |
| P6 | Profile signature. | Tampered profile rejected with `PROFILE_REJECTED`. |

## E. OpenEnv, tasks, and reward

| ID | Test | Required result |
|---|---|---|
| E1 | `reset/step/state` lifecycle. | OpenEnv semantics preserved. |
| E2 | Direct control-path lockout. | Policy cannot bypass budgets/reward/termination. |
| E3 | Known-target task success. | Trusted reward becomes 1 only after target condition. |
| E4 | Finish-before-success. | Terminates with reward 0 and public failure summary. |
| E5 | Budget exhaustion. | Truncates episode and prevents further mutation. |
| E6 | Hidden target non-leakage. | Logs/errors/timing summaries do not reveal hidden coordinates. |
| E7 | Reference policies. | Sanity baselines pass admitted easy tasks and fail controls. |

## F. Sandbox and security

| ID | Test | Required result |
|---|---|---|
| S1 | Script broker equivalence. | Script-generated calls match direct-tool semantics. |
| S2 | Filesystem isolation. | No host mounts except explicit read-only runtime assets. |
| S3 | Device/API denial. | `/dev/mem`, `/proc/pagemap`, KVM, GPU/RDMA, huge pages denied. |
| S4 | Network denial. | No outbound/inbound network from policy sandbox. |
| S5 | Resource limits. | CPU, memory, process, file, stdout, and wall-time limits enforced. |
| S6 | Sandbox attestation. | Unapproved runtime causes environment startup failure, not fallback. |
| S7 | Malicious script corpus. | Escape attempts fail without simulator-state corruption. |

## G. Mitigations and release

| ID | Test | Required result |
|---|---|---|
| M1 | `none` baseline. | No mitigation protection beyond baseline refresh. |
| M2 | `oracle` mitigation. | Target-row protection fires when expected. |
| M3 | Per-mitigation conformance. | Each admitted mitigation has paper-to-code and differential traces. |
| M4 | Unimplemented mitigation. | Fails closed; not silently equivalent to `none`. |
| L1 | Long-run determinism/concurrency. | Parallel episodes cannot share hidden state or RNG streams. |
| L2 | Performance regression. | Throughput meets release target without semantic drift. |
| L3 | No-mock scan. | CI rejects mock/fallback symbols and forbidden runtime paths. |
| L4 | Release bundle. | SBOM, source manifest, profile cards, schemas, docs, and checksums emitted. |
