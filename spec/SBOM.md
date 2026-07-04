# Release SBOM

This software bill of materials covers the P20 release-qualification surface.
Source pins are authoritative in `SOURCE_MANIFEST.yaml`; this document records
the release bundle members checked by `scripts/verify_release.py`.

## Source Components

| Component | Role | Pin | Admission |
|---|---|---|---|
| Ramulator 2 | Simulator | `38c51d40a976c6b07fbc09de869a7e08dc187d29` | admitted |
| OpenEnv | Environment protocol and HTTP transport | `7449c5dfe375c4c6e6f0827826925a46efd9249f` | admitted |
| ReadDisturbanceVTS25 | DDR4 empirical profile source | `5d734309457cc8a4ea3b1ec36b93932925548bac` | admitted |
| HBM Read Disturbance | Optional HBM2 profile source | `bc9d600e03efbe38c740ba015398510ca9bd1a60` | deferred |

## Runtime Components

| Path | Purpose |
|---|---|
| `rowhammer_env/` | OpenEnv-compatible RowHammer environment, tools, disturbance model, sandbox, HTTP client/server, and evaluation harness |
| `cpp/simulator_service/` | Ramulator worker and smoke binaries |
| `cpp/ramulator_extensions/` | Issued-event recorder plugin consumed by the worker |
| `sdk/rh_sdk/` | Script sandbox broker SDK |
| `profile_builder/` | Source ingestion, profile fitting, signing, and validation |
| `profiles/ddr4_vts25_v1/` | Signed admitted DDR4 read-disturbance profile package |

## Release Gates

The P20 gate runs all admitted phase gates, the full unit-test suite with zero
required skips, deterministic replay for fixed seeds, no-mock symbol scanning
over executable roots, source-pin checks, and tracked-file hygiene.
