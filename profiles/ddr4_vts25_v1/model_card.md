# Profile model card: ddr4_vts25_v1

- Standard: DDR4
- Labeling: sampled-not-exact-replay
- Fit method: lognormal-mle-per-stratum
- Sampling model: log N_threshold(row) = mu + module_offset + row_eps; module_offset ~ Normal(0, sigma_between_chip); row_eps ~ Normal(0, sigma_within_chip)

## Source trace

- Repository: https://github.com/CMU-SAFARI/ReadDisturbanceVTS25
- Pinned commit: `5d734309457cc8a4ea3b1ec36b93932925548bac`
- Paper: https://arxiv.org/abs/2503.16749
- Retrieved (UTC): 2026-06-28T06:18:22Z
- License / data-use: no SPDX license file upstream at pinned commit; CMU-SAFARI public research artifact; use conditioned on citing luo2025revisiting (see third_party/ReadDisturbanceVTS25/PROVENANCE.md)
- Source data anchor (sha256): `4d25bffd34f17ede7d19431278e1f82f6eb5ad84aaca6555f58bd93ee164a8f7`
- Canonical table digest (sha256): `3ea7bd1be13dfc8a3dd2eb4a3d9bcbbc59457ef9d8fb71e9166d0997a4f5ed5b`
- Source files: 48

## Supported domain

- Temperatures (°C): [50]
- Temperature extrapolation: False
- Timing: ddr4_nominal_dram_bender (extrapolation: False)
- Aggressor classes: ['single', 'double']
- RowPress: True
- Data patterns: ['all_ones', 'all_zeros']

## Families

| Family | Train chips | Held-out chips | single/double (all_ones) | Dominant direction |
|---|---|---|---|---|
| axmicr | axmicr02 | — | 5.60 | 1->0 |
| hisasa | hisasa00, hisasa01, hisasa02 | hisasa03 | 7.00 | 0->1 |
| hyhy | hyhy03, hyhy0c, hyhy13 | hyhy1e | 7.27 | 0->1 |
| sasa | sasa05, sasa23 | sasa29 | 6.32 | 0->1 |

## Held-out validation

- Held-out chips: ['hisasa03', 'hyhy1e', 'sasa29']
- Train/hold-out chip disjoint: True
- Within-module gates: KS D ≤ 0.3, median within factor 1.5
- Cross-module gate: median within factor 2.5 (KS reported as diagnostic)
- Gated checks: 64 of 77 (0 failed)
- Result: PASS

## Limitations

- Read-disturbance characterization is at a single temperature ([50] °C); out-of-domain temperatures are rejected (no extrapolation).
- Only uniform all-ones and all-zeros data patterns are characterized; striped/random patterns are not supported.
- Retention failure is a distinct mechanism and is out of scope for this v1 read-disturbance profile.
- The family key is the upstream filename prefix only; it does not assert a specific DRAM vendor.
- The profile is sampled from fitted empirical distributions; it is not an exact per-chip replay.
- Signed with the repository development trust root; production use requires an externally managed signing key.
