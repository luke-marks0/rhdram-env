# Phase 3: Empirical DDR4 Read-Disturbance Profile

Phase 3 admits the empirical profile pipeline and the first DDR4 read-disturbance
profile, `ddr4_vts25_v1`. It fits from real-chip measurements only. There is no
fabricated calibration fallback: a missing, unadmitted, or hash-mismatched source
fails closed.

This phase produces a signed, held-out-validated profile package. It does **not**
admit the disturbance engine, tasks, reward, sandbox, SDK, or mitigations; those
remain unavailable until their own gates pass.

## Source admission

The `ddr4_vts25` source in `SOURCE_MANIFEST.yaml` is now `admission: admitted` with a
resolved commit, license note, retrieval date, and content hash. The hash anchors
`profile_builder/sources/ddr4_vts25.sha256sums`, which lists the SHA-256 of every one
of the 48 vendored read-disturbance CSVs. The upstream artifact is git-ignored like
the other `third_party` sources; `scripts/fetch_phase3_sources.py` reproduces it from
the pinned commit and verifies every file hash.

- Upstream: <https://github.com/CMU-SAFARI/ReadDisturbanceVTS25> @ `5d734309457cc8a4ea3b1ec36b93932925548bac`
- Paper: Luo et al., VTS 2025, <https://arxiv.org/abs/2503.16749>
- Data-use: no SPDX license file is declared upstream; use is conditioned on citing
  the paper. Redistribution terms should be confirmed before any public release.

## Pipeline

`profile_builder/` owns the stages, each of which fails closed:

1. **manifest** — verify the source is admitted and the on-disk data matches the
   committed hashes.
2. **ingest** — parse the four measurement types (`rd_hcf`, `rd_ber`, `rd_rp`,
   `ds_ber_sweep`) into typed records; map data patterns to bitflip directions and
   aggressor types to single/double exposure classes.
3. **canonicalize** — sort into stable tables with a deterministic content digest;
   canonical record counts must reproduce the raw source row counts exactly.
4. **fit** — per family and stratum, fit a lognormal HC-first model with a two-level
   variance decomposition, plus the single/double exposure ratio, bitflip-direction
   dependence, a RowPress threshold reduction, a BER-vs-HC multiplicity table, and the
   temperature domain.
5. **validate** — score the fit on held-out units it never saw.
6. **package** — serialize canonical bytes, sign with Ed25519, and write the model
   card, validation report, and source trace.

## Fitted model

For each family (the upstream filename prefix: `axmicr`, `hisasa`, `hyhy`, `sasa`) and
stratum (`single`/`double` x `all_ones`/`all_zeros`), the threshold model is:

    log N_threshold(row) = mu + module_offset + row_eps
    module_offset ~ Normal(0, sigma_between_chip)   # module-to-module variability
    row_eps       ~ Normal(0, sigma_within_chip)    # row-to-row variability

This lets a consumer sample a persistent per-module offset at episode start and a
per-row threshold, matching the spec's latent-vulnerability requirement. The
bitflip direction is taken from the data pattern; the dominant direction is recorded
per family and flagged `biased` only when the two patterns' median HC-first differ by
at least the configured factor (so a directionally symmetric family is not over-claimed).

## Held-out validation

Held-out units are reserved two ways and the fit never sees either:

- **Held-out chips** (cross-module generalization): one chip per multi-chip family
  (`hisasa03`, `hyhy1e`, `sasa29`) is reserved entirely.
- **Held-out rows** (within-module generalization): a deterministic ~25% of victim
  rows in every training chip, selected by hashing `"<chip>:<row>"`.

Validation is two-tier, because distribution shape legitimately shifts between modules:

- **Within-module** (unseen rows of training chips): strict Kolmogorov-Smirnov and
  tight median-ratio gates.
- **Cross-module** (unseen chips): central-tendency (looser median-ratio) and
  structural gates (single > double ordering, dominant direction for biased families);
  cross-module KS is reported as a diagnostic.

If any gated check fails, the build fails closed and the profile is not admitted.

## Supported domain and limitations

- Standard: DDR4; characterization temperature: 50 °C only — out-of-domain
  temperatures are rejected (no extrapolation).
- Data patterns: uniform all-ones and all-zeros only.
- Retention failure is a distinct mechanism and is out of scope for this v1 profile.
- The profile is sampled from fitted empirical distributions; it is not an exact
  per-chip replay.
- The package is signed with the repository development trust root. Production use
  requires an externally managed signing key, a re-sign, and a published public key.
  Tampering with the signed bytes raises `PROFILE_REJECTED`.

## Local gate

```sh
python3 -B scripts/fetch_phase3_sources.py     # populate + hash-verify source data
python3 -B -m profile_builder.package.build     # build + sign the profile package
python3 -B scripts/verify_phase3.py             # phase 0 gate + admission criteria
python3 -B -m unittest tests.test_profile_package tests.test_profile_fit
```

## Test-plan coverage

`tests/test_profile_package.py` and `tests/test_profile_fit.py` cover TEST_PLAN
section D: P1 (manifest/source trace), P2 (ingestion reproduces counts), P3 (held-out
never used in fitting; train/hold-out disjoint), P4 (distribution gates pass and have
teeth), P5 (model card completeness), and P6 (tampered profile -> `PROFILE_REJECTED`),
plus replay determinism. The fit/ingest tests skip when the source data has not been
fetched; `scripts/verify_phase3.py` is the full admission gate.
