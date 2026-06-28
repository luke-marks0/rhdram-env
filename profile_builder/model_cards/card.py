"""Render a profile model card (TEST_PLAN P5).

The card states the supported domain, limitations, held-out validation statistics,
and the full source trace, so a consumer can decide whether the profile applies.
"""

from __future__ import annotations


def render_model_card(profile: dict, report: dict) -> str:
    src = profile["source"]
    domain = profile["domain"]
    fit = profile["fit"]
    lines: list[str] = []
    add = lines.append

    add(f"# Profile model card: {profile['profile_id']}")
    add("")
    add(f"- Standard: {profile['standard']}")
    add(f"- Labeling: {profile['labeling']}")
    add(f"- Fit method: {fit['method']}")
    add(f"- Sampling model: {fit['sampling_model']}")
    add("")

    add("## Source trace")
    add("")
    add(f"- Repository: {src['url']}")
    add(f"- Pinned commit: `{src['commit']}`")
    add(f"- Paper: {src['paper']}")
    add(f"- Retrieved (UTC): {src['retrieved_utc']}")
    add(f"- License / data-use: {src['license']}")
    add(f"- Source data anchor (sha256): `{src['data_sha256']}`")
    add(f"- Canonical table digest (sha256): `{profile['build_basis']['canonical_table_sha256']}`")
    add(f"- Source files: {src['n_files']}")
    add("")

    add("## Supported domain")
    add("")
    add(f"- Temperatures (°C): {domain['supported_temperatures_celsius']}")
    add(f"- Temperature extrapolation: {domain['temperature_extrapolation']}")
    add(f"- Timing: {domain['timing']} (extrapolation: {domain['timing_extrapolation']})")
    add(f"- Aggressor classes: {domain['aggressor_classes']}")
    add(f"- RowPress: {domain['rowpress']}")
    add(f"- Data patterns: {domain['data_patterns']}")
    add("")

    add("## Families")
    add("")
    add("| Family | Train chips | Held-out chips | single/double (all_ones) | Dominant direction |")
    add("|---|---|---|---|---|")
    for family, fam in fit["families"].items():
        ratio = fam["single_to_double_ratio"].get("all_ones")
        ratio_s = f"{ratio:.2f}" if ratio is not None else "n/a"
        add(
            f"| {family} | {', '.join(fam['chips_train']) or '—'} | "
            f"{', '.join(fam['chips_holdout']) or '—'} | {ratio_s} | "
            f"{fam['direction']['dominant_flip_direction']} |"
        )
    add("")

    g = report["gates"]
    add("## Held-out validation")
    add("")
    add(f"- Held-out chips: {report['holdout_chips']}")
    add(f"- Train/hold-out chip disjoint: {report['train_chip_holdout_chip_disjoint']}")
    add(f"- Within-module gates: KS D ≤ {g['row_ks_gate']}, median within factor {g['row_median_ratio_gate']}")
    add(f"- Cross-module gate: median within factor {g['chip_median_ratio_gate']} (KS reported as diagnostic)")
    add(f"- Gated checks: {report['n_gated']} of {report['n_checks']} ({report['n_failed']} failed)")
    add(f"- Result: {'PASS' if report['passed'] else 'FAIL'}")
    add("")

    add("## Limitations")
    add("")
    add(
        "- Read-disturbance characterization is at a single temperature "
        f"({domain['supported_temperatures_celsius']} °C); out-of-domain temperatures "
        "are rejected (no extrapolation)."
    )
    add("- Only uniform all-ones and all-zeros data patterns are characterized; striped/random patterns are not supported.")
    add("- Retention failure is a distinct mechanism and is out of scope for this v1 read-disturbance profile.")
    add("- The family key is the upstream filename prefix only; it does not assert a specific DRAM vendor.")
    add("- The profile is sampled from fitted empirical distributions; it is not an exact per-chip replay.")
    add("- Signed with the repository development trust root; production use requires an externally managed signing key.")
    add("")
    return "\n".join(lines)
