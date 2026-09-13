#!/usr/bin/env python3
"""Required direct-tool PoC gate. Missing worker, dependencies, or sockets fail."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from rowhammer_env.observability.experiment import provenance

# Explicit selection keeps optional sandbox/mitigation/standard/restructure gates out.
TESTS = (
    "test_poc", "test_activation_budget", "test_issue_expansion", "test_probe_signal",
    "test_discovery_geometry", "test_secret_mapping", "test_probe_shaping",
    "test_multiturn_rollout", "test_profile_package", "test_profile_fit",
    "test_phase11", "test_phase12.AddressMapperTests", "test_phase12.DisclosureTests",
    "test_phase12.HandleTableTests", "test_phase12.AddressResolverTests",
    "test_phase14.KnownFlipControlTests", "test_phase14.SingleVsDoubleTests",
    "test_phase14.RefreshDecayTests", "test_phase14.TemperatureDomainTests",
    "test_phase14.VictimAnchorTests", "test_phase14.CommandVocabularyTests",
    "test_reference_policy.ReferenceProbeIntegrationTests.test_reference_solves_easy_and_medium_within_budget",
    "test_reference_policy.ReferenceProbeIntegrationTests.test_timing_blind_control_fails_where_reference_succeeds",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=pathlib.Path, default=ROOT / "runs/poc_gate.json")
    args = ap.parse_args()
    for name in ("ramulator_worker", "p2_external_ddr4.yaml"):
        if not (ROOT / "build/phase2" / name).is_file():
            raise SystemExit("Build the native worker first: python scripts/build_phase2.py")
    suite = unittest.defaultTestLoader.loadTestsFromNames(TESTS)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and not result.skipped
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": passed, "tests": result.testsRun,
        "failures": [(str(t), e) for t, e in result.failures], "errors": [(str(t), e) for t, e in result.errors],
        "skipped": [(str(t), e) for t, e in result.skipped], "provenance": provenance(),
        "gpu_training_validated": False}, indent=2) + "\n")
    print(f"PoC {'PASS' if passed else 'FAIL'}: {result.testsRun} tests, {len(result.skipped)} skips. Report: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
