"""Profile package tests that run against committed artifacts (no source data needed).

Covers TEST_PLAN P1 (manifest/source trace), P5 (model card completeness), and
P6 (signature + tamper -> PROFILE_REJECTED), plus canonical-JSON determinism.
"""

from __future__ import annotations

import json
import os
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from profile_builder.errors import ProfileRejected
from profile_builder.manifest import load_source_pin
from profile_builder.package import canonical_bytes, canonicalize
from profile_builder.package.build import read_package, verify_package
from profile_builder.paths import OUT_DIR

CARD = OUT_DIR / "model_card.md"
VALIDATION = OUT_DIR / "validation_report.json"
SOURCES = OUT_DIR / "SOURCES.json"


class ManifestTests(unittest.TestCase):
    def test_source_admitted_with_resolved_provenance(self) -> None:
        pin = load_source_pin(verify_data=False)
        self.assertEqual(pin.source_id, "ddr4_vts25")
        self.assertTrue(pin.commit and pin.commit != "pending")
        self.assertTrue(pin.url.startswith("https://github.com/CMU-SAFARI/ReadDisturbanceVTS25"))
        self.assertNotIn("pending", (pin.license, pin.sha256, pin.retrieved_utc))
        self.assertEqual(len(pin.files), 48)

    def test_sources_json_complete(self) -> None:
        src = json.loads(SOURCES.read_text())
        for field in ("url", "commit", "license", "retrieved_utc", "data_sha256", "files"):
            self.assertIn(field, src)
        self.assertEqual(len(src["files"]), 48)
        self.assertTrue(all(len(f["sha256"]) == 64 for f in src["files"]))


class SignatureTests(unittest.TestCase):
    def test_committed_package_verifies(self) -> None:
        profile = verify_package()
        self.assertEqual(profile["profile_id"], "ddr4_vts25_v1")
        self.assertTrue(profile["validation"]["passed"])

    def test_tampered_payload_is_rejected(self) -> None:
        from profile_builder.package import verify

        payload, signature, pub = read_package()
        tampered = bytearray(payload)
        tampered[len(tampered) // 2] ^= 0x01
        with self.assertRaises(ProfileRejected) as ctx:
            verify(bytes(tampered), signature, pub)
        self.assertEqual(ctx.exception.code, "PROFILE_REJECTED")

    def test_wrong_key_is_rejected(self) -> None:
        from profile_builder.package import verify

        payload, signature, _ = read_package()
        with self.assertRaises(ProfileRejected) as ctx:
            verify(payload, signature, os.urandom(32).hex())
        self.assertEqual(ctx.exception.code, "PROFILE_REJECTED")


class ModelCardTests(unittest.TestCase):
    def test_card_has_required_sections(self) -> None:
        text = CARD.read_text()
        for section in ("## Source trace", "## Supported domain", "## Held-out validation", "## Limitations"):
            self.assertIn(section, text)

    def test_card_states_domain_and_provenance(self) -> None:
        text = CARD.read_text()
        self.assertIn("5d734309457cc8a4ea3b1ec36b93932925548bac", text)  # pinned commit
        self.assertIn("sampled-not-exact-replay", text)  # honest labeling
        self.assertIn("out-of-domain temperatures are rejected", text)


class ValidationArtifactTests(unittest.TestCase):
    def test_committed_validation_passed_and_leak_free(self) -> None:
        report = json.loads(VALIDATION.read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["n_failed"], 0)
        self.assertTrue(report["train_chip_holdout_chip_disjoint"])


class CanonicalJsonTests(unittest.TestCase):
    def test_bytes_are_stable_and_sorted(self) -> None:
        a = canonical_bytes({"b": 1.123456789, "a": [3, 2, 1]})
        b = canonical_bytes({"a": [3, 2, 1], "b": 1.123456789})
        self.assertEqual(a, b)
        self.assertEqual(a, b'{"a":[3,2,1],"b":1.123457}')

    def test_non_finite_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            canonicalize(float("inf"))
        with self.assertRaises(ValueError):
            canonicalize(float("nan"))


if __name__ == "__main__":
    unittest.main()
