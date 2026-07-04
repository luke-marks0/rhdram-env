from __future__ import annotations

import importlib.util
import pathlib
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_verify_release():
    spec = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts" / "verify_release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase20ReleaseGateTests(unittest.TestCase):
    def test_release_gate_runs_full_admitted_phase_matrix(self) -> None:
        verify_release = _load_verify_release()
        self.assertEqual(verify_release.PHASE_GATE_NUMBERS, tuple(list(range(0, 10)) + list(range(11, 20))))

    def test_symbol_scan_exempts_only_verifier_denylist_definitions(self) -> None:
        verify_release = _load_verify_release()
        self.assertIn("scripts/verify_phase0.py", verify_release.SYMBOL_SCAN_EXEMPTIONS)
        self.assertIn("scripts/verify_release.py", verify_release.SYMBOL_SCAN_EXEMPTIONS)
        self.assertIn("scripts/verify_phase20.py", verify_release.SYMBOL_SCAN_EXEMPTIONS)

    def test_release_bundle_artifacts_are_declared(self) -> None:
        manifest = yaml.safe_load((ROOT / "SOURCE_MANIFEST.yaml").read_text())
        self.assertEqual(manifest["phase"], "P20")
        self.assertTrue((ROOT / "spec" / "SBOM.md").is_file())
        self.assertTrue((ROOT / "scripts" / "verify_phase20.py").is_file())


if __name__ == "__main__":
    unittest.main()
