from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Phase0Tests(unittest.TestCase):
    def test_phase0_gate_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/verify_phase0.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_old_hidden_project_files_are_absent(self) -> None:
        for name in [".github", ".pytest_cache"]:
            self.assertFalse((ROOT / name).exists(), name)

    def test_local_artifacts_are_ignored(self) -> None:
        text = (ROOT / ".gitignore").read_text()
        self.assertIn("build/", text)
        self.assertIn("third_party/", text)


if __name__ == "__main__":
    unittest.main()
