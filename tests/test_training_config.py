"""Config loading/validation and the seed-split contract (torch-free)."""
from __future__ import annotations

import pathlib
import tempfile
import unittest

import yaml

from rowhammer_env.training import config as configlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/training.yaml"


def _write(tmp: pathlib.Path, doc: dict) -> str:
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path)


class ConfigTests(unittest.TestCase):
    def test_default_config_loads_and_validates(self) -> None:
        c = configlib.load_config(BASE)
        self.assertEqual(set(configlib.STAGES), set(c["curriculum"]))
        self.assertEqual(len(configlib.TASK_PATHS), 5)

    def test_unknown_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write(pathlib.Path(d), {"grpo": {"not_a_field": 1}})
            with self.assertRaises(ValueError):
                configlib.load_config(path)

    def test_overlapping_seed_splits_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write(pathlib.Path(d), {"seeds": {"train": [1, 100], "validation": [50, 60]}})
            with self.assertRaises(ValueError):
                configlib.load_config(path)

    def test_shaping_must_stay_below_success(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write(pathlib.Path(d), {"grpo": {"probe_shaping": 1.0}})
            with self.assertRaises(ValueError):
                configlib.load_config(path)

    def test_group_size_needs_at_least_two(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write(pathlib.Path(d), {"grpo": {"group_size": 1}})
            with self.assertRaises(ValueError):
                configlib.load_config(path)

    def test_seeds_helper_is_inclusive_and_disjoint(self) -> None:
        c = configlib.load_config(BASE)
        train, val, bench = (set(configlib.seeds(c, s)) for s in ("train", "validation", "benchmark"))
        self.assertEqual(min(train), c["seeds"]["train"][0])
        self.assertEqual(max(train), c["seeds"]["train"][1])
        self.assertFalse(train & val)
        self.assertFalse(train & bench)
        self.assertFalse(val & bench)

    def test_sampling_seed_is_deterministic_and_varies(self) -> None:
        self.assertEqual(configlib.sampling_seed(42, "a"), configlib.sampling_seed(42, "a"))
        self.assertNotEqual(configlib.sampling_seed(42, "a"), configlib.sampling_seed(42, "b"))
        self.assertNotEqual(configlib.sampling_seed(1, "a"), configlib.sampling_seed(2, "a"))


if __name__ == "__main__":
    unittest.main()
