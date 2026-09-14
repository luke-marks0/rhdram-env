"""Environment-only checks for the narrow PoC surface."""
from __future__ import annotations

import unittest

from rowhammer_env import Phase2Action
from rowhammer_env.poc import POC_TOOLS, TASK_PATHS, PoCEnv, load_task, validate_task
from rowhammer_env.tasks.compiler import TaskConfigError


class PoCEnvironmentTests(unittest.TestCase):
    def test_scoped_task_files_are_valid(self) -> None:
        self.assertEqual(len(TASK_PATHS), 5)
        for path in TASK_PATHS:
            with self.subTest(path=path):
                validate_task(load_task(path))

    def test_out_of_scope_variants_fail_closed(self) -> None:
        for task in (
            {"family": "hidden_adjacency", "difficulty": "hard"},
            {"standard": "DDR5"},
            {"mitigation": {"name": "oracle"}},
        ):
            with self.subTest(task=task), self.assertRaises(TaskConfigError):
                validate_task(task)

    def test_policy_surface_rejects_extra_tools_before_dispatch(self) -> None:
        env = PoCEnv(task=load_task(TASK_PATHS[0]))
        try:
            for tool in ("dram.read", "dram.write", "script.run"):
                with self.subTest(tool=tool):
                    obs = env.step(Phase2Action(tool=tool, args={}))
                    self.assertEqual(obs.error["code"], "UNSUPPORTED_TOOL")
                    self.assertEqual(obs.cycle, 0)
                    self.assertFalse(obs.done)
            self.assertEqual(POC_TOOLS, ("dram.info", "dram.issue", "episode.finish"))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
