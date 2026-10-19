"""APP 日志目录选择：默认值、持久化、越界与损坏状态。"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labmemory import logdirs                                   # noqa: E402


class TestLogDirs(unittest.TestCase):
    def setUp(self):
        self.base = ROOT / "tests" / "_tmp" / f"logdirs_{id(self)}"
        shutil.rmtree(self.base, ignore_errors=True)
        self.logs = self.base / "logs"
        (self.logs / "test1").mkdir(parents=True)
        (self.logs / "test2").mkdir()
        (self.logs / ".hidden").mkdir()
        (self.logs / "readme.txt").write_text("x", encoding="utf-8")
        self.state = self.base / ".labmemory" / "ui.json"

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_lists_only_visible_directories(self):
        self.assertEqual(logdirs.names(self.logs), ["test1", "test2"])

    def test_defaults_to_test1(self):
        self.assertEqual(logdirs.current(self.logs, self.state).name, "test1")

    def test_selection_is_persisted(self):
        chosen = logdirs.select(self.logs, self.state, "test2")
        self.assertEqual(chosen.name, "test2")
        self.assertEqual(logdirs.load_selected(self.state), "test2")
        self.assertEqual(logdirs.current(self.logs, self.state).name, "test2")

    def test_missing_saved_selection_falls_back(self):
        logdirs.save_selected(self.state, "test1")
        shutil.rmtree(self.logs / "test1")
        self.assertEqual(logdirs.current(self.logs, self.state).name, "test2")

    def test_path_traversal_and_unknown_name_are_rejected(self):
        for name in ("../outside", "..\\outside", "nope", ""):
            with self.assertRaises(logdirs.LogDirError):
                logdirs.select(self.logs, self.state, name)

    def test_corrupt_state_does_not_break_startup(self):
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.state.write_text("{not json", encoding="utf-8")
        self.assertIsNone(logdirs.load_selected(self.state))
        self.assertEqual(logdirs.current(self.logs, self.state).name, "test1")

    def test_empty_logs_root_is_reported(self):
        empty = self.base / "empty"
        empty.mkdir()
        with self.assertRaises(logdirs.LogDirError):
            logdirs.current(empty, self.state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
