"""周报按当前日志文件夹取卡片，并保存到独立文件。"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labmemory import capture, report                         # noqa: E402
from labmemory.store import Store                             # noqa: E402


class TestWeeklyScope(unittest.TestCase):
    def setUp(self):
        self.base = ROOT / "tests" / "_tmp" / f"weekly_scope_{id(self)}"
        shutil.rmtree(self.base, ignore_errors=True)
        self.logs = self.base / "logs"
        self.one = self.logs / "test1"
        self.two = self.logs / "test2"
        self.one.mkdir(parents=True)
        self.two.mkdir()
        self.store = Store(self.base / "test.db")
        for root, exp, run in ((self.one, "exp-one", "run1"),
                               (self.two, "exp-two", "run2")):
            self.store.save_card({
                "exp_name": exp,
                "run_id": run,
                "goal": "测试周报隔离",
                "files": [str(root / f"{exp}.log")],
                "metrics": {"loss": {"first": 1.0, "last": 0.5,
                                     "best": 0.5, "trend": "down", "n": 10}},
            })

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_recent_for_root_is_isolated(self):
        one = self.store.recent_for_root(self.one, days=3650)
        two = self.store.recent_for_root(self.two, days=3650)
        self.assertEqual([c["exp_name"] for c in one], ["exp-one"])
        self.assertEqual([c["exp_name"] for c in two], ["exp-two"])

    def test_facets_and_run_filter(self):
        facets = self.store.facets()
        self.assertIn("exp-one", facets["exp_name"])
        self.assertIn("run1", facets["run_id"])
        self.assertIn("active", facets["status"])
        hits = self.store.search("", {"run_id": "run2"}, 10)
        self.assertEqual([c["exp_name"] for c in hits], ["exp-two"])

    def test_card_browser_is_scoped_to_root(self):
        one = self.store.search("", {}, 10, root=self.one)
        two = self.store.search("", {}, 10, root=self.two)
        self.assertEqual([c["exp_name"] for c in one], ["exp-one"])
        self.assertEqual([c["exp_name"] for c in two], ["exp-two"])
        self.assertEqual(self.store.search("loss", {}, 10, root=self.one)[0]["exp_name"],
                         "exp-one")

        one_facets = self.store.facets(root=self.one)
        two_facets = self.store.facets(root=self.two)
        self.assertEqual(one_facets["exp_name"], ["exp-one"])
        self.assertEqual(one_facets["run_id"], ["run1"])
        self.assertEqual(two_facets["exp_name"], ["exp-two"])
        self.assertEqual(two_facets["run_id"], ["run2"])

    def test_report_includes_scope_and_saves_file(self):
        cards = self.store.recent_for_root(self.one, days=3650)
        md = report.build_weekly(cards, days=7, scope="`test1` · 测试目录")
        self.assertIn("数据来源：`test1`", md)
        self.assertIn("exp-one", md)
        self.assertNotIn("exp-two", md)

        out = report.save_weekly(md, self.one / "weekly")
        self.assertTrue(out.exists())
        self.assertEqual(out, self.one / "weekly" / "weekly.md")
        self.assertEqual(out.read_text(encoding="utf-8"), md)

    def test_weekly_report_dir_is_not_ingested_as_data(self):
        self.assertFalse(capture.tracked(Path("weekly") / "weekly.md"))
        self.assertFalse(capture.tracked_dir(Path("weekly")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
