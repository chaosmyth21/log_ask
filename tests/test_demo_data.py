"""demo/make_demo_data.py 的顺序布局：第一次生成到 test1，第二次到 test2。"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.make_demo_data import generate  # noqa: E402

TMP_ROOT = ROOT / "tests" / "_tmp"


def _new_tmp() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    for i in range(10_000):
        d = TMP_ROOT / f"demo{i:03d}"
        if not d.exists():
            d.mkdir()
            return d
    raise RuntimeError("临时目录太多，先清一下 tests/_tmp")


class TestDemoData(unittest.TestCase):
    def setUp(self):
        self.base = _new_tmp()
        self.target = self.base / "demo_data"

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_first_run_goes_to_test1(self):
        generate(self.target)
        self.assertTrue((self.target / "logs" / "test1").is_dir())
        self.assertFalse((self.target / "logs" / "test2").exists())
        self.assertTrue(list((self.target / "logs" / "test1").rglob("*.log")))
        self.assertTrue((self.target / "logs" / "test1" / "notes").is_dir())

    def test_second_run_goes_to_test2(self):
        generate(self.target)
        generate(self.target)
        one = {p.name for p in (self.target / "logs" / "test1").rglob("*.log")}
        two = {p.name for p in (self.target / "logs" / "test2").rglob("*.log")}
        self.assertTrue(one, "test1 应有实验日志")
        self.assertEqual(one, two, "test2 应生成同样结构的实验日志")
        self.assertTrue((self.target / "logs" / "test2" / "notes").is_dir())

    def test_third_run_raises(self):
        generate(self.target)
        generate(self.target)
        with self.assertRaises(ValueError):
            generate(self.target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
