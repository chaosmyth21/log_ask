"""验收器本身的测试。

上一轮真模型跑出来的那次，验收器自己有两个 bug（路径比较、假 PASS），
所以把它也纳入回归：用假卡片直接调四个 check_*，断言 PASS/WARN/FAIL 判定正确。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labmemory import parsing  # noqa: E402

_spec = importlib.util.spec_from_file_location("acceptance", ROOT / "scripts" / "acceptance.py")
acceptance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acceptance)


def run(checks) -> acceptance.Report:
    rep = acceptance.Report()
    with contextlib.redirect_stdout(io.StringIO()):
        checks(rep)
    return rep


def status_of(rep: acceptance.Report) -> str:
    return rep.rows[-1][1]


TMP_ROOT = ROOT / "tests" / "_tmp"


def _new_tmp() -> Path:
    """仓库内造临时目录：系统 temp 在受限令牌下写不进去（mkdtemp 的权限坑）。"""
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    for i in range(10_000):
        d = TMP_ROOT / f"acc{i:03d}"
        if not d.exists():
            d.mkdir()
            return d
    raise RuntimeError("tests/_tmp 下目录太多")


class TestCoverage(unittest.TestCase):
    def setUp(self):
        self.root = _new_tmp()
        (self.root / "logs").mkdir()
        self.f1 = self.root / "logs" / "a.log"
        self.f2 = self.root / "logs" / "b.log"
        for p in (self.f1, self.f2):
            p.write_text("step 1 loss=1.0\n", encoding="utf-8")

    def test_relative_paths_are_accepted(self):
        """模型给相对路径时，覆盖检查不能误判成"漏文件"（上一轮就栽在这）。"""
        cards = [
            {"id": "c-1", "files": ["logs/a.log"]},
            {"id": "c-2", "files": [str(self.f2)]},          # 混合绝对路径
        ]
        rep = run(lambda r: acceptance.check_coverage(cards, [self.f1, self.f2], self.root, r))
        self.assertEqual(status_of(rep), acceptance.PASS, rep.rows)

    def test_missing_file_fails(self):
        cards = [{"id": "c-1", "files": ["logs/a.log"]}]
        rep = run(lambda r: acceptance.check_coverage(cards, [self.f1, self.f2], self.root, r))
        self.assertEqual(status_of(rep), acceptance.FAIL)

    def test_too_many_cards_warns(self):
        cards = [{"id": f"c-{i}", "files": [str(self.f1)]} for i in range(5)]
        rep = run(lambda r: acceptance.check_coverage(cards, [self.f1], self.root, r))
        self.assertEqual(status_of(rep), acceptance.WARN)


class TestNumbers(unittest.TestCase):
    def setUp(self):
        self.root = _new_tmp()
        self.f = self.root / "run.log"
        self.f.write_text("step 1 loss=1.00\nstep 2 loss=0.50\nstep 3 loss=0.25\n",
                          encoding="utf-8")
        self.truth = parsing.series_stats("loss", parsing.iter_metrics(self.f)["loss"])

    def _card(self, metrics):
        return [{"id": "c-1", "files": [str(self.f)], "metrics": metrics}]

    def test_exact_numbers_pass(self):
        m = {"loss": {"first": self.truth["first"], "last": self.truth["last"],
                      "best": self.truth["best"], "n": self.truth["n"]}}
        rep = run(lambda r: acceptance.check_numbers(self._card(m), r, self.root))
        self.assertEqual(status_of(rep), acceptance.PASS)

    def test_wrong_numbers_fail(self):
        m = {"loss": {"first": 9.9, "last": 9.9, "best": 9.9, "n": 3}}
        rep = run(lambda r: acceptance.check_numbers(self._card(m), r, self.root))
        self.assertEqual(status_of(rep), acceptance.FAIL)

    def test_no_comparable_metric_is_warn_not_pass(self):
        """一个指标都比不上时必须 WARN —— 这正是上一版"假 PASS"的回归。"""
        m = {"somethingelse": {"first": 1, "last": 1, "best": 1, "n": 1}}
        rep = run(lambda r: acceptance.check_numbers(self._card(m), r, self.root))
        self.assertEqual(status_of(rep), acceptance.WARN, rep.rows)

    def test_bare_number_metric_is_warn(self):
        rep = run(lambda r: acceptance.check_numbers(self._card({"loss": 0.25}), r, self.root))
        self.assertEqual(status_of(rep), acceptance.WARN)


class TestEvidence(unittest.TestCase):
    def setUp(self):
        self.root = _new_tmp()
        self.f = self.root / "run.log"
        self.f.write_text("\n".join(f"line {i}" for i in range(10)) + "\n", encoding="utf-8")

    def test_evidence_with_valid_line_passes(self):
        cards = [{"id": "c-1", "files": [str(self.f)],
                  "observations": [{"kind": "crash", "text": "x", "evidence": "run.log:3"}]}]
        rep = run(lambda r: acceptance.check_evidence(cards, r, self.root))
        self.assertEqual(status_of(rep), acceptance.PASS)

    def test_line_beyond_file_fails(self):
        cards = [{"id": "c-1", "files": [str(self.f)],
                  "observations": [{"kind": "crash", "text": "x", "evidence": "run.log:999"}]}]
        rep = run(lambda r: acceptance.check_evidence(cards, r, self.root))
        self.assertEqual(status_of(rep), acceptance.FAIL)

    def test_missing_evidence_fails(self):
        cards = [{"id": "c-1", "files": [str(self.f)],
                  "observations": [{"kind": "crash", "text": "x", "evidence": ""}]}]
        rep = run(lambda r: acceptance.check_evidence(cards, r, self.root))
        self.assertEqual(status_of(rep), acceptance.FAIL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
