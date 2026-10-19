"""P1 三项的回归测试：目录深度限制、失败态区分、卡片写入的事务性。"""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labmemory import llm, tools                      # noqa: E402
from labmemory.store import Store                      # noqa: E402
from labmemory.tools import Ctx, dispatch              # noqa: E402

TMP_ROOT = ROOT / "tests" / "_tmp"


def _new_tmp() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    for i in range(10_000):
        d = TMP_ROOT / f"lim{i:03d}"
        if not d.exists():
            d.mkdir()
            return d
    raise RuntimeError("tests/_tmp 下目录太多")


def _ctx(root: Path) -> Ctx:
    return Ctx(root=root, store=Store(root / "t.db"), state_file=root / ".state.json")


class TestDepthLimit(unittest.TestCase):
    """list_files 要能在越界前剪枝，并且支持下钻。"""

    def setUp(self):
        self.root = _new_tmp()
        (self.root / "a" / "b" / "c").mkdir(parents=True)
        (self.root / "a" / "g.log").write_text("step 1 loss=1.0\n", encoding="utf-8")
        (self.root / "a" / "b" / "c" / "f.log").write_text("step 1 loss=2.0\n", encoding="utf-8")
        self.ctx = _ctx(self.root)

    def test_shallow_call_prunes_and_reports(self):
        r = dispatch(self.ctx, "list_files", {"since_hours": 24, "max_depth": 2})
        names = {f["name"] for f in r["files"]}
        self.assertIn("g.log", names)
        self.assertNotIn("f.log", names)                 # 太深，被剪掉
        self.assertTrue(r.get("pruned"), "被剪枝的目录必须报出来，否则用户不知道还有东西")
        self.assertIn("a/b", [p["path"] for p in r["pruned"]])

    def test_drill_down_by_subdir(self):
        r = dispatch(self.ctx, "list_files",
                     {"since_hours": 24, "max_depth": 2, "subdir": "a/b"})
        names = {f["name"] for f in r["files"]}
        self.assertIn("f.log", names, "下钻后深度从新起点重算")
        self.assertLessEqual(r["visited"], 20, "下钻不应该遍历全树")

    def test_beyond_root_is_rejected(self):
        r = dispatch(self.ctx, "list_files", {"subdir": "../../etc"})
        self.assertFalse(r["ok"])


class TestErrorClassification(unittest.TestCase):
    """失败态要能分辨：鉴权 / 限流 / 超时 / 连接 / 请求错 / 缺依赖。"""

    @staticmethod
    def _err(name: str, status=None):
        cls = type(name, (Exception,), {"status_code": status})
        return cls(f"{name} happened")

    def test_kinds(self):
        cases = [
            ("AuthenticationError", None, "auth"),
            ("PermissionDeniedError", 403, "auth"),
            ("RateLimitError", 429, "rate_limit"),
            ("APITimeoutError", None, "timeout"),
            ("APIConnectionError", None, "connection"),
            ("BadRequestError", 400, "bad_request"),
            ("InsufficientQuotaError", 402, "quota"),
        ]
        for name, status, expect in cases:
            with self.subTest(name=name):
                self.assertEqual(llm.classify_error(self._err(name, status)), expect)

    def test_import_error_means_not_installed(self):
        self.assertEqual(llm.classify_error(ImportError("no openai")), "not_installed")

    def test_unknown_falls_back(self):
        self.assertEqual(llm.classify_error(ValueError("??")), "unknown")

    def test_wrap_adds_actionable_hint(self):
        e = llm.wrap_error(self._err("AuthenticationError"))
        self.assertIsInstance(e, llm.LLMError)
        self.assertEqual(e.kind, "auth")
        self.assertIn("api_key", e.hint)                 # 提示必须能直接照着做

    def test_agent_surfaces_kind(self):
        """Agent 要把错误类型透出去，不能只给一句 '调用失败'。"""
        from labmemory.agent import Agent

        class Boom:
            def chat(self, messages, tools):
                raise self._e()

            @staticmethod
            def _e():
                return TestErrorClassification._err("RateLimitError", 429)

        root = _new_tmp()
        ctx = _ctx(root)
        res = Agent(Boom(), ctx, "x", verbose=False).run("hi")
        self.assertFalse(res["ok"])
        self.assertEqual(res["error_kind"], "rate_limit")
        self.assertTrue(res["hint"])


class _FailOn:
    """代理 connection：命中指定 SQL 就抛错，用来验证事务回滚。"""

    def __init__(self, conn, needle: str):
        self._conn = conn
        self._needle = needle

    def execute(self, sql, *args):
        if self._needle in sql:
            raise sqlite3.OperationalError(f"injected failure: {self._needle}")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)


class TestTransaction(unittest.TestCase):
    """写卡片要原子：中间失败不能留下"有卡无索引"的半成品。"""

    def setUp(self):
        self.root = _new_tmp()
        self.store = Store(self.root / "t.db")

    def _card(self, exp: str) -> dict:
        return {"exp_name": exp, "goal": "g", "files": [],
                "setup": {"lr": 0.001}, "metrics": {"loss": {"last": 0.5, "n": 1}}}

    def test_failure_rolls_back_everything(self):
        real = self.store.conn
        self.store.conn = _FailOn(real, "INSERT INTO params")
        with self.assertRaises(sqlite3.OperationalError):
            self.store.save_card(self._card("exp-rollback"))
        self.store.conn = real

        cards = real.execute("SELECT COUNT(*) c FROM cards").fetchone()["c"]
        fts = real.execute("SELECT COUNT(*) c FROM cards_fts").fetchone()["c"]
        self.assertEqual(cards, 0, "回滚后不该留下卡片行")
        self.assertEqual(fts, 0, "回滚后不该留下索引行")

    def test_success_still_commits(self):
        self.store.save_card(self._card("exp-ok"))
        self.assertEqual(len(self.store.search("", {}, 5)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSameEvidenceUpdate(unittest.TestCase):
    """同一份证据重复保存必须原地更新，否则模型会陷入"再存一遍"的死循环。"""

    def setUp(self):
        self.root = _new_tmp()
        self.store = Store(self.root / "t.db")
        self.f = self.root / "run.log"
        self.f.write_text("step 1 loss=1.0\n", encoding="utf-8")

    def _card(self):
        return {"exp_name": "exp", "run_id": "run1", "goal": "g",
                "files": [str(self.f)], "metrics": {"loss": {"last": 0.5, "n": 1}}}

    def test_repeat_save_updates_in_place(self):
        first = self.store.save_card(self._card())
        second = self.store.save_card(self._card())
        self.assertFalse(first["_updated"])
        self.assertTrue(second["_updated"], "第二次保存应标记为原地更新")
        self.assertEqual(first["id"], second["id"], "卡片 id 必须不变")
        self.assertEqual(len(self.store.recent(days=3650)), 1, "不能新建第二张")

    def test_different_evidence_still_creates_chain(self):
        self.store.save_card(self._card())
        other = self.root / "run_retry.log"
        other.write_text("step 1 loss=0.9\n", encoding="utf-8")
        card = self._card()
        card["files"] = [str(other)]
        second = self.store.save_card(card)
        self.assertFalse(second["_updated"])
        cards = self.store.recent(days=3650)
        self.assertEqual(len(cards), 2)

    def test_chained_card_links_survive(self):
        self.store.save_card(self._card())
        other = self.root / "run_retry.log"
        other.write_text("step 1 loss=0.9\n", encoding="utf-8")
        card = self._card()
        card["files"] = [str(other)]
        self.store.save_card(card)
        active = [c for c in self.store.recent(days=3650) if c["status"] != "superseded"]
        old = [c for c in self.store.recent(days=3650) if c["status"] == "superseded"]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["links"].get("supersedes"), old[0]["id"])
        self.assertEqual(old[0]["links"].get("superseded_by"), active[0]["id"])


class TestRepeatGuard(unittest.TestCase):
    """模型连续三次用同样参数调同一个工具时，要被打断。"""

    def test_third_identical_call_is_blocked(self):
        from labmemory.agent import Agent

        class Stuck:
            """永远请求同一个 inspect_file。"""

            def __init__(self):
                self.n = 0

            def chat(self, messages, tools):
                self.n += 1
                return {"content": None, "tool_calls": [
                    {"id": f"c{self.n}", "name": "list_files", "args": {"since_hours": 24}}]}

        root = _new_tmp()
        res = Agent(Stuck(), _ctx(root), "x", max_turns=6, verbose=False).run("go")
        blocked = [t for t in res["trace"] if t["tool"] == "list_files" and not t["ok"]]
        self.assertTrue(blocked, f"重复调用没有被拦下：{res['trace']}")
        # 第 1、2 次放行，第 3 次起全部拦下：6 轮里最多 2 次真正执行
        executed = [t for t in res["trace"] if t["tool"] == "list_files" and t["ok"]]
        self.assertLessEqual(len(executed), 2, f"只应有前两次被真正执行：{res['trace']}")
        self.assertGreaterEqual(len(blocked), 3)


class TestCoverageRepair(unittest.TestCase):
    """模型漏建卡时，规则兜底必须把文件补齐，并明确标成 #补漏。"""

    def setUp(self):
        self.root = _new_tmp()
        self.ctx = _ctx(self.root)
        self.a = self.root / "a.log"
        self.b = self.root / "b.log"
        for f, v in ((self.a, 1.0), (self.b, 2.0)):
            f.write_text(f"step 1 loss={v}\n", encoding="utf-8")

    def test_repair_builds_missing_card(self):
        from labmemory.cli import repair_uncovered
        from labmemory.llm import card_from_inspect
        from labmemory import parsing

        # 模拟模型：列了两个文件、都读了，但只给 a 建卡
        dispatch(self.ctx, "list_files", {"since_hours": 24})
        for f in (self.a, self.b):
            dispatch(self.ctx, "inspect_file", {"path": str(f)})
        dispatch(self.ctx, "save_card", {"card": {
            "exp_name": "x", "goal": "g", "files": [str(self.a)]}})

        made = repair_uncovered(self.ctx, verbose=False)
        self.assertEqual(len(made), 1, f"应该只补 b：{made}")
        card = self.ctx.store.to_dict(made[0])
        self.assertIn("#补漏", card["tags"])
        self.assertEqual(card["status"], "needs_review")
        self.assertEqual(Path(card["files"][0]).name, "b.log")

    def test_repair_is_noop_when_all_covered(self):
        from labmemory.cli import repair_uncovered

        dispatch(self.ctx, "list_files", {"since_hours": 24})
        for f in (self.a, self.b):
            dispatch(self.ctx, "inspect_file", {"path": str(f)})
            dispatch(self.ctx, "save_card", {"card": {
                "exp_name": Path(f).stem, "goal": "g", "files": [str(f)]}})
        self.assertEqual(repair_uncovered(self.ctx, verbose=False), [])
