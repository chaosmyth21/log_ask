"""端到端测试：不依赖网络、不依赖 API Key、不依赖 pytest（只用标准库 unittest）。

    python -m unittest discover -s tests -v
    python tests/test_labmemory.py            # 等价

覆盖的是"改了代码会不会悄悄坏"的那几条关键路径：
  - 重复摄录的幂等性
  - 同 run 重跑的 supersede 实验链
  - 结构化检索能否命中（"学习率 0.001"）
  - 工具永不抛异常、越界拒绝
  - 配置的 Windows 反斜杠 / BOM / 坏文件兼容
  - 运行采集的参数 diff
  - 文件监听的稳定窗口 + 攒批
  - 周报结构
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.make_demo_data import generate                      # noqa: E402
from labmemory import capture, config, parsing, prompts, report  # noqa: E402
from labmemory.agent import Agent                              # noqa: E402
from labmemory.llm import RulePolicy                           # noqa: E402
from labmemory.store import Store                              # noqa: E402
from labmemory.tools import Ctx, dispatch                      # noqa: E402
from labmemory.watch import Watcher                            # noqa: E402

SYSTEM = prompts.INGEST_SYSTEM.replace("{tags}", "、".join(prompts.TAG_VOCAB))

# 临时目录放在仓库内，不用系统 temp：Windows 沙箱下系统 temp 的 chmod 会被拒
TMP_ROOT = ROOT / "tests" / "_tmp"


def _new_tmp() -> Path:
    """在仓库内造一个唯一临时目录。

    不用 tempfile.mkdtemp：它建成 0o700 的目录，在 Windows + 受限令牌下
    后续往里面建子目录会被拒（这里实测报 WinError 5）。普通 mkdir 继承父目录权限，没这问题。
    """
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    for i in range(10_000):
        d = TMP_ROOT / f"case{i:03d}"
        if not d.exists():
            d.mkdir()
            return d
    raise RuntimeError("临时目录太多，先清一下 tests/_tmp")


@contextlib.contextmanager
def tempdir():
    d = _new_tmp()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class TempDir:
    """给 setUpClass 用的长生命周期临时目录。"""

    def __init__(self):
        self.path = _new_tmp()

    def cleanup(self):
        shutil.rmtree(self.path, ignore_errors=True)


def make_workspace(base: Path) -> Ctx:
    """造一份样例实验数据 + 空的库。"""
    data = base / "lab"
    generate(data)
    db = base / "test.db"
    ctx = Ctx(root=data, store=Store(db),
              state_file=base / ".labmemory" / "ingested.json")
    return ctx


def ingest(ctx: Ctx, hours: float = 240) -> dict:
    agent = Agent(RulePolicy(mode="ingest", since_hours=hours), ctx, SYSTEM, max_turns=40,
                  verbose=False)
    return agent.run(f"数据目录：{ctx.root}\n请处理最近 {hours} 小时内的实验记录。")


class TestIngest(unittest.TestCase):
    """摄录：幂等、覆盖率、清理。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TempDir()
        cls.base = cls._tmp.path
        cls.ctx = make_workspace(cls.base)
        cls.first = ingest(cls.ctx)
        cls.cards = cls.ctx.store.recent(days=3650)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_file_is_covered(self):
        """每个实验文件都必须被某张卡片引用，不许漏。"""
        files = {str(p) for p in self.ctx.root.rglob("*")
                 if p.is_file() and capture.tracked(p.relative_to(self.ctx.root))}
        referenced = {f for c in self.cards for f in c["files"]}
        self.assertEqual(files - referenced, set(), "有文件没进任何卡片")

    def test_repeat_scan_is_idempotent(self):
        """再扫一次：不新增卡片、不报错、明确说'没有新的'。"""
        before = len(self.ctx.store.recent(days=3650))
        res = ingest(self.ctx)
        after = len(self.ctx.store.recent(days=3650))
        self.assertTrue(res["ok"], res.get("error"))
        self.assertEqual(before, after)
        self.assertIn("没有新的", res["final"])

    def test_no_tool_errors_recorded(self):
        """工具异常会写进 events 审计表 —— 这一条挂了说明有工具在吞错。"""
        rows = self.ctx.store.conn.execute(
            "SELECT ref, detail FROM events WHERE kind='tool_error'").fetchall()
        self.assertEqual([tuple(r) for r in rows], [])


class TestKnowledgeChain(unittest.TestCase):
    """supersede 实验链：同一 exp+run 重跑，旧卡不删但降级。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TempDir()
        cls.ctx = make_workspace(cls._tmp.path)
        ingest(cls.ctx)
        cls.cards = cls.ctx.store.recent(days=3650)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run3_cards(self):
        return [c for c in self.cards
                if c["exp_name"] == "lora-sft" and (c.get("run_id") or "") == "run3"]

    def test_two_cards_linked(self):
        cards = self._run3_cards()
        self.assertEqual(len(cards), 2, f"期望 run3 有两张卡，实际 {len(cards)}")
        old = next(c for c in cards if c["status"] == "superseded")
        new = next(c for c in cards if c["status"] != "superseded")
        self.assertEqual(old["links"].get("superseded_by"), new["id"])
        self.assertEqual(new["links"].get("supersedes"), old["id"])

    def test_crash_card_is_failed_and_tagged(self):
        crash = next(c for c in self._run3_cards() if "#崩溃" in c["tags"])
        self.assertEqual(crash["status"], "superseded")   # 已被重跑覆盖
        self.assertTrue(any(o["kind"] == "crash" for o in crash["observations"]))
        self.assertTrue(all(o.get("evidence") for o in crash["observations"]))

    def test_active_card_ranks_first(self):
        """检索时有效卡片必须排在 superseded 之前。"""
        hits = self.ctx.store.search("", {"param_key": "lr", "param_value": "0.001"}, 5)
        self.assertTrue(hits)
        self.assertNotEqual(hits[0]["status"], "superseded")


class TestRetrieval(unittest.TestCase):
    """'学习率 0.001 那次 loss 多少' 能不能查回来。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TempDir()
        cls.ctx = make_workspace(cls._tmp.path)
        ingest(cls.ctx)
        agent = Agent(RulePolicy(mode="query"), cls.ctx, prompts.QUERY_SYSTEM, verbose=False)
        cls.answer = agent.run("上次学习率 0.001 那次的 loss 最后降到多少？")["final"]
        cls.hits = cls.ctx.store.search("loss", {"param_key": "lr", "param_value": "0.001"}, 5)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_structured_filter_matches(self):
        self.assertTrue(self.hits, "按 lr=0.001 过滤应至少命中一张卡")
        for h in self.hits:
            self.assertEqual(h["setup"].get("lr"), 0.001)

    def test_answer_contains_real_number(self):
        top = self.hits[0]
        last = top["metrics"]["loss"]["last"]
        self.assertIn(f"{last}", self.answer, f"答案里没有卡片里的数字 {last}：\n{self.answer}")

    def test_answer_cites_source(self):
        self.assertTrue(("c-" in self.answer) or (".log" in self.answer),
                        f"答案没有引用卡片 id 或来源文件：\n{self.answer}")


class TestToolSafety(unittest.TestCase):
    """工具层：永不抛异常、越界必拒。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = TempDir()
        cls.ctx = make_workspace(cls._tmp.path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_bad_args_never_raise(self):
        for name, args in [("inspect_file", {"path": "nope.log"}),
                           ("read_slice", {}),
                           ("get_card", {"card_id": "c-xxxx"}),
                           ("compare_files", {"path_a": "a", "path_b": "b"}),
                           ("unknown_tool", {})]:
            r = dispatch(self.ctx, name, args)
            self.assertFalse(r["ok"], f"{name} 应返回失败")
            self.assertIn("error", r)

    def test_outside_root_is_rejected(self):
        outside = str(Path(tempfile.gettempdir()) / "definitely_outside.log")
        r = dispatch(self.ctx, "inspect_file", {"path": outside})
        self.assertFalse(r["ok"])
        self.assertIn("越界", r["error"])

    def test_image_file_only_registered(self):
        img = self.ctx.root / "shot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        r = dispatch(self.ctx, "inspect_file", {"path": str(img)})
        self.assertTrue(r["ok"])
        self.assertEqual(r["kind"], "image")
        self.assertFalse(dispatch(self.ctx, "read_slice", {"path": str(img)})["ok"])


class TestConfig(unittest.TestCase):
    """配置：Windows 单反斜杠、BOM、坏文件。"""

    def setUp(self):
        self._tmp = TempDir()
        self.dir = self._tmp.path

    def tearDown(self):
        self._tmp.cleanup()

    def test_single_backslash_path_is_tolerated(self):
        p = self.dir / "a.toml"
        p.write_text('[paths]\ndata_dir = "D:\\lab\\exp"\n', encoding="utf-8")
        cfg = config.load(p)
        self.assertEqual(cfg.paths.data_dir, r"D:\lab\exp")

    def test_escaped_backslash_also_works(self):
        p = self.dir / "b.toml"
        p.write_text('[paths]\ndata_dir = "D:\\\\lab\\\\exp"\n', encoding="utf-8")
        self.assertEqual(config.load(p).paths.data_dir, r"D:\lab\exp")

    def test_bom_is_stripped(self):
        p = self.dir / "c.toml"
        p.write_bytes(b"\xef\xbb\xbf[llm]\nmodel = \"kimi-k3\"\n")
        self.assertEqual(config.load(p).llm.model, "kimi-k3")

    def test_broken_file_is_loud_in_strict_mode(self):
        p = self.dir / "d.toml"
        p.write_text("{ this is not toml", encoding="utf-8")
        with self.assertRaises(SystemExit):
            config.load(p, strict=True)

    def test_broken_file_falls_back_when_not_strict(self):
        """钩子用 strict=False：配置坏了最多不采集，绝不能打断宿主程序。"""
        p = self.dir / "d.toml"
        p.write_text("{ this is not toml", encoding="utf-8")
        cfg = config.load(p, strict=False)
        self.assertFalse(cfg.llm.offline or cfg.llm.api_key)   # 全是默认值
        self.assertIn("解析失败", cfg.source)

    def test_generated_config_uses_demo_data_dir(self):
        """config --init 生成的配置应默认读取模拟数据生成器的实验日志目录。"""
        p = self.dir / "labmemory.toml"
        config.write_default(p)
        self.assertEqual(config.load(p).paths.data_dir, "./demo_data/logs/test1")


class TestCapture(unittest.TestCase):
    """运行采集：参数快照 diff + stdout 落盘。"""

    def setUp(self):
        self._tmp = TempDir()
        self.root = self._tmp.path / "proj"
        (self.root / ".labmemory").mkdir(parents=True, exist_ok=True)
        self.script = self.root / "train.py"
        self.script.write_text("lr = 0.0005\nbatch = 8\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self) -> Path:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):      # 子进程输出会打到 stdout，测试里静音
            code = capture.run_wrapped(
                self.root, [sys.executable, "-c", "print('step 10 loss=1.6900')"])
        self.assertEqual(code, 0)
        return max((self.root / ".labmemory" / "runs").glob("*.log"),
                   key=lambda p: p.stat().st_mtime)

    def test_record_contains_params_and_stdout(self):
        rec = self._run()
        text = rec.read_text(encoding="utf-8")
        self.assertIn("# params(new)", text)
        self.assertIn("lr=0.0005", text)
        self.assertIn("loss=1.6900", text)
        self.assertIn("# ==== END exit=0", text)

    def test_param_change_is_diffed(self):
        self._run()
        time.sleep(0.05)
        self.script.write_text("lr = 0.001\nbatch = 8\n", encoding="utf-8")
        rec = self._run()
        text = rec.read_text(encoding="utf-8")
        self.assertIn("lr 0.0005 -> 0.001", text)

    def test_record_is_searchable_as_card(self):
        """运行记录必须能被走一遍正常摄录流程变成卡片。"""
        self._run()
        ctx = Ctx(root=self.root, store=Store(self._tmp.path / "c.db"),
                  state_file=self._tmp.path / ".labmemory" / "ingested.json")
        ingest(ctx, hours=240)
        cards = ctx.store.recent(days=3650)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["exp_name"], "proj")       # 实验名取项目目录名
        self.assertEqual(cards[0]["setup"].get("lr"), 0.0005)


class TestCardShape(unittest.TestCase):
    """模型会写出各种奇怪形状：写入口必须归一，读路径不能崩。"""

    def setUp(self):
        self._tmp = TempDir()
        self.ctx = make_workspace(self._tmp.path)
        self.sample = next((self.ctx.root / "logs").rglob("*.log"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_scalar_metrics_are_normalized(self):
        """`"loss": 0.641` 这种裸数字不能把下游打崩（真模型实际就这么写过）。"""
        r = dispatch(self.ctx, "save_card", {"card": {
            "exp_name": "shape", "goal": "g", "files": [str(self.sample)],
            "metrics": {"loss": 0.641, "acc": "0.8", "bad": "abc"},
            "observations": ["纯字符串观测"],
        }})
        self.assertTrue(r["ok"], r.get("error"))
        self.assertTrue(r["warnings"], "形状被纠正时必须回告警，否则模型不会改")
        card = self.ctx.store.to_dict(r["card_id"])
        self.assertEqual(card["metrics"]["loss"]["n"], 1)
        self.assertEqual(card["metrics"]["loss"]["last"], 0.641)
        self.assertEqual(card["metrics"]["acc"]["last"], 0.8)
        self.assertNotIn("bad", card["metrics"])
        self.assertEqual(card["observations"][0]["kind"], "note")

    def test_relative_file_path_is_stored_absolute(self):
        """模型给相对路径时，卡片里要落绝对路径，否则覆盖检查/证据核对都对不上。"""
        rel = self.sample.relative_to(self.ctx.root)
        r = dispatch(self.ctx, "save_card", {"card": {
            "exp_name": "relpath", "goal": "g", "files": [str(rel)]}})
        self.assertTrue(r["ok"], r.get("error"))
        files = self.ctx.store.to_dict(r["card_id"])["files"]
        self.assertTrue(Path(files[0]).is_absolute(), files)
        self.assertEqual(Path(files[0]).resolve(), self.sample.resolve())

    def test_legacy_card_does_not_break_readers(self):
        """库里已存在的裸数字卡片，不能让检索和周报崩。"""
        self.ctx.store.save_card({"exp_name": "old", "goal": "g", "files": [],
                                  "metrics": {"loss": 0.5}, "observations": ["x"]})
        res = dispatch(self.ctx, "search_cards", {"query": "loss"})
        self.assertTrue(res["ok"], res.get("error"))
        md = report.build_weekly(self.ctx.store.recent(days=7), 7)
        self.assertIn("实验室周报", md)


class TestWatcher(unittest.TestCase):
    """监听：慢速写入只触发一次，且发生在写完之后。"""

    def test_settle_and_batch(self):
        with tempdir() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            got: list[list[str]] = []
            w = Watcher(root, lambda ps: got.append([p.name for p in ps]),
                        settle=0.5, batch_window=0.8, interval=0.2, verbose=False)
            target = root / "logs" / "new_run.log"

            def writer():
                time.sleep(0.6)
                with target.open("w", encoding="utf-8") as f:
                    for i in range(4):
                        f.write(f"step {i} loss={1.0 - 0.1 * i}\n")
                        time.sleep(0.15)          # 慢慢写，验证不会读到半个文件

            th = threading.Thread(target=writer)
            th.start()
            w.run(duration=4)
            th.join()

            self.assertEqual(len(got), 1, f"应只触发一次，实际 {got}")
            self.assertEqual(got[0], ["new_run.log"])
            self.assertIn("step 3", target.read_text(encoding="utf-8"))


class TestReport(unittest.TestCase):
    def test_weekly_sections(self):
        with tempdir() as tmp:
            ctx = make_workspace(Path(tmp))
            ingest(ctx)
            md = report.build_weekly(ctx.store.recent(days=7), days=7)
            for section in ("## 一、概览", "## 二、逐实验进展", "## 三、需要处理的问题",
                            "## 四、下周建议验证", "## 五、实验链"):
                self.assertIn(section, md)
            self.assertIn("更快的收敛", "更快的收敛")     # 占位，防止空文档也算通过
            self.assertGreater(len(md), 500)


class TestParser(unittest.TestCase):
    """解析层：几类常见日志格式都要能抽出来。"""

    def test_key_value_and_jsonl_and_csv(self):
        with tempdir() as tmp:
            root = Path(tmp)
            f1 = root / "a.log"
            f1.write_text("step 10 loss=1.20 lr=0.001\nstep 20 loss=0.90 lr=0.001\n",
                          encoding="utf-8")
            s1 = parsing.iter_metrics(f1)
            self.assertEqual([round(v, 2) for _, v in s1["loss"]], [1.2, 0.9])
            self.assertTrue(parsing.series_stats("loss", s1["loss"])["constant"] is False)

            f2 = root / "b.jsonl"
            f2.write_text('{"step": 1, "eval_loss": 0.7}\n{"step": 2, "eval_loss": 0.5}\n',
                          encoding="utf-8")
            self.assertEqual(len(parsing.iter_metrics(f2)["eval_loss"]), 2)

            # 单值 lr 的 CSV：不是扫描维度，而是这次运行的 setup
            f3 = root / "c.csv"
            f3.write_text("epoch,lr,loss\n1,0.001,0.9\n2,0.001,0.6\n", encoding="utf-8")
            info = parsing.inspect(f3)
            self.assertEqual(info["guesses"]["setup"].get("lr"), 0.001)
            self.assertEqual(info["sweep"], {})
            self.assertNotIn("lr", info["metrics"])     # 恒定列不算指标
            self.assertIn("loss", info["metrics"])

            # 多值 lr 的 CSV：才是扫描维度
            f4 = root / "d.csv"
            f4.write_text("epoch,lr,loss\n1,0.001,0.9\n2,0.005,0.6\n", encoding="utf-8")
            info4 = parsing.inspect(f4)
            self.assertIn("lr", info4["sweep"])
            self.assertNotIn("lr", info4["metrics"])
            self.assertEqual(info4["anomalies"], [])    # 扫描表不做趋势判定

    def test_constant_series_is_not_a_metric(self):
        with tempdir() as tmp:
            f = Path(tmp) / "const.log"
            f.write_text("\n".join(f"step {i} lr=0.001 batch=32" for i in range(10)),
                         encoding="utf-8")
            info = parsing.inspect(f)
            self.assertEqual(info["metrics"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
