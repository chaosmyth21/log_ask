"""真模型验收：接上 API Key 之后，跑一遍判断"能不能放心用"。

    python scripts/acceptance.py                 # 用生成的样例数据
    python scripts/acceptance.py --data D:/lab/logs
    python scripts/acceptance.py --keep          # 保留工作目录以便人工检查

它会真的调模型跑一次摄录 + 一次检索，然后逐条检查五件事：
    0  配置可用（key / base_url / model / 能不能连通）
    1  覆盖：每个实验文件都进了卡片，不许漏
    2  数字：卡片里的指标必须与源文件一致，不许编
    3  证据：每条观测都要有 evidence，且行号能对上
    4  检索：一句人话能命中正确卡片并给出数字
    5  降级：key 失效时必须明确说"降级"，而不是静默出低质量结果

退出码 = 失败项数（0 表示全过）。
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _preflight() -> None:
    """在 import 项目代码之前先证明这个解释器本身是好的。

    最常见的事故：IDE 里配的 Python 和终端里不是同一个，而前者环境是坏的
    （典型报错 `ImportError: DLL load failed while importing _sqlite3`），
    看起来像项目崩了，其实是解释器崩了。这里要第一时间把话说清楚。
    """
    try:
        import sqlite3  # noqa: F401
    except Exception as e:                      # noqa: BLE001 - 什么错都要报出来
        print(
            "\n[环境检查失败] 当前解释器连 sqlite3 都加载不了，和项目代码无关。\n"
            f"  解释器：{sys.executable}\n"
            f"  错误　：{type(e).__name__}: {e}\n"
            "\nLabMemory 依赖标准库 sqlite3，所以这是解释器/环境的问题。三种修法任选：\n"
            "  1) IDE 里换解释器（Settings → Project → Python Interpreter），\n"
            "     选终端里能跑的那个。终端查路径：\n"
            "        python -c \"import sys, sqlite3; print(sys.executable, sqlite3.sqlite_version)\"\n"
            "  2) 修 conda 环境：conda install -c conda-forge sqlite\n"
            "     （或新建一个 Python 3.11+ 的干净环境）\n"
            "  3) 直接用能跑的解释器执行：\n"
            "        <能跑的python> LabMemory/scripts/acceptance.py\n"
        )
        raise SystemExit(2)


_preflight()

from labmemory import config, parsing, prompts          # noqa: E402
from labmemory.agent import Agent                        # noqa: E402
from labmemory.llm import OpenAICompat, RulePolicy       # noqa: E402
from labmemory.store import Store                        # noqa: E402
from labmemory.tools import Ctx                          # noqa: E402

TMP_ROOT = ROOT / "tests" / "_tmp"
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.rows.append((name, status, detail))
        icon = {PASS: "✓", WARN: "!", FAIL: "✗"}[status]
        print(f"  {icon} [{status}] {name}" + (f"\n        {detail}" if detail else ""))

    def failures(self) -> int:
        return sum(1 for _, s, _ in self.rows if s == FAIL)

    def summary(self) -> None:
        print("\n" + "=" * 68)
        table = {"PASS": 0, "WARN": 0, "FAIL": 0}
        for _, s, _ in self.rows:
            table[s] += 1
        print(f"结果：{table['PASS']} 通过 / {table['WARN']} 警告 / {table['FAIL']} 失败")
        for name, s, _ in self.rows:
            if s != PASS:
                print(f"  {s}  {name}")
        print("=" * 68)


def _new_workdir() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    for i in range(1000):
        d = TMP_ROOT / f"accept{i:03d}"
        if not d.exists():
            d.mkdir()
            return d
    raise SystemExit("tests/_tmp 下目录太多，先清理")


# ------------------------------ 检查项 ------------------------------

def check_connectivity(cfg: config.Config, rep: Report) -> bool:
    """0. 一次最小调用，确认 key / base_url / model 真的能通。"""
    name = "0 配置与连通性"
    if cfg.llm.offline:
        rep.add(name, FAIL, "配置里 offline = true，验收必须走真模型；先改成 false")
        return False
    if not cfg.llm.api_key:
        rep.add(name, FAIL, "没有 api_key；在 labmemory.toml 的 [llm] 里填，或设 LM_API_KEY")
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        rep.add(name, FAIL, "没有安装 openai：pip install -r requirements.txt")
        return False
    try:
        llm = OpenAICompat(model=cfg.llm.model, base_url=cfg.llm.base_url,
                           api_key=cfg.llm.api_key, timeout=cfg.llm.timeout)
        t0 = time.time()
        resp = llm.chat([{"role": "user", "content": "只回复 ok"}], tools=[])
        dt = time.time() - t0
        text = (resp.get("content") or "").strip()[:40]
        rep.add(name, PASS, f"{cfg.llm.model} @ {cfg.llm.base_url}　{dt:.1f}s　回复：{text!r}")
        return True
    except Exception as e:
        rep.add(name, FAIL, f"调用失败：{type(e).__name__}: {e}")
        return False


def _resolve(f, root: Path) -> Path:
    """卡片里的 files 可能是绝对路径，也可能是模型给的相对路径，统一成绝对。"""
    p = Path(str(f))
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def check_coverage(cards: list[dict], files: list[Path], root: Path, rep: Report) -> None:
    referenced = {str(_resolve(f, root)) for c in cards for f in c.get("files") or []}
    missing = [f for f in files if str(f.resolve()) not in referenced]
    extra = len(cards) - len(files)
    detail = f"{len(files)} 个文件 → {len(cards)} 张卡片"
    repaired = [c for c in cards if "#补漏" in (c.get("tags") or [])]
    if missing:
        detail += f"；漏了 {len(missing)} 个：{[p.name for p in missing][:5]}"
        dup: dict = {}
        for c in cards:
            key = (c.get("exp_name"), c.get("run_id"))
            dup[key] = dup.get(key, 0) + 1
        if dup:
            (exp, run), n = max(dup.items(), key=lambda kv: kv[1])
            if n >= 3:
                detail += (f"\n        其中 {n} 张都是 {exp}/{run} —— 模型很可能在反复重存同一张卡"
                           f"（这一版应该已被 save_card 的原地更新挡住，若仍出现说明护栏没生效）")
        rep.add("1 覆盖：每个文件都进卡片", FAIL, detail)
    elif repaired:
        # 有卡就说明产品兜住了，但那是规则补的，模型自己漏了 —— 必须标成警告
        rep.add("1 覆盖：每个文件都进卡片", WARN,
                detail + f"；其中 {len(repaired)} 张是补漏生成的（模型漏读了："
                f"{'、'.join(sorted({Path(f).name for c in repaired for f in c.get('files') or []})[:5])}）")
    elif extra > 1:
        rep.add("1 覆盖：每个文件都进卡片", WARN,
                detail + f"；卡片数比文件数多 {extra}，模型可能把一次 run 拆成了多张")
    else:
        rep.add("1 覆盖：每个文件都进卡片", PASS, detail)


def check_numbers(cards: list[dict], rep: Report, root: Path) -> None:
    """2. 卡片里的指标必须能由源文件重算出来。"""
    bad, warn, ok, odd = [], [], 0, []
    comparable, unreadable = 0, []
    for c in cards:
        for f in c.get("files") or []:
            p = _resolve(f, root)
            if not p.exists():
                unreadable.append(f"{c['id']} → {f}")
                continue
            try:
                series = parsing.iter_metrics(p)
            except Exception:
                continue
            for name, m in (c.get("metrics") or {}).items():
                if not isinstance(m, dict):
                    # 只有"根本不是对象"才算形状问题；缺 n 的字典仍然是合法的统计对象
                    odd.append(f"{c['id']} {name}={m!r}")
                    continue
                if name not in series:
                    continue
                comparable += 1
                truth = parsing.series_stats(name, series[name])
                for key in ("first", "last", "best"):
                    if key not in m or key not in truth:
                        continue
                    a, b = float(m[key]), float(truth[key])
                    if abs(a - b) <= 1e-6:
                        ok += 1
                    elif any(abs(a - round(b, d)) <= 1e-9 for d in (1, 2, 3, 4)):
                        warn.append(f"{c['id']} {name}.{key}: 卡片 {a} vs 源文件 {b}（四舍五入）")
                    else:
                        bad.append(f"{c['id']} {name}.{key}: 卡片 {a} vs 源文件 {b}")
    if bad:
        rep.add("2 数字：卡片指标与源文件一致", FAIL,
                f"{len(bad)} 处不一致，例如：\n        " + "\n        ".join(bad[:4]))
    elif comparable == 0:
        # 一个都没比对上不能算通过 —— 这正是上一版"假 PASS"的原因
        rep.add("2 数字：卡片指标与源文件一致", WARN,
                f"无法校验：卡片里的指标名与源文件抽出的序列对不上（{ok} 项）"
                + (f"；另有 {len(unreadable)} 个文件路径打不开，例如 {unreadable[0]}" if unreadable else "")
                + (f"；{len(odd)} 个指标不是统计对象" if odd else ""))
    elif odd:
        rep.add("2 数字：卡片指标与源文件一致", WARN,
                f"{ok} 项数值一致（覆盖 {comparable} 个指标）；另有 {len(odd)} 个指标不是统计对象"
                f"（已按单点归一）：{'；'.join(odd[:2])}")
    elif warn:
        rep.add("2 数字：卡片指标与源文件一致", WARN,
                f"{ok} 项一致（覆盖 {comparable} 个指标）；{len(warn)} 项被四舍五入，例如：{warn[0]}")
    else:
        rep.add("2 数字：卡片指标与源文件一致", PASS,
                f"{ok} 项数值全部逐位一致（{comparable} 个指标 × first/last/best）")


def check_evidence(cards: list[dict], rep: Report, root: Path) -> None:
    """3. 每条观测都要有可核对的 evidence。"""
    total, missing, bad_line = 0, [], []
    for c in cards:
        for o in c.get("observations") or []:
            if not isinstance(o, dict) or o.get("kind") in (None, "note"):
                continue
            total += 1
            ev = (o.get("evidence") or "").strip()
            if not ev:
                missing.append(f"{c['id']} [{o.get('kind')}] 无 evidence")
                continue
            m = re.match(r"(.+?)[:：](\d+)$", ev)
            if not m:
                continue
            fname, line_no = m.group(1), int(m.group(2))
            src = next((p for p in (_resolve(f, root) for f in c.get("files") or [])
                        if p.name == Path(fname).name and p.exists()), None)
            if src is None:
                continue
            n_lines = len(src.read_text("utf-8", errors="replace").splitlines())
            if line_no > n_lines:
                bad_line.append(f"{c['id']} {fname}:{line_no} 超出文件行数 {n_lines}")
    if total == 0:
        rep.add("3 证据：每条观测都有 evidence", WARN, "所有卡片都没有 observations，无法判定")
    elif missing or bad_line:
        rep.add("3 证据：每条观测都有 evidence", FAIL,
                f"{len(missing)} 条缺证据、{len(bad_line)} 条行号越界\n        "
                + "\n        ".join((missing + bad_line)[:4]))
    else:
        rep.add("3 证据：每条观测都有 evidence", PASS, f"{total} 条观测全部带可核对证据")


def check_retrieval(cfg: config.Config, ctx: Ctx, cards: list[dict], rep: Report) -> None:
    """4. 一句人话能不能查回来。"""
    target = None
    for c in cards:
        if (c.get("setup") or {}).get("lr") == 0.001 and (c.get("metrics") or {}).get("loss"):
            if c["status"] != "superseded":
                target = c
                break
    if target is None:
        rep.add("4 检索：一句话查到正确答案", WARN, "样例数据里没有 lr=0.001 的卡片，跳过")
        return
    expect = target["metrics"]["loss"]["last"]
    q = "上次学习率 0.001 那次的 loss 最后降到多少？"
    llm = OpenAICompat(model=cfg.llm.model, base_url=cfg.llm.base_url,
                       api_key=cfg.llm.api_key, timeout=cfg.llm.timeout)
    res = Agent(llm, ctx, prompts.QUERY_SYSTEM,
                max_turns=cfg.agent.max_turns_query, verbose=False).run(q)
    ans = res.get("final") or ""
    hit_num = any(tok in ans for tok in (f"{expect}", f"{expect:.4f}", f"{expect:.2f}"))
    hit_src = (target["id"] in ans) or any(Path(f).name in ans for f in target.get("files") or [])
    if not res.get("ok"):
        rep.add("4 检索：一句话查到正确答案", FAIL, f"模型调用失败：{res.get('error')}")
    elif hit_num and hit_src:
        rep.add("4 检索：一句话查到正确答案", PASS,
                f"{res.get('turns')} 轮找到 {target['id']}，答案含 {expect} 与来源")
    elif hit_num or hit_src:
        rep.add("4 检索：一句话查到正确答案", WARN,
                f"命中了数字={hit_num} 来源={hit_src}，但另一项缺失\n        答案：{ans[:200]}")
    else:
        rep.add("4 检索：一句话查到正确答案", FAIL,
                f"答案里既没有 {expect} 也没有卡片 id/文件名\n        答案：{ans[:300]}")


def check_degradation(rep: Report) -> None:
    """5. key 不可用时必须明确降级/报错，而不是静默出低质量结果。"""
    problems: list[str] = []
    notes: list[str] = []

    # a) 没 key：必须切规则模式并打印提示
    buf = io.StringIO()
    with redirect_stdout(buf):
        llm, tag = _make_llm_like_cli(config.Config())
    if not isinstance(llm, RulePolicy) or not tag.startswith("rule"):
        problems.append(f"没有 api_key 时应走规则模式，实际 tag={tag}")
    if "降级" not in buf.getvalue() and "没有" not in buf.getvalue():
        problems.append("没有 api_key 时没有打印任何提示，用户不知道自己在用降级模式")
    else:
        notes.append("无 key → 规则模式并提示")

    # b) 坏 key：客户端能构造出来，但首次调用必须报错（而不是返回空内容糊弄过去）
    try:
        bad = config.Config()
        bad.llm.api_key = "sk-invalid-for-acceptance"
        client = OpenAICompat(model=bad.llm.model, base_url=bad.llm.base_url,
                              api_key=bad.llm.api_key, timeout=10)
        try:
            client.chat([{"role": "user", "content": "hi"}], tools=[])
            problems.append("坏 key 竟然调用成功，检查 base_url 是否指到了本地可信端点")
        except Exception as e:
            notes.append(f"坏 key → 明确报错 {type(e).__name__}")
    except ImportError:
        notes.append("未安装 openai，跳过坏 key 测试")
    except Exception as e:                                  # 构造就失败也算明确报错
        notes.append(f"坏 key → 构造即失败 {type(e).__name__}")

    if problems:
        rep.add("5 降级：key 不可用时说清楚", FAIL,
                "；".join(problems) + "\n        " + "；".join(notes))
    else:
        rep.add("5 降级：key 不可用时说清楚", PASS, "；".join(notes))


def _make_llm_like_cli(cfg: config.Config):
    """复刻 cli.make_llm 的判定逻辑（不 import cli，避免循环）。"""
    if cfg.llm.offline or not cfg.llm.api_key:
        print("[!] 配置里没有 api_key，自动降级为规则模式")
        return RulePolicy(mode="query"), "rule(fallback)"
    return OpenAICompat(model=cfg.llm.model, base_url=cfg.llm.base_url,
                        api_key=cfg.llm.api_key), f"llm({cfg.llm.model})"


# ------------------------------ 主流程 ------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LabMemory 真模型验收")
    ap.add_argument("--data", default=None, help="用你自己的实验目录（默认用生成的样例数据）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--keep", action="store_true", help="保留工作目录")
    ap.add_argument("--selftest", action="store_true",
                    help="不调模型，用规则策略跑一遍检查逻辑本身（验证验收器没写错）")
    args = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    cfg = config.load(args.config)
    import sqlite3
    print(f"解释器：{sys.executable}")
    print(f"配置：{cfg.source}")
    print(f"模型：{cfg.llm.model} @ {cfg.llm.base_url}")
    print(f"SQLite：{sqlite3.sqlite_version}")
    rep = Report()

    if not args.selftest and not check_connectivity(cfg, rep):
        rep.summary()
        return rep.failures()

    work = _new_workdir()
    try:
        if args.data:
            data = Path(args.data).expanduser().resolve()
            if not data.exists():
                print(f"数据目录不存在：{data}")
                return 1
            print(f"数据：{data}（你自己的目录，只读，不会改动）")
        else:
            from demo.make_demo_data import generate
            data = work / "lab"
            n = generate(data)
            print(f"数据：{data}（临时生成的 {n} 个样例文件）")

        ctx = Ctx(root=data, store=Store(work / "acc.db"),
                  state_file=work / ".labmemory" / "ingested.json")
        system = prompts.INGEST_SYSTEM.replace("{tags}", "、".join(prompts.TAG_VOCAB))

        if args.selftest:
            llm = RulePolicy(mode="ingest", since_hours=cfg.scan.hours)
            print("\n[selftest] 用规则策略摄录，只验证检查逻辑…")
        else:
            llm = OpenAICompat(model=cfg.llm.model, base_url=cfg.llm.base_url,
                               api_key=cfg.llm.api_key, timeout=cfg.llm.timeout)
            print(f"\n开始摄录（max_turns={cfg.agent.max_turns_ingest}）…")
        t0 = time.time()
        res = Agent(llm, ctx, system, max_turns=cfg.agent.max_turns_ingest,
                    max_calls=cfg.agent.max_calls,
                    max_result_chars=cfg.agent.max_result_chars).run(
            f"数据目录：{data}\n请处理其中最近 {cfg.scan.hours:g} 小时内新增或改动过的实验记录。")
        dt = time.time() - t0
        print(f"摄录结束：{res.get('turns')} 轮 / {len(res.get('trace', []))} 次工具调用 / {dt:.1f}s")
        print((res.get("final") or "")[:400])

        if not res.get("ok"):
            rep.add("0 摄录流程", FAIL, f"模型调用失败：{res.get('error')}")
            rep.summary()
            return rep.failures()

        cards = ctx.store.recent(days=3650, limit=200)
        files = [p for p in data.rglob("*") if p.is_file() and _tracked(p, data)]
        print(f"\n检查 {len(cards)} 张卡片 / {len(files)} 个文件：\n")

        check_coverage(cards, files, data, rep)
        check_numbers(cards, rep, data)
        check_evidence(cards, rep, data)
        if not args.selftest:
            check_retrieval(cfg, ctx, cards, rep)
        check_degradation(rep)

        rep.summary()
        if args.keep:
            print(f"\n工作目录保留在：{work}")
        return rep.failures()
    finally:
        if not args.keep and not args.data:
            shutil.rmtree(work, ignore_errors=True)


def _tracked(p: Path, root: Path) -> bool:
    from labmemory import capture
    from labmemory import parsing as ps
    if p.suffix.lower() not in ps.TEXT_SUFFIX | ps.IMAGE_SUFFIX:
        return False
    try:
        return capture.tracked(p.relative_to(root))
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
