"""命令行入口：demo / scan / ask / weekly / cards / show / watch / run / hook。"""

from __future__ import annotations

import json
import os
import shutil
import site
import sys
import time
from pathlib import Path

from . import capture, config, parsing, prompts, report
from .agent import Agent
from .llm import OpenAICompat, RulePolicy
from .store import Store
from .tools import Ctx
from .tools import dispatch
from .watch import Watcher

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


# ------------------------------ 基础设施 ------------------------------

def _banner(title: str) -> None:
    print(f"\n== {title} " + "=" * max(8, 62 - len(title)))


def load_cfg(args) -> config.Config:
    return config.load(getattr(args, "config", None))


def resolve_dir(args, cfg: config.Config) -> Path:
    """命令行 --dir > 环境变量/config 里的 data_dir"""
    return Path(args.dir or cfg.paths.data_dir).expanduser().resolve()


def resolve_db(args, cfg: config.Config) -> Path:
    return Path(args.db or cfg.paths.db).expanduser()


def make_llm(offline: bool, mode: str, cfg: config.Config, since_hours: float | None = None):
    since_hours = cfg.scan.hours if since_hours is None else since_hours
    if offline or cfg.llm.offline:
        why = "offline" if (offline or cfg.llm.offline) else ""
        return RulePolicy(mode=mode, since_hours=since_hours), f"rule({why})"
    if not cfg.llm.api_key:
        print("[!] 配置里没有 api_key（环境变量 LM_API_KEY 也没有），自动降级为规则模式")
        return RulePolicy(mode=mode, since_hours=since_hours), "rule(fallback)"
    return (OpenAICompat(model=cfg.llm.model, base_url=cfg.llm.base_url,
                         api_key=cfg.llm.api_key, temperature=cfg.llm.temperature,
                         timeout=cfg.llm.timeout),
            f"llm({cfg.llm.model})")


def make_ctx(args, root: Path, cfg: config.Config | None = None) -> Ctx:
    cfg = cfg or load_cfg(args)
    db = resolve_db(args, cfg)
    store = Store(db)
    # 状态文件统一放到 <db 所在目录>/.labmemory/ 下：
    # 这是点目录，且 capture.tracked() 只放行 .labmemory/runs 下的 .log/.diff，
    # 所以 LabMemory 自己产生的状态与轨迹不会被当成"实验记录"再摄入一遍。
    state = db.resolve().parent / ".labmemory" / "ingested.json"
    return Ctx(root=root, store=store, state_file=state,
               interactive=not getattr(args, "no_input", False))


def _dump_run(name: str, result: dict, db: Path) -> Path:
    out = Path(db).resolve().parent / ".labmemory" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.json"
    path.write_text(json.dumps({"argv": sys.argv[1:], **result}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def repair_uncovered(ctx: Ctx, verbose: bool = True) -> list[str]:
    """兜底：把"列出过或读过、但没有任何卡片引用"的文件用规则补建一张卡。

    为什么不指望模型自觉：真模型跑 7 个文件时漏掉过 eval_*.jsonl，
    而"每个文件都有归属"是这套工具对用户的承诺，不能靠模型记性。
    补出来的卡会打上 #补漏 并降到 needs_review，一眼看得出不是模型写的。
    """
    from .llm import card_from_inspect

    covered = ctx.saved
    pending = sorted((ctx.listed | ctx.inspected) - covered)
    made: list[str] = []
    for path in pending:
        p = Path(path)
        if not p.exists():
            continue
        try:
            info = parsing.inspect(p)
        except Exception:
            continue
        card = card_from_inspect(info)
        tags = list(card.get("tags") or [])
        card["tags"] = list(dict.fromkeys([*tags, "#补漏"]))
        card["status"] = "needs_review"
        card["confidence"] = min(float(card.get("confidence") or 0.4), 0.4)
        card["goal"] = f"[补漏] 这个文件没有进任何卡片，由规则补建：{p.name}"
        res = dispatch(ctx, "save_card", {"card": card})
        if res.get("ok"):
            made.append(res["card_id"])
    if verbose and made:
        print(f"\n[!] 覆盖自检：有 {len(made)} 个文件读过却没建卡，已用规则补建（标记 #补漏）")
        for cid in made:
            print(f"    {cid}")
    return made


def _run_agent(args, mode: str, system: str, user_input: str, root: Path) -> dict:
    cfg = load_cfg(args)
    ctx = make_ctx(args, root, cfg)
    turns = args.max_turns or (cfg.agent.max_turns_ingest if mode == "ingest"
                               else cfg.agent.max_turns_query)
    llm, tag = make_llm(getattr(args, "offline", False), mode, cfg,
                        getattr(args, "hours", None))
    agent = Agent(llm, ctx, system, max_turns=turns, max_calls=cfg.agent.max_calls,
                  max_result_chars=cfg.agent.max_result_chars)
    _banner(f"{mode} · {tag} · 根目录 {root}")
    res = agent.run(user_input)
    degraded = False
    if not res.get("ok") and not (getattr(args, "offline", False) or cfg.llm.offline):
        kind = res.get("error_kind") or "unknown"
        print(f"\n[!] 模型不可用（{kind}）")
        if res.get("hint"):
            print(f"    怎么办：{res['hint']}")
        print(f"    原文　：{res.get('error')}")
        print("    → 降级为规则模式重跑同一任务（卡片照样建，结论会比模型版朴素）")
        llm2 = RulePolicy(mode=mode, since_hours=getattr(args, "hours", None) or cfg.scan.hours)
        res = Agent(llm2, ctx, system, max_turns=turns, verbose=True).run(user_input)
        tag += " → 降级"
        degraded = True
    _banner("Agent 输出")
    print(res.get("final") or f"(无输出) {res.get('error','')}")
    print(f"\n-- {tag}｜{res.get('turns')} 轮｜{len(res.get('trace', []))} 次工具调用｜{res.get('elapsed')}s")
    if degraded:
        print("-- 注意：本次结果由规则模式生成（模型不可用），严格来说不可用于正式结论")
    if res.get("error") and not degraded:
        print(f"-- 错误：{res['error']}")
    res["degraded"] = degraded
    if mode == "ingest" and getattr(args, "repair", True):
        res["repaired"] = repair_uncovered(ctx)
        if res["repaired"]:
            print("-- 提示：这些卡片是补漏生成的，建议人工看一眼再引用")
    return res


# ------------------------------ 子命令 ------------------------------

def cmd_scan(args) -> int:
    cfg = load_cfg(args)
    root = resolve_dir(args, cfg)
    if not root.exists():
        print(f"数据目录不存在：{root}\n请在 {cfg.source} 的 [paths] data_dir 里改，或用 --dir 指定")
        return 2
    hours = args.hours or cfg.scan.hours
    system = prompts.INGEST_SYSTEM.replace("{tags}", "、".join(prompts.TAG_VOCAB))
    prompt = (f"数据目录：{root}\n"
              f"请处理其中最近 {hours} 小时内新增或改动过的实验记录。")
    res = _run_agent(args, "ingest", system, prompt, root)
    _dump_run("scan", res, resolve_db(args, cfg))
    return 0 if res.get("ok") else 1


def cmd_ask(args) -> int:
    cfg = load_cfg(args)
    root = resolve_dir(args, cfg)
    ctx = make_ctx(args, root, cfg)
    stats = ctx.store.counts()
    if not stats:
        print("知识库还是空的。先运行：python main.py scan")
        return 2
    res = _run_agent(args, "query", prompts.QUERY_SYSTEM, args.question, root)
    _dump_run("ask", {**res, "question": args.question}, resolve_db(args, cfg))
    return 0 if res.get("ok") else 1


def cmd_weekly(args) -> int:
    cfg = load_cfg(args)
    root = resolve_dir(args, cfg)
    db = resolve_db(args, cfg)
    store = Store(db)
    cards = store.recent_for_root(root, days=args.days)
    md = report.build_weekly(cards, days=args.days, scope=f"`{root.name}` · {root}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    else:
        out = report.save_weekly(md, root / "weekly")
    print(md)
    print(f"\n-- 已写入 {out}")
    return 0


def cmd_cards(args) -> int:
    cfg = load_cfg(args)
    store = Store(resolve_db(args, cfg))
    cards = store.recent(days=3650, limit=200)
    if not cards:
        print("还没有卡片。先运行 scan。")
        return 0
    print(f"{'id':<12}{'实验':<20}{'run':<8}{'状态':<14}{'标签'}")
    print("-" * 88)
    for c in cards:
        print(f"{c['id']:<12}{c['exp_name'][:18]:<20}{str(c.get('run_id') or '—'):<8}"
              f"{c['status']:<14}{' '.join(c.get('tags') or [])}")
    return 0


def cmd_show(args) -> int:
    store = Store(resolve_db(args, load_cfg(args)))
    card = store.to_dict(args.card_id)
    if not card:
        print(f"没有这张卡片：{args.card_id}")
        return 2
    print(json.dumps(card, ensure_ascii=False, indent=2))
    return 0


def cmd_config(args) -> int:
    """查看 / 生成 / 定位配置文件。"""
    if args.init:
        target = Path(args.path or (PROJECT / config.DEFAULT_NAME))
        out = config.write_default(target, force=args.force)
        print(f"已生成 {out}\n请填入 [llm] api_key"
              f"（[paths] data_dir 已默认指向 demo_data/logs/test1），然后：python main.py config --show")
        return 0
    cfg = config.load(args.config)
    if args.path_only:
        print(cfg.source)
        return 0
    print(cfg.describe())
    if cfg.source == "(内置默认值)":
        print(f"\n[!] 没有找到配置文件。运行 `python main.py config --init` 生成 "
              f"{PROJECT / config.DEFAULT_NAME}")
    return 0


def cmd_demo(args) -> int:
    from demo.make_demo_data import generate  # 局部导入，避免循环

    demo_root = Path(args.demo_root).resolve()
    db = demo_root / "demo.db"
    if demo_root.exists() and args.reset:
        shutil.rmtree(demo_root)
    demo_root.mkdir(parents=True, exist_ok=True)
    if db.exists() and args.reset:
        db.unlink()
    data = demo_root / "lab"
    n = generate(data)
    print(f"[demo] 已生成 {n} 个样例实验文件 → {data}")

    _banner("第 1 幕：摄录（Agent 自己决定看什么、怎么切卡）")
    ctx = Ctx(root=data, store=Store(db), state_file=demo_root / ".labmemory" / "ingested.json")
    system = prompts.INGEST_SYSTEM.replace("{tags}", "、".join(prompts.TAG_VOCAB))
    agent = Agent(RulePolicy(mode="ingest", since_hours=args.hours), ctx, system, max_turns=60)
    r1 = agent.run(f"数据目录：{data}\n请处理其中最近 {args.hours} 小时内新增或改动过的实验记录。")
    print("\n" + (r1.get("final") or ""))

    store = Store(db)
    print("\n-- 库内卡片：")
    for c in store.recent(days=3650, limit=50):
        print(f"   {c['id']}  {c['exp_name']:<18} {str(c.get('run_id') or '—'):<6} "
              f"{c['status']:<13} {' '.join(c.get('tags') or [])}")

    for q in ["上次学习率 0.001 那次的 loss 最后降到多少？",
              "最近有没有出现异常或者崩溃的实验？"]:
        _banner(f"第 2 幕：自然语言检索 —— {q}")
        ctx2 = Ctx(root=data, store=store, state_file=demo_root / ".labmemory" / "ingested.json")
        r = Agent(RulePolicy(mode="query"), ctx2, prompts.QUERY_SYSTEM).run(q)
        print("\n" + (r.get("final") or ""))

    _banner("第 3 幕：周报（确定性生成，断网可用）")
    md = report.build_weekly(store.recent(days=7), days=7)
    out = demo_root / "weekly.md"
    out.write_text(md, encoding="utf-8")
    print("\n".join(md.splitlines()[:26]))
    print(f"...\n-- 完整周报：{out}")
    print(f"-- 数据库：{db}")
    return 0


# ------------------------------ 监听 / 采集 ------------------------------

def _ingest_paths(args, root: Path, paths: list[Path]) -> None:
    """把"这几个文件刚变了"交给摄录 Agent（不靠 list_files 自己找）。"""
    cfg = load_cfg(args)
    ctx = make_ctx(args, root, cfg)
    llm, tag = make_llm(getattr(args, "offline", False), "ingest", cfg)
    system = prompts.INGEST_SYSTEM.replace("{tags}", "、".join(prompts.TAG_VOCAB))
    rels = []
    for p in paths:
        try:
            rels.append(str(p.relative_to(root)).replace("\\", "/"))
        except ValueError:
            rels.append(str(p))
    prompt = ("刚刚检测到这些文件发生变化，请处理并建卡：\n"
              + "\n".join(f"- {r}" for r in rels)
              + "\n如果是运行记录（.labmemory/runs 下的 .log），请连同同名 .diff 一起看，"
                "参数以记录头部的 params(new)/changes 为准，指标以 stdout 段为准。")
    turns = args.max_turns or cfg.agent.max_turns_ingest
    res = Agent(llm, ctx, system, max_turns=turns, max_calls=cfg.agent.max_calls,
                max_result_chars=cfg.agent.max_result_chars).run(prompt)
    print(res.get("final") or res.get("error", ""))
    print(f"-- {tag}｜{res.get('turns')} 轮｜{res.get('elapsed')}s")
    if getattr(args, "repair", True):
        repair_uncovered(ctx)


def cmd_watch(args) -> int:
    cfg = load_cfg(args)
    root = resolve_dir(args, cfg)
    if not root.exists():
        print(f"目录不存在：{root}\n请在 {cfg.source} 的 [paths] data_dir 里改，或用 --dir 指定")
        return 2
    settle = args.settle if args.settle is not None else cfg.watch.settle
    batch = args.batch if args.batch is not None else cfg.watch.batch_window
    interval = args.interval if args.interval is not None else cfg.watch.interval
    duration = args.duration if args.duration is not None else (cfg.watch.duration or None)
    print(f"[watch] 监听 {root}　（settle={settle}s, batch={batch}s）")

    def handle(paths: list[Path]) -> None:
        # 运行记录旁边通常还有一个同名 .diff，一起送进去
        extra = [p.with_suffix(".diff") for p in paths
                 if p.suffix == ".log" and p.with_suffix(".diff").exists()]
        batch = paths + [p for p in extra if p not in paths]
        _banner(f"摄录 {len(batch)} 个文件")
        _ingest_paths(args, root, batch)

    Watcher(root, handle, settle=settle, batch_window=batch,
            interval=interval).run(duration=duration)
    return 0


def cmd_run(args) -> int:
    """包装运行：python main.py run -- python train.py --lr 0.001"""
    cfg = load_cfg(args)
    root = resolve_dir(args, cfg)
    cmd = list(args.cmd or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print('用法：python main.py run -- python train.py   （-- 之后是要执行的命令）')
        return 2
    # 先建 .labmemory/ 标记，让解释器钩子也能在这个项目上生效
    (root / capture.STATE_DIR).mkdir(parents=True, exist_ok=True)
    code = capture.run_wrapped(root, cmd)
    record = max((root / capture.RUN_DIR).glob("*.log"), key=lambda p: p.stat().st_mtime, default=None)
    if record and not args.no_ingest:
        _banner("摄录本次运行")
        extra = [record.with_suffix(".diff")] if record.with_suffix(".diff").exists() else []
        _ingest_paths(args, root, [record, *extra])
    return code


def cmd_hook(args) -> int:
    """安装/卸载解释器级采集钩子（IDE 里按 F5 也能自动落盘）。"""
    src = PROJECT / "integrations" / "sitecustomize.py"
    if not src.exists():
        print(f"找不到模板：{src}")
        return 2
    targets = []
    if args.target:
        targets = [Path(args.target)]
    else:
        try:
            targets = [Path(site.getusersitepackages())]
        except AttributeError:
            targets = []
        if args.system:
            targets += [Path(p) for p in site.getsitepackages()]
    if args.uninstall:
        for d in targets:
            f = d / "sitecustomize.py"
            if f.exists() and "LABMEMORY_HOME" in f.read_text("utf-8", errors="replace"):
                f.unlink()
                print(f"已卸载 {f}")
        return 0
    installed = []
    for d in targets:
        try:
            d.mkdir(parents=True, exist_ok=True)
            dst = d / "sitecustomize.py"
            if dst.exists() and "LABMEMORY_HOME" not in dst.read_text("utf-8", errors="replace"):
                print(f"[!] {dst} 已存在且不是本工具的，未覆盖")
                continue
            dst.write_text(src.read_text("utf-8"), encoding="utf-8")
            installed.append(dst)
        except OSError as e:
            print(f"[!] 写入 {d} 失败：{e}")
    if not installed:
        print("没有安装到任何目录，可用 --target <目录> 指定（该目录需在 sys.path 上）")
        return 1
    for f in installed:
        print(f"已安装 {f}")
    print("\n用法：\n"
          f"  1) export LABMEMORY_HOME={PROJECT}        # 或写进系统环境变量\n"
          "  2) 在项目里建标记：mkdir .labmemory\n"
          "  3) 之后该项目下任何 python 进程（含 IDE F5 调试）都会自动写运行记录到 .labmemory/runs/")
    return 0
    return 0
