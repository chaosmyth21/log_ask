#!/usr/bin/env python
"""LabMemory 命令行入口。

所有需要配置的东西都在 labmemory.toml 里（见 `python main.py config --show`）：

    python main.py config --show             # 看当前生效配置 + 文件位置
    python main.py config --init             # 从模板生成配置文件
    python main.py demo                      # 一键三幕演示（离线可跑）
    python main.py scan                      # 摄入实验记录，生成卡片
    python main.py ask "学习率0.001那次loss到多少"
    python main.py weekly --days 7
    python main.py watch                     # 常驻监听
    python main.py run -- python train.py    # 包装运行并记录参数变化
    python main.py hook                      # 装解释器钩子（IDE F5 也能记录）

命令行参数优先级最高，其次是环境变量（LM_*），最后才是配置文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from labmemory import cli  # noqa: E402


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def build_parser():
    import argparse

    common = argparse.ArgumentParser(add_help=False)
    # 默认值一律 None：真正的默认值来自 labmemory.toml，留 None 才能区分"没传"与"传了默认值"
    common.add_argument("--config", default=None,
                        help="配置文件路径，默认用 <仓库>/labmemory.toml")
    common.add_argument("--db", default=None, help="SQLite 路径，覆盖配置里的 paths.db")
    common.add_argument("--no-input", action="store_true", help="禁止 Agent 向你提问（批处理）")

    p = argparse.ArgumentParser(prog="labmemory", description="实验记录 → 知识卡片 Agent")
    p.add_argument("--config", dest="config_global", default=None, help="同子命令后的 --config")
    p.add_argument("--db", dest="db_global", default=None, help="同子命令后的 --db")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("config", parents=[common], help="查看 / 生成配置文件")
    s.add_argument("--show", action="store_true", help="打印生效配置（默认行为）")
    s.add_argument("--init", action="store_true", help="从 labmemory.toml.example 生成")
    s.add_argument("--force", action="store_true", help="生成时覆盖已存在的文件")
    s.add_argument("--path", default=None, help="--init 的输出路径；或 --path-only 要查的路径")
    s.add_argument("--path-only", dest="path_only", action="store_true", help="只打印当前配置文件的路径")
    s.set_defaults(func=cli.cmd_config)

    s = sub.add_parser("scan", parents=[common], help="摄入数据目录里的实验记录")
    s.add_argument("--dir", default=None, help="覆盖配置里的 paths.data_dir")
    s.add_argument("--hours", type=float, default=None, help="回看多少小时内的改动")
    s.add_argument("--offline", action="store_true", help="强制用确定性策略（不调 LLM）")
    s.add_argument("--no-repair", dest="repair", action="store_false",
                   help="关闭覆盖补漏（模型漏建卡时不自动补齐）")
    s.add_argument("--max-turns", type=int, default=None)
    s.set_defaults(func=cli.cmd_scan)

    s = sub.add_parser("ask", parents=[common], help="自然语言检索知识卡片")
    s.add_argument("question")
    s.add_argument("--dir", default=None)
    s.add_argument("--offline", action="store_true")
    s.add_argument("--max-turns", type=int, default=None)
    s.set_defaults(func=cli.cmd_ask)

    s = sub.add_parser("weekly", parents=[common], help="生成 Markdown 周报")
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--dir", default=None,
                   help="按这个日志文件夹生成周报，默认用配置里的 data_dir")
    s.add_argument("--out", default=None,
                   help="输出路径；默认保存到 <当前日志文件夹>/weekly/weekly.md")
    s.set_defaults(func=cli.cmd_weekly)

    s = sub.add_parser("cards", parents=[common], help="列出所有卡片")
    s.set_defaults(func=cli.cmd_cards)

    s = sub.add_parser("show", parents=[common], help="查看一张卡片的完整内容")
    s.add_argument("card_id")
    s.set_defaults(func=cli.cmd_show)

    s = sub.add_parser("demo", parents=[common], help="一键演示：摄入 → 检索 → 周报")
    s.add_argument("--demo-root", default="./demo_out")
    s.add_argument("--hours", type=float, default=240)
    s.add_argument("--reset", action="store_true", default=True)
    s.set_defaults(func=cli.cmd_demo)

    # ---- 监听与采集 ----
    s = sub.add_parser("watch", parents=[common], help="监听目录，文件稳定后自动摄录")
    s.add_argument("--dir", default=None)
    s.add_argument("--settle", type=float, default=None, help="文件多久不变才算写完（秒）")
    s.add_argument("--batch", type=float, default=None, help="攒批窗口（秒）")
    s.add_argument("--interval", type=float, default=None, help="轮询间隔（秒）")
    s.add_argument("--duration", type=float, default=None, help="只跑这么多秒就退出，默认常驻")
    s.add_argument("--offline", action="store_true")
    s.add_argument("--no-repair", dest="repair", action="store_false")
    s.add_argument("--max-turns", type=int, default=None)
    s.set_defaults(func=cli.cmd_watch)

    s = sub.add_parser("run", parents=[common], help="包装运行一条命令，自动记录参数变化与输出")
    s.add_argument("--dir", default=None)
    s.add_argument("--no-ingest", action="store_true", help="只记录，不立即摄录")
    s.add_argument("--offline", action="store_true")
    s.add_argument("--max-turns", type=int, default=None)
    s.add_argument("cmd", nargs=argparse.REMAINDER, help="-- 之后是要执行的命令")
    s.set_defaults(func=cli.cmd_run)

    s = sub.add_parser("hook", parents=[common], help="安装解释器级采集钩子（IDE F5 也能记录）")
    s.add_argument("--uninstall", action="store_true")
    s.add_argument("--system", action="store_true", help="同时装到系统 site-packages")
    s.add_argument("--target", default=None, help="装到指定目录（需在 sys.path 上）")
    s.set_defaults(func=cli.cmd_hook)

    return p


def main(argv=None) -> int:
    _utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    # 允许 --config / --db 写在子命令前（argparse 的 parents 只能吃子命令后的那份）
    if getattr(args, "config_global", None):
        args.config = args.config_global
    if getattr(args, "db_global", None):
        args.db = args.db_global
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
