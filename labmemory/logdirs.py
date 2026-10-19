"""日志数据集（logs 下的子文件夹）的发现、校验与选择持久化。

APP 右上角齿轮只负责修改“读哪个文件夹”，真正的路径解析放在这里：

    logs/
      test1/     <- 当前实验数据
      test2/     <- 另一组实验数据

选择结果写入 ``.labmemory/ui.json``，下次启动 APP 仍然生效。
所有用户输入都会被收敛到 logs 根目录内部，避免通过目录名越界读取。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

STATE_KEY = "log_dir"
PREFERRED = ("test1", "test2")


class LogDirError(ValueError):
    """日志目录不存在、不可用或试图越界时抛出。"""


def list_dirs(root: str | Path) -> list[Path]:
    """列出 logs 根目录下可选的子文件夹（忽略隐藏目录）。"""
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    try:
        entries = list(base.iterdir())
    except OSError:
        return []
    return sorted(
        (p for p in entries if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )


def names(root: str | Path) -> list[str]:
    return [p.name for p in list_dirs(root)]


def resolve(root: str | Path, name: str) -> Path:
    """把用户选择的文件夹名解析成绝对路径，并确认它仍在 logs 根目录内。"""
    name = (name or "").strip()
    if not name or name in (".", ".."):
        raise LogDirError("请选择一个日志文件夹")
    if "/" in name or "\\" in name:
        raise LogDirError("只能选择 logs 根目录下的一级文件夹")
    base = Path(root).expanduser().resolve()
    target = (base / name).resolve()
    if base != target and base not in target.parents:
        raise LogDirError(f"目录越界：{name}")
    if not target.is_dir():
        raise LogDirError(f"日志文件夹不存在：{target}")
    return target


def load_selected(state_file: str | Path) -> str | None:
    """读取上次选择；损坏或缺失时返回 None，不影响 APP 启动。"""
    path = Path(state_file)
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get(STATE_KEY) if isinstance(data, dict) else None
    return value if isinstance(value, str) and value.strip() else None


def save_selected(state_file: str | Path, name: str) -> None:
    """持久化选择。先写临时文件再替换，避免并发读到半个 JSON。"""
    path = Path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    if path.exists():
        try:
            old = json.loads(path.read_text("utf-8"))
            if isinstance(old, dict):
                payload = old
        except (OSError, ValueError):
            payload = {}
    payload[STATE_KEY] = name
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    os.replace(tmp, path)


def current(root: str | Path, state_file: str | Path) -> Path:
    """返回当前应读取的日志文件夹。

    优先级：上次保存的选择（仍存在）> test1 > test2 > 名称排序后的第一个。
    """
    base = Path(root).expanduser()
    available = list_dirs(base)
    if not available:
        raise LogDirError(f"{base} 下没有可读取的日志文件夹，请在 logs 里新建 test1 / test2")

    saved = load_selected(state_file)
    if saved:
        try:
            return resolve(base, saved)
        except LogDirError:
            pass
    by_name = {p.name: p for p in available}
    for name in PREFERRED:
        if name in by_name:
            return by_name[name].resolve()
    return by_name[sorted(by_name, key=str.lower)[0]].resolve()


def select(root: str | Path, state_file: str | Path, name: str) -> Path:
    """校验并保存新的日志文件夹，返回其绝对路径。"""
    target = resolve(root, name)
    save_selected(state_file, target.name)
    return target
