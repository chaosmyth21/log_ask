"""文件监听：把"目录里发生了改动"变成"触发一次增量摄录"。

两种后端：
  - watchdog（ReadDirectoryChangesW / inotify）：有就用；
  - 轮询快照比对：没装 watchdog 时自动退化，行为等价（略耗 IO）。

两者共用同一套"稳定窗口"逻辑：文件大小/mtime 连续 N 秒不再变化才算写完，
避免 agent 读到半个日志。一次运行往往会同时产生多个文件（记录 + diff + 输出表），
所以还要一个"攒批窗口"把它们合并成一次摄录。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

from . import parsing
from .capture import tracked

WATCH_SUFFIX = parsing.TEXT_SUFFIX | parsing.IMAGE_SUFFIX | {".diff"}


def _eligible(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    if path.suffix.lower() not in WATCH_SUFFIX:
        return False
    return tracked(rel)


class Watcher:
    def __init__(self, root: Path | str, on_batch: Callable[[list[Path]], None],
                 settle: float = 2.0, batch_window: float = 6.0,
                 interval: float = 1.5, verbose: bool = True):
        self.root = Path(root)
        self.on_batch = on_batch
        self.settle = settle
        self.batch_window = batch_window
        self.interval = interval
        self.verbose = verbose
        self._pending: dict[Path, tuple[float, tuple[int, float]]] = {}
        self._ready: dict[Path, None] = {}
        self._batch_since: float | None = None
        self._snap: dict[Path, tuple[int, float]] = self._snapshot()
        self.stats = {"events": 0, "batches": 0}

    # ---------------- 快照与事件 ----------------
    def _snapshot(self) -> dict[Path, tuple[int, float]]:
        snap: dict[Path, tuple[int, float]] = {}
        if not self.root.exists():
            return snap
        for p in self.root.rglob("*"):
            if p.is_file() and _eligible(p, self.root):
                try:
                    st = p.stat()
                except OSError:
                    continue
                snap[p] = (st.st_size, st.st_mtime)
        return snap

    def _mark(self, p: Path) -> None:
        try:
            st = p.stat()
        except OSError:
            return
        self._pending[p] = (time.time(), (st.st_size, st.st_mtime))

    def poll_once(self) -> list[Path]:
        """轮询：与上一次快照比对，返回"已稳定"的文件。"""
        cur = self._snapshot()
        for p, sig in cur.items():
            if self._snap.get(p) != sig:
                self._mark(p)
        self._snap = cur
        return self._settle()

    def _settle(self) -> list[Path]:
        now = time.time()
        done = []
        for p, (t0, sig) in list(self._pending.items()):
            try:
                cur = (p.stat().st_size, p.stat().st_mtime)
            except OSError:
                self._pending.pop(p, None)
                continue
            if cur != sig:                       # 还在写
                self._pending[p] = (now, cur)
                continue
            if now - t0 >= self.settle:          # 稳定够久
                done.append(p)
                self._pending.pop(p, None)
        for p in done:
            self._ready[p] = None
            self.stats["events"] += 1
        return done

    # ---------------- 攒批 ----------------
    def _flush(self) -> list[Path] | None:
        if not self._ready:
            self._batch_since = None
            return None
        now = time.time()
        if self._batch_since is None:
            self._batch_since = now
        if now - self._batch_since < self.batch_window:
            return None
        items = sorted(self._ready)
        self._ready.clear()
        self._batch_since = None
        self.stats["batches"] += 1
        return items

    # ---------------- 主循环 ----------------
    def run(self, duration: float | None = None, once: bool = False) -> None:
        observer = self._start_watchdog()
        started = time.time()
        try:
            while True:
                if observer is None:
                    self.poll_once()
                else:
                    self._settle()
                batch = self._flush()
                if batch:
                    if self.verbose:
                        print(f"\n[watch] 检测到 {len(batch)} 个文件稳定，开始摄录…")
                    self.on_batch(batch)
                if once:
                    pending = [p for p in list(self._pending) + list(self._ready)]
                    if not pending:
                        break
                if duration and time.time() - started >= duration:
                    break
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n[watch] 收到中断，退出。")
        finally:
            if observer is not None:
                observer.stop()
                observer.join(timeout=3)

    def _start_watchdog(self):
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            if self.verbose:
                print("[watch] 未安装 watchdog，使用轮询后端（行为等价，略耗 IO）")
            return None

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                self._touch(event)

            def on_modified(self, event):
                self._touch(event)

            def on_moved(self, event):
                self._touch(event, getattr(event, "dest_path", None))

            def _touch(self, event, path=None):
                if getattr(event, "is_directory", False):
                    return
                p = Path(path or event.src_path)
                if p.is_file() and _eligible(p, watcher.root):
                    watcher._mark(p)

        obs = Observer()
        obs.schedule(_Handler(), str(self.root), recursive=True)
        obs.start()
        if self.verbose:
            print(f"[watch] watchdog 后端已启动，监听 {self.root}")
        return obs
