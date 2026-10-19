"""运行采集：在没有日志文件的场景下，重建"这一次跑了什么"。

三个通道，按可靠性从高到低：
  A. 包装运行（labmemory run -- cmd）：argv / 退出码 / 耗时 / stdout / 参数变化，全都有；
  B. 解释器级 tee（sitecustomize 钩子）：不改启动方式，IDE 里按 F5 也能自动落盘；
  C. 外部工具自己写日志：什么都不用做，走原有 scan 流程。

关键设计：**没有参数日志时，代码本身就是参数台账。**
每次运行开始时对代码做一次参数抽取，与上一次运行的快照比对，
得到 "lr 0.0005 -> 0.001" 这种变更记录 —— 比日志里手写的超参更可信，因为代码是唯一事实来源。
"""

from __future__ import annotations

import atexit
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RUN_DIR = ".labmemory/runs"
STATE_DIR = ".labmemory"
IGNORE_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".idea",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".ipynb_checkpoints",
    ".DS_Store",
    # APP 生成的周报目录，不能再被 scan 当成实验笔记摄入
    "weekly",
}
CODE_SUFFIX = {".py", ".yaml", ".yml", ".toml", ".json", ".sh", ".ipynb"}


def tracked(rel: Path) -> bool:
    """这个相对路径算不算"实验痕迹"。

    `.labmemory/` 默认不扫（里面有状态文件），唯一例外是 `runs/` 下的运行记录 ——
    那是采集钩子产出的、最原始的实验数据，必须能被摄录。
    """
    parts = rel.parts
    if not parts:
        return False
    if parts[0] == STATE_DIR:
        return (len(parts) >= 3 and parts[1] == "runs"
                and rel.suffix.lower() in {".log", ".diff"})
    if any(p in IGNORE_DIRS or p.startswith(".") for p in parts[:-1]):
        return False
    return not parts[-1].startswith(".")


def tracked_dir(rel: Path) -> bool:
    """目录能不能进去看。跟 tracked() 的区别：`.labmemory` 允许进（里面有 runs/）。"""
    parts = rel.parts
    if not parts:
        return True
    if parts[0] == STATE_DIR:
        return True
    return not any(p in IGNORE_DIRS or p.startswith(".") for p in parts)

# 只关心这些东西的赋值；其余变量不记录，避免把业务代码里的数字全拖进来
_INTEREST = re.compile(
    r"(lr|learning_rate|batch|bs|epoch|seed|dropout|warmup|weight_decay|"
    r"max_len|seq_len|max_seq_len|top_p|temperature|rank|alpha|gamma|"
    r"n_layers|hidden|dim|steps|k\b)", re.I)
_ASSIGN = re.compile(r"^\s*(?:self\.)?([A-Za-z_][\w.]{0,31})\s*=\s*(.+?)\s*(?:#.*)?$")
_LITERAL = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


# ------------------------------ 参数抽取 ------------------------------

def _literal(raw: str):
    v = raw.strip().strip("'\"")
    if _LITERAL.match(v):
        try:
            return float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            return None
    if v in ("True", "False"):
        return v == "True"
    return None


def collect_params(root: Path, max_files: int = 200) -> dict[str, dict]:
    """扫代码文件里"看起来是超参"的赋值：{相对路径: {键: 值}}。"""
    out: dict[str, dict] = {}
    n = 0
    for p in sorted(Path(root).rglob("*")):
        if n >= max_files:
            break
        if not p.is_file() or p.suffix not in CODE_SUFFIX:
            continue
        if any(part in IGNORE_DIRS or part.startswith(".") for part in p.relative_to(root).parts):
            continue
        try:
            if p.stat().st_size > 2_000_000:
                continue
            text = p.read_text("utf-8", errors="replace")
        except OSError:
            continue
        found: dict = {}
        for line in text.splitlines()[:4000]:
            m = _ASSIGN.match(line)
            if not m:
                continue
            key = m.group(1).split(".")[-1]
            if not _INTEREST.search(key):
                continue
            val = _literal(m.group(2))
            if val is not None:
                found[key] = val
        if found:
            out[str(p.relative_to(root)).replace("\\", "/")] = found
            n += 1
    return out


def diff_params(old: dict, new: dict) -> list[str]:
    """返回 ['train.py: lr 0.0005 -> 0.001', ...]"""
    rows = []
    for f, kv in sorted(new.items()):
        prev = old.get(f) or {}
        for k, v in kv.items():
            if k in prev and prev[k] != v:
                rows.append(f"{f}: {k} {prev[k]} -> {v}")
            elif k not in prev and old:
                rows.append(f"{f}: {k} = {v}(新增)")
    for f, kv in sorted(old.items()):
        if f not in new:
            rows.append(f"{f}: 整个文件不再被扫描(可能被删或改名)")
    return rows


def flatten_params(params: dict) -> dict:
    """多文件的同名参数以最后一次出现为准，取成一张扁平表。"""
    flat: dict = {}
    for kv in params.values():
        flat.update(kv)
    return flat


# ------------------------------ 进程输出 tee ------------------------------

class _Tee(io.TextIOBase):
    """把 stdout/stderr 同时写到原流和文件。

    多进程护栏：fork 出来的 DataLoader worker 会继承同一个文件句柄，
    只允许创建时那个 PID 写文件，其余进程仍然正常输出到终端。
    """

    def __init__(self, original, handle, owner_pid: int):
        self._orig = original
        self._file = handle
        self._pid = owner_pid

    def write(self, s):
        try:
            self._orig.write(s)
        except (ValueError, OSError):
            pass
        if os.getpid() == self._pid:
            try:
                self._file.write(s)
                self._file.flush()
            except (ValueError, OSError):
                pass
        return len(s)

    def flush(self):
        for target in (self._orig, self._file):
            try:
                target.flush()
            except (ValueError, OSError, AttributeError):
                pass

    @property
    def encoding(self):          # 某些库会读这个属性
        return getattr(self._orig, "encoding", "utf-8")


# ------------------------------ 运行记录 ------------------------------

def _git(root: Path, *args: str) -> str:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=10)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def snapshot_diff(root: Path, limit: int = 12000) -> str:
    """优先 git diff（能拿到"改了哪几行"）；不是 git 仓库就返回空，靠参数快照兜底。"""
    body = _git(root, "diff", "--stat")
    body += "\n" + _git(root, "diff", "-U2")
    if not body.strip():
        body = _git(root, "diff", "-U2", "HEAD")
    return body[:limit]


def record_path(root: Path, suffix: str = ".log") -> Path:
    d = Path(root) / RUN_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}{suffix}"


def write_header(root: Path, argv: list[str], started: datetime) -> tuple[Path, dict]:
    """写入运行记录的头部（参数快照 + 变更 + git diff），返回记录文件与上下文。"""
    root = Path(root)
    state = root / STATE_DIR / "params.json"
    prev = {}
    if state.exists():
        try:
            prev = json.loads(state.read_text("utf-8"))
        except json.JSONDecodeError:
            prev = {}
    cur = collect_params(root)
    changes = diff_params(prev, cur)
    flat = flatten_params(cur)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(cur, ensure_ascii=False, indent=1), "utf-8")

    path = record_path(root)
    head = [
        "# ==== LABMEMORY RUN RECORD ====",
        f"# argv: {' '.join(argv) or '(unknown)'}",
        f"# cwd: {root}",
        f"# started: {started.isoformat(timespec='seconds')}",
        f"# params(new): " + (", ".join(f"{k}={v}" for k, v in sorted(flat.items())) or "(未识别到超参赋值)"),
        f"# changes: " + (" | ".join(changes) if changes else "(与上次相比无参数变化)"),
        f"# files_changed: " + (", ".join(sorted(cur)) if cur else "-"),
    ]
    diff = snapshot_diff(root)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(head) + "\n")
        if diff:
            f.write("# git_diff_stat: 见同名 .diff 文件\n")
        f.write("# ==== stdout ====\n")
    if diff:
        record_path(root, ".diff").write_text(diff, encoding="utf-8")
    return path, {"flat": flat, "changes": changes}


def write_footer(path: Path, exit_code: int, started: datetime) -> None:
    dur = (datetime.now() - started).total_seconds()
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n# ==== END exit={exit_code} duration={dur:.1f}s "
                    f"ended={datetime.now().isoformat(timespec='seconds')} ====\n")
    except OSError:
        pass


# ------------------------------ 模式 A：包装运行 ------------------------------

def run_wrapped(root: Path, cmd: list[str]) -> int:
    root = Path(root)
    started = datetime.now()
    path, _ = write_header(root, cmd, started)
    with path.open("a", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in proc.stdout:          # 边跑边看，同时落盘
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
        except KeyboardInterrupt:
            proc.terminate()
            f.write("\n# 用户中断\n")
        finally:
            proc.stdout.close()
        code = proc.wait()
    write_footer(path, code, started)
    print(f"\n[labmemory] 运行记录 → {path}")
    return code


# ------------------------------ 模式 B：解释器级钩子 ------------------------------

def find_project_root(start: Path | None = None) -> Path | None:
    """从 cwd 和主脚本所在目录分别向上找 `.labmemory/`。

    只看 cwd 不够：`python D:\\proj\\train.py` 从别的目录启动时 cwd 不是项目根；
    IDE 调试时 cwd 通常等于 workspaceFolder，但脚本目录更接近"项目"这个概念。
    """
    cands: list[Path] = []
    if start:
        cands.append(Path(start))
    cands.append(Path.cwd())
    argv0 = sys.argv[0] if sys.argv and sys.argv[0] else ""
    if argv0 and not argv0.startswith("-"):
        try:
            cands.append(Path(argv0).resolve().parent)
        except OSError:
            pass
    for c in cands:
        cur = c.resolve()
        for p in [cur, *list(cur.parents)[:7]]:
            try:
                if (p / STATE_DIR).is_dir():
                    return p
            except OSError:
                continue
    return None


def install_hook(root: Path | None = None) -> Path | None:
    """把当前解释器的 stdout/stderr 接一份到运行记录。失败必须静默，不能影响用户程序。"""
    try:
        root = root or find_project_root()
        if not root:
            return None
        # 配置里可以整体关掉采集（capture.enabled = false）。
        # 注意 strict=False 且吞 BaseException：钩子运行时 Python 还在初始化，
        # 这里抛出任何东西（包括 SystemExit）都会让用户的解释器直接起不来。
        try:
            from . import config
            if not config.load(strict=False).capture.enabled:
                return None
        except BaseException:
            pass
        started = datetime.now()
        path, _ = write_header(root, [sys.executable, *sys.argv], started)
        handle = path.open("a", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, handle, os.getpid())
        sys.stderr = _Tee(sys.stderr, handle, os.getpid())

        def _hook(exc_type, exc, tb):
            import traceback
            traceback.print_exception(exc_type, exc, tb)
            write_footer(path, 1, started)
            atexit._run_exitfuncs()          # 让 IDE 的调试器看到完整栈
        sys.excepthook = _hook
        atexit.register(lambda: (write_footer(path, 0, started), handle.close()))
        return path
    except Exception:                         # 钩子绝不允许打断用户程序
        return None
