"""配置：一个 TOML 文件 + 环境变量覆盖 + 命令行最高优先。

优先级（低的被高的覆盖）：
    内置默认  <  labmemory.toml  <  环境变量(LM_*)  <  命令行参数
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

try:                                    # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:             # pragma: no cover - 兜底解析
    _toml = None

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_NAME = "labmemory.toml"
EXAMPLE_NAME = "labmemory.toml.example"


# ------------------------------ 结构 ------------------------------

@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    timeout: float = 60.0
    offline: bool = False               # true = 强制规则策略，不调模型


@dataclass
class PathConfig:
    data_dir: str = "./demo_data/logs/test1"  # demo/make_demo_data.py 第一次生成的实验数据目录
    logs_root: str = "./demo_data/logs"       # APP 齿轮从中挑选 test1 / test2 的父目录
    db: str = "labmemory.db"


@dataclass
class AgentConfig:
    max_turns_ingest: int = 20
    max_turns_query: int = 10
    max_calls: int = 40
    max_result_chars: int = 5000


@dataclass
class ScanConfig:
    hours: float = 72.0                 # 只看多少小时内的改动


@dataclass
class WatchConfig:
    settle: float = 2.0                 # 文件多久不变才算写完（秒）
    batch_window: float = 6.0           # 攒批窗口（秒）
    interval: float = 1.5               # 轮询间隔（秒）
    duration: float = 0.0               # 0 = 常驻


@dataclass
class CaptureConfig:
    enabled: bool = True                # 解释器钩子是否采集


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    watch: WatchConfig = field(default_factory=WatchConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    source: str = "(内置默认值)"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source", None)
        return d

    def describe(self) -> str:
        key = self.llm.api_key
        masked = f"{key[:6]}…{key[-4:]}" if len(key) > 12 else ("(未设置)" if not key else "***")
        rows = [
            f"配置文件：{self.source}",
            "",
            "[llm]",
            f"  api_key     = {masked}",
            f"  base_url    = {self.llm.base_url}",
            f"  model       = {self.llm.model}",
            f"  temperature = {self.llm.temperature}",
            f"  timeout     = {self.llm.timeout}",
            f"  offline     = {self.llm.offline}",
            "",
            "[paths]",
            f"  data_dir = {self.paths.data_dir}",
            f"  logs_root = {self.paths.logs_root or '(未设置，使用 data_dir)'}",
            f"  db       = {self.paths.db}",
            "",
            "[agent]",
            f"  max_turns_ingest = {self.agent.max_turns_ingest}",
            f"  max_turns_query  = {self.agent.max_turns_query}",
            f"  max_calls        = {self.agent.max_calls}",
            f"  max_result_chars = {self.agent.max_result_chars}",
            "",
            "[scan]",
            f"  hours = {self.scan.hours}",
            "",
            "[watch]",
            f"  settle = {self.watch.settle}   batch_window = {self.watch.batch_window}"
            f"   interval = {self.watch.interval}   duration = {self.watch.duration}",
            "",
            "[capture]",
            f"  enabled = {self.capture.enabled}",
        ]
        return "\n".join(rows)


# ------------------------------ 查找与解析 ------------------------------

def find_config(explicit: str | Path | None = None) -> Path | None:
    """显式路径 > 环境变量 LM_CONFIG > 仓库根 labmemory.toml > 当前目录 labmemory.toml"""
    if explicit:
        return Path(explicit).expanduser()          # 存在与否交给 load() 判断并报错
    env = os.environ.get("LM_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    for cand in (PROJECT / DEFAULT_NAME, Path.cwd() / DEFAULT_NAME):
        if cand.exists():
            return cand
    return None


def _mini_toml(text: str) -> dict:
    """极简 TOML 子集（供 Python < 3.11）：只支持本配置用到的语法。"""
    data: dict = {}
    cur = data
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = data
            for part in line[1:-1].split("."):
                cur = cur.setdefault(part.strip(), {})
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        cur[key.strip()] = _parse_value(val.strip())
    return data


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out)


def _parse_value(v: str):
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_parse_value(x.strip()) for x in inner.split(",")] if inner else []
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1].replace("\\\\", "\\")
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return float(v) if re.search(r"[.eE]", v) else int(v)
    except ValueError:
        return v


def _merge(obj, data: dict):
    """把 dict 合并进 dataclass 实例，未知键忽略（并在 source 里提示）。"""
    if not is_dataclass(obj):
        return obj
    for key, val in (data or {}).items():
        if not hasattr(obj, key):
            continue
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val)
        elif isinstance(val, dict):
            continue
        else:
            try:
                setattr(obj, key, type(cur)(val) if not isinstance(cur, str) else str(val))
            except (TypeError, ValueError):
                setattr(obj, key, val)
    return obj


def _double_backslashes(raw: str) -> str:
    r"""把双引号字符串里"落单"的反斜杠补成两个。

    Windows 路径是 TOML 的头号坑：``"D:\\lab"`` 合法，``"D:\lab"`` 非法（``\l`` 不是合法转义），
    而所有人第一次都会写成后者。这里做一次兜底重写，让两种写法都能用。
    """
    out: list[str] = []
    in_str = False
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch == '"':
            escaped = i > 0 and raw[i - 1] == "\\"
            if not escaped:
                in_str = not in_str
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and in_str:
            if i + 1 < n and raw[i + 1] == "\\":
                out.append("\\\\")
                i += 2
            else:
                out.append("\\\\")
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load(explicit: str | Path | None = None, strict: bool = True) -> Config:
    """读取配置。strict=False 时解析失败会退回默认值（钩子必须用这个，不能影响宿主程序）。"""
    cfg = Config()
    path = find_config(explicit)
    if path and path.exists():
        raw = path.read_text("utf-8-sig")            # 顺手吃掉 Windows 编辑器加的 BOM
        data = None
        errors = []
        for attempt in (raw, _double_backslashes(raw)):
            try:
                data = _toml.loads(attempt) if _toml else _mini_toml(attempt)
                break
            except Exception as e:
                errors.append(str(e))
        if data is None:
            if not strict:
                cfg.source = f"{path}(解析失败，已用默认值)"
                _apply_env(cfg)
                return cfg
            raise SystemExit(
                f"配置文件解析失败：{path}\n  {errors[-1]}\n"
                f"提示：Windows 路径里的反斜杠写两个（D:\\\\lab\\\\)，或直接用正斜杠（D:/lab）。")
        _merge(cfg, data)
        cfg.source = str(path)
    elif explicit:
        raise SystemExit(f"找不到配置文件：{explicit}\n"
                         f"先运行 `python main.py config --init --path {explicit}` 生成，或检查路径")
    _apply_env(cfg)
    return cfg


def _apply_env(cfg: Config) -> None:
    """环境变量覆盖文件（命令行仍可再覆盖）。"""
    env = os.environ
    key = env.get("LM_API_KEY") or env.get("LLM_API_KEY")
    if key:
        cfg.llm.api_key = key
        cfg.source = f"{cfg.source} + 环境变量"
    if env.get("LM_BASE_URL"):
        cfg.llm.base_url = env["LM_BASE_URL"]
    if env.get("LM_MODEL"):
        cfg.llm.model = env["LM_MODEL"]
    if env.get("LM_OFFLINE"):
        cfg.llm.offline = env["LM_OFFLINE"].lower() in ("1", "true", "yes", "on")
    if env.get("LM_DATA"):
        cfg.paths.data_dir = env["LM_DATA"]
    if env.get("LM_DB"):
        cfg.paths.db = env["LM_DB"]
    if env.get("LM_TEMPERATURE"):
        try:
            cfg.llm.temperature = float(env["LM_TEMPERATURE"])
        except ValueError:
            pass


def write_default(target: Path, force: bool = False) -> Path:
    """从模板生成配置文件。"""
    target = Path(target)
    if target.exists() and not force:
        raise SystemExit(f"{target} 已存在，未覆盖（要覆盖加 --force）")
    src = PROJECT / EXAMPLE_NAME
    if not src.exists():
        raise SystemExit(f"缺少模板：{src}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(src.read_text("utf-8"), encoding="utf-8")
    return target
