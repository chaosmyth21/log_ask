"""工具层：Agent 能动用的全部动作。

每个工具都有一个"何时调用"的说明（description），这是 Agent 自主性的来源 ——
模型靠它决定下一步做什么，而不是靠代码里的 if/else 顺序。

约定：工具**永不抛异常**，失败一律返回 {"ok": false, "error": "..."}，
让模型看到错误并自行改策略；这比崩掉整个 loop 更接近真实 Agent 行为。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import capture, parsing
from .store import Store


@dataclass
class Ctx:
    """工具运行时上下文：数据根目录、库、已处理文件清单、预算计数。"""
    root: Path
    store: Store
    state_file: Path
    interactive: bool = False
    calls: dict[str, int] = field(default_factory=dict)
    # 本次运行里被列出 / 被 inspect / 已进卡片的文件，用来做"覆盖自检与补漏"
    listed: set[str] = field(default_factory=set)
    inspected: set[str] = field(default_factory=set)
    saved: set[str] = field(default_factory=set)
    limit: dict[str, int] = field(default_factory=lambda: {
        "read_slice": 12, "inspect_file": 20, "list_files": 8,
        "search_cards": 10, "get_card": 12, "save_card": 24,
        "compare_files": 4, "ask_user": 2,
    })

    def bump(self, name: str) -> bool:
        n = self.calls.get(name, 0) + 1
        self.calls[name] = n
        return n <= self.limit.get(name, 10)


def _rel(ctx: Ctx, p: str) -> Path | None:
    """只允许访问数据根目录内的文件，防止模型乱读盘。"""
    try:
        path = Path(p)
        if not path.is_absolute():
            path = ctx.root / path
        path = path.resolve()
        root = ctx.root.resolve()
        if root != path and root not in path.parents:
            return None
        return path
    except (OSError, ValueError):
        return None


def _ingested(ctx: Ctx) -> dict:
    if not ctx.state_file.exists():
        return {}
    try:
        return json.loads(ctx.state_file.read_text("utf-8"))
    except json.JSONDecodeError:
        return {}


def _mark(ctx: Ctx, path: Path) -> None:
    data = _ingested(ctx)
    data[str(path)] = {"mtime": path.stat().st_mtime, "size": path.stat().st_size,
                       "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ctx.state_file.parent.mkdir(parents=True, exist_ok=True)
    ctx.state_file.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")


# ------------------------------- 工具实现 -------------------------------

MAX_VISITED = 20000          # 单次遍历的硬上限，防止指向盘符根时卡死


def _count_candidates(d: Path, cap: int = 200) -> int:
    """数一下被剪枝的目录里大概有多少候选文件（有上限，不会为了数数遍历全树）。"""
    n = 0
    stack = [d]
    while stack and n < cap:
        cur = stack.pop()
        try:
            entries = list(os.scandir(cur))
        except OSError:
            continue
        for e in entries:
            if n >= cap:
                break
            if e.is_dir():
                stack.append(Path(e.path))
            elif Path(e.path).suffix.lower() in parsing.TEXT_SUFFIX | parsing.IMAGE_SUFFIX:
                n += 1
    return n


def list_files(ctx: Ctx, since_hours: float = 24, limit: int = 40, include_done: bool = False,
               subdir: str | None = None, max_depth: int | None = None) -> dict:
    """列候选实验文件。

    两个参数是为了"别为了看一眼就遍历整棵树"：
      - max_depth：从本次起点往下最多看几层，更深的目录只报个数不进去；
      - subdir：被剪枝的目录用它下钻（深度从新的起点重新算）。
    遍历用 os.scandir 手写而不是 rglob，才能在越界前就把整棵子树剪掉。
    """
    data_root = ctx.root
    if not data_root.exists():
        return {"ok": False, "error": f"目录不存在：{data_root}"}
    if subdir:
        base = _rel(ctx, subdir)
        if not base or not base.is_dir():
            return {"ok": False, "error": f"子目录不可读或越界：{subdir}"}
    else:
        base = data_root
    max_depth = 4 if max_depth is None else int(max_depth)
    cutoff = datetime.now().timestamp() - float(since_hours) * 3600
    done = _ingested(ctx)

    items: list[dict] = []
    pruned: dict[str, int] = {}
    visited = 0
    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            visited += 1
            if visited > MAX_VISITED:
                break
            p = Path(e.path)
            try:
                rel_base = p.relative_to(base)
                rel_root = p.relative_to(data_root)
            except ValueError:
                continue
            if e.is_dir():
                if not capture.tracked_dir(rel_root):
                    continue
                # 目录深度从本次起点算：进到第 max_depth 层就停，更深的只报个数
                if len(rel_base.parts) + 1 > max_depth:
                    pruned[str(rel_base).replace("\\", "/")] = _count_candidates(p)
                else:
                    stack.append((p, depth + 1))
                continue
            if not capture.tracked(rel_root):
                continue
            if len(rel_base.parts) > max_depth:        # 文件本身超过深度上限
                key = str(rel_base.parent).replace("\\", "/")
                pruned[key] = pruned.get(key, 0) + 1
                continue
            if p.suffix.lower() not in parsing.TEXT_SUFFIX | parsing.IMAGE_SUFFIX:
                continue
            # .diff 是 .log 运行记录的辅助证据：成对出现时只把 .log 当实验文件，
            # 避免同一次运行被扫描成两张卡。没有同名 .log 时才单独保留 .diff。
            if p.suffix.lower() == ".diff" and p.with_suffix(".log").exists():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime < cutoff:
                continue
            rec = done.get(str(p))
            changed = not rec or rec["mtime"] != st.st_mtime or rec["size"] != st.st_size
            if not changed and not include_done:
                continue
            items.append({
                "path": str(p), "name": p.name, "kind": parsing.classify(p),
                "bytes": st.st_size, "_mtime": st.st_mtime,
                "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "already_processed": not changed,
            })

    # 精确 mtime 倒序；同秒的文件按路径升序 —— 否则"哪个 run 先入库"会随机，
    # supersede 的方向（谁覆盖谁）就跟着随机。
    items.sort(key=lambda x: (-x["_mtime"], x["path"]))
    for it in items:
        it.pop("_mtime", None)
        ctx.listed.add(it["path"])
    result = {
        "ok": True, "root": str(base), "count": len(items), "files": items[: int(limit)],
        "visited": visited,
        "hint": "对新文件先 inspect_file；已处理的文件除非 suspected 变了否则不必重读",
    }
    if pruned:
        result["pruned"] = [{"path": k, "candidates": v} for k, v in sorted(pruned.items())[:20]]
        result["hint"] += ("；另有 {} 个目录因 max_depth={} 没进去（见 pruned），"
                           "需要时用 subdir 参数下钻，比如 list_files(subdir='logs/2026-10')"
                           .format(len(pruned), max_depth))
    if visited > MAX_VISITED:
        result["truncated_scan"] = True
        result["hint"] += f"；遍历已达 {MAX_VISITED} 个条目上限，结果可能不全，请用 subdir 缩小范围"
    return result


def inspect_file(ctx: Ctx, path: str) -> dict:
    p = _rel(ctx, path)
    if not p or not p.exists():
        return {"ok": False, "error": f"文件不可读或越界：{path}"}
    ctx.inspected.add(str(p))
    if parsing.classify(p) == "image":
        return {"ok": True, "path": str(p), "kind": "image", "note": "图片类无文本指标；可仅在卡片中登记其存在与文件名语义"}
    return {"ok": True, **parsing.inspect(p)}


def read_slice(ctx: Ctx, path: str, start_line: int = 1, max_lines: int = 80) -> dict:
    p = _rel(ctx, path)
    if not p or not p.exists():
        return {"ok": False, "error": f"文件不可读或越界：{path}"}
    if parsing.classify(p) == "image":
        return {"ok": False, "error": "图片文件无文本行，请用 inspect_file"}
    return {"ok": True, **parsing.read_slice(p, start_line, max_lines)}


def compare_files(ctx: Ctx, path_a: str, path_b: str) -> dict:
    a, b = _rel(ctx, path_a), _rel(ctx, path_b)
    if not a or not b or not a.exists() or not b.exists():
        return {"ok": False, "error": "任一文件不可读"}
    sa, sb = parsing.iter_metrics(a), parsing.iter_metrics(b)
    keys = sorted(set(sa) | set(sb))
    rows = []
    for k in keys:
        if k.startswith("__"):
            continue
        qa = parsing.series_stats(k, sa.get(k, []))
        qb = parsing.series_stats(k, sb.get(k, []))
        if not qa.get("n") and not qb.get("n"):
            continue
        rows.append({
            "metric": k,
            "a_last": qa.get("last"), "b_last": qb.get("last"),
            "a_best": qa.get("best"), "b_best": qb.get("best"),
            "delta_last": (round(qb["last"] - qa["last"], 6)
                           if qa.get("last") is not None and qb.get("last") is not None else None),
            "better": _better(k, qa, qb),
        })
    return {"ok": True, "a": a.name, "b": b.name, "rows": rows[:12]}


def _better(name: str, qa: dict, qb: dict) -> str | None:
    if qa.get("last") is None or qb.get("last") is None:
        return None
    hib = parsing._higher_is_better(name)
    if abs(qb["last"] - qa["last"]) < 1e-9:
        return "tie"
    return "b" if (qb["last"] > qa["last"]) == hib else "a"


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def normalize_metrics(metrics) -> tuple[dict, list[str]]:
    """把 metrics 收敛成 {名字: {first,last,best,worst,trend,n}}。

    真模型经常写成 `"loss": 0.641`（要一个数就够），下游一切按统计对象读就会崩。
    这里在唯一的写入口做归一：数字按"单点"处理，并回一条警告让模型下轮自己改。
    """
    out: dict = {}
    warns: list[str] = []
    if not isinstance(metrics, dict):
        return {}, ["metrics 必须是对象 {指标名: {first,last,best,trend,n}}，已忽略"]
    for k, v in metrics.items():
        if isinstance(v, dict):
            out[str(k)] = v
            continue
        num = _num(v)
        if num is None:
            warns.append(f"metrics.{k} 无法解析为数字，已忽略")
            continue
        out[str(k)] = {"first": num, "last": num, "best": num, "worst": num,
                       "trend": "flat", "n": 1}
        warns.append(f"metrics.{k} 应该是统计对象而不是单个数字，已按单点处理；"
                     f"正确写法：{{\"first\":..,\"last\":..,\"best\":..,\"trend\":\"down\",\"n\":120}}")
    return out, warns


def normalize_observations(obs) -> tuple[list[dict], list[str]]:
    """observation 允许写成字符串，补成 {kind,text,evidence} 结构。"""
    out, warns = [], []
    if not isinstance(obs, list):
        return [], []
    for o in obs:
        if isinstance(o, dict):
            out.append(o)
        elif isinstance(o, str) and o.strip():
            out.append({"kind": "note", "text": o.strip(), "evidence": ""})
            warns.append("observations 里有纯字符串，已按 kind=note 处理；建议带上 evidence")
    return out, warns


def save_card(ctx: Ctx, card: dict) -> dict:
    if not isinstance(card, dict):
        return {"ok": False, "error": "card 必须是对象"}
    for req in ("exp_name", "goal", "files"):
        if not card.get(req):
            return {"ok": False, "error": f"缺少必填字段 {req}"}
    if not isinstance(card.get("files"), list):
        return {"ok": False, "error": "files 必须是数组"}
    warns: list[str] = []
    card["metrics"], w = normalize_metrics(card.get("metrics"))
    warns += w
    card["observations"], w = normalize_observations(card.get("observations"))
    warns += w
    if card.get("setup") is not None and not isinstance(card.get("setup"), dict):
        card["setup"] = {}
        warns.append("setup 必须是对象，已清空")
    tags = [str(t) for t in (card.get("tags") or [])]
    card["tags"] = list(dict.fromkeys(t if t.startswith("#") else f"#{t}" for t in tags))
    if card.get("confidence") is not None:
        try:
            card["confidence"] = max(0.0, min(1.0, float(card["confidence"])))
        except (TypeError, ValueError):
            card["confidence"] = 0.5
    # files 必须统一成绝对路径：模型经常给相对路径（logs\xxx.log），
    # 原样存下来会让"这个文件属于哪张卡"在后续无法对齐（覆盖检查、证据核对都会错）。
    resolved: list[str] = []
    for f in card["files"]:
        p = _rel(ctx, str(f))
        if p and p.exists():
            _mark(ctx, p)
            resolved.append(str(p))
        else:
            resolved.append(str(f))
            warns.append(f"files 里的 {f} 在数据目录里找不到，已按原样保留")
    card["files"] = list(dict.fromkeys(resolved))
    ctx.saved.update(str(f) for f in card["files"])
    saved = ctx.store.save_card(card)
    updated = bool(saved.pop("_updated", False))
    ctx.store.log_event("save_card", saved["id"], saved["exp_name"])
    if updated:
        note = "同一份证据已存在，本次是原地更新（卡片 id 不变，不需要再存一遍）"
    elif saved["links"].get("supersedes"):
        note = "新证据比旧卡新，旧卡已标记 superseded 并连成实验链"
    else:
        note = ""
    return {"ok": True, "card_id": saved["id"], "status": saved["status"],
            "updated": updated,
            "supersedes": saved["links"].get("supersedes"),
            "warnings": warns, "note": note}


def search_cards(ctx: Ctx, query: str = "", filters: dict | None = None, k: int = 5) -> dict:
    cards = ctx.store.search(query or "", filters or {}, int(k))
    return {"ok": True, "count": len(cards), "cards": [_brief(c) for c in cards]}


def get_card(ctx: Ctx, card_id: str) -> dict:
    c = ctx.store.to_dict(card_id)
    return {"ok": True, "card": c} if c else {"ok": False, "error": f"无此卡片：{card_id}"}


def ask_user(ctx: Ctx, question: str) -> dict:
    if not ctx.interactive:
        return {"ok": True, "answer": None,
                "note": "非交互环境（演示/批处理），用户未回复；请基于现有证据继续，并把不确定项写进 observations 的 kind=uncertain"}
    print(f"\n[Agent 提问] {question}\n> ", end="", flush=True)
    try:
        ans = input().strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    return {"ok": True, "answer": ans or None}


def _brief(c: dict) -> dict:
    """紧凑卡片视图：只在必要的字段上花 token。

    注意要容忍"老卡片"：早期模型可能把 metrics 写成 `"loss": 0.641`，
    读到这种形状不能让检索直接崩掉。
    """
    m = c.get("metrics") or {}
    items = [(k, v) for k, v in m.items() if isinstance(v, dict) and v.get("n")]
    items.sort(key=lambda kv: (not kv[1].get("higher_is_better"), kv[0]))
    return {
        "id": c["id"], "exp_name": c["exp_name"], "run_id": c["run_id"],
        "created_at": c["created_at"][:10], "status": c["status"],
        "goal": c["goal"],
        "setup": c["setup"],
        "metrics": {k: {"first": v.get("first"), "last": v.get("last"), "best": v.get("best")}
                    for k, v in items[:3]},
        "tags": c["tags"],
        "anomalies": [o.get("kind") for o in c.get("observations") or []
                      if isinstance(o, dict) and o.get("kind") not in (None, "note")],
        "files": [Path(f).name for f in c.get("files") or []],
        "links": c.get("links") or {},
    }


# ------------------------------- 注册表 -------------------------------

TOOLS = {
    "list_files": {
        "fn": list_files,
        "description": "列出数据目录里最近改动过、还没处理过的实验文件。每次开工先调用它；"
                       "返回 already_processed=true 的文件不必重读。"
                       "如果返回里带 pruned（因目录太深被跳过），用 subdir 下钻再看那一层。",
        "parameters": {
            "type": "object",
            "properties": {
                "since_hours": {"type": "number", "description": "回看多少小时内的改动，默认 24"},
                "limit": {"type": "integer", "description": "最多返回多少个文件，默认 40"},
                "include_done": {"type": "boolean", "description": "是否包含已处理但未变动的文件，默认 false"},
                "subdir": {"type": "string", "description": "只看这个子目录（相对于数据目录），用于下钻 pruned 里列出的目录"},
                "max_depth": {"type": "integer", "description": "从本次起点往下最多看几层，默认 4；更深的目录只报个数"},
            },
        },
    },
    "inspect_file": {
        "fn": inspect_file,
        "description": "一眼看清一个文件的骨架：文件类型、行数、抽出的指标统计（first/last/best/trend）、"
                       "规则发现的异常信号（带行号证据）、头尾片段。"
                       "**先调它**，只有需要看具体上下文时才用 read_slice 局部重读。",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径（绝对路径或相对数据目录）"}},
            "required": ["path"],
        },
    },
    "read_slice": {
        "fn": read_slice,
        "description": "按行号分片读取文件的一段（带行号）。用于查看报错栈、确认某个指标出现的上下文、"
                       "核对指标陡降前后的日志。不要用它整份通读大文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "description": "起始行，1-based，默认 1"},
                "max_lines": {"type": "integer", "description": "最多读多少行，默认 80，建议 ≤120"},
            },
            "required": ["path"],
        },
    },
    "compare_files": {
        "fn": compare_files,
        "description": "横向对比两个实验文件的同名指标（末值/最佳值/差值/谁更好）。"
                       "当发现两个 run 只差一个超参、或想解释某次为什么更好/更差时调用。",
        "parameters": {
            "type": "object",
            "properties": {"path_a": {"type": "string"}, "path_b": {"type": "string"}},
            "required": ["path_a", "path_b"],
        },
    },
    "search_cards": {
        "fn": search_cards,
        "description": "检索已有知识卡片。query 传简短关键词（如 'OOM'、'loss 停滞'），不要传整句问题；"
                       "结构化条件放进 filters（exp_name/tag/param_key+param_value/status/days）。"
                       "写新卡片前先搜一遍，避免重复建卡。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "简短关键词，可空"},
                "filters": {"type": "object", "description": "exp_name / tag / status / days / param_key / param_value"},
                "k": {"type": "integer", "description": "返回条数，默认 5"},
            },
        },
    },
    "get_card": {
        "fn": get_card,
        "description": "读取一张卡片的完整内容（含观测细节、结论、来源文件、实验链 links）。",
        "parameters": {
            "type": "object",
            "properties": {"card_id": {"type": "string"}},
            "required": ["card_id"],
        },
    },
    "save_card": {
        "fn": save_card,
        "description": "落库一张知识卡片。这是唯一的写操作。"
                       "同一 exp_name + run_id 重复保存会自动把旧卡标记 superseded 并把两卡用 links 串起来。",
        "parameters": {
            "type": "object",
            "properties": {
                "card": {
                    "type": "object",
                    "description": "卡片对象",
                    "properties": {
                        "exp_name": {"type": "string", "description": "实验/任务名，同组实验必须同名"},
                        "run_id": {"type": "string", "description": "run 标识，如 run3 / 20261012-1912"},
                        "goal": {"type": "string", "description": "这次实验想验证什么，一句话中文"},
                        "setup": {"type": "object", "description": "超参与环境，如 {lr:0.001, batch:32, model:'qwen3-7b'}"},
                        "metrics": {"type": "object", "description": "从 inspect_file 的 stats 原样摘取关键指标，不要自己算"},
                        "observations": {"type": "array", "description": "[{kind, text, evidence}]，kind 用 crash/regression/stagnation/divergence/note/uncertain"},
                        "conclusion": {"type": "string", "description": "结论或待验证的下一步"},
                        "tags": {"type": "array", "description": "标签，如 ['#lr', '#OOM']"},
                        "status": {"type": "string", "description": "active | failed | needs_review"},
                        "confidence": {"type": "number", "description": "0-1，信息不足就给低分"},
                        "files": {"type": "array", "description": "来源文件路径"},
                    },
                    "required": ["exp_name", "goal", "files"],
                }
            },
            "required": ["card"],
        },
    },
    "ask_user": {
        "fn": ask_user,
        "description": "当关键信息（如超参、实验目的）无法从文件推出、且会显著影响卡片质量时，向用户提问。"
                       "一次运行最多问两次，能猜的不要问，猜的写进 observations(kind=uncertain)。",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
}


def schemas() -> list[dict]:
    """OpenAI function-calling 格式的工具声明。"""
    return [
        {"type": "function",
         "function": {"name": name, "description": spec["description"],
                      "parameters": spec["parameters"]}}
        for name, spec in TOOLS.items()
    ]


def dispatch(ctx: Ctx, name: str, args: dict) -> dict:
    spec = TOOLS.get(name)
    if not spec:
        return {"ok": False, "error": f"未知工具 {name}，可用：{list(TOOLS)}"}
    if not ctx.bump(name):
        return {"ok": False, "error": f"{name} 调用次数已达上限，请基于已有信息收尾"}
    try:
        return spec["fn"](ctx, **(args or {}))
    except TypeError as e:
        ctx.store.log_event("tool_error", name, f"参数不匹配：{e}")
        return {"ok": False, "error": f"参数不匹配：{e}"}
    except Exception as e:  # 工具内部异常不得中断循环
        # 静默吞掉会让 bug 藏起来，所以同时记进审计表，`events` 里能查
        ctx.store.log_event("tool_error", name, f"{type(e).__name__}: {e}")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
