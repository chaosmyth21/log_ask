"""LLM 适配层。

- OpenAICompat：任何 OpenAI 兼容端点（DeepSeek / Kimi / GLM / 本地 vLLM 都能用）；
- RulePolicy：无 API Key 时的确定性策略，跑同一套工具、同一条循环，
  保证断网、评委环境、录屏兜底都能完整演示。它不是"假模型"，
  而是一条降级路径 —— 输出质量略低，但结构、证据链、可检索性完全一致。
"""

from __future__ import annotations

import json
import os
import re


class LLMError(RuntimeError):
    """带"病因"的模型调用错误，让上层能给出对症的提示而不是一句"调用失败"。"""

    def __init__(self, kind: str, message: str, hint: str = ""):
        super().__init__(message)
        self.kind = kind
        self.hint = hint


KIND_HINTS = {
    "auth": "key 无效或已过期：改 labmemory.toml 的 [llm] api_key，或设环境变量 LM_API_KEY",
    "quota": "额度/余额不足：去服务商控制台确认",
    "rate_limit": "被限流：等几秒重试，或调小 max_turns 让单轮更短",
    "timeout": "超时：把 [llm] timeout 调大（现在偏小），或换更快的模型",
    "connection": "连不上：检查 base_url、代理和网络",
    "bad_request": "请求被拒：模型名可能不对，或这个模型不支持 tools/function calling",
    "not_installed": "没装 openai：pip install -r requirements.txt",
    "unknown": "未知错误，看下面的原文",
}


def classify_error(e: BaseException) -> str:
    """按类名 / 状态码判定错误类型。

    刻意不 import openai：这个函数要能在没装 openai 的环境里跑，
    也方便用假的异常类做单测。
    """
    name = type(e).__name__
    status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
    text = f"{name} {e}".lower()

    if isinstance(e, ImportError):
        return "not_installed"
    if "authentication" in text or "permissiondenied" in text or "unauthorized" in text \
            or status in (401, 403):
        return "auth"
    if "insufficientquota" in text or "quota" in text or status == 402:
        return "quota"
    if "ratelimit" in text or "too many requests" in text or status == 429:
        return "rate_limit"
    if "timeout" in text or "timed out" in text or status in (408, 504):
        return "timeout"
    if "connection" in text or "connecterror" in text or "network" in text or "dns" in text:
        return "connection"
    if "badrequest" in text or "unprocessable" in text or status in (400, 404, 422):
        return "bad_request"
    return "unknown"


def wrap_error(e: BaseException) -> LLMError:
    if isinstance(e, LLMError):
        return e
    kind = classify_error(e)
    return LLMError(kind, f"{type(e).__name__}: {e}", KIND_HINTS.get(kind, ""))


class OpenAICompat:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.2, timeout: float = 60):
        from openai import OpenAI  # 延迟导入：离线模式不需要这个包
        self.model = model or os.environ.get("LM_MODEL", "deepseek-chat")
        self.client = OpenAI(
            api_key=api_key or os.environ.get("LM_API_KEY") or os.environ.get("LLM_API_KEY"),
            base_url=base_url or os.environ.get("LM_BASE_URL", "https://api.deepseek.com/v1"),
            timeout=timeout,
        )
        self.temperature = temperature

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools,
                tool_choice="auto", temperature=self.temperature,
            )
        except Exception as e:                     # 统一转成带病因的错误
            raise wrap_error(e) from e
        msg = resp.choices[0].message
        calls = [
            {"id": c.id, "name": c.function.name,
             "args": json.loads(c.function.arguments or "{}")}
            for c in (msg.tool_calls or [])
        ]
        return {"content": msg.content, "tool_calls": calls,
                "usage": getattr(resp, "usage", None)}



class RulePolicy:
    """确定性策略：在 "无可用 LLM" 时扮演决策者，走完全相同的工具集。"""

    def __init__(self, mode: str = "ingest", since_hours: float = 72):
        self.mode = mode
        self.since_hours = since_hours
        self.queue: list[dict] = []
        self.current: dict | None = None
        self.saved = 0
        self.failed = 0

    # -- 对外接口，与 OpenAICompat 一致 --
    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        if self.mode == "query":
            return self._query(messages)
        return self._ingest(messages)

    # ---------------- 摄录模式 ----------------
    def _ingest(self, messages: list[dict]) -> dict:
        last = _last_tool_payload(messages)
        if last is None:
            return _call("list_files", {"since_hours": self.since_hours, "limit": 40})
        name, payload = last

        if name == "list_files":
            self.queue = [f for f in payload.get("files", []) if not f.get("already_processed")]
            if not self.queue:
                return _say("没有新的实验文件需要处理。")
            self.current = self.queue.pop(0)
            return _call("inspect_file", {"path": self.current["path"]})

        if name == "inspect_file":
            return _call("save_card", {"card": card_from_inspect(payload)})

        if name == "save_card":
            ok = payload.get("ok")
            self.saved += ok is True
            self.failed += ok is False
            if self.queue:
                self.current = self.queue.pop(0)
                return _call("inspect_file", {"path": self.current["path"]})
            return _say(f"处理完成：新建/更新 {self.saved} 张卡片"
                        + (f"，{self.failed} 个文件失败" if self.failed else "")
                        + "。")

        return _say("流程结束。")

    # ---------------- 问答模式 ----------------
    def _query(self, messages: list[dict]) -> dict:
        last = _last_tool_payload(messages)
        if last is None:
            q = _first_user(messages)
            self.question = q
            return _call("search_cards", _query_args(q))

        name, payload = last
        if name == "search_cards":
            cards = payload.get("cards") or []
            if not cards:
                return _say("库里没有匹配的卡片。换个说法，或先跑 `python main.py scan` 摄录实验日志。")
            self.hits = cards
            return _call("get_card", {"card_id": cards[0]["id"]})

        if name == "get_card":
            return _say(_compose_answer(getattr(self, "question", ""),
                                        getattr(self, "hits", []) or [],
                                        (payload or {}).get("card") or {}))

        return _say("未找到结果。")


def _query_args(q: str) -> dict:
    filters: dict = {}
    m = re.search(r"(?:lr|学习率)\s*[=为:：]?\s*([0-9.]+(?:[eE][-+]?\d+)?)", q)
    if not m and ("lr" in q.lower() or "学习率" in q):
        # "lr 放大到 0.005 效果怎么样" 这类说法：值不在 lr 后面紧跟
        m = re.search(r"(0?\.\d+|\d+(?:\.\d+)?[eE][-+]?\d+)", q)
    if m:
        v = float(m.group(1))
        filters["param_key"] = "lr"
        filters["param_value"] = str(int(v)) if v.is_integer() else str(v)
    for kw, tag in (("异常", "#异常"), ("报错", "#异常"), ("OOM", "#OOM"), ("崩", "#异常")):
        if kw.lower() in q.lower():
            filters["tag"] = tag
            break
    m = re.search(r"(?:experiment|exp|实验)\s*[:：]?\s*([\w\-]{3,})", q, re.I)
    if m:
        filters["exp_name"] = m.group(1)
    query = " ".join(w for w in ("loss", "acc", "oom", "异常", "停滞") if w.lower() in q.lower())
    return {"query": query, "filters": filters, "k": 5}


def _compose_answer(question: str, hits: list[dict], card: dict) -> str:
    if not card:
        return "没有取到卡片详情。"
    out = [f"问题：{question}", ""]
    if len(hits) > 1:
        out.append(f"命中 {len(hits)} 张卡片，按相关度排序：")
        for h in hits[:5]:
            m = next(iter((h.get("metrics") or {}).items()), None)
            metric = f"{m[0]} {m[1].get('first')}→{m[1].get('last')}" if m else "无指标"
            out.append(f"  · `{h['id']}` {h['exp_name']}/{h.get('run_id') or '—'}"
                       f"（{h.get('created_at', '')}，{h.get('status')}）：{metric}"
                       f"  {' '.join(h.get('tags') or [])}")
        out.append("")

    out.append(f"最相关：**{card['exp_name']} / {card.get('run_id') or '—'}**"
               f"（{card['created_at'][:10]}，状态 `{card['status']}`，置信度 {card.get('confidence')}）")
    if card.get("setup"):
        out.append("· 配置：" + "，".join(f"{k}={v}" for k, v in card["setup"].items()))
    for k, v in (card.get("metrics") or {}).items():
        if isinstance(v, dict) and v.get("n"):
            out.append(f"· {k}：{v.get('first')} → {v.get('last')}（最佳 {v.get('best')}，趋势 {v.get('trend')}）")
    for o in (card.get("observations") or [])[:4]:
        if not isinstance(o, dict):
            continue
        out.append(f"· [{o.get('kind')}] {o.get('text')}  ← {o.get('evidence')}")
    if card.get("conclusion"):
        out.append(f"· 结论：{card['conclusion']}")
    if card.get("files"):
        out.append("· 证据文件：" + "，".join(f.replace("\\", "/").split("/")[-1] for f in card["files"]))
    if (card.get("links") or {}).get("supersedes"):
        out.append(f"· 该卡片覆盖了旧卡片 `{card['links']['supersedes']}`（同实验同 run 的新结果）")
    if card.get("status") == "needs_review" or "#待复核" in (card.get("tags") or []):
        out.append("· 注意：该卡片标记为待复核，结论置信度有限。")
    return "\n".join(out)


def card_from_inspect(p: dict) -> dict:
    """规则模式下的建卡逻辑：能确定的写死，不能确定的标注出来。"""
    g = p.get("guesses") or {}
    setup = g.get("setup") or {}
    anomalies = p.get("anomalies") or []
    stats = {k: v for k, v in (p.get("metrics") or {}).items() if v.get("n")}
    name = (p.get("path") or "").replace("\\", "/").split("/")[-1]
    kind = p.get("kind")
    crash = any(a.get("kind") == "crash" for a in anomalies)

    if kind == "note":
        first = ""
        for line in (p.get("head") or "").splitlines():
            body = line.split("|", 1)[-1].strip().lstrip("#-* ").strip()
            if body:
                first = body
                break
        return {
            "exp_name": g.get("exp_name", "notes"),
            "run_id": g.get("run_id"),
            "goal": f"手写笔记：{name}",
            "setup": {},
            "metrics": {},
            "observations": [{"kind": "note", "text": first[:120], "evidence": name},
                             {"kind": "uncertain",
                              "text": "笔记内容未与具体 run 绑定，需人工确认归属哪个实验记录",
                              "evidence": name}],
            "conclusion": "手写笔记已入库，建议与对应 run 的卡片建立关联后复述结论。",
            "tags": ["#自动摄录", "#笔记", "#待复核"],
            "status": "needs_review",
            "confidence": 0.35,
            "files": [p.get("path")],
        }

    tags = ["#自动摄录"]
    if anomalies:
        tags.append("#异常")
    if crash:
        tags.append("#崩溃")
    if setup.get("lr") is not None:
        tags.append("#调参")
    sweep = p.get("sweep") or {}
    if sweep:
        tags.append("#扫描")
        setup = {**setup, "sweep": "；".join(f"{k}:{v[0]}~{v[-1]}" for k, v in sweep.items())}

    if sweep:
        mains = sorted(stats.items(), key=lambda kv: kv[1].get("n", 0), reverse=True)[:2]
        spread = "，".join(f"{k} 最佳 {m.get('best')}" for k, m in mains)
        conclusion = ("这是一张超参扫描表（" + "，".join(f"{k} 取 {len(v)} 个值" for k, v in sweep.items())
                      + "），不能按单条时间序列解读；需按扫描维度分组后比较"
                      + (f"，当前 {spread}。" if mains else "。"))
    elif anomalies:
        conclusion = "存在异常信号，需人工复核：" + "；".join(a.get("text", "") for a in anomalies[:2])
    else:
        mains = sorted(stats.items(), key=lambda kv: kv[1].get("n", 0), reverse=True)[:2]
        conclusion = ""
        if mains:
            k, v = mains[0]
            conclusion = f"{k} 由 {v['first']} 变为 {v['last']}（{v['trend']}），未见异常，建议换 seed 复现验证。"
        else:
            conclusion = "文件里没有可提取的指标，需人工确认该文件的用途。"

    return {
        "exp_name": g.get("exp_name", "untitled"),
        "run_id": g.get("run_id"),
        "goal": f"从 {name} 归纳：该次运行想验证什么（自动推测，待确认）",
        "setup": setup,
        "metrics": {k: {kk: v[kk] for kk in ("first", "last", "best", "worst", "trend", "n") if kk in v}
                    for k, v in stats.items()},
        "observations": [{"kind": a.get("kind"), "text": a.get("text"), "evidence": a.get("evidence")}
                         for a in anomalies],
        "conclusion": conclusion,
        "tags": tags,
        "status": "failed" if crash else ("needs_review" if anomalies else "active"),
        "confidence": 0.45 if anomalies else (0.6 if stats else 0.3),
        "files": [p.get("path")],
    }


def _call(name: str, args: dict) -> dict:
    return {"content": None,
            "tool_calls": [{"id": f"call_{name}_{abs(hash(json.dumps(args, sort_keys=True))) % 10000}",
                            "name": name, "args": args}]}


def _say(text: str) -> dict:
    return {"content": text, "tool_calls": []}


def _last_tool_payload(messages: list[dict]) -> tuple[str, dict] | None:
    for m in reversed(messages):
        if m.get("role") == "tool":
            try:
                return m["name"], json.loads(m["content"])
            except (json.JSONDecodeError, KeyError):
                return m.get("name", "?"), {}
    return None


def _first_user(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""
