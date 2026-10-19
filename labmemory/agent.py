"""Agent 主循环：观察 → 决策 → 调工具 → 再观察，直到模型给出最终答复。

三条护栏是 Agent 与"脚本"的分界线，也是黑客松现场最容易翻车的地方：
1. max_turns  —— 最多几轮模型交互；
2. 工具级配额 —— 单个工具最多调几次（见 tools.Ctx.limit）；
3. 结果截断   —— 单次工具输出限长，防止一次读大文件把上下文打爆。
"""

from __future__ import annotations

import json
import time

from . import tools as T
from .llm import wrap_error

MAX_RESULT_CHARS = 5000


class Agent:
    def __init__(self, llm, ctx: T.Ctx, system_prompt: str, max_turns: int = 14,
                 verbose: bool = True, max_calls: int = 40,
                 max_result_chars: int = MAX_RESULT_CHARS):
        self.llm = llm
        self.ctx = ctx
        self.system = system_prompt
        self.max_turns = max_turns
        self.max_calls = max_calls
        self.max_result_chars = max_result_chars
        self.verbose = verbose
        self.trace: list[dict] = []
        self.calls = 0
        self._recent: list[str] = []

    def run(self, user_input: str) -> dict:
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_input},
        ]
        started = time.time()
        for turn in range(1, self.max_turns + 1):
            try:
                resp = self.llm.chat(messages, T.schemas())
            except Exception as e:                       # 网络/鉴权失败 -> 交给上层降级
                err = wrap_error(e)
                return {"ok": False, "error": str(err), "error_kind": err.kind,
                        "hint": err.hint, "turns": turn, "trace": self.trace,
                        "elapsed": round(time.time() - started, 1)}

            calls = resp.get("tool_calls") or []
            if not calls:
                text = resp.get("content") or ""
                self._log(turn, "final", {}, True, 0)
                return {"ok": True, "final": text, "turns": turn, "trace": self.trace,
                        "elapsed": round(time.time() - started, 1)}

            if self.calls + len(calls) > self.max_calls:
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [self._tc(c) for c in calls]})
                for c in calls:
                    messages.append(self._tool_msg(c["id"], c["name"],
                                                   {"ok": False, "error": "总调用次数已达上限"}))
                return {"ok": True, "final": "工具调用预算用尽，已按现有信息收尾。",
                        "turns": turn, "trace": self.trace,
                        "elapsed": round(time.time() - started, 1)}

            messages.append({"role": "assistant", "content": resp.get("content"),
                             "tool_calls": [self._tc(c) for c in calls]})

            for c in calls:
                t0 = time.time()
                sig = c["name"] + "::" + json.dumps(c.get("args") or {},
                                                    sort_keys=True, ensure_ascii=False)
                repeat = sum(1 for s in self._recent[-4:] if s == sig)
                self._recent.append(sig)
                if repeat >= 2:
                    # 同一个调用连着来第三遍：模型多半以为自己没成功，得打断它
                    result = {"ok": False,
                              "error": "重复调用：完全相同的参数已经执行过两次，结果就是你上面拿到的那个。"
                                       "如果那张卡已经是你要的状态，请直接处理下一个文件或结束；"
                                       "不要再用同样的参数重试。"}
                else:
                    result = T.dispatch(self.ctx, c["name"], c.get("args") or {})
                self.calls += 1
                payload = json.dumps(result, ensure_ascii=False)
                self._log(turn, c["name"], c.get("args") or {}, bool(result.get("ok")),
                          round((time.time() - t0) * 1000), len(payload))
                if len(payload) > self.max_result_chars:
                    payload = payload[:self.max_result_chars] + "\n…（结果过长已截断，需要更多细节请用 read_slice 局部读取）"
                messages.append(self._tool_msg(c["id"], c["name"], payload))

        return {"ok": True, "final": f"已达 {self.max_turns} 轮上限，停留在未完成状态。",
                "turns": self.max_turns, "trace": self.trace,
                "elapsed": round(time.time() - started, 1)}

    # ---- 内部 ----
    @staticmethod
    def _tc(c: dict) -> dict:
        return {"id": c.get("id") or f"call_{c['name']}", "type": "function",
                "function": {"name": c["name"],
                             "arguments": json.dumps(c.get("args") or {}, ensure_ascii=False)}}

    @staticmethod
    def _tool_msg(call_id: str, name: str, payload) -> dict:
        if not isinstance(payload, str):
            payload = json.dumps(payload, ensure_ascii=False)
        return {"role": "tool", "tool_call_id": call_id, "name": name, "content": payload}

    def _log(self, turn: int, tool: str, args: dict, ok: bool, ms: int, size: int = 0) -> None:
        rec = {"turn": turn, "tool": tool, "ok": ok, "ms": ms, "result_chars": size}
        if tool not in ("final",):
            rec["args"] = _shrink(args)
        self.trace.append(rec)
        if self.verbose and tool != "final":
            print(f"  │ step{turn:<2} {tool:<14} {_fmt_args(args):<46} {ms:>4}ms")


def _shrink(args: dict) -> dict:
    out = {}
    for k, v in (args or {}).items():
        s = v if isinstance(v, (int, float, bool)) else str(v)
        if isinstance(s, str) and len(s) > 120:
            s = s[:120] + "…"
        out[k] = s
    return out


def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if k == "card" and isinstance(v, dict):
            s = f"{v.get('exp_name')}/{v.get('run_id') or '—'}"
        elif k == "filters":
            s = json.dumps(v, ensure_ascii=False)
        else:
            s = str(v)
        parts.append(f"{k}={s[:60]}")
    return ", ".join(parts)[:90]
