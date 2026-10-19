"""周报组装：确定性的 Markdown，不依赖 LLM，断网也能出。

分工与卡片一致：规则负责"排版与计数"，LLM 只在需要润色时介入（可选）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def _fmt_metrics(metrics: dict, limit: int = 3) -> str:
    # 兼容早期/异常形状：metrics 里可能是裸数字而不是统计对象
    items = [(k, v) for k, v in (metrics or {}).items() if isinstance(v, dict) and v.get("n")]
    items.sort(key=lambda kv: kv[1].get("n", 0), reverse=True)
    out = []
    for k, v in items[:limit]:
        arrow = {"down": "↓", "up": "↑", "flat": "→"}.get(v.get("trend"), "→")
        out.append(f"{k} {v.get('first')}{arrow}{v.get('last')}（best {v.get('best')}）")
    return "；".join(out)


def build_weekly(cards: list[dict], days: int = 7, scope: str | None = None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    by_exp: dict[str, list[dict]] = {}
    for c in cards:
        by_exp.setdefault(c["exp_name"], []).append(c)

    abnormal = [c for c in cards if "#异常" in (c.get("tags") or [])]
    failed = [c for c in cards if c.get("status") == "failed"]
    review = [c for c in cards if c.get("status") in ("needs_review",) or "#待复核" in (c.get("tags") or [])]

    L = [f"# 实验室周报（近 {days} 天）", ""]
    if scope:
        L += [f"> 数据来源：{scope}", ""]
    L += [f"> 生成时间：{now}　|　卡片 {len(cards)} 张　|　实验 {len(by_exp)} 个　|　"
          f"异常 {len(abnormal)} 处　|　失败 {len(failed)} 次",
          "", "---", ""]

    L += ["## 一、概览", ""]
    if not cards:
        L += ["本周没有新增卡片。可以往数据目录里放实验日志，然后 `python main.py scan`。", ""]
        return "\n".join(L)
    L += ["| 实验 | 卡片 | 最新状态 | 主要指标 |", "|---|---|---|---|"]
    for exp, cs in sorted(by_exp.items(), key=lambda kv: -len(kv[1])):
        latest = max(cs, key=lambda c: c["created_at"])
        L.append(f"| `{exp}` | {len(cs)} | {latest['status']} | {_fmt_metrics(latest.get('metrics') or {}, 2)} |")
    L += [""]

    L += ["## 二、逐实验进展", ""]
    for exp, cs in sorted(by_exp.items(), key=lambda kv: -len(kv[1])):
        L += [f"### `{exp}`（{len(cs)} 次记录）", ""]
        for c in sorted(cs, key=lambda c: c["created_at"]):
            run = c.get("run_id") or "—"
            setup = "，".join(f"{k}={v}" for k, v in (c.get("setup") or {}).items()) or "未记录超参"
            L.append(f"- **{run}**（`{c['id']}`，{c['created_at'][:10]}，{c['status']}）")
            L.append(f"  - 配置：{setup}")
            if c.get("metrics"):
                L.append(f"  - 指标：{_fmt_metrics(c['metrics'])}")
            if c.get("conclusion"):
                L.append(f"  - 结论：{c['conclusion']}")
            files = "，".join(Path(f).name for f in c.get("files") or [])
            if files:
                L.append(f"  - 来源：{files}")
        L += [""]

    L += ["## 三、需要处理的问题", ""]
    if not (abnormal or failed):
        L += ["本周没有异常信号。", ""]
    else:
        for c in sorted(abnormal + failed, key=lambda c: c["created_at"], reverse=True):
            for o in (c.get("observations") or []):
                if not isinstance(o, dict) or o.get("kind") in ("note", None):
                    continue
                L.append(f"- `[{o.get('kind')}]` {o.get('text')}　—　`{c['id']}` / {c['exp_name']}"
                         + (f"　（证据 {o.get('evidence')}）" if o.get("evidence") else ""))
        L += [""]

    L += ["## 四、下周建议验证", ""]
    cand = [c for c in (review or cards) if c.get("conclusion")]
    if not cand:
        L += ["—（卡片里还没有可执行的下一步）", ""]
    else:
        for c in sorted(cand, key=lambda c: c["created_at"], reverse=True)[:3]:
            L.append(f"- {c['conclusion']}　—　来自 `{c['id']}`")
        L += [""]

    chain = [c for c in cards if (c.get("links") or {}).get("supersedes")]
    if chain:
        L += ["## 五、实验链（后一次覆盖前一次）", ""]
        for c in chain:
            L.append(f"- `{c['links']['supersedes']}` → `{c['id']}`（{c['exp_name']} {c.get('run_id') or ''}）")
        L += [""]
    return "\n".join(L)


def save_weekly(md: str, out_dir: str | Path) -> Path:
    """把周报保存到日志文件夹下的 weekly/weekly.md。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "weekly.md"
    path.write_text(md, encoding="utf-8")
    return path
