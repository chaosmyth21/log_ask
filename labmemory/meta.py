"""从文件名与文件头"贪心"推断实验名/run/超参。

这不是要替代模型，而是给 Agent 一个**起点**：命名规范（exp_lr0001_bs32_run3.log）
在真实实验室里普遍存在，先解析出来，模型只需要纠错而不是从零猜。
"""

from __future__ import annotations

import re
from typing import Any

STATE_MARK = ".labmemory"          # 与 capture.STATE_DIR 保持一致（此处避免循环导入）


def num_from_token(tok: str) -> float | None:
    """lr0001 -> 0.001，lr1e-4 -> 0.0001，bs32 -> 32。"""
    m = re.match(r"^(0\d{3,})$", tok)
    if m:
        return float("0." + m.group(1)[1:])
    try:
        return float(tok)
    except ValueError:
        return None


def guess_setup(name: str, text: str = "") -> dict[str, Any]:
    setup: dict[str, Any] = {}
    for tok in re.split(r"[^A-Za-z0-9.+-]+", name):
        low = tok.lower()
        m = re.match(r"^(lr|learning_?rate)[-_]?([0-9.eE+-]+)$", low)
        if m and (v := num_from_token(m.group(2))) is not None:
            setup["lr"] = v
        m = re.match(r"^(bs|batch|batch_?size)[-_]?(\d+)$", low)
        if m:
            setup["batch"] = int(m.group(2))
        m = re.match(r"^seed[-_]?(\d+)$", low)
        if m:
            setup["seed"] = int(m.group(1))

    head = text[:8000]
    for key, pat in (("lr", r"\blr\s*[=:]\s*([0-9.eE+-]+)"),
                     ("batch", r"\bbatch(?:_size)?\s*[=:]\s*(\d+)"),
                     ("seed", r"\bseed\s*[=:]\s*(\d+)")):
        if key in setup:
            continue
        m = re.search(pat, head)
        if m:
            try:
                setup[key] = float(m.group(1)) if key == "lr" else int(m.group(1))
            except ValueError:
                pass
    m = re.search(r"\bmodel\s*[=:]\s*([\w.\-/:]+)", head)
    if m:
        setup["model"] = m.group(1)
    return setup


def guess_meta(path: str, text: str = "") -> dict:
    name = re.split(r"[\\/]", path)[-1]
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", name)
    parts = re.split(r"[\\/]", path)

    # 采集钩子/包装运行产出的记录：文件名是时间戳，实验名要从项目目录名取，
    # 这样同一个项目里的每次 F5 调试都会归到同一个 exp_name，按时间排成一条链。
    if STATE_MARK in parts:
        idx = parts.index(STATE_MARK)
        proj = parts[idx - 1] if idx > 0 else "project"
        run = re.sub(r"-\d{2,6}$", "", stem)          # 去掉尾部 pid
        return {"exp_name": proj.lower(), "run_id": run, "setup": guess_setup("", text)}

    run_id = None
    # 注意用 (?<![A-Za-z0-9]) 而非 \b：下划线在 \w 里，lora-sft_..._run2 之间没有 \b
    m = re.search(r"(?<![A-Za-z0-9])(run[-_]?\d+|r\d{1,2})(?![0-9])", stem, re.I)
    if m:
        run_id = m.group(1).lower().replace("run_", "run")
    base = re.sub(r"(?:lr|bs|batch|seed)[-_]?[0-9.eE+-]+", "", stem, flags=re.I)
    base = re.sub(r"(?<![A-Za-z0-9])(?:run[-_]?\d+|r\d{1,2})(?![0-9])", "", base, flags=re.I)
    base = re.sub(r"\d{4}[-_]?\d{2}[-_]?\d{2}", "", base)      # 2026-10-13 / 20261013
    base = re.sub(r"[-_.\s]{2,}", "_", base).strip("-_ .")
    return {"exp_name": (base or "untitled").lower(), "run_id": run_id,
            "setup": guess_setup(stem, text)}
