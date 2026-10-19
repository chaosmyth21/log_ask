"""文件分类、文本切片、指标抽取、异常规则。

分工原则：**规则负责发现信号，LLM 负责解释信号**。
数值、行号、报警类型这些"事实"一律由确定性代码产出，
不让模型去记数字 —— 既避免幻觉，也让每张卡片都能回指到具体行。
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path

from . import meta

TEXT_SUFFIX = {".log", ".txt", ".out", ".err", ".md", ".json", ".jsonl", ".csv",
               ".tsv", ".yaml", ".yml", ".diff", ".patch"}
IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}

# 指标名过滤：这些是过程量/环境量，不进卡片指标区
NOISE_KEYS = {
    "step", "global_step", "epoch", "iter", "iteration", "idx", "index",
    "time", "secs", "sec", "second", "seconds", "ms", "elapsed", "eta",
    "speed", "it_s", "samples_s", "s_it", "pid", "port", "rank", "world_size",
    "line", "errno", "code", "exit", "worker", "thread", "gpu", "gpu_id",
    "mem", "memory", "vram", "memsize", "token", "tokens", "n", "count",
    # 运行时诊断量：能说明"跑得快不快/稳不稳"，但不是实验结论，不进指标区
    "grad_norm", "grad", "l2_norm", "norm", "clip", "tokens_per_s", "tps",
    "throughput", "samples_per_s", "steps_per_s", "steps_s", "lr_scale",
}
AXIS_KEYS = {"step", "global_step", "epoch", "iter", "iteration"}
# 这些键在表格里出现多个不同值，说明是"扫描维度"而不是"指标"
HYPERPARAM_KEYS = {
    "lr", "learning_rate", "batch", "batch_size", "weight_decay", "dropout",
    "temperature", "seq_len", "max_len", "max_seq_len", "rank", "alpha",
    "top_p", "beam", "n_layers", "hidden", "dim",
}

_KV = re.compile(
    r"(?<![\w./-])(?P<k>[A-Za-z_][A-Za-z0-9_./\-]{0,31})\s*[=:：]\s*"
    r"(?P<v>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|inf|NaN|Inf)"
)
_CRASH = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|RuntimeError|"
    r"Segmentation fault|Killed|core dumped|OOMKilled|KeyboardInterrupt|"
    r"^ERROR[:\s]|^CRITICAL[:\s]|Exception raised"
)


def classify(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in IMAGE_SUFFIX:
        return "image"
    if suf in {".csv", ".tsv"}:
        return "table"
    if suf in {".json", ".jsonl"}:
        return "json"
    if suf in {".md", ".markdown"}:
        return "note"
    if suf in {".log", ".out", ".err", ".txt"}:
        return "log"
    return "text" if suf in TEXT_SUFFIX else "other"


def read_text(path: Path, limit_bytes: int = 4_000_000) -> tuple[str, bool]:
    raw = path.read_bytes()[:limit_bytes]
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc), len(path.read_bytes()) > limit_bytes
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace"), False


def read_slice(path: Path, start: int = 1, max_lines: int = 80, width: int = 220) -> dict:
    """分片读取：给 Agent 一个可以反复"局部重读"的能力，避免整份塞进上下文。"""
    text, truncated = read_text(path)
    lines = text.splitlines()
    start = max(1, int(start))
    chunk = lines[start - 1: start - 1 + int(max_lines)]
    numbered = [f"{start + i:>6}| {ln[:width]}" for i, ln in enumerate(chunk)]
    return {
        "path": str(path),
        "total_lines": len(lines),
        "start": start,
        "end": start + len(chunk) - 1,
        "truncated_by_size": truncated,
        "text": "\n".join(numbered),
    }


def _value(tok: str) -> float:
    low = tok.lower()
    if low in ("nan", "inf", "-inf"):
        return float(low) if low != "-inf" else -math.inf
    return float(tok)


def _clean_key(k: str) -> str:
    return k.strip().strip("./-").replace("/", "_").lower()


def iter_metrics(path: Path) -> dict[str, list[tuple[float, float]]]:
    """抽取 {指标名: [(x, value)]}。x 取 step/epoch，否则取累计出现序号。"""
    kind = classify(path)
    text, _ = read_text(path)
    series: dict[str, list[tuple[float, float]]] = {}
    axis: list[float] = []
    counter = 0

    if kind == "table":
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        rows = list(csv.reader(io.StringIO(text)))
        if len(rows) >= 2:
            header = [h.strip().lower() for h in rows[0]]
            xs = next((i for i, h in enumerate(header) if h in AXIS_KEYS), None)
            for r in rows[1:]:
                if len(r) < 2:
                    continue
                x = _value(r[xs]) if xs is not None and _value_ok(r[xs]) else float(counter)
                for i, h in enumerate(header):
                    if i >= len(r) or i == xs or h in NOISE_KEYS or not h:
                        continue
                    if _value_ok(r[i]):
                        series.setdefault(_clean_key(h), []).append((x, _value(r[i])))
                counter += 1
        return series

    if kind == "json":
        try:
            obj = json.loads(text)
            _flatten_json(obj, series, counter)
            return series
        except json.JSONDecodeError:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _flatten_json(obj, series, counter)
                counter += 1
            return series

    # log / note / text：逐行 key=value
    for line in text.splitlines():
        counter += 1
        found: dict[str, float] = {}
        for m in _KV.finditer(line):
            k = _clean_key(m.group("k"))
            if not k or k in NOISE_KEYS:
                continue
            try:
                v = _value(m.group("v"))
            except ValueError:
                continue
            if not math.isfinite(v):
                found[f"__nonfinite__{k}"] = v
                continue
            found.setdefault(k, v)
        x = next((found[a] for a in AXIS_KEYS if a in found), float(counter))
        for k, v in found.items():
            if k.startswith("__nonfinite__"):
                series.setdefault(k, []).append((x, v))
            else:
                series.setdefault(k, []).append((x, v))
    return series


def _value_ok(tok: str) -> bool:
    try:
        _value(tok)
        return True
    except (TypeError, ValueError):
        return False


def _flatten_json(obj, series, x, prefix="") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}"
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                ck = _clean_key(key)
                if ck and ck not in NOISE_KEYS:
                    series.setdefault(ck, []).append((float(x), float(v)))
            elif isinstance(v, (dict, list)):
                _flatten_json(v, series, x, f"{key}.")
    elif isinstance(obj, list):
        for item in obj:
            _flatten_json(item, series, x, prefix)


def _higher_is_better(name: str) -> bool:
    n = name.lower()
    for kw in ("acc", "f1", "auc", "map", "bleu", "rouge", "score", "precision", "recall", "success", "rewards"):
        if kw in n:
            return True
    return False


def series_stats(name: str, pts: list[tuple[float, float]]) -> dict:
    vals = [v for _, v in pts if math.isfinite(v)]
    if not vals:
        return {"n": len(pts), "nonfinite": True}
    best = max(vals) if _higher_is_better(name) else min(vals)
    k = min(5, len(vals))
    head, tail = sum(vals[:k]) / k, sum(vals[-k:]) / k
    if abs(head) < 1e-12:
        trend = "flat"
    else:
        rel = (tail - head) / abs(head)
        trend = "up" if rel > 0.02 else "down" if rel < -0.02 else "flat"
    return {
        "n": len(vals), "first": round(vals[0], 6), "last": round(vals[-1], 6),
        "best": round(best, 6), "worst": round(max(vals) if not _higher_is_better(name) else min(vals), 6),
        "trend": trend, "higher_is_better": _higher_is_better(name),
        # 全程不变的序列（lr、batch、seed 这类超参）不是"指标"，不参与异常判定
        "constant": len(set(vals)) == 1,
    }


def anomaly_signals(path: Path, series: dict, text: str) -> list[dict]:
    """规则发现"哪里不对"，不解释"为什么"。每条带行号证据。"""
    out: list[dict] = []
    lines = text.splitlines()
    is_note = classify(path) == "note"

    # 手写笔记里出现"崩了/OOM"是在叙述历史，不是本次运行的事实，不做崩溃判定
    if not is_note:
        for i, ln in enumerate(lines, 1):
            if _CRASH.search(ln):
                out.append({"kind": "crash", "text": f"日志出现崩溃/异常标记：{ln.strip()[:120]}",
                            "evidence": f"{path.name}:{i}"})
                break

    for name, pts in series.items():
        if name.startswith("__nonfinite__"):
            out.append({"kind": "nonfinite", "text": f"指标 {name[14:]} 出现 NaN/Inf",
                        "evidence": path.name})
            continue
        s = series_stats(name, pts)
        if s.get("nonfinite") or s.get("constant"):
            continue
        if abs(s["first"]) > 1e-12:
            ratio = s["last"] / s["first"] if s["first"] else 0
            if ratio > 5:
                out.append({"kind": "divergence",
                            "text": f"{name} 从 {s['first']} 升到 {s['last']}（{ratio:.1f}×），疑似发散",
                            "evidence": path.name})
        if s["higher_is_better"] and s["last"] < s["best"] * 0.8 and s["n"] >= 10:
            out.append({"kind": "regression",
                        "text": f"{name} 末值 {s['last']} 比最佳 {s['best']} 回退 {(1 - s['last'] / s['best']) * 100:.0f}%",
                        "evidence": path.name})
        if not s["higher_is_better"] and s["last"] > s["best"] * 1.5 and s["n"] >= 10:
            out.append({"kind": "regression",
                        "text": f"{name} 末值 {s['last']} 比最佳 {s['best']} 恶化 {(s['last'] / s['best'] - 1) * 100:.0f}%",
                        "evidence": path.name})
        if s["n"] >= 8:
            tail = [v for _, v in pts[-8:]]
            span = abs(tail[0]) * 0.015
            if max(tail) - min(tail) <= max(1e-9, span):
                out.append({"kind": "stagnation", "text": f"{name} 最近 8 个点几乎不变（{tail[-1]}），疑似停滞",
                            "evidence": path.name})
        if s["n"] >= 20 and s["trend"] == "flat":
            out.append({"kind": "no_learning", "text": f"{name} 全程无明显变化（{s['first']} → {s['last']}）",
                        "evidence": path.name})
    # 同类信号去重，保留前 4 条
    seen, uniq = set(), []
    for a in out:
        key = (a["kind"], re.sub(r"\d+(\.\d+)?", "#", a["text"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return uniq[:4]


def inspect(path: Path) -> dict:
    """工具 extract_metrics 的实现：一次调用给出"这个文件里有什么值得看的东西"。"""
    text, truncated = read_text(path)
    kind = classify(path)
    series = iter_metrics(path)
    stats = {k: series_stats(k, v) for k, v in series.items() if not k.startswith("__")}
    # 只保留真正的"指标"：至少 2 个点，且数值有变化（排除 lr/batch/seed 这类恒定超参）
    stats = {k: v for k, v in stats.items()
             if v.get("n", 0) >= 2 and not v.get("constant") and not v.get("nonfinite")}
    # 在一张表里跨多个取值的超参 = 扫描维度，单独收集，不当成指标
    sweep = {k: sorted({round(v, 8) for _, v in series[k]})[:8]
             for k in list(stats) if k in HYPERPARAM_KEYS}
    for k in sweep:
        stats.pop(k, None)
    # 扫描表是"横向对比"而不是"一次运行的时间序列"，不做趋势异常判定
    anomalies = [] if (kind == "table" and sweep) else anomaly_signals(path, series, text)
    guesses = meta.guess_meta(str(path), text)
    # 表里恒定的超参列（只有一个 lr 的那种 CSV）：不是扫描维度，但属于这次运行的 setup
    for k, pts in series.items():
        if k in HYPERPARAM_KEYS and len({round(v, 8) for _, v in pts}) == 1:
            guesses.setdefault("setup", {}).setdefault(k, round(pts[0][1], 8))
    lines = text.splitlines()
    # 大文件只回传 head/tail 采样，样本点也做抽稀，控制上下文体积
    samples = {k: [{"x": round(x, 3), "v": round(v, 6)} for x, v in (v[:3] + v[-3:])[:6]]
               for k, v in series.items() if not k.startswith("__")}
    return {
        "path": str(path), "kind": kind,
        "bytes": path.stat().st_size, "lines": len(lines), "truncated_by_size": truncated,
        "guesses": guesses,
        "metrics": stats, "samples": samples, "sweep": sweep,
        "anomalies": anomalies,
        "head": "\n".join(f"{i:>5}| {ln[:160]}" for i, ln in enumerate(lines[:8], 1)),
        "tail": "\n".join(f"{len(lines) - 5 + i:>5}| {ln[:160]}" for i, ln in enumerate(lines[-6:], 1)) if len(lines) > 8 else "",
    }
