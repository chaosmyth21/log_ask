"""可选：Gradio 网页外壳（评审"点开就能用"的 Demo 链接）。

    pip install gradio
    python app.py        # 默认 http://127.0.0.1:7860

只是薄壳：所有逻辑仍在 labmemory 里，网页层不承担任何判断。
右上角齿轮用于切换 logs 下的日志文件夹（默认 test1 / test2），
选择会持久化到 .labmemory/ui.json，扫描与检索都按当前选择读取。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import gradio as gr
    GRADIO_MAJOR = int(str(gr.__version__).split(".", 1)[0])
except ImportError:                                          # 允许作为普通模块被测试导入
    gr = None
    GRADIO_MAJOR = 0

from labmemory import config, logdirs, prompts, report       # noqa: E402
from labmemory.agent import Agent                            # noqa: E402
from labmemory.llm import OpenAICompat, RulePolicy           # noqa: E402
from labmemory.store import Store                            # noqa: E402
from labmemory.tools import Ctx                              # noqa: E402

CFG = config.load()                                          # 读 labmemory.toml
DB = Path(CFG.paths.db)
UI_STATE = DB.resolve().parent / ".labmemory" / "ui.json"
LOGS_ROOT = Path(CFG.paths.logs_root or CFG.paths.data_dir).expanduser()
CARD_HEADERS = ["ID", "实验", "run", "状态", "标签", "指标", "来源文件", "创建日期"]

CUSTOM_CSS = """
:root {
    --lm-ink: #0f172a;
    --lm-muted: #64748b;
    --lm-line: #e2e8f0;
    --lm-soft: #f8fafc;
    --lm-primary: #4f46e5;
    --lm-primary-deep: #3730a3;
    --lm-cyan: #22d3ee;
    --lm-card-shadow: 0 12px 34px rgba(15, 23, 42, .07);
}

html, body {
    background:
        radial-gradient(circle at 8% 0%, rgba(99, 102, 241, .10), transparent 28rem),
        radial-gradient(circle at 96% 12%, rgba(14, 165, 233, .08), transparent 24rem),
        #f5f7fb !important;
}

body, .gradio-container {
    font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif !important;
    color: var(--lm-ink);
}

.gradio-container {
    width: 100% !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 24px 24px 44px !important;
}

footer { display: none !important; }

/* ---------- hero ---------- */
#hero-card {
    position: relative;
    overflow: hidden;
    padding: 28px 30px 24px !important;
    border: 1px solid rgba(255, 255, 255, .10) !important;
    border-radius: 26px !important;
    background:
        radial-gradient(circle at 84% -10%, rgba(34, 211, 238, .26), transparent 34%),
        radial-gradient(circle at 68% 110%, rgba(124, 58, 237, .36), transparent 40%),
        linear-gradient(135deg, #0f172a 0%, #171b3b 58%, #312e81 100%) !important;
    box-shadow: 0 22px 50px rgba(15, 23, 42, .20) !important;
}

#hero-card::after {
    content: "";
    position: absolute;
    right: -86px;
    bottom: -180px;
    width: 330px;
    height: 330px;
    border: 1px solid rgba(255, 255, 255, .10);
    border-radius: 50%;
    box-shadow: 0 0 0 54px rgba(255, 255, 255, .025),
                0 0 0 108px rgba(255, 255, 255, .018);
    pointer-events: none;
}

#hero-row { align-items: center !important; }
.hero-copy { position: relative; z-index: 1; }
.hero-kicker {
    margin-bottom: 9px;
    color: #a5b4fc;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
}
.hero-title {
    margin: 0 !important;
    color: #f8fafc;
    font-size: clamp(30px, 3.5vw, 44px);
    font-weight: 750;
    letter-spacing: -.035em;
    line-height: 1.1;
}
.hero-title span {
    margin-left: 10px;
    background: linear-gradient(90deg, #a5b4fc, #67e8f9);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: .46em;
    font-weight: 600;
    letter-spacing: .02em;
    white-space: nowrap;
}
.hero-sub {
    max-width: 720px;
    margin: 13px 0 0 !important;
    color: #cbd5e1;
    font-size: 14px;
    line-height: 1.7;
}

#active-log {
    position: relative;
    z-index: 1;
    width: fit-content;
    max-width: 100%;
    margin-top: 16px !important;
    padding: 9px 13px !important;
    border: 1px solid rgba(255, 255, 255, .13);
    border-radius: 12px;
    background: rgba(255, 255, 255, .075);
    backdrop-filter: blur(8px);
}
#active-log p {
    margin: 0 !important;
    color: #dbeafe !important;
    font-size: 12.5px !important;
    line-height: 1.7 !important;
}
#active-log p + p {
    margin-top: 2px !important;
    color: #94a3b8 !important;
    font-size: 11px !important;
}
#active-log code {
    padding: 2px 6px;
    border: 0;
    border-radius: 6px;
    background: rgba(255, 255, 255, .13);
    color: #f8fafc;
    font-size: .94em;
}
#active-log strong { color: #fff !important; }

/* ---------- settings ---------- */
#gear-btn {
    width: 48px;
    height: 48px;
    min-width: 48px;
    max-width: 48px;
    padding: 0 !important;
    border-radius: 50% !important;
    border: 1px solid rgba(255, 255, 255, .16) !important;
    background: rgba(255, 255, 255, .10) !important;
    color: #fff !important;
    font-size: 22px;
    line-height: 1;
    transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
}
#gear-btn:hover {
    transform: rotate(30deg);
    background: rgba(255, 255, 255, .18) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, .20);
}
#settings-panel {
    position: fixed !important;
    top: 96px;
    right: max(24px, calc((100vw - 1280px) / 2 + 24px));
    z-index: 9999;
    width: min(380px, calc(100vw - 32px));
    padding: 16px 18px 12px;
    background: var(--background-fill-primary, #fff);
    border: 1px solid var(--lm-line);
    border-radius: 18px;
    box-shadow: 0 24px 64px rgba(15, 23, 42, .22);
}
#settings-panel h3 { margin-top: 0; }
#settings-panel .prose { max-width: none; }

/* ---------- tabs ---------- */
.tabs { margin-top: 18px; }
.tab-container {
    gap: 6px !important;
    padding: 5px !important;
    border: 1px solid var(--lm-line) !important;
    border-radius: 14px !important;
    background: rgba(255, 255, 255, .72) !important;
    box-shadow: 0 5px 18px rgba(15, 23, 42, .035);
}
.tab-container button {
    min-height: 38px;
    padding: 8px 18px !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    color: #64748b !important;
    font-weight: 650 !important;
    transition: color .16s ease, background .16s ease, transform .16s ease;
}
.tab-container button:hover {
    background: #f1f5f9 !important;
    color: #334155 !important;
}
.tab-container button.selected {
    border-color: #c7d2fe !important;
    background: linear-gradient(135deg, #eef2ff, #e0f2fe) !important;
    color: #4338ca !important;
    box-shadow: 0 3px 12px rgba(79, 70, 229, .10);
}
.tabitem { padding: 0 !important; }

/* ---------- panels ---------- */
#ask-panel, #scan-panel, #cards-panel, #weekly-panel {
    gap: 14px !important;
    padding-top: 16px !important;
}

.section-head h3 {
    margin: 2px 0 6px !important;
    color: var(--lm-ink) !important;
    font-size: 21px !important;
    font-weight: 750 !important;
    letter-spacing: -.02em;
}
.section-head p {
    margin: 0 !important;
    color: var(--lm-muted) !important;
    font-size: 13.5px !important;
    line-height: 1.7 !important;
}

#filter-card {
    padding: 16px 16px 14px !important;
    border: 1px solid var(--lm-line) !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, .90) !important;
    box-shadow: var(--lm-card-shadow);
}
#filter-card .styler,
#filter-card .form {
    background: transparent !important;
}
#filter-card .row { align-items: flex-end !important; }
#filter-card .block,
#detail-card {
    border-color: var(--lm-line) !important;
    background: #fff !important;
}
#filter-card .block {
    border: 1px solid var(--lm-line) !important;
    box-shadow: none !important;
}
#filter-card span[data-testid="block-info"] {
    padding: 0 !important;
    background: transparent !important;
    color: #475569 !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    letter-spacing: .03em;
}
#filter-actions { justify-content: flex-end !important; }
#filter-actions button {
    min-width: 132px !important;
    min-height: 42px;
}

/* ---------- inputs and buttons ---------- */
.block {
    border-radius: 14px !important;
}
span[data-testid="block-info"] {
    padding: 0 !important;
    background: transparent !important;
    color: #475569 !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
    letter-spacing: .02em;
}
input, textarea {
    border-radius: 10px !important;
}
textarea:focus, input:focus {
    border-color: #a5b4fc !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, .10) !important;
}
#query-box textarea {
    padding: 12px 14px !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
}
#query-box {
    border: 1px solid var(--lm-line) !important;
    background: #fff !important;
    box-shadow: var(--lm-card-shadow);
}

button.primary {
    border: 0 !important;
    background: linear-gradient(135deg, #4f46e5, #6366f1 55%, #0ea5e9) !important;
    color: #fff !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 22px rgba(79, 70, 229, .22) !important;
    transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
}
button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(79, 70, 229, .28) !important;
    filter: saturate(1.08);
}
button.secondary {
    border: 1px solid var(--lm-line) !important;
    background: #fff !important;
    color: #334155 !important;
    font-weight: 650 !important;
}
button.secondary:hover {
    border-color: #cbd5e1 !important;
    background: #f8fafc !important;
}
#ask-btn, #scan-btn, #weekly-btn { width: 100% !important; min-height: 46px; }

/* ---------- examples ---------- */
.gallery-item {
    border: 1px solid var(--lm-line) !important;
    border-radius: 12px !important;
    background: #fff !important;
    transition: border .16s ease, background .16s ease, transform .16s ease;
}
.gallery-item:hover {
    border-color: #a5b4fc !important;
    background: #f8faff !important;
    transform: translateY(-1px);
}
#answer-box, #scan-result, #weekly-content {
    padding: 18px 20px !important;
    border: 1px solid var(--lm-line) !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, .92) !important;
    box-shadow: var(--lm-card-shadow);
}
#answer-box:empty, #scan-result:empty { display: none; }

/* ---------- card browser ---------- */
#card-count {
    margin: 2px 2px 0 !important;
    padding: 9px 13px !important;
    border: 1px solid #dbeafe;
    border-radius: 10px;
    background: #f0f9ff;
}
#card-count p {
    margin: 0 !important;
    color: #0369a1 !important;
    font-size: 12.5px !important;
}
#card-count code {
    color: #075985;
    background: #e0f2fe;
}
#card-table {
    overflow: hidden;
    border: 1px solid var(--lm-line) !important;
    border-radius: 18px !important;
    background: #fff !important;
    box-shadow: var(--lm-card-shadow);
}
#card-table table { font-size: 12.5px !important; }
#card-table thead th {
    padding: 11px 10px !important;
    background: #f8fafc !important;
    color: #475569 !important;
    font-size: 11.5px !important;
    font-weight: 750 !important;
    letter-spacing: .03em;
}
#card-table tbody td { padding: 10px !important; border-color: #eef2f7 !important; }
#card-table tbody tr:hover td { background: #f8faff !important; }

#detail-card {
    gap: 10px !important;
    padding: 16px 18px 18px !important;
    border: 1px solid var(--lm-line) !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, .92) !important;
    box-shadow: var(--lm-card-shadow);
}
#detail-card .styler,
#detail-card .form {
    background: transparent !important;
}
#detail-card span[data-testid="block-info"] {
    padding: 0 !important;
    background: transparent !important;
    color: #475569 !important;
    font-size: 11.5px !important;
    font-weight: 700 !important;
}
#card-detail {
    padding: 18px 20px !important;
    border: 1px solid var(--lm-line) !important;
    border-radius: 14px !important;
    background: #fff !important;
    box-shadow: none !important;
}
#card-detail h3 {
    margin-top: 2px !important;
    color: var(--lm-ink) !important;
}
#card-detail h3 code {
    padding: 3px 7px;
    border: 1px solid #e0e7ff;
    border-radius: 7px;
    background: #eef2ff;
    color: #3730a3;
}
#card-detail table { width: 100%; }
#card-detail th,
#card-detail td {
    padding: 7px 9px !important;
    border: 1px solid #e2e8f0 !important;
}
#card-detail thead th,
#card-detail th {
    background: #f8fafc !important;
    color: #475569 !important;
    font-size: 11.5px !important;
}

#weekly-content h2 {
    padding-bottom: 10px;
    border-bottom: 1px solid var(--lm-line);
    color: var(--lm-ink);
    font-size: 21px;
}
#weekly-content table {
    display: table !important;
    width: 100% !important;
}
#weekly-content th,
#weekly-content td {
    padding: 8px 10px !important;
    border: 1px solid #e2e8f0 !important;
}
#weekly-content thead th,
#weekly-content th {
    background: #f8fafc !important;
    color: #475569 !important;
}

@media (max-width: 760px) {
    .gradio-container { padding: 14px 12px 28px !important; }
    main.contain {
        padding-left: 4px !important;
        padding-right: 4px !important;
    }
    #hero-card { padding: 22px 18px 20px !important; border-radius: 20px !important; }
    .hero-title span { display: block; margin: 7px 0 0; }
    .hero-sub { font-size: 13px; }
    #settings-panel { top: 80px; right: 12px; }
    .tab-container { overflow-x: auto; flex-wrap: nowrap !important; }
}
"""

HERO_HTML = """
<div class="hero-copy">
  <div class="hero-kicker">Experiment knowledge workspace</div>
  <h1 class="hero-title">LabMemory <span>实验知识库</span></h1>
  <p class="hero-sub">把散落的日志、指标与结论，整理成可检索、可追溯的知识卡片。</p>
</div>
"""


def _theme():
    """低饱和蓝紫主题；字体使用系统字体，离线也能正常显示。"""
    if gr is None:
        return None
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="sky",
        neutral_hue="slate",
        radius_size="lg",
        text_size="md",
        font=["Segoe UI", "Microsoft YaHei", "PingFang SC", "Noto Sans SC",
              "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=["Consolas", "ui-monospace", "SFMono-Regular", "monospace"],
    )


def _llm(mode: str):
    if CFG.llm.offline or not CFG.llm.api_key:
        return RulePolicy(mode=mode, since_hours=CFG.scan.hours), "rule(offline)"
    return OpenAICompat(model=CFG.llm.model, base_url=CFG.llm.base_url,
                        api_key=CFG.llm.api_key, temperature=CFG.llm.temperature,
                        timeout=CFG.llm.timeout), f"llm({CFG.llm.model})"


def _data_dir() -> Path:
    """每次请求都重新解析，齿轮切换后无需重启 APP。"""
    return logdirs.current(LOGS_ROOT, UI_STATE)


def _ctx() -> Ctx:
    return Ctx(root=_data_dir(), store=Store(DB),
               state_file=DB.resolve().parent / ".labmemory" / "ingested.json",
               interactive=False)


def _folder_names() -> list[str]:
    return logdirs.names(LOGS_ROOT)


def _selected_name() -> str | None:
    try:
        return _data_dir().name
    except logdirs.LogDirError:
        return None


def _current_label() -> str:
    try:
        path = _data_dir()
    except logdirs.LogDirError as exc:
        return f"⚠️ {exc}"
    try:
        cards = Store(DB).search("", {}, 10000, root=path)
        stats = (f"　·　**{len(cards)}** 张卡片　·　"
                 f"**{len({c['exp_name'] for c in cards})}** 个实验")
    except Exception:                    # 统计失败不能影响数据集切换和页面显示
        stats = ""
    return f"**当前数据集** `{path.name}`{stats}\n\n`{path}`"


def _open_settings():
    return (gr.update(visible=True),
            gr.update(choices=_folder_names(), value=_selected_name()),
            _current_label(),
            "")


def _refresh_settings():
    return (gr.update(choices=_folder_names(), value=_selected_name()),
            _current_label(),
            "")


def _close_settings():
    return gr.update(visible=False)


def _save_settings(name: str | None):
    try:
        path = logdirs.select(LOGS_ROOT, UI_STATE, name or "")
    except logdirs.LogDirError as exc:
        return (gr.update(visible=True), _current_label(), f"❌ {exc}",
                load_weekly(), *_open_cards())
    return (gr.update(visible=False),
            f"**当前日志文件夹：** `{path.name}` · {path}",
            f"✅ 已切换到 `{path.name}`，扫描与知识卡片都会使用这个目录。",
            load_weekly(), *_open_cards())


def ask(question: str, trace: bool = True) -> str:
    question = (question or "").strip()
    if not question:
        return "请先输入问题。"
    try:
        ctx = _ctx()
    except logdirs.LogDirError as exc:
        return f"❌ {exc}"
    llm, tag = _llm("query")
    res = Agent(llm, ctx, prompts.QUERY_SYSTEM,
                max_turns=CFG.agent.max_turns_query, verbose=False).run(question)
    body = res.get("final") or res.get("error", "无输出")
    return body + (f"\n\n---\n`{tag}` · {res.get('turns')} 轮 · {res.get('elapsed')}s" if trace else "")


def scan() -> str:
    try:
        ctx = _ctx()
    except logdirs.LogDirError as exc:
        return f"❌ {exc}"
    llm, tag = _llm("ingest")
    system = prompts.INGEST_SYSTEM.replace("{tags}", "、".join(prompts.TAG_VOCAB))
    res = Agent(llm, ctx, system, max_turns=CFG.agent.max_turns_ingest,
                max_calls=CFG.agent.max_calls, verbose=False).run(
        f"数据目录：{ctx.root}\n当前日志文件夹：{ctx.root.name}\n"
        f"请处理其中最近 {CFG.scan.hours:g} 小时内新增或改动过的实验记录。")
    return (res.get("final") or res.get("error", "")) + f"\n\n---\n`{tag}`"


def weekly():
    """按当前日志文件夹生成周报，并保存到 <文件夹>/weekly/weekly.md。"""
    try:
        data = _data_dir()
    except logdirs.LogDirError as exc:
        return f"❌ {exc}"
    cards = Store(DB).recent_for_root(data, days=7)
    md = report.build_weekly(cards, days=7, scope=f"`{data.name}` · {data}")
    if not cards:
        md += ("\n\n> 当前日志文件夹还没有近 7 天的知识卡片。"
               "请先到「扫描实验记录」点击「扫描当前日志文件夹」，再生成周报。")
    out = report.save_weekly(md, data / "weekly")
    return md + f"\n\n---\n已自动保存到 `{out}`"


def load_weekly() -> str:
    """进入周报页时直接读取当前文件夹已保存的周报。"""
    try:
        data = _data_dir()
    except logdirs.LogDirError as exc:
        return f"❌ {exc}"
    path = data / "weekly" / "weekly.md"
    if not path.exists():
        return (f"当前日志文件夹 `{data.name}` 还没有生成周报。"
                "点击下方「生成当前文件夹周报」即可生成并自动保存。")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"❌ 读取周报失败：{exc}"


def _metric_summary(metrics: dict, limit: int = 2) -> str:
    items = [(k, v) for k, v in (metrics or {}).items()
             if isinstance(v, dict) and v.get("n")]
    items.sort(key=lambda kv: kv[1].get("n", 0), reverse=True)
    out = []
    for key, val in items[:limit]:
        out.append(f"{key} {val.get('first')}→{val.get('last')}")
    return "；".join(out)


def _card_row(card: dict) -> list[str]:
    return [
        card["id"],
        card["exp_name"],
        card.get("run_id") or "—",
        card["status"],
        " ".join(card.get("tags") or []),
        _metric_summary(card.get("metrics") or {}),
        "，".join(Path(f).name for f in card.get("files") or []),
        (card.get("created_at") or "")[:10],
    ]


def _card_detail(card: dict | None) -> str:
    if not card:
        return "请从上方列表选择一张卡片查看详情。"
    lines = [f"### `{card['exp_name']}` / {card.get('run_id') or '—'}", ""]
    lines.append(f"- 状态：`{card.get('status')}`　置信度：`{card.get('confidence')}`")
    lines.append(f"- 卡片 ID：`{card['id']}`　创建：`{(card.get('created_at') or '')[:19]}`")
    if card.get("goal"):
        lines += ["", f"**目标：** {card['goal']}"]
    if card.get("setup"):
        setup = "，".join(f"{k}={v}" for k, v in card["setup"].items())
        lines += ["", f"**配置：** {setup}"]
    metrics = [(k, v) for k, v in (card.get("metrics") or {}).items()
               if isinstance(v, dict) and v.get("n")]
    if metrics:
        lines += ["", "| 指标 | first | last | best | worst | trend | n |",
                  "|---|---|---|---|---|---|---|"]
        for key, val in sorted(metrics):
            lines.append(f"| {key} | {val.get('first')} | {val.get('last')} | "
                         f"{val.get('best')} | {val.get('worst')} | "
                         f"{val.get('trend')} | {val.get('n')} |")
    observations = card.get("observations") or []
    if observations:
        lines += ["", "**观测与证据：**"]
        for obs in observations:
            if isinstance(obs, dict):
                kind = obs.get("kind") or "note"
                text = obs.get("text") or ""
                evidence = f"　—　`{obs['evidence']}`" if obs.get("evidence") else ""
                lines.append(f"- **[{kind}]** {text}{evidence}")
            else:
                lines.append(f"- {obs}")
    if card.get("conclusion"):
        lines += ["", f"**结论：** {card['conclusion']}"]
    if card.get("tags"):
        lines += ["", f"**标签：** {' '.join(card['tags'])}"]
    links = card.get("links") or {}
    if links.get("supersedes"):
        lines += ["", f"**覆盖旧卡：** `{links['supersedes']}`"]
    if links.get("superseded_by"):
        lines += ["", f"**已被新卡覆盖：** `{links['superseded_by']}`"]
    if card.get("files"):
        lines += ["", "**来源文件：**"]
        lines += [f"- `{f}`" for f in card["files"]]
    return "\n".join(lines)


def _opt(value: str | None) -> str | None:
    return None if not value or value == "全部" else value


def _browse_cards(query: str, exp_name: str, run_id: str,
                  tag: str, status: str):
    filters = {
        "exp_name": _opt(exp_name),
        "run_id": _opt(run_id),
        "tag": _opt(tag),
        "status": _opt(status),
    }
    filters = {k: v for k, v in filters.items() if v}
    if filters.get("exp_name"):
        filters["exp_exact"] = True
    try:
        root = _data_dir()
    except logdirs.LogDirError as exc:
        return (f"❌ {exc}", [], gr.update(choices=[], value=None),
                "请先在右上角选择日志文件夹。")
    cards = Store(DB).search(query or "", filters, 200, root=root)
    if not cards:
        return (f"当前文件夹 `{root.name}` 没有匹配的卡片。",
                [],
                gr.update(choices=[], value=None),
                "请调整筛选条件，或先扫描实验记录。")
    rows = [_card_row(c) for c in cards]
    choices = [c["id"] for c in cards]
    summary = (f"当前文件夹 `{root.name}`：共 **{len(cards)}** 张卡片"
               "（最多显示 200 张）")
    return (summary, rows,
            gr.update(choices=choices, value=choices[0]),
            _card_detail(cards[0]))


def _open_cards():
    try:
        root = _data_dir()
    except logdirs.LogDirError as exc:
        return (gr.update(choices=["全部"], value="全部"),
                gr.update(choices=["全部"], value="全部"),
                gr.update(choices=["全部"], value="全部"),
                gr.update(choices=["全部"], value="全部"),
                f"❌ {exc}", [], gr.update(choices=[], value=None),
                "请先在右上角选择日志文件夹。")
    facets = Store(DB).facets(root=root)
    exp = ["全部"] + facets["exp_name"]
    run = ["全部"] + facets["run_id"]
    tag = ["全部"] + facets["tag"]
    status = ["全部"] + facets["status"]
    summary, rows, pick, detail = _browse_cards("", "全部", "全部", "全部", "全部")
    return (gr.update(choices=exp, value="全部"),
            gr.update(choices=run, value="全部"),
            gr.update(choices=tag, value="全部"),
            gr.update(choices=status, value="全部"),
            summary, rows, pick, detail)


def _show_card(card_id: str | None) -> str:
    if not card_id:
        return "请从上方列表选择一张卡片查看详情。"
    return _card_detail(Store(DB).to_dict(card_id))


def build_demo():
    """构建 Gradio 界面；单独成函数便于无浏览器测试。"""
    blocks_kwargs = {"title": "LabMemory · 实验记录知识卡片"}
    if GRADIO_MAJOR < 6:                 # Gradio 6 起 css 改为 launch() 参数
        blocks_kwargs.update(css=CUSTOM_CSS, theme=_theme())
    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_id="hero-card", variant="panel"):
            with gr.Row(elem_id="hero-row"):
                with gr.Column(scale=1, min_width=460):
                    gr.HTML(HERO_HTML, container=False)
                    active_md = gr.Markdown(_current_label(), elem_id="active-log",
                                            container=False)
                with gr.Column(scale=0, min_width=64):
                    settings_btn = gr.Button("⚙", elem_id="gear-btn",
                                             variant="secondary")

        with gr.Column(visible=False, elem_id="settings-panel") as settings_panel:
            gr.Markdown("### 读取设置\n选择 `logs` 下的实验文件夹。保存后，扫描与知识卡片都会使用该路径。")
            dir_dd = gr.Dropdown(choices=_folder_names(), value=_selected_name(),
                                 label="日志文件夹", allow_custom_value=False)
            with gr.Row():
                save_btn = gr.Button("保存并切换", variant="primary")
                refresh_btn = gr.Button("刷新列表")
                close_btn = gr.Button("关闭")
            settings_status = gr.Markdown()

        with gr.Tab("自然语言检索"):
            with gr.Column(elem_id="ask-panel"):
                gr.Markdown(
                    "### 从实验记录里找答案\n"
                    "直接描述你要找的实验、参数或现象，回答会附上卡片 ID 与来源文件。",
                    elem_classes="section-head", container=False)
                q = gr.Textbox(label="你的问题",
                               placeholder="上次学习率 0.001 那次的 loss 最后降到多少？",
                               elem_id="query-box", lines=2)
                gr.Examples([["上次学习率 0.001 那次的 loss 最后降到多少？"],
                             ["最近有没有出现异常或者崩溃的实验？"],
                             ["lr 放大到 0.005 效果怎么样？"]], inputs=q,
                            label="试试这些问题", elem_id="query-examples")
                ask_btn = gr.Button("开始检索", elem_id="ask-btn", variant="primary")
                a = gr.Markdown("*回答会显示在这里，并附上卡片 ID 与证据文件。*",
                                elem_id="answer-box")
            ask_btn.click(ask, inputs=q, outputs=a)
            q.submit(ask, inputs=q, outputs=a)
        with gr.Tab("扫描实验记录"):
            with gr.Column(elem_id="scan-panel"):
                gr.Markdown(
                    "### 把新记录收进知识库\n"
                    "只处理当前数据集内新增或改动过的日志、指标表与笔记，重复扫描不会重复建卡。",
                    elem_classes="section-head", container=False)
                scan_btn = gr.Button("扫描当前日志文件夹", elem_id="scan-btn",
                                     variant="primary")
                s = gr.Markdown("*准备就绪 · 扫描结果会显示在这里。*",
                                elem_id="scan-result")
            scan_btn.click(scan, outputs=s)
        with gr.Tab("知识卡片") as cards_tab:
            with gr.Column(elem_id="cards-panel"):
                gr.Markdown(
                    "### 当前数据集的知识卡片\n"
                    "按关键词、实验、run、标签或状态缩小范围，点击下方详情查看完整证据链。",
                    elem_classes="section-head", container=False)
                with gr.Group(elem_id="filter-card"):
                    with gr.Row():
                        card_q = gr.Textbox(label="关键词", scale=2,
                                            placeholder="loss / OOM / run4 / 组会")
                        card_exp = gr.Dropdown(choices=["全部"], value="全部",
                                               label="实验", scale=1)
                    with gr.Row():
                        card_run = gr.Dropdown(choices=["全部"], value="全部", label="run")
                        card_tag = gr.Dropdown(choices=["全部"], value="全部", label="标签")
                        card_status = gr.Dropdown(choices=["全部"], value="全部", label="状态")
                    with gr.Row(elem_id="filter-actions"):
                        card_search_btn = gr.Button("查询", variant="primary",
                                                    scale=0, min_width=132)
                        card_refresh_btn = gr.Button("刷新选项", variant="secondary",
                                                     scale=0, min_width=132)
                card_count = gr.Markdown("进入本页后自动加载卡片。", elem_id="card-count")
                card_table = gr.Dataframe(headers=CARD_HEADERS,
                                          datatype=["str"] * len(CARD_HEADERS),
                                          interactive=False, wrap=True,
                                          max_height=430,
                                          column_widths=[104, 146, 84, 134, 156, 220, 224, 108],
                                          elem_id="card-table",
                                          label="卡片列表", show_label=False)
                with gr.Group(elem_id="detail-card"):
                    card_pick = gr.Dropdown(choices=[], label="查看卡片详情")
                    card_detail = gr.Markdown(elem_id="card-detail")
        with gr.Tab("周报") as weekly_tab:
            with gr.Column(elem_id="weekly-panel"):
                gr.Markdown(
                    "### 当前数据集的周报\n"
                    "汇总本周实验、关键指标与异常，并自动保存到当前日志文件夹。",
                    elem_classes="section-head", container=False)
                weekly_btn = gr.Button("生成当前文件夹周报", elem_id="weekly-btn",
                                       variant="primary")
                w = gr.Markdown(elem_id="weekly-content")
            weekly_btn.click(weekly, outputs=w)

        settings_btn.click(_open_settings,
                           outputs=[settings_panel, dir_dd, active_md, settings_status])
        save_btn.click(_save_settings, inputs=dir_dd,
                       outputs=[settings_panel, active_md, settings_status, w,
                                card_exp, card_run, card_tag, card_status,
                                card_count, card_table, card_pick, card_detail])
        refresh_btn.click(_refresh_settings,
                          outputs=[dir_dd, active_md, settings_status])
        close_btn.click(_close_settings, outputs=settings_panel)
        weekly_tab.select(load_weekly, outputs=w)
        cards_tab.select(_open_cards,
                         outputs=[card_exp, card_run, card_tag, card_status,
                                  card_count, card_table, card_pick, card_detail])
        card_search_btn.click(_browse_cards,
                              inputs=[card_q, card_exp, card_run, card_tag, card_status],
                              outputs=[card_count, card_table, card_pick, card_detail])
        card_refresh_btn.click(_open_cards,
                               outputs=[card_exp, card_run, card_tag, card_status,
                                        card_count, card_table, card_pick, card_detail])
        card_pick.change(_show_card, inputs=card_pick, outputs=card_detail)
    return demo


def main() -> int:
    if gr is None:
        print("需要先安装：pip install gradio")
        return 1

    demo = build_demo()
    if GRADIO_MAJOR >= 6:
        demo.launch(server_name="127.0.0.1", server_port=7860,
                    css=CUSTOM_CSS, theme=_theme())
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
