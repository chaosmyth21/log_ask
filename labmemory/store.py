"""LabMemory 存储层：SQLite + FTS5，零外部依赖。

设计要点：
1. cards 表存结构化卡片，params 表把 setup 摊平成可精确过滤的键值对；
2. cards_fts 对"中文逐字空格化"后的文本建索引，使中文可按字匹配，
   再用 bm25 排序 —— 不引入向量库也能有可用的相关性排序；
3. 同一 (exp_name, run_id) 重复保存时生成新卡并把旧卡标记 superseded，
   形成"实验链"而不是一堆孤立笔记。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_DEFAULT = Path(__file__).resolve().parent.parent / "labmemory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    exp_name    TEXT NOT NULL,
    run_id      TEXT,
    goal        TEXT,
    setup_json  TEXT,      -- {"lr": 0.001, "batch": 32, ...}
    metrics_json TEXT,     -- {"loss": {"first":..,"last":..,"best":..,"worst":..,"trend":"down","n":120}}
    obs_json    TEXT,      -- [{"kind":"crash","text":"...","evidence":"file:1187"}]
    conclusion  TEXT,
    tags_json   TEXT,      -- ["#异常","#lr","#OOM"]
    status      TEXT,      -- active | failed | needs_review | superseded
    confidence  REAL,
    files_json  TEXT,      -- ["E:\\...\\exp_lr001_run3.log"]
    links_json  TEXT       -- {"supersedes": "...", "superseded_by": "...", "related": [...]}
);
CREATE INDEX IF NOT EXISTS idx_cards_exp ON cards(exp_name, run_id);
CREATE INDEX IF NOT EXISTS idx_cards_created ON cards(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
    id UNINDEXED, body, tags, tokenize="unicode61 remove_diacritics 2"
);

CREATE TABLE IF NOT EXISTS params (
    card_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT, num REAL
);
CREATE INDEX IF NOT EXISTS idx_params_kv ON params(key, value);

CREATE TABLE IF NOT EXISTS events (
    ts TEXT, kind TEXT, ref TEXT, detail TEXT
);
"""

_CJK = re.compile(r"[\u3400-\u4dff\u4e00-\u9fff]")


def _cq(text: str) -> str:
    """把中文逐字用空格隔开，让 unicode61 分词器产出单字 token。"""
    return re.sub(r"\s+", " ", _CJK.sub(lambda m: f" {m.group(0)} ", text or "")).strip()


def _fts_query(q: str) -> str:
    toks = [t for t in _cq(q).split() if t]
    # 单字 AND 组合：召回优先，排序交给 bm25
    return " AND ".join(f'"{t}"' for t in toks)


def _num_tokenize(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _inside(path: str, root: Path) -> bool:
    """判断卡片来源文件是否位于某个日志文件夹内。"""
    try:
        p = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return False
    return p == root or root in p.parents


class Store:
    def __init__(self, path: Path | str = DB_DEFAULT):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        try:                                  # 读写并发（watch 在写、ask 在读）更稳
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:
            pass
        self.conn.commit()

    # ---------------- 写入 ----------------
    def save_card(self, card: dict) -> dict:
        # 微秒精度：样例/批量导入时多个源文件可能拿到完全相同的文件 mtime，
        # 若这里只记到秒，实验链会退化成按插入顺序决胜，结果不稳定。
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        exp = (card.get("exp_name") or "untitled").strip()
        run = card.get("run_id")
        setup = card.get("setup") or {}
        metrics = card.get("metrics") or {}
        obs = card.get("observations") or []
        tags = list(dict.fromkeys(card.get("tags") or []))
        status = card.get("status") or "active"
        files = card.get("files") or []
        links = card.get("links") or {}

        # 同一 exp+run 且来源文件完全相同 = 同一份证据，原地更新那张卡。
        # 不这么做，模型每次"再存一遍"都会新建一张、把上一张标成 superseded，
        # 它看到自己被覆盖又会再存 —— 一个真实的死循环就是这么来的。
        cid = card.get("id")
        updated = False
        if not cid:
            cid = self._find_same_evidence(exp, run, files)
            updated = cid is not None
        cid = cid or f"c-{uuid.uuid4().hex[:8]}"

        with self.conn:                      # 事务：任一步失败整体回滚，不留半张卡
            self._write_card(cid, now, exp, run, card, setup, metrics, obs,
                             tags, status, files, links)
        out = self.to_dict(cid)
        out["_updated"] = updated          # 给工具层判断是新建还是原地更新
        return out

    def _write_card(self, cid, now, exp, run, card, setup, metrics, obs,
                    tags, status, files, links):
        self.conn.execute(
            """INSERT OR REPLACE INTO cards
               (id, created_at, exp_name, run_id, goal, setup_json, metrics_json,
                obs_json, conclusion, tags_json, status, confidence, files_json, links_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                cid, now, exp, run, card.get("goal", ""),
                json.dumps(setup, ensure_ascii=False),
                json.dumps(metrics, ensure_ascii=False),
                json.dumps(obs, ensure_ascii=False),
                card.get("conclusion", ""), json.dumps(tags, ensure_ascii=False),
                status, float(card.get("confidence", 0.6) or 0.6),
                json.dumps(files, ensure_ascii=False),
                json.dumps(links, ensure_ascii=False),
            ),
        )

        body = " ".join([
            exp, run or "", card.get("goal", ""), card.get("conclusion", ""),
            " ".join(f"{k}={v}" for k, v in setup.items()),
            # 指标名必须进检索正文，否则"loss 降到多少"这类问句无法命中
            " ".join(metrics.keys()),
            " ".join(f"{k} {v.get('last')}" for k, v in metrics.items() if isinstance(v, dict)),
            " ".join(o.get("text", "") if isinstance(o, dict) else str(o) for o in obs),
        ])
        self.conn.execute("DELETE FROM cards_fts WHERE id=?", (cid,))
        self.conn.execute(
            "INSERT INTO cards_fts (id, body, tags) VALUES (?,?,?)",
            (cid, _cq(body), _cq(" ".join(tags))),
        )

        self.conn.execute("DELETE FROM params WHERE card_id=?", (cid,))
        for k, v in setup.items():
            self.conn.execute(
                "INSERT INTO params (card_id, key, value, num) VALUES (?,?,?,?)",
                (cid, str(k).lower(), str(v), _num_tokenize(v)),
            )
        # 同 exp+run 的卡片统一对账：证据最新的那张留下，其余标 superseded。
        # 按"谁最后写入"决定胜负是不对的 —— 写入顺序取决于遍历顺序，不代表文件新旧。
        self._reconcile_run(exp, run)


    def _find_same_evidence(self, exp: str, run, files: list) -> str | None:
        """找一张"同实验同 run 且来源文件集合相同"的有效卡片。"""
        want = {str(f) for f in files}
        if not want:
            return None
        rows = self.conn.execute(
            "SELECT id, files_json FROM cards "
            "WHERE exp_name=? AND IFNULL(run_id,'')=IFNULL(?,'') AND status!='superseded'",
            (exp, run),
        ).fetchall()
        for r in rows:
            try:
                have = {str(f) for f in json.loads(r["files_json"] or "[]")}
            except (json.JSONDecodeError, TypeError):
                continue
            if have == want:
                return r["id"]
        return None

    def _reconcile_run(self, exp: str, run) -> None:
        """同一 exp+run 下只留证据最新的那张卡，其余 superseded 并双向连线。"""
        rows = self.conn.execute(
            "SELECT id, files_json, created_at FROM cards "
            "WHERE exp_name=? AND IFNULL(run_id,'')=IFNULL(?,'')",
            (exp, run),
        ).fetchall()
        if len(rows) < 2:
            return

        def evidence_time(row):
            try:
                paths = json.loads(row["files_json"] or "[]")
                stamps = [os.path.getmtime(f) for f in paths if isinstance(f, str)]
                return (max(stamps) if stamps else 0.0, row["created_at"])
            except (OSError, ValueError, TypeError):
                return (0.0, row["created_at"])

        ranked = sorted(rows, key=evidence_time, reverse=True)
        keep, losers = ranked[0]["id"], [r["id"] for r in ranked[1:]]
        for oid in losers:
            self.conn.execute(
                "UPDATE cards SET status='superseded', links_json=? WHERE id=?",
                (json.dumps({"superseded_by": keep}, ensure_ascii=False), oid),
            )
        row = self.conn.execute("SELECT links_json FROM cards WHERE id=?", (keep,)).fetchone()
        links = json.loads(row["links_json"] or "{}") if row else {}
        links["supersedes"] = losers[0]
        if len(losers) > 1:
            links["supersedes_all"] = losers
        self.conn.execute("UPDATE cards SET status='active', links_json=? WHERE id=?",
                          (json.dumps(links, ensure_ascii=False), keep))

    def log_event(self, kind: str, ref: str = "", detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO events (ts, kind, ref, detail) VALUES (?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), kind, ref, detail[:500]),
        )
        self.conn.commit()

    # ---------------- 读取 ----------------
    def to_dict(self, cid: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM cards WHERE id=?", (cid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        return {
            "id": d["id"], "created_at": d["created_at"], "exp_name": d["exp_name"],
            "run_id": d["run_id"], "goal": d["goal"],
            "setup": json.loads(d["setup_json"] or "{}"),
            "metrics": json.loads(d["metrics_json"] or "{}"),
            "observations": json.loads(d["obs_json"] or "[]"),
            "conclusion": d["conclusion"], "tags": json.loads(d["tags_json"] or "[]"),
            "status": d["status"], "confidence": d["confidence"],
            "files": json.loads(d["files_json"] or "[]"),
            "links": json.loads(d["links_json"] or "{}"),
        }

    def search(self, query: str = "", filters: dict | None = None, k: int = 5,
               root: str | Path | None = None) -> list[dict]:
        """结构化过滤 + FTS 排序。

        filters 支持 exp_name / tag / status / param_key / param_value / days。
        传入 root 时，只保留来源文件位于该日志文件夹内的卡片。
        """
        filters = filters or {}
        sql = "SELECT id FROM cards WHERE 1=1"
        args: list = []
        if filters.get("status"):
            sql += " AND status = ?"
            args.append(filters["status"])
        if filters.get("exp_name"):
            if filters.get("exp_exact"):
                sql += " AND exp_name = ?"
                args.append(str(filters["exp_name"]))
            else:
                sql += " AND exp_name LIKE ?"
                args.append(f"%{filters['exp_name']}%")
        if filters.get("run_id"):
            sql += " AND IFNULL(run_id,'') = ?"
            args.append(str(filters["run_id"]))
        if filters.get("days"):
            cutoff = (datetime.now(timezone.utc) - timedelta(days=int(filters["days"]))).isoformat()
            sql += " AND created_at >= ?"
            args.append(cutoff)
        rows = {r["id"] for r in self.conn.execute(sql, args)}
        if root is not None:
            rows &= self._ids_under_root(root)

        if filters.get("tag"):
            tag = str(filters["tag"])
            tag = tag if tag.startswith("#") else f"#{tag}"
            rows &= {c["id"] for c in self._all_cards() if tag in (c["tags"] or [])}
        if filters.get("param_key"):
            q = "SELECT card_id FROM params WHERE key=? AND value=?"
            args = [str(filters["param_key"]).lower(), str(filters.get("param_value"))]
            rows &= {r["card_id"] for r in self.conn.execute(q, args)}

        if not query:
            ordered = self._order(rows)
            return [self.to_dict(i) for i in ordered[:k]]

        fts = _fts_query(query)
        if not fts:
            return [self.to_dict(i) for i in sorted(rows, reverse=True)[:k]]
        scored = []
        try:
            for r in self.conn.execute(
                "SELECT id, bm25(cards_fts) AS s FROM cards_fts "
                "WHERE cards_fts MATCH ? ORDER BY s",
                (fts,),
            ):
                if r["id"] in rows:
                    scored.append((r["s"], r["id"]))
                    if len(scored) >= max(k, 200):
                        break
        except sqlite3.OperationalError:
            scored = []
        if scored:
            return [self.to_dict(i) for _, i in self._rank(scored)[:k]]
        # 中文单字 AND 全不命中时退化为"任意字命中"，避免零结果
        toks = [t for t in _cq(query).split() if t]
        if not toks:
            return []
        or_q = " OR ".join(f'"{t}"' for t in toks)
        scored = [
            (r["s"], r["id"]) for r in self.conn.execute(
                "SELECT id, bm25(cards_fts) AS s FROM cards_fts "
                "WHERE cards_fts MATCH ? ORDER BY s",
                (or_q,),
            ) if r["id"] in rows
        ]
        return [self.to_dict(i) for _, i in self._rank(scored)[:k]]

    def _rank(self, scored: list[tuple[float, str]]) -> list[tuple[float, str]]:
        """相关度为主、有效状态为辅、同档内按时间新→旧。

        被覆盖的旧卡仍然可检索（"run3 崩过"是历史事实，不该丢），
        但排在有效卡片之后，回答里会标出它的状态。
        """
        meta = {
            r["id"]: (r["status"], r["created_at"])
            for r in self.conn.execute("SELECT id, status, created_at FROM cards")
        }
        scored = sorted(scored, key=lambda it: meta.get(it[1], ("", ""))[1], reverse=True)
        # round(...,1)：把相近的 bm25 视作同一档，档内保留"新的在前"
        return sorted(scored, key=lambda it: (meta.get(it[1], ("", ""))[0] == "superseded",
                                              round(it[0], 1)))

    def _order(self, rows: set[str]) -> list[str]:
        """有效卡片在前，各自按时间倒序。"""
        active, old = [], []
        for cid in rows:
            row = self.conn.execute("SELECT status FROM cards WHERE id=?", (cid,)).fetchone()
            (old if row and row["status"] == "superseded" else active).append(cid)
        key = lambda i: self._created(i)
        return sorted(active, key=key, reverse=True) + sorted(old, key=key, reverse=True)

    def _created(self, cid: str) -> str:
        row = self.conn.execute("SELECT created_at FROM cards WHERE id=?", (cid,)).fetchone()
        return row["created_at"] if row else ""

    def _all_cards(self) -> list[dict]:
        return [
            {"id": r["id"], "tags": json.loads(r["tags_json"] or "[]")}
            for r in self.conn.execute("SELECT id, tags_json FROM cards")
        ]

    def recent(self, days: int = 7, limit: int = 100) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return [
            self.to_dict(r["id"]) for r in self.conn.execute(
                "SELECT id FROM cards WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (cutoff, limit),
            )
        ]

    def recent_for_root(self, root: str | Path, days: int = 7,
                        limit: int = 100) -> list[dict]:
        """只取来源文件位于指定日志文件夹内的卡片。

        周报必须跟着 APP 右上角当前选择的 test1 / test2 / test3 走，
        不能在切换文件夹后仍把其他数据集的卡片混进来。
        """
        base = Path(root).expanduser().resolve()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        out: list[dict] = []
        rows = self.conn.execute(
            "SELECT id, files_json FROM cards WHERE created_at >= ? "
            "ORDER BY created_at DESC",
            (cutoff,),
        )
        for row in rows:
            try:
                files = json.loads(row["files_json"] or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            if not any(isinstance(f, str) and _inside(f, base) for f in files):
                continue
            card = self.to_dict(row["id"])
            if card:
                out.append(card)
            if len(out) >= limit:
                break
        return out

    def _ids_under_root(self, root: str | Path) -> set[str]:
        """返回来源文件位于指定日志文件夹内的卡片 ID。"""
        base = Path(root).expanduser().resolve()
        out: set[str] = set()
        for row in self.conn.execute("SELECT id, files_json FROM cards"):
            try:
                files = json.loads(row["files_json"] or "[]")
            except (json.JSONDecodeError, TypeError):
                continue
            if any(isinstance(f, str) and _inside(f, base) for f in files):
                out.add(row["id"])
        return out

    def counts(self) -> dict:
        out = {}
        for r in self.conn.execute("SELECT status, COUNT(*) n FROM cards GROUP BY status"):
            out[r["status"]] = r["n"]
        return out

    def facets(self, root: str | Path | None = None) -> dict:
        """卡片浏览页的筛选项：实验、run、标签、状态。

        传入 root 时，只从该日志文件夹内的卡片收集选项。
        """
        scoped = self._ids_under_root(root) if root is not None else None
        exps: set[str] = set()
        runs: set[str] = set()
        statuses: set[str] = set()
        tags: set[str] = set()
        for row in self.conn.execute(
            "SELECT id, exp_name, run_id, status, tags_json FROM cards"
        ):
            if scoped is not None and row["id"] not in scoped:
                continue
            exps.add(row["exp_name"])
            if row["run_id"]:
                runs.add(row["run_id"])
            if row["status"]:
                statuses.add(row["status"])
            try:
                tags.update(t for t in json.loads(row["tags_json"] or "[]") if t)
            except (json.JSONDecodeError, TypeError):
                continue
        return {
            "exp_name": sorted(exps, key=str.lower),
            "run_id": sorted(runs, key=str.lower),
            "status": sorted(statuses, key=str.lower),
            "tag": sorted(tags),
        }
