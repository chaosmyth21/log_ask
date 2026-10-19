"""生成一套可复现的样例实验数据（不含真实实验内容，用于演示与自测）。

场景覆盖知识管理最典型的四类情况：
  1. 一次正常收敛的 run；
  2. 一次学得太慢、后期停滞的 run（换了个更大的 lr）；
  3. 一次跑到一半 OOM 崩掉的 run；
  4. 同一条命令隔天重跑成功的 run —— 与 3 同名同 run，会形成"实验链"；
  5. 一个超参扫描表 + 一份手写组会笔记 + 一次评测的 jsonl。
"""

from __future__ import annotations

import json
import random
from pathlib import Path


DATASET_NAMES = ("test1", "test2")


def _train_log(path: Path, steps: int, l0: float, l1: float, a0: float, a1: float,
               crash_at: int | None = None, plateau_from: int | None = None,
               header_note: str = "", dataset: str = "sft-v3",
               loss_delta: float = 0.0, acc_delta: float = 0.0,
               lr: float = 0.001, batch: int = 32, seq_len: int | None = None,
               seed: int = 42, date: str = "2026-10-13") -> None:
    rnd = random.Random(hash(path.name) & 0xFFFF)
    seq = f" seq_len={seq_len}" if seq_len else ""
    lines = [
        f"[{date} 09:12:03] config: model=qwen3-7b lr={lr:g} batch_size={batch} "
        f"seed={seed} dataset={dataset} max_steps={steps}{seq}"
        + (f' note="{header_note}"' if header_note else ""),
        f"[{date} 09:12:04] start training",
    ]
    for s in range(10, steps + 1, 10):
        t = s / steps
        loss = max(0.0, l0 + (l1 - l0) * (t ** 0.55) + loss_delta
                   + rnd.uniform(-0.012, 0.012))
        acc = min(1.0, max(0.0, a0 + (a1 - a0) * (t ** 0.7) + acc_delta
                           + rnd.uniform(-0.004, 0.004)))
        if plateau_from and s >= plateau_from:
            loss = max(0.0, l1 + loss_delta + 0.004 * ((s // 10) % 2))  # 停滞：在末值附近抖动
            acc = min(1.0, max(0.0, a1 + acc_delta - 0.002 * ((s // 10) % 2)))
        lines.append(f"step {s} loss={loss:.4f} acc={acc:.4f} lr={lr:g} grad_norm={rnd.uniform(0.4, 2.1):.3f} "
                     f"tokens_per_s={rnd.randint(1800, 2400)}")
        if crash_at and s >= crash_at:
            lines += [
                "Traceback (most recent call last):",
                '  File "train.py", line 212, in <module>',
                "    trainer.fit(model, datamodule)",
                '  File "train.py", line 178, in fit',
                "    loss = model(batch)",
                "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB "
                "(GPU 0; 23.65 GiB total capacity; 21.98 GiB already allocated)",
            ]
            break
    if not crash_at:
        lines.append(f"[{date} 11:02:44] training finished, "
                     f"eval_loss={max(0.0, l1 + loss_delta):.4f}, "
                     f"eval_acc={min(1.0, max(0.0, a1 + acc_delta)):.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _next_dataset_dir(logs_root: Path) -> Path:
    """返回下一个可用的数据集目录：logs/test1，其次 logs/test2。

    前两次生成分别落在 test1、test2；两次都用掉之后再生成会报错，
    避免把第三份数据混进已有数据集。
    """
    logs_root = Path(logs_root)
    for name in DATASET_NAMES:
        candidate = logs_root / name
        if not candidate.exists():
            return candidate
    raise ValueError(
        f"{logs_root} 下 {', '.join(DATASET_NAMES)} 都已存在；"
        f"请换一个 target，或先清理旧数据再重新生成。")


def generate(root: Path, suffix: str = "", dataset: str | None = None,
             loss_delta: float = 0.0, acc_delta: float = 0.0,
             flat: bool = False) -> int:
    """生成一套样例数据。

    默认第一次生成到 ``<root>/logs/test1``，第二次到 ``<root>/logs/test2``；
    flat=True 时把日期目录直接放在 root 下（旧布局）。
    suffix 用于区分不同数据集（例如 ``_test2``），会同时进入文件名和指标。
    """
    root = Path(root)
    if flat:
        data_root = root
    else:
        data_root = _next_dataset_dir(root / "logs")
    data_root.mkdir(parents=True, exist_ok=True)
    notes_root = data_root / "notes"
    notes_root.mkdir(parents=True, exist_ok=True)
    dataset = dataset or f"sft-v3{suffix}"
    made = 0

    # 1. 正常收敛
    p = data_root / "2026-10-11" / f"lora-sft{suffix}_lr0001_bs32_run2.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    _train_log(p, 1200, 1.821, 0.641, 0.312, 0.712, dataset=dataset,
               loss_delta=loss_delta, acc_delta=acc_delta)
    made += 1

    # 2. lr 放大 5 倍：下降更慢、后期停滞
    p = data_root / "2026-10-12" / f"lora-sft{suffix}_lr0005_bs32_run1.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    _train_log(p, 1200, 1.804, 0.724, 0.298, 0.661, plateau_from=900,
               dataset=dataset, loss_delta=loss_delta, acc_delta=acc_delta)
    made += 1

    # 3. 崩在 1180 步
    p = data_root / "2026-10-13" / f"lora-sft{suffix}_lr0001_bs32_run3.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    _train_log(p, 2000, 1.792, 0.312, 0.305, 0.845, crash_at=1180,
               dataset=dataset, loss_delta=loss_delta, acc_delta=acc_delta)
    made += 1

    # 4. 隔天改 seq_len 重跑成功（同名同 run -> 与 3 形成实验链）
    p = data_root / "2026-10-14" / f"lora-sft{suffix}_lr0001_bs32_run3.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    _train_log(p, 1600, 1.744, 0.287, 0.310, 0.861,
               header_note="retry of run3 after CUDA OOM; seq_len 2048 -> 1024",
               dataset=dataset, loss_delta=loss_delta, acc_delta=acc_delta)
    made += 1

    # 5. 超参扫描表
    p = data_root / "2026-10-12" / f"sweep_lr{suffix}.csv"
    rows = ["epoch,lr,loss,f1,acc,steps_per_s"]
    for lr, base in ((0.0005, 0.98), (0.001, 0.66), (0.005, 0.93)):
        for e in range(1, 5):
            loss = max(0.0, base - 0.08 * e + loss_delta)
            acc = min(1.0, max(0.0, 0.61 + 0.04 * e + acc_delta))
            rows.append(f"{e},{lr},{loss:.4f},{0.42 + 0.03 * e:.4f},"
                        f"{acc:.4f},{18 + e}")
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    made += 1

    # 6. 评测结果（jsonl）
    p = data_root / "2026-10-13" / f"eval_20261013{suffix}.jsonl"
    recs = []
    for i, step in enumerate((200, 600, 1000, 1400)):
        recs.append({"step": step,
                     "eval_loss": round(max(0.0, 0.72 - 0.09 * i + loss_delta), 4),
                     "map": round(0.38 + 0.05 * i, 4),
                     "acc": round(min(1.0, max(0.0, 0.63 + 0.06 * i + acc_delta)), 4)})
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    made += 1

    # 7. 手写组会笔记
    p = notes_root / f"2026-10-13{suffix}_组会笔记.md"
    p.write_text(
        "# 组会笔记 2026-10-13\n\n"
        "- 这周主要调 lr。lr=0.001 收敛最快，1200 步就能到 0.64；放大到 0.005 反而停滞在 0.72。\n"
        "- run3 跑到 1180 步 OOM 崩了，怀疑是 seq_len=2048 太长，显存吃满。\n"
        "- 下一步：把 seq_len 降到 1024 重跑 run3，看能不能到 0.3 以下；顺便把 batch 降到 16 试试。\n"
        "- 老张建议加上 grad accumulation，别直接放大 batch。\n",
        encoding="utf-8")
    made += 1

    return made


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成可复现的样例实验数据")
    parser.add_argument("target", nargs="?", default="./demo_data")
    parser.add_argument("--suffix", default="", help="文件名与实验名后缀，如 _test2")
    parser.add_argument("--dataset", default=None, help="日志里 dataset= 的值")
    parser.add_argument("--loss-delta", type=float, default=0.0)
    parser.add_argument("--acc-delta", type=float, default=0.0)
    parser.add_argument("--flat", action="store_true",
                        help="日期目录直接放在 target 下（APP 的 test1/test2 布局）")
    args = parser.parse_args()
    target = Path(args.target)
    out_dir = target if args.flat else _next_dataset_dir(target / "logs")
    n = generate(target, suffix=args.suffix, dataset=args.dataset,
                 loss_delta=args.loss_delta, acc_delta=args.acc_delta,
                 flat=args.flat)
    print(f"已生成 {n} 个文件 → {out_dir.resolve()}")
