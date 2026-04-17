#!/usr/bin/env python3
"""
解析 verl/megatron 训练日志中的 [DEBUG MEM] 行,画出显存随 step 变化的图。

日志格式示例:
  [DEBUG MEM] actor-C  after_offload_param | gstep=1 rollout=0 rank=10 \
      dev_used=5.69GB dev_free=59.75GB | proc_alloc=0.19GB proc_reserved=0.24GB

典型用法:
  # 1. 画 rank 0 所有阶段 (actor-A/B/C/D, rollout-A/B/C/D ...) 跨 step 的变化 —— 最常用
  python plot_mem.py train.log

  # 2. 把所有阶段串成一条时间轴,看一个 step 内部的波形
  python plot_mem.py train.log --mode timeline

  # 3. 只看 rollout 阶段
  python plot_mem.py train.log --tag-prefix rollout

  # 4. 多 rank 在同一阶段的对比
  python plot_mem.py train.log --compare-ranks 0,4,8,12 --tag "actor-C  after_offload_param"
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LINE_RE = re.compile(
    r"\[DEBUG MEM\]\s+"
    r"(?P<tag>\S+(?:\s+\S+)*?)\s+\|\s+"
    r"gstep=(?P<gstep>\d+)\s+"
    r"rollout=(?P<rollout>\d+)\s+"
    r"rank=(?P<rank>\d+)\s+"
    r"(?P<rest>.+)$"
)
FIELD_RE = re.compile(r"(\w+)=([\d.]+)GB")


def parse_log(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "[DEBUG MEM]" not in line:
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            row = {
                "tag": m.group("tag").strip(),
                "gstep": int(m.group("gstep")),
                "rollout": int(m.group("rollout")),
                "rank": int(m.group("rank")),
            }
            for key, val in FIELD_RE.findall(m.group("rest")):
                row[key] = float(val)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No [DEBUG MEM] lines parsed from {path}")
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} samples from {path}")
    print(f"  ranks: {sorted(df['rank'].unique())}")
    print(f"  tags:")
    for t in sorted(df['tag'].unique()):
        print(f"    - {t!r}")
    print(f"  gstep range: {df['gstep'].min()} .. {df['gstep'].max()}")
    return df


def _tag_sort_key(t):
    """让 actor-A / actor-B / actor-C / actor-D 按字母顺序排,rollout 同理。"""
    phase = t.split()[0] if t else ""
    m = re.search(r"-([A-Z])(\d*)", t)
    letter = m.group(1) if m else "Z"
    num = int(m.group(2)) if m and m.group(2) else 0
    return (phase, letter, num, t)


def _color_for_tag(tag: str) -> str:
    first = tag.split()[0].lower() if tag else ""
    m = re.search(r"-([A-Z])", tag)
    letter = m.group(1) if m else None
    if "actor" in first:
        palette = {"A": "#ffb74d", "B": "#fb8c00", "C": "#e65100", "D": "#bf360c"}
        return palette.get(letter, "#ff9800")
    if "rollout" in first:
        palette = {"A": "#90caf9", "B": "#42a5f5", "C": "#1565c0", "D": "#0d47a1"}
        return palette.get(letter, "#2196f3")
    return "#4caf50"


def plot_single_rank(df, rank, metrics, tags, tag_prefix, out):
    sub = df[df["rank"] == rank].copy()
    if tags:
        sub = sub[sub["tag"].isin(tags)]
    if tag_prefix:
        sub = sub[sub["tag"].str.startswith(tag_prefix)]
    if sub.empty:
        raise RuntimeError(f"No data for rank={rank}")

    tag_order = sorted(sub["tag"].unique(), key=_tag_sort_key)

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True, squeeze=False)

    for ax_i, metric in enumerate(metrics):
        ax = axes[ax_i][0]
        for tag in tag_order:
            g = sub[sub["tag"] == tag].sort_values("gstep")
            if metric not in g.columns or g[metric].isna().all():
                continue
            ax.plot(g["gstep"], g[metric],
                    marker="o", markersize=4, linewidth=1.3,
                    color=_color_for_tag(tag), label=tag, alpha=0.85)
        ax.set_ylabel(f"{metric} (GB)")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"rank={rank}  {metric}")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=8, frameon=False)
    axes[-1][0].set_xlabel("gstep")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}  ({len(tag_order)} tags)")


def plot_phase_timeline(df, rank, metric, out):
    """一条线从头走到尾,x 轴 = step * n_tags + tag 序号。看 step 内部波形。"""
    sub = df[df["rank"] == rank].copy()
    if sub.empty:
        raise RuntimeError(f"No data for rank={rank}")

    tag_order = sorted(sub["tag"].unique(), key=_tag_sort_key)
    tag_idx = {t: i for i, t in enumerate(tag_order)}
    n_tags = len(tag_order)

    sub["x"] = sub["gstep"] * n_tags + sub["tag"].map(tag_idx)
    sub = sub.sort_values("x")

    fig, ax = plt.subplots(figsize=(max(14, n_tags * 0.8), 5))
    ax.plot(sub["x"], sub[metric], marker="o", markersize=3.5, linewidth=1.0,
            color="#1976d2")

    # step 分隔线
    steps = sorted(sub["gstep"].unique())
    for gs in steps:
        ax.axvline(gs * n_tags - 0.5, color="gray", alpha=0.4, linewidth=0.7)

    # x 轴:step 标签
    ticks = [gs * n_tags + n_tags / 2 for gs in steps]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"step {gs}" for gs in steps])

    # 副坐标轴标 tag (只在第一个 step 下方标一次)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    first_step = steps[0]
    ax2.set_xticks([first_step * n_tags + i for i in range(n_tags)])
    ax2.set_xticklabels(tag_order, rotation=45, ha="left", fontsize=8)
    ax2.tick_params(axis="x", length=0)

    ax.set_ylabel(f"{metric} (GB)")
    ax.set_title(f"rank={rank}  {metric}  —  phase timeline across {len(steps)} steps")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


def plot_compare_ranks(df, ranks, tag, metrics, out):
    sub = df[(df["tag"] == tag) & (df["rank"].isin(ranks))].copy()
    if sub.empty:
        raise RuntimeError(f"No data for tag={tag!r} ranks={ranks}")
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True, squeeze=False)
    for ax_i, metric in enumerate(metrics):
        ax = axes[ax_i][0]
        for rank, g in sub.groupby("rank"):
            g = g.sort_values("gstep")
            if metric in g.columns and not g[metric].isna().all():
                ax.plot(g["gstep"], g[metric], marker="o", markersize=4,
                        label=f"rank={rank}", linewidth=1.2)
        ax.set_ylabel(f"{metric} (GB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"tag={tag}  {metric}")
    axes[-1][0].set_xlabel("gstep")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tags", default="", help="逗号分隔,精确匹配")
    ap.add_argument("--tag-prefix", default="",
                    help="只保留以此开头的 tag,例如 'rollout' 或 'actor'")
    ap.add_argument("--metrics", default="proc_alloc,proc_reserved,dev_used,dev_free")
    ap.add_argument("--mode", choices=["per_tag", "timeline"], default="per_tag",
                    help="per_tag: 每个 tag 一条线跨 step; timeline: 串成单条时间轴")
    ap.add_argument("--timeline-metric", default="proc_reserved",
                    help="timeline 模式下画哪个指标")
    ap.add_argument("--compare-ranks", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump-csv", type=Path, default=None)
    args = ap.parse_args()

    df = parse_log(args.log)
    if args.dump_csv:
        df.to_csv(args.dump_csv, index=False)
        print(f"Dumped CSV to {args.dump_csv}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    if args.compare_ranks:
        ranks = [int(x) for x in args.compare_ranks.split(",")]
        if not args.tag:
            raise SystemExit("--compare-ranks 需要配合 --tag")
        out = args.out or Path(f"mem_compare_{args.tag.replace(' ', '_')}.png")
        plot_compare_ranks(df, ranks, args.tag, metrics, out)
    elif args.mode == "timeline":
        out = args.out or Path(f"mem_rank{args.rank}_timeline.png")
        plot_phase_timeline(df, args.rank, args.timeline_metric, out)
    else:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        out = args.out or Path(f"mem_rank{args.rank}.png")
        plot_single_rank(df, args.rank, metrics, tags, args.tag_prefix, out)


if __name__ == "__main__":
    main()
