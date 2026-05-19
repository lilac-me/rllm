#!/usr/bin/env python3
"""Parse KernelGym replay-training logs and plot precision traces for custom runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+) - (.*)")

DEFAULT_RUNS = {
    "gpu": "rllm/gpu_training.log",
    "npu_attempt1": "rllm/npu_training.log",
    "npu_attempt2": "rllm/npu_training_v2.log",
}

INPUT_INVARIANT_METRICS = [
    "batch/solve_none",
    "batch/solve_all",
    "batch/solve_partial",
    "critic/full-score/mean",
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "critic/returns/mean",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
    "response/aborted_ratio",
    "prompt_length/mean",
    "global_seqlen/mean",
    "global_seqlen/minmax_diff",
]

PROBABILITY_METRICS = [
    "rollout_corr/rollout_ppl",
    "rollout_corr/training_ppl",
    "rollout_corr/log_ppl_diff",
    "rollout_corr/log_ppl_abs_diff",
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
    "training/rollout_probs_diff_mean",
    "training/rollout_probs_diff_std",
    "training/rollout_probs_diff_max",
]

UPDATE_METRICS = [
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/entropy",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/pg_clipfrac_lower",
    "actor/lr",
]

CORE_PLOT_METRICS = [
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/entropy",
    "actor/ppo_kl",
    "rollout_corr/kl",
    "training/rollout_probs_diff_mean",
]

STYLE_CYCLE = [
    {"marker": "o", "linestyle": "-", "color": "#1f77b4"},
    {"marker": "s", "linestyle": "--", "color": "#ff7f0e"},
    {"marker": "^", "linestyle": "-.", "color": "#2ca02c"},
    {"marker": "D", "linestyle": ":", "color": "#d62728"},
    {"marker": "P", "linestyle": "-", "color": "#9467bd"},
    {"marker": "X", "linestyle": "--", "color": "#8c564b"},
    {"marker": "v", "linestyle": "-.", "color": "#e377c2"},
    {"marker": "h", "linestyle": ":", "color": "#7f7f7f"},
]


def parse_step_metrics(path: Path) -> dict[int, dict[str, float]]:
    metrics_by_step: dict[int, dict[str, float]] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = STEP_RE.search(line)
        if not match:
            continue

        step = int(match.group(1))
        values: dict[str, float] = {}
        for chunk in match.group(2).split(" - "):
            if ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            try:
                values[key] = float(value)
            except ValueError:
                continue
        metrics_by_step[step] = values
    return metrics_by_step


def write_step_csv(out_path: Path, data: dict[str, dict[int, dict[str, float]]]) -> None:
    all_metrics = sorted({key for steps in data.values() for row in steps.values() for key in row})
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "step", *all_metrics])
        for run_name, steps in data.items():
            for step in sorted(steps):
                row = steps[step]
                writer.writerow([run_name, step, *[row.get(metric, "") for metric in all_metrics]])


def summarize_pair(
    base_name: str,
    other_name: str,
    base: dict[int, dict[str, float]],
    other: dict[int, dict[str, float]],
    metrics: list[str],
) -> list[dict[str, object]]:
    common_steps = sorted(set(base) & set(other))
    rows: list[dict[str, object]] = []
    for metric in metrics:
        diffs: list[tuple[int, float, float, float]] = []
        for step in common_steps:
            if metric not in base[step] or metric not in other[step]:
                continue
            base_value = base[step][metric]
            other_value = other[step][metric]
            diffs.append((step, abs(base_value - other_value), base_value, other_value))

        if not diffs:
            continue

        max_step, max_abs_diff, base_value, other_value = max(diffs, key=lambda item: item[1])
        mean_abs_diff = sum(item[1] for item in diffs) / len(diffs)
        rows.append(
            {
                "pair": f"{base_name}_vs_{other_name}",
                "metric": metric,
                "common_steps": len(diffs),
                "max_abs_diff": max_abs_diff,
                "max_diff_step": max_step,
                "base_run": base_name,
                "base_value_at_max": base_value,
                "other_run": other_name,
                "other_value_at_max": other_value,
                "mean_abs_diff": mean_abs_diff,
            }
        )
    return rows


def write_summary_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "pair",
        "metric",
        "common_steps",
        "max_abs_diff",
        "max_diff_step",
        "base_run",
        "base_value_at_max",
        "other_run",
        "other_value_at_max",
        "mean_abs_diff",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_display_offsets(run_names: list[str], line_offset_dots: float) -> dict[str, float]:
    if len(run_names) <= 1 or line_offset_dots <= 0:
        return {run_name: 0.0 for run_name in run_names}
    center_index = (len(run_names) - 1) / 2.0
    return {run_name: (index - center_index) * line_offset_dots for index, run_name in enumerate(run_names)}


def style_map_for_runs(run_names: list[str]) -> dict[str, dict[str, object]]:
    styles: dict[str, dict[str, object]] = {}
    for idx, run_name in enumerate(run_names):
        styles[run_name] = STYLE_CYCLE[idx % len(STYLE_CYCLE)]
    return styles


def plot_metric_grid(
    out_path: Path,
    data: dict[str, dict[int, dict[str, float]]],
    metrics: list[str],
    title: str,
    line_offset_dots: float,
    max_cols: int = 3,
) -> None:
    run_names = list(data.keys())
    if not run_names:
        return

    styles = style_map_for_runs(run_names)
    display_offsets = build_display_offsets(run_names, line_offset_dots)

    n_cols = min(max_cols, len(metrics))
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.4 * n_rows), squeeze=False)
    for index, metric in enumerate(metrics):
        ax = axes[index // n_cols][index % n_cols]
        min_step: int | None = None
        max_step: int | None = None
        min_y: float | None = None
        max_y: float | None = None
        has_series = False
        for run_index, run_name in enumerate(run_names):
            steps = data[run_name]
            x_values = [step for step in sorted(steps) if metric in steps[step]]
            y_values = [steps[step][metric] for step in x_values]
            if x_values:
                has_series = True
                min_step = min(x_values) if min_step is None else min(min_step, min(x_values))
                max_step = max(x_values) if max_step is None else max(max_step, max(x_values))
                y_lo = min(y_values)
                y_hi = max(y_values)
                min_y = y_lo if min_y is None else min(min_y, y_lo)
                max_y = y_hi if max_y is None else max(max_y, y_hi)
                style = styles[run_name]
                x_offset = display_offsets[run_name]
                transform = (
                    ax.transData
                    if x_offset == 0.0
                    else offset_copy(ax.transData, fig=fig, x=x_offset, y=0, units="dots")
                )
                ax.plot(
                    x_values,
                    y_values,
                    linewidth=1.9,
                    markersize=4.4,
                    markeredgewidth=0.75,
                    markeredgecolor="white",
                    alpha=0.95,
                    label=run_name,
                    transform=transform,
                    zorder=10 + run_index,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    color=style["color"],
                )
        if has_series:
            ax.set_title(metric)
            ax.set_xlabel("step")
            if min_step is not None and max_step is not None:
                pad = 0.35
                ax.set_xlim(min_step - pad, max_step + pad)
            if min_y is not None and max_y is not None:
                if min_y == max_y:
                    y_pad = max(abs(min_y) * 0.02, 1e-6)
                else:
                    y_pad = max((max_y - min_y) * 0.08, 1e-6)
                ax.set_ylim(min_y - y_pad, max_y + y_pad)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{metric} (no data)")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

    for index in range(len(metrics), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(run_names)), frameon=False)
    fig.suptitle(title, y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_grid(
    out_path: Path,
    data: dict[str, dict[int, dict[str, float]]],
    metrics: list[str],
    baseline_name: str,
    line_offset_dots: float,
) -> None:
    if baseline_name not in data:
        return

    compare_runs = [name for name in data if name != baseline_name]
    if not compare_runs:
        return

    styles = style_map_for_runs(compare_runs)
    display_offsets = build_display_offsets(compare_runs, line_offset_dots)
    baseline = data[baseline_name]

    n_cols = 3
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.4 * n_rows), squeeze=False)

    for index, metric in enumerate(metrics):
        ax = axes[index // n_cols][index % n_cols]
        min_step: int | None = None
        max_step: int | None = None
        min_y: float | None = None
        max_y: float | None = None
        has_series = False
        for run_index, run_name in enumerate(compare_runs):
            common_steps = sorted(set(baseline) & set(data[run_name]))
            x_values = [
                step
                for step in common_steps
                if metric in baseline[step] and metric in data[run_name][step]
            ]
            y_values = [abs(baseline[step][metric] - data[run_name][step][metric]) for step in x_values]
            if x_values:
                has_series = True
                min_step = min(x_values) if min_step is None else min(min_step, min(x_values))
                max_step = max(x_values) if max_step is None else max(max_step, max(x_values))
                y_lo = min(y_values)
                y_hi = max(y_values)
                min_y = y_lo if min_y is None else min(min_y, y_lo)
                max_y = y_hi if max_y is None else max(max_y, y_hi)
                style = styles[run_name]
                x_offset = display_offsets[run_name]
                transform = (
                    ax.transData
                    if x_offset == 0.0
                    else offset_copy(ax.transData, fig=fig, x=x_offset, y=0, units="dots")
                )
                ax.plot(
                    x_values,
                    y_values,
                    linewidth=1.9,
                    markersize=4.4,
                    markeredgewidth=0.75,
                    markeredgecolor="white",
                    alpha=0.95,
                    label=run_name,
                    transform=transform,
                    zorder=10 + run_index,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    color=style["color"],
                )
        if has_series:
            ax.set_title(f"abs delta vs {baseline_name}: {metric}")
            ax.set_xlabel("step")
            if min_step is not None and max_step is not None:
                pad = 0.35
                ax.set_xlim(min_step - pad, max_step + pad)
            if min_y is not None and max_y is not None:
                if min_y == max_y:
                    y_pad = max(abs(min_y) * 0.02, 1e-9)
                else:
                    y_pad = max((max_y - min_y) * 0.08, 1e-9)
                ax.set_ylim(min_y - y_pad, max_y + y_pad)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"{metric} (no common data)")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

    for index in range(len(metrics), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(compare_runs)), frameon=False)
    fig.suptitle(f"Absolute deltas on common steps (baseline={baseline_name})", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_invariant_summary(
    out_path: Path,
    data: dict[str, dict[int, dict[str, float]]],
    metrics: list[str],
) -> None:
    pairs = list(itertools.combinations(data.keys(), 2))
    if not pairs:
        return

    labels: list[str] = []
    values: list[float] = []
    for base_name, other_name in pairs:
        common_steps = sorted(set(data[base_name]) & set(data[other_name]))
        for metric in metrics:
            diffs = [
                abs(data[base_name][step][metric] - data[other_name][step][metric])
                for step in common_steps
                if metric in data[base_name][step] and metric in data[other_name][step]
            ]
            if not diffs:
                continue
            labels.append(f"{base_name} vs {other_name}\n{metric}")
            values.append(max(diffs))

    if not labels:
        return

    fig_height = max(5.0, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("max absolute delta across common steps")
    ax.set_xscale("symlog", linthresh=1e-12)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Replay/input-side invariants")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_run_pairs(run_items: list[list[str]] | None) -> list[tuple[str, Path]]:
    if not run_items:
        return []
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in run_items:
        if len(item) != 2:
            raise ValueError(f"Each --run must be '<name> <path>', got: {item}")
        name, path_str = item
        if name in seen:
            raise ValueError(f"Duplicate run name: {name}")
        seen.add(name)
        parsed.append((name, Path(path_str)))
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run",
        dest="run_pairs",
        action="append",
        nargs=2,
        metavar=("NAME", "LOG_PATH"),
        help="Custom run entry; repeat to add more lines.",
    )
    parser.add_argument("--gpu-log", type=Path, default=None)
    parser.add_argument("--npu1-log", type=Path, default=None)
    parser.add_argument("--npu2-log", type=Path, default=None)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["gpu", "npu_attempt1", "npu_attempt2"],
        help="Run names to include when using legacy fixed inputs.",
    )
    parser.add_argument(
        "--delta-baseline",
        type=str,
        default=None,
        help="Baseline run name for absolute-delta plots. Defaults to first run name.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("rllm/examples/kernelgym/precision_compare_outputs"),
    )
    parser.add_argument(
        "--line-offset-dots",
        type=float,
        default=1.6,
        help="Visual horizontal offset in display dots to separate overlapping lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else (repo_root / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_specs = parse_run_pairs(args.run_pairs)
    if not run_specs:
        run_paths = {
            "gpu": args.gpu_log if args.gpu_log is not None else Path(DEFAULT_RUNS["gpu"]),
            "npu_attempt1": args.npu1_log if args.npu1_log is not None else Path(DEFAULT_RUNS["npu_attempt1"]),
            "npu_attempt2": args.npu2_log if args.npu2_log is not None else Path(DEFAULT_RUNS["npu_attempt2"]),
        }
        unknown = [name for name in args.runs if name not in run_paths]
        if unknown:
            raise ValueError(
                f"Unknown run names without --run: {unknown}. Use --run <name> <path> for custom runs."
            )
        run_specs = [(name, run_paths[name]) for name in args.runs]

    data: dict[str, dict[int, dict[str, float]]] = {}
    for run_name, maybe_rel_path in run_specs:
        path = maybe_rel_path if maybe_rel_path.is_absolute() else (repo_root / maybe_rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Log not found for run '{run_name}': {path}")
        data[run_name] = parse_step_metrics(path)
        print(f"{run_name}: {len(data[run_name])} steps from {path}")

    if not data:
        raise ValueError("No runs provided.")

    summary_metrics = INPUT_INVARIANT_METRICS + PROBABILITY_METRICS + UPDATE_METRICS
    all_summary_rows: list[dict[str, object]] = []
    for base_name, other_name in itertools.combinations(data.keys(), 2):
        all_summary_rows.extend(
            summarize_pair(base_name, other_name, data[base_name], data[other_name], summary_metrics)
        )

    write_step_csv(out_dir / "step_metrics.csv", data)
    write_summary_csv(out_dir / "precision_summary.csv", all_summary_rows)

    plot_metric_grid(
        out_dir / "core_training_metrics.png",
        data,
        CORE_PLOT_METRICS,
        "Core replay-training precision metrics",
        line_offset_dots=args.line_offset_dots,
    )
    plot_metric_grid(
        out_dir / "probability_recompute_metrics.png",
        data,
        PROBABILITY_METRICS,
        "Probability/logprob recomputation metrics",
        line_offset_dots=args.line_offset_dots,
    )

    baseline = args.delta_baseline if args.delta_baseline else next(iter(data.keys()))
    if baseline not in data:
        raise ValueError(f"--delta-baseline '{baseline}' is not in run names: {list(data.keys())}")
    plot_delta_grid(
        out_dir / "abs_delta_common_steps.png",
        data,
        CORE_PLOT_METRICS,
        baseline_name=baseline,
        line_offset_dots=args.line_offset_dots,
    )
    plot_invariant_summary(out_dir / "input_replay_invariants.png", data, INPUT_INVARIANT_METRICS)

    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
