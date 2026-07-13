#!/usr/bin/env python3
"""Plot rollout tracking and train-set action distribution comparisons.

This script complements A_rollout_controller_tracking.py.

It compares:
1. Rollout command aggressiveness vs tracking error vs success.
2. Success vs failure episodes for rollout metrics.
3. RLDS train-set action distributions for original_joint, pure_ik, liu_ik,
   th_ik, and hrr_ik.

Outputs are written under:
  <root_dir>/tracking_and_trainset_analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds


_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROLLOUT_ROOT = _REPO_ROOT / "rollouts" / "save_result"
DEFAULT_RLDS_ROOT = _REPO_ROOT / "modified_libero_rlds"
DEFAULT_METHODS = ["original_joint", "pure_ik", "liu_ik", "th_ik", "hrr_ik"]
DEFAULT_SUITES = ["libero_10", "libero_spatial", "libero_goal", "libero_object"]
METHOD_COLORS = {
    "original_joint": "#4C78A8",
    "pure_ik": "#F58518",
    "liu_ik": "#E45756",
    "th_ik": "#54A24B",
    "hrr_ik": "#B279A2",
}
PANDA_JOINT_LIMITS = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float32,
)
NEAR_LIMIT_THRESHOLD = 0.10


@dataclass
class RolloutEpisodeMetrics:
    method: str
    suite: str
    success: bool
    task_name: str
    next_step_mae: float
    next_step_rmse: float
    next_step_maxabs: float
    command_step_l2: float
    command_step_maxabs: float
    env_joint_max_mean: float
    env_sat_frac: float
    num_steps: int


@dataclass
class TrainStats:
    method: str
    suite: str
    episodes: int
    steps: int
    step_l2_mean: float
    step_l2_p90: float
    jerk_l2_mean: float
    jerk_l2_p90: float
    near_limit_frac: float
    min_margin_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rollout / train-set comparison plots.")
    parser.add_argument("--rollout_root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--rlds_root", type=Path, default=DEFAULT_RLDS_ROOT)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def latest_run_dir(suite_dir: Path, run_name: Optional[str]) -> Optional[Path]:
    if not suite_dir.is_dir():
        return None
    if run_name is not None:
        candidate = suite_dir / run_name
        return candidate if candidate.is_dir() else None
    runs = sorted(path for path in suite_dir.iterdir() if path.is_dir())
    return runs[-1] if runs else None


def rollout_npzs(run_dir: Path) -> List[Path]:
    data_dir = run_dir / "rollout_data"
    search_dir = data_dir if data_dir.is_dir() else run_dir
    return sorted(search_dir.glob("*.npz"))


def safe_mean(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def safe_quantile(array: np.ndarray, q: float) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, q))


def compute_rollout_metrics(method: str, suite: str, npz_path: Path) -> Optional[RolloutEpisodeMetrics]:
    data = np.load(npz_path, allow_pickle=True)
    if "joint_pos" not in data or "model_joint_targets" not in data:
        return None

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    targets = np.asarray(data["model_joint_targets"], dtype=np.float32)
    actions_env = np.asarray(data["actions_env"], dtype=np.float32)
    num_steps = min(len(joint_pos), len(targets))
    if num_steps <= 1:
        return None

    next_err = targets[: num_steps - 1] - joint_pos[1:num_steps]
    next_abs = np.abs(next_err)
    cmd_step = np.diff(targets[:num_steps], axis=0)
    env_joint = np.abs(actions_env[:num_steps, :7])

    success = bool(data["success"].item()) if "success" in data else ("success=True" in npz_path.name)
    task_name = str(data["task_description"].item()) if "task_description" in data else npz_path.stem
    return RolloutEpisodeMetrics(
        method=method,
        suite=suite,
        success=success,
        task_name=task_name,
        next_step_mae=safe_mean(next_abs),
        next_step_rmse=float(np.sqrt(np.mean(np.square(next_err)))),
        next_step_maxabs=float(np.max(next_abs)),
        command_step_l2=safe_mean(np.linalg.norm(cmd_step, axis=1)),
        command_step_maxabs=safe_mean(np.max(np.abs(cmd_step), axis=1)),
        env_joint_max_mean=safe_mean(np.max(env_joint, axis=1)),
        env_sat_frac=safe_mean((env_joint >= 0.999).astype(np.float32)),
        num_steps=num_steps,
    )


def rollout_to_dict(row: RolloutEpisodeMetrics) -> Dict[str, object]:
    return {
        "method": row.method,
        "suite": row.suite,
        "success": row.success,
        "task_name": row.task_name,
        "next_step_mae": row.next_step_mae,
        "next_step_rmse": row.next_step_rmse,
        "next_step_maxabs": row.next_step_maxabs,
        "command_step_l2": row.command_step_l2,
        "command_step_maxabs": row.command_step_maxabs,
        "env_joint_max_mean": row.env_joint_max_mean,
        "env_sat_frac": row.env_sat_frac,
        "num_steps": row.num_steps,
    }


def dataset_dir_for_method_suite(root: Path, method: str, suite: str) -> Optional[Path]:
    if method == "original_joint":
        dataset_dir = root / f"{suite}_joint_no_noops" / "1.0.0"
    else:
        dataset_dir = root / method / f"{suite}_humanized_no_noops" / "1.0.0"
    return dataset_dir if dataset_dir.is_dir() else None


def dataset_decoders() -> Dict[str, object]:
    return {
        "steps": {
            "observation": {
                "image": tfds.decode.SkipDecoding(),
                "wrist_image": tfds.decode.SkipDecoding(),
            }
        }
    }


def compute_limit_margin(q: np.ndarray) -> np.ndarray:
    low = PANDA_JOINT_LIMITS[:, 0]
    high = PANDA_JOINT_LIMITS[:, 1]
    low_margin = (q - low) / (high - low)
    high_margin = (high - q) / (high - low)
    return np.minimum(low_margin, high_margin)


def analyze_train_dataset(dataset_dir: Path, method: str, suite: str) -> Tuple[TrainStats, Dict[str, np.ndarray]]:
    builder = tfds.builder_from_directory(str(dataset_dir))
    ds = builder.as_dataset(split="train", decoders=dataset_decoders())

    episodes = 0
    steps = 0
    step_l2_values = []
    jerk_l2_values = []
    margin_values = []

    for episode in ds:
        episodes += 1
        targets = []
        for step in episode["steps"]:
            action = step["action"].numpy().astype(np.float32)
            targets.append(action[:7])
        if not targets:
            continue
        q = np.asarray(targets, dtype=np.float32)
        steps += len(q)
        if len(q) > 1:
            dq = np.diff(q, axis=0)
            step_l2_values.append(np.linalg.norm(dq, axis=1))
        if len(q) > 2:
            ddq = np.diff(q, axis=0, n=2)
            jerk_l2_values.append(np.linalg.norm(ddq, axis=1))
        margin_values.append(np.min(compute_limit_margin(q), axis=1))

    def concat(values: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(values) if values else np.empty((0,), dtype=np.float32)

    step_l2 = concat(step_l2_values)
    jerk_l2 = concat(jerk_l2_values)
    margins = concat(margin_values)
    stats = TrainStats(
        method=method,
        suite=suite,
        episodes=episodes,
        steps=steps,
        step_l2_mean=safe_mean(step_l2),
        step_l2_p90=safe_quantile(step_l2, 0.9),
        jerk_l2_mean=safe_mean(jerk_l2),
        jerk_l2_p90=safe_quantile(jerk_l2, 0.9),
        near_limit_frac=safe_mean((margins < NEAR_LIMIT_THRESHOLD).astype(np.float32)),
        min_margin_mean=safe_mean(margins),
    )
    payload = {"step_l2": step_l2, "jerk_l2": jerk_l2, "min_margin": margins}
    return stats, payload


def train_to_dict(row: TrainStats) -> Dict[str, object]:
    return {
        "method": row.method,
        "suite": row.suite,
        "episodes": row.episodes,
        "steps": row.steps,
        "step_l2_mean": row.step_l2_mean,
        "step_l2_p90": row.step_l2_p90,
        "jerk_l2_mean": row.jerk_l2_mean,
        "jerk_l2_p90": row.jerk_l2_p90,
        "near_limit_frac": row.near_limit_frac,
        "min_margin_mean": row.min_margin_mean,
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rollout_scatter(rows: List[Dict[str, object]], suites: Sequence[str], methods: Sequence[str], output_dir: Path) -> None:
    metrics = [
        ("command_step_l2", "next_step_mae", "Command Aggressiveness vs Tracking Error", "rollout_command_vs_tracking.png"),
        ("env_joint_max_mean", "next_step_mae", "Env Command Size vs Tracking Error", "rollout_env_vs_tracking.png"),
    ]
    for x_key, y_key, title, filename in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        for ax, suite in zip(axes, suites):
            suite_rows = [row for row in rows if row["suite"] == suite]
            for method in methods:
                method_rows = [row for row in suite_rows if row["method"] == method]
                if not method_rows:
                    continue
                x = np.array([row[x_key] for row in method_rows], dtype=np.float32)
                y = np.array([row[y_key] for row in method_rows], dtype=np.float32)
                success = np.array([row["success"] for row in method_rows], dtype=bool)
                ax.scatter(x[~success], y[~success], s=18, alpha=0.22, color=METHOD_COLORS.get(method))
                ax.scatter(x[success], y[success], s=28, alpha=0.65, color=METHOD_COLORS.get(method), label=method)
                ax.scatter([float(np.mean(x))], [float(np.mean(y))], marker="*", s=160, color=METHOD_COLORS.get(method), edgecolors="black", linewidths=0.6)
            ax.set_title(suite)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=min(5, len(uniq)), frameon=False)
        fig.suptitle(title + "\nsolid = success, faint = failure, star = method mean")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def plot_success_failure_bars(rows: List[Dict[str, object]], suites: Sequence[str], methods: Sequence[str], metric: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(suites), figsize=(4.4 * len(suites), 4.2), sharey=True)
    if len(suites) == 1:
        axes = [axes]
    for ax, suite in zip(axes, suites):
        suite_rows = [row for row in rows if row["suite"] == suite]
        succ = []
        fail = []
        for method in methods:
            m_rows = [row for row in suite_rows if row["method"] == method]
            succ_vals = [row[metric] for row in m_rows if row["success"]]
            fail_vals = [row[metric] for row in m_rows if not row["success"]]
            succ.append(float(np.mean(succ_vals)) if succ_vals else np.nan)
            fail.append(float(np.mean(fail_vals)) if fail_vals else np.nan)
        x = np.arange(len(methods), dtype=np.float32)
        ax.bar(x - 0.18, succ, width=0.36, color="#4C78A8", label="success")
        ax.bar(x + 0.18, fail, width=0.36, color="#E45756", label="failure")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_title(suite)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(metric)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Success vs Failure: {metric}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_train_bars(rows: List[Dict[str, object]], suites: Sequence[str], methods: Sequence[str], metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(suites), dtype=np.float32)
    width = 0.8 / max(len(methods), 1)
    for idx, method in enumerate(methods):
        vals = []
        for suite in suites:
            row = next((item for item in rows if item["method"] == method and item["suite"] == suite), None)
            vals.append(float(row[metric]) if row else np.nan)
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=method, color=METHOD_COLORS.get(method))
    ax.set_xticks(x)
    ax.set_xticklabels(suites)
    ax.set_ylabel(metric)
    ax.set_title(f"Train-set {metric} by method and suite")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(3, len(methods)))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distribution_histograms(payloads: Dict[Tuple[str, str], Dict[str, np.ndarray]], suites: Sequence[str], methods: Sequence[str], key: str, output_dir: Path) -> None:
    for suite in suites:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        suite_values = []
        per_method = {}
        for method in methods:
            values = payloads.get((method, suite), {}).get(key, np.empty((0,), dtype=np.float32))
            if values.size > 0:
                per_method[method] = values
                suite_values.append(values)
        if not suite_values:
            plt.close(fig)
            continue
        all_values = np.concatenate(suite_values)
        bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 70)
        if bins[0] == bins[-1]:
            bins = np.linspace(bins[0], bins[0] + 1e-4, 10)
        for method in methods:
            values = per_method.get(method)
            if values is None:
                continue
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2.0, label=method, color=METHOD_COLORS.get(method))
        ax.set_title(f"{suite}: {key} distribution")
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"train_{suite}_{key}_hist.png", dpi=180)
        plt.close(fig)


def format_float(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.5f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def main() -> None:
    args = parse_args()
    rollout_root = args.rollout_root.resolve()
    rlds_root = args.rlds_root.resolve()
    output_dir = (args.output_dir or (rollout_root / "tracking_and_trainset_analysis")).resolve()
    plots_dir = output_dir / "plots"
    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    rollout_rows: List[Dict[str, object]] = []
    for method in args.methods:
        for suite in args.suites:
            run_dir = latest_run_dir(rollout_root / method / suite, args.run_name)
            if run_dir is None:
                continue
            for npz_path in rollout_npzs(run_dir):
                row = compute_rollout_metrics(method, suite, npz_path)
                if row is not None:
                    rollout_rows.append(rollout_to_dict(row))

    train_rows: List[Dict[str, object]] = []
    train_payloads: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for method in args.methods:
        for suite in args.suites:
            dataset_dir = dataset_dir_for_method_suite(rlds_root, method, suite)
            if dataset_dir is None:
                continue
            stats, payload = analyze_train_dataset(dataset_dir, method, suite)
            train_rows.append(train_to_dict(stats))
            train_payloads[(method, suite)] = payload

    write_csv(output_dir / "rollout_per_episode_metrics.csv", rollout_rows)
    write_csv(output_dir / "train_action_stats.csv", train_rows)

    plot_rollout_scatter(rollout_rows, args.suites, args.methods, plots_dir)
    plot_success_failure_bars(rollout_rows, args.suites, args.methods, "command_step_l2", plots_dir / "rollout_success_failure_command_step_l2.png")
    plot_success_failure_bars(rollout_rows, args.suites, args.methods, "next_step_mae", plots_dir / "rollout_success_failure_next_step_mae.png")
    plot_success_failure_bars(rollout_rows, args.suites, args.methods, "env_joint_max_mean", plots_dir / "rollout_success_failure_env_joint_max_mean.png")

    plot_train_bars(train_rows, args.suites, args.methods, "step_l2_mean", plots_dir / "train_step_l2_mean.png")
    plot_train_bars(train_rows, args.suites, args.methods, "jerk_l2_mean", plots_dir / "train_jerk_l2_mean.png")
    plot_train_bars(train_rows, args.suites, args.methods, "near_limit_frac", plots_dir / "train_near_limit_frac.png")
    plot_distribution_histograms(train_payloads, args.suites, args.methods, "step_l2", plots_dir)
    plot_distribution_histograms(train_payloads, args.suites, args.methods, "jerk_l2", plots_dir)
    plot_distribution_histograms(train_payloads, args.suites, args.methods, "min_margin", plots_dir)

    summary = {
        "rollout_rows": rollout_rows,
        "train_rows": train_rows,
        "joint_limit_definition": {
            "robot": "Franka Panda",
            "near_limit_threshold": NEAR_LIMIT_THRESHOLD,
            "joint_limits": PANDA_JOINT_LIMITS.tolist(),
        },
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    summary_lines = []
    summary_lines.append("TRACKING AND TRAIN-SET ANALYSIS")
    summary_lines.append("")
    summary_lines.append("Rollout metrics")
    summary_lines.append("method,suite,success,next_step_mae,command_step_l2,env_joint_max_mean")
    for row in rollout_rows[:20]:
        summary_lines.append(
            ",".join(
                [
                    str(row["method"]),
                    str(row["suite"]),
                    str(row["success"]),
                    format_float(row["next_step_mae"]),
                    format_float(row["command_step_l2"]),
                    format_float(row["env_joint_max_mean"]),
                ]
            )
        )
    summary_lines.append("")
    summary_lines.append("Train-set metrics")
    summary_lines.append("method,suite,step_l2_mean,jerk_l2_mean,near_limit_frac,min_margin_mean")
    for row in train_rows:
        summary_lines.append(
            ",".join(
                [
                    str(row["method"]),
                    str(row["suite"]),
                    format_float(row["step_l2_mean"]),
                    format_float(row["jerk_l2_mean"]),
                    format_float(row["near_limit_frac"]),
                    format_float(row["min_margin_mean"]),
                ]
            )
        )
    with (output_dir / "summary.txt").open("w") as file:
        file.write("\n".join(summary_lines) + "\n")

    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()