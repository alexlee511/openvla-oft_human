#!/usr/bin/env python3
"""Plot per-joint train-set angle and step distributions across methods.

This script is intended to make the action-distribution differences easier to
inspect than the aggregate plots in A_rollout_tracking_and_trainset_analysis.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RLDS_ROOT = REPO_ROOT / "modified_libero_rlds"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "rollouts/save_result/joint_distribution_diagnostics"
DEFAULT_METHODS = ["original_joint", "pure_ik", "liu_ik", "hrr_ik", "th_ik"]
DEFAULT_SUITES = ["libero_10", "libero_spatial", "libero_goal", "libero_object"]
METHOD_DISPLAY_NAMES = {
    "original_joint": "Original",
    "pure_ik": "Pure-IK",
    "liu_ik": "Liu-IK",
    "hrr_ik": "HRR-IK",
    "th_ik": "TH-IK (Ours)",
}
METHOD_PLOT_ORDER = ["original_joint", "pure_ik", "liu_ik", "hrr_ik", "th_ik"]
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
ANGLE_PLOT_TITLE = "Robot Joint Target Distribution"
DELTA_PLOT_TITLE = "Per-Step Joint Target Change Magnitude"
NEAR_LIMIT_PLOT_TITLE = "Per-Joint Limit-Proximity Frequency"


def suite_display_name(suite: str) -> str:
    return "" if suite == "all_suites_combined" else suite


def method_display_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method.replace("_", "-"))


def ordered_methods(methods: Sequence[str]) -> List[str]:
    ordered = [method for method in METHOD_PLOT_ORDER if method in methods]
    extras = [method for method in methods if method not in METHOD_PLOT_ORDER]
    return ordered + extras


def format_plot_title(suite: str, title: str) -> str:
    suite_name = suite_display_name(suite)
    return title if not suite_name else f"{suite_name}: {title}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-joint train-set diagnostics.")
    parser.add_argument("--rlds_root", type=Path, default=DEFAULT_RLDS_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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


def load_joint_targets(dataset_dir: Path) -> np.ndarray:
    builder = tfds.builder_from_directory(str(dataset_dir))
    ds = builder.as_dataset(split="train", decoders=dataset_decoders())
    trajectories = []
    for episode in ds:
        q = []
        for step in episode["steps"]:
            q.append(step["action"].numpy().astype(np.float32)[:7])
        if q:
            trajectories.append(np.asarray(q, dtype=np.float32))
    if not trajectories:
        return np.empty((0, 7), dtype=np.float32)
    return np.concatenate(trajectories, axis=0)


def load_joint_targets_and_deltas(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    builder = tfds.builder_from_directory(str(dataset_dir))
    ds = builder.as_dataset(split="train", decoders=dataset_decoders())
    trajectories = []
    deltas = []
    for episode in ds:
        q = []
        for step in episode["steps"]:
            q.append(step["action"].numpy().astype(np.float32)[:7])
        if q:
            q = np.asarray(q, dtype=np.float32)
            trajectories.append(q)
            if len(q) > 1:
                deltas.append(np.diff(q, axis=0))
    q_all = np.concatenate(trajectories, axis=0) if trajectories else np.empty((0, 7), dtype=np.float32)
    dq_all = np.concatenate(deltas, axis=0) if deltas else np.empty((0, 7), dtype=np.float32)
    return q_all, dq_all


def compute_joint_near_limit_frac(q: np.ndarray) -> np.ndarray:
    low = PANDA_JOINT_LIMITS[:, 0]
    high = PANDA_JOINT_LIMITS[:, 1]
    normalized = (q - low) / (high - low)
    margin = np.minimum(normalized, 1.0 - normalized)
    return np.mean(margin < NEAR_LIMIT_THRESHOLD, axis=0)


def wasserstein_1d(x: np.ndarray, y: np.ndarray, bins: int = 256) -> float:
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    hx, _ = np.histogram(x, bins=edges, density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    dx = edges[1] - edges[0]
    cdx = np.cumsum(hx) * dx
    cdy = np.cumsum(hy) * dx
    return float(np.sum(np.abs(cdx - cdy)) * dx)


def plot_joint_histograms(
    suite: str,
    title: str,
    data: Dict[str, np.ndarray],
    xlabel: str,
    output_path: Path,
    methods: Sequence[str],
) -> None:
    methods = ordered_methods(methods)
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    axes = axes.flatten()
    for joint_idx in range(7):
        ax = axes[joint_idx]
        joint_values = [data[method][:, joint_idx] for method in methods if method in data and len(data[method])]
        if not joint_values:
            ax.set_visible(False)
            continue
        all_values = np.concatenate(joint_values)
        bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 80)
        if bins[0] == bins[-1]:
            bins = np.linspace(bins[0], bins[0] + 1e-4, 10)
        for method in methods:
            if method not in data or not len(data[method]):
                continue
            ax.hist(
                data[method][:, joint_idx],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=method_display_name(method),
                color=METHOD_COLORS.get(method),
            )
        ax.set_title(f"Joint {joint_idx + 1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.grid(alpha=0.25)
    axes[7].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.suptitle(format_plot_title(suite, title), y=0.985)
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(5, len(uniq)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output_path)


def plot_joint_histograms_separate(
    suite: str,
    title: str,
    data: Dict[str, np.ndarray],
    xlabel: str,
    output_dir: Path,
    methods: Sequence[str],
    file_prefix: str,
) -> None:
    methods = ordered_methods(methods)
    ensure_dir(output_dir)
    for joint_idx in range(7):
        joint_values = [data[method][:, joint_idx] for method in methods if method in data and len(data[method])]
        if not joint_values:
            continue
        all_values = np.concatenate(joint_values)
        bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 80)
        if bins[0] == bins[-1]:
            bins = np.linspace(bins[0], bins[0] + 1e-4, 10)
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for method in methods:
            if method not in data or not len(data[method]):
                continue
            ax.hist(
                data[method][:, joint_idx],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                label=method_display_name(method),
                color=METHOD_COLORS.get(method),
            )
        ax.set_title(f"{format_plot_title(suite, title)} - Joint {joint_idx + 1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability Density")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{file_prefix}_joint{joint_idx + 1}.png")


def plot_joint_histograms_averaged(
    suite: str,
    title: str,
    data: Dict[str, np.ndarray],
    xlabel: str,
    output_path: Path,
    methods: Sequence[str],
) -> None:
    available_methods = [method for method in ordered_methods(methods) if method in data and len(data[method])]
    if not available_methods:
        return
    pooled_values = [data[method].reshape(-1) for method in available_methods]
    all_values = np.concatenate(pooled_values)
    bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 120)
    if bins[0] == bins[-1]:
        bins = np.linspace(bins[0], bins[0] + 1e-4, 10)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for method in available_methods:
        ax.hist(
            data[method].reshape(-1),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=method_display_name(method),
            color=METHOD_COLORS.get(method),
        )
    ax.set_title(f"{format_plot_title(suite, title)} (Averaged Across 7 Joints)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_joint_near_limit_bars(
    suite: str,
    near_limit: Dict[str, np.ndarray],
    methods: Sequence[str],
    output_path: Path,
) -> None:
    methods = ordered_methods(methods)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(7, dtype=np.float32)
    width = 0.8 / max(len(methods), 1)
    for idx, method in enumerate(methods):
        if method not in near_limit:
            continue
        vals = near_limit[method]
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=method_display_name(method),
            color=METHOD_COLORS.get(method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{i}" for i in range(1, 8)])
    ax.set_ylabel("near-limit fraction")
    ax.set_title(format_plot_title(suite, NEAR_LIMIT_PLOT_TITLE))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(3, len(methods)))
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_method_near_limit_across_suites(
    method: str,
    suites: Sequence[str],
    suite_near_limit: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    available_suites = [suite for suite in suites if suite in suite_near_limit and method in suite_near_limit[suite]]
    if not available_suites:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(7, dtype=np.float32)
    width = 0.8 / max(len(available_suites), 1)
    suite_colors = plt.cm.Set2(np.linspace(0.15, 0.85, len(available_suites)))
    for idx, suite in enumerate(available_suites):
        vals = suite_near_limit[suite][method]
        offset = (idx - (len(available_suites) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=suite_display_name(suite),
            color=suite_colors[idx],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{i}" for i in range(1, 8)])
    ax.set_ylabel("near-limit fraction")
    ax.set_title(f"{method_display_name(method)}: {NEAR_LIMIT_PLOT_TITLE} Across Suites")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(4, len(available_suites)))
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_method_joint_distribution_across_suites(
    method: str,
    suites: Sequence[str],
    suite_q_data: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    available_suites = [suite for suite in suites if suite in suite_q_data and method in suite_q_data[suite]]
    if not available_suites:
        return
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    axes = axes.flatten()
    suite_colors = plt.cm.Set2(np.linspace(0.15, 0.85, len(available_suites)))
    for joint_idx in range(7):
        ax = axes[joint_idx]
        joint_values = [suite_q_data[suite][method][:, joint_idx] for suite in available_suites if len(suite_q_data[suite][method])]
        if not joint_values:
            ax.set_visible(False)
            continue
        all_values = np.concatenate(joint_values)
        bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 80)
        if bins[0] == bins[-1]:
            bins = np.linspace(bins[0], bins[0] + 1e-4, 10)
        for color, suite in zip(suite_colors, available_suites):
            ax.hist(
                suite_q_data[suite][method][:, joint_idx],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=suite_display_name(suite),
                color=color,
            )
        ax.set_title(f"Joint {joint_idx + 1}")
        ax.set_xlabel("joint target angle (rad)")
        ax.set_ylabel("Probability Density")
        ax.grid(alpha=0.25)
    axes[7].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.suptitle(f"{method_display_name(method)}: {ANGLE_PLOT_TITLE}", y=0.985)
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(4, len(uniq)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output_path)


def plot_method_joint_distribution_across_suites_separate(
    method: str,
    suites: Sequence[str],
    suite_q_data: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    file_prefix: str,
) -> None:
    available_suites = [suite for suite in suites if suite in suite_q_data and method in suite_q_data[suite]]
    if not available_suites:
        return
    ensure_dir(output_dir)
    suite_colors = plt.cm.Set2(np.linspace(0.15, 0.85, len(available_suites)))
    for joint_idx in range(7):
        joint_values = [suite_q_data[suite][method][:, joint_idx] for suite in available_suites if len(suite_q_data[suite][method])]
        if not joint_values:
            continue
        all_values = np.concatenate(joint_values)
        bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 80)
        if bins[0] == bins[-1]:
            bins = np.linspace(bins[0], bins[0] + 1e-4, 10)
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for color, suite in zip(suite_colors, available_suites):
            ax.hist(
                suite_q_data[suite][method][:, joint_idx],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                label=suite_display_name(suite),
                color=color,
            )
        ax.set_title(f"{method_display_name(method)}: {ANGLE_PLOT_TITLE} - Joint {joint_idx + 1}")
        ax.set_xlabel("joint target angle (rad)")
        ax.set_ylabel("Probability Density")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, output_dir / f"{file_prefix}_joint{joint_idx + 1}.png")


def plot_method_joint_distribution_across_suites_averaged(
    method: str,
    suites: Sequence[str],
    suite_q_data: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    available_suites = [suite for suite in suites if suite in suite_q_data and method in suite_q_data[suite]]
    if not available_suites:
        return
    pooled_values = [suite_q_data[suite][method].reshape(-1) for suite in available_suites if len(suite_q_data[suite][method])]
    if not pooled_values:
        return
    all_values = np.concatenate(pooled_values)
    bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 120)
    if bins[0] == bins[-1]:
        bins = np.linspace(bins[0], bins[0] + 1e-4, 10)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    suite_colors = plt.cm.Set2(np.linspace(0.15, 0.85, len(available_suites)))
    for color, suite in zip(suite_colors, available_suites):
        ax.hist(
            suite_q_data[suite][method].reshape(-1),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=suite_display_name(suite),
            color=color,
        )
    ax.set_title(f"{method_display_name(method)}: {ANGLE_PLOT_TITLE} (Averaged Across 7 Joints)")
    ax.set_xlabel("joint target angle (rad)")
    ax.set_ylabel("Probability Density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_path)


def suite_output_dir(root: Path, suite: str) -> Path:
    return root / suite


def combined_output_dirs(root: Path) -> Dict[str, Path]:
    return {
        "root": root,
        "joint_angle_hist": root / "joint_angle_hist",
        "joint_delta_hist": root / "joint_delta_hist",
    }


def summarize_suite_against_original(
    suite: str,
    methods: Sequence[str],
    q_data: Dict[str, np.ndarray],
    dq_data: Dict[str, np.ndarray],
    near_limit: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    if "original_joint" not in q_data:
        return summary_rows
    q0 = q_data["original_joint"]
    dq0 = dq_data.get("original_joint", np.empty((0, 7), dtype=np.float32))
    for method in methods:
        if method not in q_data:
            continue
        row = {
            "suite": suite,
            "method": method,
        }
        for joint_idx in range(7):
            row[f"joint{joint_idx + 1}_near_limit_frac"] = float(near_limit[method][joint_idx])
            row[f"joint{joint_idx + 1}_w1_vs_original"] = (
                0.0 if method == "original_joint" else wasserstein_1d(q_data[method][:, joint_idx], q0[:, joint_idx])
            )
            row[f"joint{joint_idx + 1}_delta_w1_vs_original"] = (
                0.0
                if method == "original_joint" or not len(dq_data.get(method, np.empty((0, 7)))) or not len(dq0)
                else wasserstein_1d(dq_data[method][:, joint_idx], dq0[:, joint_idx])
            )
        summary_rows.append(row)
    return summary_rows


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    summary_rows: List[Dict[str, object]] = []
    combined_q_data: Dict[str, List[np.ndarray]] = {method: [] for method in args.methods}
    combined_dq_data: Dict[str, List[np.ndarray]] = {method: [] for method in args.methods}
    suite_near_limit_map: Dict[str, Dict[str, np.ndarray]] = {}
    suite_q_map: Dict[str, Dict[str, np.ndarray]] = {}

    for suite in args.suites:
        q_data: Dict[str, np.ndarray] = {}
        dq_data: Dict[str, np.ndarray] = {}
        near_limit: Dict[str, np.ndarray] = {}
        for method in args.methods:
            dataset_dir = dataset_dir_for_method_suite(args.rlds_root, method, suite)
            if dataset_dir is None:
                continue
            q, dq = load_joint_targets_and_deltas(dataset_dir)
            if not len(q):
                continue
            q_data[method] = q
            dq_data[method] = np.abs(dq)
            near_limit[method] = compute_joint_near_limit_frac(q)
            combined_q_data[method].append(q)
            if len(dq):
                combined_dq_data[method].append(np.abs(dq))

        if not q_data:
            continue

        suite_near_limit_map[suite] = near_limit
        suite_q_map[suite] = q_data

        suite_dir = suite_output_dir(args.output_dir, suite)
        ensure_dir(suite_dir)
        suite_joint_angle_dir = suite_dir / "joint_angle_hist"
        suite_joint_delta_dir = suite_dir / "joint_delta_hist"

        plot_joint_histograms(
            suite,
            title=ANGLE_PLOT_TITLE,
            data=q_data,
            xlabel="joint target angle (rad)",
            output_path=suite_dir / f"{suite}_joint_angle_hist.png",
            methods=args.methods,
        )
        plot_joint_histograms_separate(
            suite,
            title=ANGLE_PLOT_TITLE,
            data=q_data,
            xlabel="joint target angle (rad)",
            output_dir=suite_joint_angle_dir,
            methods=args.methods,
            file_prefix=f"{suite}_joint_angle_hist",
        )
        plot_joint_histograms(
            suite,
            title=DELTA_PLOT_TITLE,
            data=dq_data,
            xlabel="|q[t+1] - q[t]| (rad)",
            output_path=suite_dir / f"{suite}_joint_delta_hist.png",
            methods=args.methods,
        )
        plot_joint_histograms_separate(
            suite,
            title=DELTA_PLOT_TITLE,
            data=dq_data,
            xlabel="|q[t+1] - q[t]| (rad)",
            output_dir=suite_joint_delta_dir,
            methods=args.methods,
            file_prefix=f"{suite}_joint_delta_hist",
        )
        plot_joint_near_limit_bars(
            suite,
            near_limit=near_limit,
            methods=args.methods,
            output_path=suite_dir / f"{suite}_joint_near_limit_bar.png",
        )
        summary_rows.extend(summarize_suite_against_original(suite, args.methods, q_data, dq_data, near_limit))

    plot_method_near_limit_across_suites(
        method="liu_ik",
        suites=args.suites,
        suite_near_limit=suite_near_limit_map,
        output_path=args.output_dir / "liu_ik_near_limit_across_suites.png",
    )
    plot_method_joint_distribution_across_suites(
        method="pure_ik",
        suites=args.suites,
        suite_q_data=suite_q_map,
        output_path=args.output_dir / "pure_ik_joint_distribution_across_suites.png",
    )
    plot_method_joint_distribution_across_suites_separate(
        method="pure_ik",
        suites=args.suites,
        suite_q_data=suite_q_map,
        output_dir=args.output_dir / "pure_ik_joint_distribution_across_suites",
        file_prefix="pure_ik_joint_distribution_across_suites",
    )
    plot_method_joint_distribution_across_suites_averaged(
        method="pure_ik",
        suites=args.suites,
        suite_q_data=suite_q_map,
        output_path=args.output_dir / "pure_ik_joint_distribution_across_suites_avg7.png",
    )

    combined_q = {
        method: np.concatenate(values, axis=0)
        for method, values in combined_q_data.items()
        if values
    }
    combined_dq = {
        method: np.concatenate(values, axis=0)
        for method, values in combined_dq_data.items()
        if values
    }
    if combined_q:
        combined_near_limit = {
            method: compute_joint_near_limit_frac(q)
            for method, q in combined_q.items()
        }
        dirs = combined_output_dirs(args.output_dir)
        plot_joint_histograms(
            "all_suites_combined",
            title=ANGLE_PLOT_TITLE,
            data=combined_q,
            xlabel="joint target angle (rad)",
            output_path=dirs["root"] / "all_suites_joint_angle_hist.png",
            methods=args.methods,
        )
        plot_joint_histograms_separate(
            "all_suites_combined",
            title=ANGLE_PLOT_TITLE,
            data=combined_q,
            xlabel="joint target angle (rad)",
            output_dir=dirs["joint_angle_hist"],
            methods=args.methods,
            file_prefix="all_suites_joint_angle_hist",
        )
        plot_joint_histograms_averaged(
            "all_suites_combined",
            title=ANGLE_PLOT_TITLE,
            data=combined_q,
            xlabel="joint target angle (rad)",
            output_path=dirs["root"] / "all_suites_joint_angle_hist_avg7.png",
            methods=args.methods,
        )
        plot_joint_histograms(
            "all_suites_combined",
            title=DELTA_PLOT_TITLE,
            data=combined_dq,
            xlabel="|q[t+1] - q[t]| (rad)",
            output_path=dirs["root"] / "all_suites_joint_delta_hist.png",
            methods=args.methods,
        )
        plot_joint_histograms_separate(
            "all_suites_combined",
            title=DELTA_PLOT_TITLE,
            data=combined_dq,
            xlabel="|q[t+1] - q[t]| (rad)",
            output_dir=dirs["joint_delta_hist"],
            methods=args.methods,
            file_prefix="all_suites_joint_delta_hist",
        )
        plot_joint_histograms_averaged(
            "all_suites_combined",
            title=DELTA_PLOT_TITLE,
            data=combined_dq,
            xlabel="|q[t+1] - q[t]| (rad)",
            output_path=dirs["root"] / "all_suites_joint_delta_hist_avg7.png",
            methods=args.methods,
        )
        plot_joint_near_limit_bars(
            "all_suites_combined",
            near_limit=combined_near_limit,
            methods=args.methods,
            output_path=dirs["root"] / "all_suites_joint_near_limit_bar.png",
        )
        summary_rows.extend(
            summarize_suite_against_original(
                "all_suites_combined",
                args.methods,
                combined_q,
                combined_dq,
                combined_near_limit,
            )
        )

    if summary_rows:
        csv_path = args.output_dir / "joint_distribution_summary.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        with (args.output_dir / "joint_distribution_summary.json").open("w") as file:
            json.dump(summary_rows, file, indent=2)

    print(f"Saved joint diagnostics to: {args.output_dir}")


if __name__ == "__main__":
    main()