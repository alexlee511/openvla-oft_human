#!/usr/bin/env python3
"""Analyze controller tracking gaps in saved LIBERO rollout traces.

This script scans rollout npz files produced by run_libero_eval.py and compares
the policy's joint-space targets against the realized robot joint trajectory.

By default it looks under:
  <repo_root>/rollouts/save_result

Expected folder structure:
  <root>/<method>/<suite>/<timestamp>/rollout_data/*.npz

The main tracking metric is the one-step gap between:
  model_joint_targets[t] and joint_pos[t+1]

because run_libero_eval.py stores joint_pos[t] before executing
model_joint_targets[t]. The script also reports command aggressiveness and env
action saturation to help distinguish controller-tracking issues from harder or
less stable policy outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "rollouts" / "save_result"
DEFAULT_METHODS = ["original_joint", "pure_ik", "liu_ik", "th_ik", "hrr_ik"]
DEFAULT_SUITES = ["libero_10", "libero_spatial", "libero_goal", "libero_object"]
EPS_SAT = 0.999


@dataclass
class EpisodeMetrics:
    method: str
    suite: str
    run_name: str
    episode_file: str
    task_name: str
    success: bool
    num_steps: int
    same_step_mae: float
    next_step_mae: float
    next_step_rmse: float
    next_step_maxabs: float
    next_step_p90abs: float
    lag_improvement: float
    command_step_l2: float
    command_step_maxabs: float
    env_joint_abs_mean: float
    env_joint_max_mean: float
    env_sat_frac: float
    env_any_sat_step_frac: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare controller tracking gaps across saved rollout methods."
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing method folders. Default: %(default)s",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Method folders to analyze.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=DEFAULT_SUITES,
        help="Task suites to analyze.",
    )
    parser.add_argument(
        "--baseline_method",
        default="original_joint",
        help="Reference method used for relative comparisons.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional explicit timestamp folder name to use for every method/suite when present. If omitted, use latest run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root_dir>/controller_tracking_analysis",
    )
    return parser.parse_args()


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


def scalar_string(value: np.ndarray, default: str = "unknown") -> str:
    try:
        return str(value.item())
    except Exception:
        return default


def scalar_bool(value: np.ndarray, default: bool = False) -> bool:
    try:
        return bool(value.item())
    except Exception:
        return default


def safe_mean(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def safe_rmse(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(array))))


def safe_quantile(array: np.ndarray, q: float) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, q))


def episode_task_name(data: np.lib.npyio.NpzFile, npz_path: Path) -> str:
    if "task_description" in data:
        return scalar_string(data["task_description"])
    if "task_name" in data:
        return scalar_string(data["task_name"])
    return npz_path.stem


def compute_episode_metrics(method: str, suite: str, run_dir: Path, npz_path: Path) -> Optional[EpisodeMetrics]:
    data = np.load(npz_path, allow_pickle=True)
    if "joint_pos" not in data or "model_joint_targets" not in data:
        return None

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    model_targets = np.asarray(data["model_joint_targets"], dtype=np.float32)
    actions_env = np.asarray(data["actions_env"], dtype=np.float32) if "actions_env" in data else np.empty((0, 8), dtype=np.float32)

    num_steps = min(len(joint_pos), len(model_targets))
    if num_steps == 0:
        return None

    joint_pos = joint_pos[:num_steps]
    model_targets = model_targets[:num_steps]
    same_step_err = model_targets - joint_pos
    same_step_abs = np.abs(same_step_err)

    if num_steps > 1:
        next_step_err = model_targets[:-1] - joint_pos[1:]
        next_step_abs = np.abs(next_step_err)
    else:
        next_step_err = np.empty((0, 7), dtype=np.float32)
        next_step_abs = np.empty((0, 7), dtype=np.float32)

    if num_steps > 1:
        cmd_step = np.diff(model_targets, axis=0)
        cmd_step_l2 = np.linalg.norm(cmd_step, axis=1)
        cmd_step_maxabs = np.max(np.abs(cmd_step), axis=1)
    else:
        cmd_step_l2 = np.empty((0,), dtype=np.float32)
        cmd_step_maxabs = np.empty((0,), dtype=np.float32)

    env_joint = np.abs(actions_env[:, :7]) if actions_env.ndim == 2 and actions_env.shape[1] >= 7 else np.empty((0, 7), dtype=np.float32)
    env_sat = env_joint >= EPS_SAT if env_joint.size else np.empty((0, 7), dtype=bool)
    env_any_sat = np.any(env_sat, axis=1) if env_sat.size else np.empty((0,), dtype=bool)
    env_max = np.max(env_joint, axis=1) if env_joint.size else np.empty((0,), dtype=np.float32)

    same_step_mae = safe_mean(same_step_abs)
    next_step_mae = safe_mean(next_step_abs)
    return EpisodeMetrics(
        method=method,
        suite=suite,
        run_name=run_dir.name,
        episode_file=npz_path.name,
        task_name=episode_task_name(data, npz_path),
        success=scalar_bool(data["success"], default="success=True" in npz_path.name) if "success" in data else ("success=True" in npz_path.name),
        num_steps=num_steps,
        same_step_mae=same_step_mae,
        next_step_mae=next_step_mae,
        next_step_rmse=safe_rmse(next_step_err),
        next_step_maxabs=float(np.max(next_step_abs)) if next_step_abs.size else float("nan"),
        next_step_p90abs=safe_quantile(next_step_abs, 0.9),
        lag_improvement=(same_step_mae - next_step_mae) if np.isfinite(same_step_mae) and np.isfinite(next_step_mae) else float("nan"),
        command_step_l2=safe_mean(cmd_step_l2),
        command_step_maxabs=safe_mean(cmd_step_maxabs),
        env_joint_abs_mean=safe_mean(env_joint),
        env_joint_max_mean=safe_mean(env_max),
        env_sat_frac=safe_mean(env_sat.astype(np.float32)),
        env_any_sat_step_frac=safe_mean(env_any_sat.astype(np.float32)),
    )


def metrics_to_dict(metrics: EpisodeMetrics) -> Dict[str, object]:
    return {
        "method": metrics.method,
        "suite": metrics.suite,
        "run_name": metrics.run_name,
        "episode_file": metrics.episode_file,
        "task_name": metrics.task_name,
        "success": metrics.success,
        "num_steps": metrics.num_steps,
        "same_step_mae": metrics.same_step_mae,
        "next_step_mae": metrics.next_step_mae,
        "next_step_rmse": metrics.next_step_rmse,
        "next_step_maxabs": metrics.next_step_maxabs,
        "next_step_p90abs": metrics.next_step_p90abs,
        "lag_improvement": metrics.lag_improvement,
        "command_step_l2": metrics.command_step_l2,
        "command_step_maxabs": metrics.command_step_maxabs,
        "env_joint_abs_mean": metrics.env_joint_abs_mean,
        "env_joint_max_mean": metrics.env_joint_max_mean,
        "env_sat_frac": metrics.env_sat_frac,
        "env_any_sat_step_frac": metrics.env_any_sat_step_frac,
    }


def mean_of(records: Iterable[EpisodeMetrics], attr: str) -> float:
    values = [float(getattr(record, attr)) for record in records if np.isfinite(getattr(record, attr))]
    return float(np.mean(values)) if values else float("nan")


def summarize_group(records: List[EpisodeMetrics]) -> Dict[str, object]:
    successes = sum(int(record.success) for record in records)
    success_records = [record for record in records if record.success]
    failure_records = [record for record in records if not record.success]
    return {
        "episodes": len(records),
        "success_rate": successes / len(records) if records else float("nan"),
        "mean_steps": mean_of(records, "num_steps"),
        "same_step_mae": mean_of(records, "same_step_mae"),
        "next_step_mae": mean_of(records, "next_step_mae"),
        "next_step_rmse": mean_of(records, "next_step_rmse"),
        "next_step_maxabs": mean_of(records, "next_step_maxabs"),
        "next_step_p90abs": mean_of(records, "next_step_p90abs"),
        "lag_improvement": mean_of(records, "lag_improvement"),
        "command_step_l2": mean_of(records, "command_step_l2"),
        "command_step_maxabs": mean_of(records, "command_step_maxabs"),
        "env_joint_abs_mean": mean_of(records, "env_joint_abs_mean"),
        "env_joint_max_mean": mean_of(records, "env_joint_max_mean"),
        "env_sat_frac": mean_of(records, "env_sat_frac"),
        "env_any_sat_step_frac": mean_of(records, "env_any_sat_step_frac"),
        "success_next_step_mae": mean_of(success_records, "next_step_mae"),
        "failure_next_step_mae": mean_of(failure_records, "next_step_mae"),
        "success_command_step_l2": mean_of(success_records, "command_step_l2"),
        "failure_command_step_l2": mean_of(failure_records, "command_step_l2"),
        "success_env_joint_max_mean": mean_of(success_records, "env_joint_max_mean"),
        "failure_env_joint_max_mean": mean_of(failure_records, "env_joint_max_mean"),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.5f}"
    return str(value)


def render_summary_text(
    root_dir: Path,
    summaries: Dict[str, Dict[str, Dict[str, object]]],
    deltas: List[Dict[str, object]],
) -> str:
    lines = []
    lines.append("ROLL OUT CONTROLLER TRACKING ANALYSIS")
    lines.append(f"root_dir: {root_dir}")
    lines.append("")
    lines.append("Per-method / per-suite summary")
    header = (
        "method,suite,episodes,success_rate,next_step_mae,next_step_rmse,"
        "next_step_maxabs,command_step_l2,command_step_maxabs,env_joint_max_mean,env_sat_frac"
    )
    lines.append(header)
    for method in sorted(summaries):
        for suite in sorted(summaries[method]):
            row = summaries[method][suite]
            lines.append(
                ",".join(
                    [
                        method,
                        suite,
                        format_float(row["episodes"]),
                        format_float(row["success_rate"]),
                        format_float(row["next_step_mae"]),
                        format_float(row["next_step_rmse"]),
                        format_float(row["next_step_maxabs"]),
                        format_float(row["command_step_l2"]),
                        format_float(row["command_step_maxabs"]),
                        format_float(row["env_joint_max_mean"]),
                        format_float(row["env_sat_frac"]),
                    ]
                )
            )

    if deltas:
        lines.append("")
        lines.append("Relative to baseline")
        lines.append(
            "method,suite,baseline_method,delta_success_rate,delta_next_step_mae,delta_command_step_l2,delta_env_joint_max_mean"
        )
        for row in deltas:
            lines.append(
                ",".join(
                    [
                        str(row["method"]),
                        str(row["suite"]),
                        str(row["baseline_method"]),
                        format_float(row["delta_success_rate"]),
                        format_float(row["delta_next_step_mae"]),
                        format_float(row["delta_command_step_l2"]),
                        format_float(row["delta_env_joint_max_mean"]),
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    output_dir = (args.output_dir or (root_dir / "controller_tracking_analysis")).resolve()
    ensure_dir(output_dir)

    all_episode_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    summaries: Dict[str, Dict[str, Dict[str, object]]] = {}

    for method in args.methods:
        method_dir = root_dir / method
        if not method_dir.is_dir():
            continue
        summaries[method] = {}
        for suite in args.suites:
            suite_dir = method_dir / suite
            run_dir = latest_run_dir(suite_dir, args.run_name)
            if run_dir is None:
                continue
            episode_metrics = []
            for npz_path in rollout_npzs(run_dir):
                metrics = compute_episode_metrics(method, suite, run_dir, npz_path)
                if metrics is None:
                    continue
                episode_metrics.append(metrics)
                all_episode_rows.append(metrics_to_dict(metrics))

            if not episode_metrics:
                continue

            summary = summarize_group(episode_metrics)
            summary["method"] = method
            summary["suite"] = suite
            summary["run_name"] = run_dir.name
            summaries[method][suite] = summary
            summary_rows.append(summary)

    baseline = args.baseline_method
    delta_rows: List[Dict[str, object]] = []
    if baseline in summaries:
        for method, per_suite in summaries.items():
            if method == baseline:
                continue
            for suite, row in per_suite.items():
                if suite not in summaries[baseline]:
                    continue
                base = summaries[baseline][suite]
                delta_rows.append(
                    {
                        "method": method,
                        "suite": suite,
                        "baseline_method": baseline,
                        "delta_success_rate": row["success_rate"] - base["success_rate"],
                        "delta_next_step_mae": row["next_step_mae"] - base["next_step_mae"],
                        "delta_command_step_l2": row["command_step_l2"] - base["command_step_l2"],
                        "delta_env_joint_max_mean": row["env_joint_max_mean"] - base["env_joint_max_mean"],
                    }
                )

    write_csv(output_dir / "per_episode_tracking.csv", all_episode_rows)
    write_csv(output_dir / "summary_by_method_suite.csv", summary_rows)
    write_csv(output_dir / "baseline_deltas.csv", delta_rows)

    json_payload = {
        "root_dir": str(root_dir),
        "baseline_method": baseline,
        "summaries": summaries,
        "baseline_deltas": delta_rows,
    }
    with (output_dir / "summary.json").open("w") as file:
        json.dump(json_payload, file, indent=2)

    summary_text = render_summary_text(root_dir, summaries, delta_rows)
    with (output_dir / "summary.txt").open("w") as file:
        file.write(summary_text)

    print(summary_text, end="")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()