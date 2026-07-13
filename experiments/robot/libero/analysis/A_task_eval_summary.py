#!/usr/bin/env python3
"""Generate suite-level task evaluation summaries from rollout results.

This script scans result roots such as rollouts/save_result/th_ik or
rollouts/save_result/original_joint. For each suite directory, it selects the
latest run directory containing human_likeness/summary.csv and writes a
task_eval_<suite>.txt file at the suite level.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from LIBERO.libero.libero.benchmark.libero_suite_task_map import libero_task_map


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULT_ROOTS = [
    REPO_ROOT / "rollouts/save_result/th_ik",
    REPO_ROOT / "rollouts/save_result/original_joint",
]
SUITE_NAMES = ["libero_10", "libero_spatial", "libero_goal", "libero_object"]
METRIC_COLUMNS = ["HJL", "MJE", "SOAq", "SOAx", "EEA", "noMJE", "UNIFIED", "COMBINED"]


def strip_scene_prefix(task_name: str) -> str:
    return re.sub(r"^[A-Z_]+_SCENE\d+_", "", task_name)


def task_alias_map(suite: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for task in libero_task_map.get(suite, []):
        aliases[task] = task
        aliases[strip_scene_prefix(task)] = task
    return aliases


def normalize_task_name(task_name: str, suite: str) -> str:
    if task_name in libero_task_map.get(suite, []):
        return task_name

    alias_to_full = task_alias_map(suite)
    matches = []
    for alias, full_name in alias_to_full.items():
        if alias.startswith(task_name):
            matches.append(full_name)

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return task_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_roots",
        nargs="+",
        type=Path,
        default=DEFAULT_RESULT_ROOTS,
        help="Result roots to scan, e.g. rollouts/save_result/th_ik",
    )
    return parser.parse_args()


def list_run_dirs(suite_dir: Path) -> List[Path]:
    run_dirs = [path for path in suite_dir.iterdir() if path.is_dir()]
    return sorted(run_dirs, key=lambda path: path.name)


def find_latest_run_with_task_results(suite_dir: Path) -> Optional[Path]:
    latest_match: Optional[Path] = None
    for run_dir in list_run_dirs(suite_dir):
        task_csv = run_dir / "aggregate" / "task_results.csv"
        human_csv = run_dir / "human_likeness" / "summary.csv"
        if task_csv.is_file() and human_csv.is_file():
            latest_match = run_dir
    return latest_match


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def official_task_order(suite: str, task_names: List[str]) -> List[str]:
    ordered = [task for task in libero_task_map.get(suite, []) if task in task_names]
    extras = sorted(task for task in task_names if task not in ordered)
    return ordered + extras


def read_human_likeness_task_rows(csv_path: Path, suite: str) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            task_name = str(raw_row.get("task", "")).strip()
            if not task_name:
                continue
            try:
                unified = float(raw_row["unified"])
                unified_no_mje = float(raw_row["unified_noMJE"])
                hlj = float(raw_row["m_HJL"])
                mje = float(raw_row["m_MJE"])
                soaq = float(raw_row["m_SOAq"])
                soax = float(raw_row["m_SOAx"])
                eea = float(raw_row["m_EEA"])
            except (KeyError, TypeError, ValueError):
                continue
            success = parse_bool(raw_row.get("success", False))
            grouped[normalize_task_name(task_name, suite)].append(
                {
                    "success": 1.0 if success else 0.0,
                    "UNIFIED": unified,
                    "noMJE": unified_no_mje,
                    "HJL": hlj,
                    "MJE": mje,
                    "SOAq": soaq,
                    "SOAx": soax,
                    "EEA": eea,
                    "COMBINED": unified if success else 0.0,
                }
            )

    rows: List[Dict[str, object]] = []
    for task_name in official_task_order(suite, list(grouped.keys())):
        entries = grouped[task_name]
        count = len(entries)
        row: Dict[str, object] = {
            "task": task_name,
            "n_demos": count,
            "success_rate_pct": 100.0 * sum(entry["success"] for entry in entries) / max(count, 1),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = sum(float(entry[metric]) for entry in entries) / max(count, 1)
        rows.append(row)
    return rows


def read_results_header(results_path: Path) -> Dict[str, str]:
    header: Dict[str, str] = {}
    if not results_path.is_file():
        return header
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip()] = value.strip()
    return header


def format_task_eval_lines(result_root: Path, suite: str, run_dir: Path, task_rows: List[Dict[str, object]]) -> List[str]:
    header = read_results_header(run_dir / "results.txt")
    headers = ["task", "success_rate_pct", "HJL", "MJE", "SOAq", "SOAx", "EEA", "noMJE", "UNIFIED", "COMBINED", "n_demos"]
    display_rows = []
    for row in task_rows:
        display_rows.append(
            {
                "task": str(row["task"]),
                "success_rate_pct": f"{float(row['success_rate_pct']):.1f}",
                "HJL": f"{float(row['HJL']):.6f}",
                "MJE": f"{float(row['MJE']):.6f}",
                "SOAq": f"{float(row['SOAq']):.6f}",
                "SOAx": f"{float(row['SOAx']):.6f}",
                "EEA": f"{float(row['EEA']):.6f}",
                "noMJE": f"{float(row['noMJE']):.6f}",
                "UNIFIED": f"{float(row['UNIFIED']):.6f}",
                "COMBINED": f"{float(row['COMBINED']):.6f}",
                "n_demos": str(int(row['n_demos'])),
            }
        )

    widths = {name: len(name) for name in headers}
    for row in display_rows:
        for name in headers:
            widths[name] = max(widths[name], len(str(row[name])))

    lines = [
        f"Task Evaluation Summary ({result_root.name})",
        f"suite={suite}",
        f"run={run_dir.name}",
    ]
    if "Checkpoint" in header:
        lines.append(f"checkpoint={header['Checkpoint']}")
    if "Controller" in header:
        lines.append(f"controller={header['Controller']}")
    lines.extend(
        [
            "",
            " ".join(name.ljust(widths[name]) for name in headers),
            " ".join("-" * widths[name] for name in headers),
        ]
    )
    for row in display_rows:
        lines.append(" ".join(str(row[name]).ljust(widths[name]) for name in headers))
    return lines


def format_task_eval_compare_lines(
    result_root: Path,
    suite: str,
    run_dir: Path,
    task_rows: List[Dict[str, object]],
    original_rows: List[Dict[str, object]],
    original_run_dir: Path,
) -> List[str]:
    original_by_task = {str(row["task"]): row for row in original_rows}
    headers = [
        "task",
        "human_sr_pct",
        "orig_sr_pct",
        "delta_sr_pct",
        "human_UNIFIED",
        "orig_UNIFIED",
        "delta_UNIFIED",
        "human_noMJE",
        "orig_noMJE",
        "delta_noMJE",
        "human_COMBINED",
        "orig_COMBINED",
        "delta_COMBINED",
        "human_HJL",
        "orig_HJL",
        "human_MJE",
        "orig_MJE",
        "human_SOAq",
        "orig_SOAq",
        "human_SOAx",
        "orig_SOAx",
        "human_EEA",
        "orig_EEA",
        "n_demos",
    ]

    display_rows = []
    for row in task_rows:
        task = str(row["task"])
        original_row = original_by_task.get(task)
        if original_row is None:
            continue
        display_rows.append(
            {
                "task": task,
                "human_sr_pct": f"{float(row['success_rate_pct']):.1f}",
                "orig_sr_pct": f"{float(original_row['success_rate_pct']):.1f}",
                "delta_sr_pct": f"{float(row['success_rate_pct']) - float(original_row['success_rate_pct']):+.1f}",
                "human_UNIFIED": f"{float(row['UNIFIED']):.6f}",
                "orig_UNIFIED": f"{float(original_row['UNIFIED']):.6f}",
                "delta_UNIFIED": f"{float(row['UNIFIED']) - float(original_row['UNIFIED']):+.6f}",
                "human_noMJE": f"{float(row['noMJE']):.6f}",
                "orig_noMJE": f"{float(original_row['noMJE']):.6f}",
                "delta_noMJE": f"{float(row['noMJE']) - float(original_row['noMJE']):+.6f}",
                "human_COMBINED": f"{float(row['COMBINED']):.6f}",
                "orig_COMBINED": f"{float(original_row['COMBINED']):.6f}",
                "delta_COMBINED": f"{float(row['COMBINED']) - float(original_row['COMBINED']):+.6f}",
                "human_HJL": f"{float(row['HJL']):.6f}",
                "orig_HJL": f"{float(original_row['HJL']):.6f}",
                "human_MJE": f"{float(row['MJE']):.6f}",
                "orig_MJE": f"{float(original_row['MJE']):.6f}",
                "human_SOAq": f"{float(row['SOAq']):.6f}",
                "orig_SOAq": f"{float(original_row['SOAq']):.6f}",
                "human_SOAx": f"{float(row['SOAx']):.6f}",
                "orig_SOAx": f"{float(original_row['SOAx']):.6f}",
                "human_EEA": f"{float(row['EEA']):.6f}",
                "orig_EEA": f"{float(original_row['EEA']):.6f}",
                "n_demos": str(int(row["n_demos"])),
            }
        )

    widths = {name: len(name) for name in headers}
    for row in display_rows:
        for name in headers:
            widths[name] = max(widths[name], len(str(row[name])))

    lines = [
        f"Task Evaluation Compare Summary ({result_root.name} vs original_joint)",
        f"suite={suite}",
        f"run={run_dir.name}",
        f"baseline_run={original_run_dir.name}",
        "",
        " ".join(name.ljust(widths[name]) for name in headers),
        " ".join("-" * widths[name] for name in headers),
    ]
    for row in display_rows:
        lines.append(" ".join(str(row[name]).ljust(widths[name]) for name in headers))
    return lines


def write_task_eval(suite_dir: Path, suite: str, lines: List[str]) -> None:
    legacy_output_path = suite_dir / "task_eval.txt"
    if legacy_output_path.exists():
        legacy_output_path.unlink()
    output_path = suite_dir / f"task_eval_{suite}.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_task_eval_compare(suite_dir: Path, suite: str, lines: List[str]) -> None:
    output_path = suite_dir / f"task_eval_compare_{suite}.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_result_root_data(result_root: Path) -> Dict[str, Dict[str, object]]:
    suite_data: Dict[str, Dict[str, object]] = {}
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root not found: {result_root}")
    for suite in SUITE_NAMES:
        suite_dir = result_root / suite
        if not suite_dir.is_dir():
            continue
        run_dir = find_latest_run_with_task_results(suite_dir)
        if run_dir is None:
            continue
        task_rows = read_human_likeness_task_rows(run_dir / "human_likeness" / "summary.csv", suite)
        if not task_rows:
            continue
        suite_data[suite] = {
            "suite_dir": suite_dir,
            "run_dir": run_dir,
            "task_rows": task_rows,
        }
    return suite_data


def process_result_root(result_root: Path) -> None:
    suite_data = collect_result_root_data(result_root)
    for suite, data in suite_data.items():
        lines = format_task_eval_lines(result_root, suite, data["run_dir"], data["task_rows"])
        write_task_eval(data["suite_dir"], suite, lines)


def process_compare_outputs(result_roots: List[Path]) -> None:
    original_root = next((path for path in result_roots if path.name == "original_joint"), None)
    if original_root is None:
        return
    original_data = collect_result_root_data(original_root)
    for result_root in result_roots:
        if result_root == original_root:
            continue
        suite_data = collect_result_root_data(result_root)
        for suite, data in suite_data.items():
            if suite not in original_data:
                continue
            compare_lines = format_task_eval_compare_lines(
                result_root,
                suite,
                data["run_dir"],
                data["task_rows"],
                original_data[suite]["task_rows"],
                original_data[suite]["run_dir"],
            )
            write_task_eval_compare(data["suite_dir"], suite, compare_lines)


def main() -> None:
    args = parse_args()
    for result_root in args.result_roots:
        process_result_root(result_root)
    process_compare_outputs(args.result_roots)
    joined_roots = ", ".join(str(path) for path in args.result_roots)
    print(f"Generated suite-level task_eval_<suite>.txt and task_eval_compare_<suite>.txt under: {joined_roots}")


if __name__ == "__main__":
    main()