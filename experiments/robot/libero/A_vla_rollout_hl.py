#!/usr/bin/env python3
"""
A_vla_rollout_hl.py

Batch-evaluate human-likeness metrics on all rollout .npz files from an
eval run (produced by run_libero_eval.py).

Usage:
  # Evaluate all episodes in a rollout directory:
  python experiments/robot/libero/A_vla_rollout_hl.py \
      --rollout_dir ./rollouts/libero_10/<checkpoint>/<datetime>/ \
      --suite libero_10

  # Or point at a specific rollout_data/ folder:
  python experiments/robot/libero/A_vla_rollout_hl.py \
      --rollout_dir ./rollouts/libero_10/<checkpoint>/<datetime>/rollout_data/ \
      --suite libero_10

  # Optionally compare against humanized reference NPZs:
  python experiments/robot/libero/A_vla_rollout_hl.py \
      --rollout_dir ./rollouts/libero_10/<checkpoint>/<datetime>/ \
      --humanized_dir /path/to/humanized_npzs/ \
      --suite libero_10

Outputs:
  <rollout_dir>/human_likeness/
    per_episode/          — per-episode eval_rollout/ results
    summary.csv           — one row per episode with all metrics
    aggregate_summary.txt — overall averages across all episodes
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

# Add LIBERO scripts to path so we can import the evaluator functions
LIBERO_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "LIBERO", "scripts")
if not os.path.isdir(LIBERO_SCRIPTS):
    # Try the user's known path
    LIBERO_SCRIPTS = "/home/vsp1323/alex/LIBERO/scripts"
sys.path.insert(0, LIBERO_SCRIPTS)

from A_human_likeness_evaluate import (
    METRIC_KEYS,
    WEIGHTS,
    WEIGHTS_NO_MJE,
    build_env,
    evaluate_trajectory,
    save_results,
)

# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def find_rollout_npzs(rollout_dir):
    """Find all episode .npz files under rollout_dir."""
    # Check for rollout_data/ subdirectory
    data_dir = os.path.join(rollout_dir, "rollout_data")
    if os.path.isdir(data_dir):
        search_dir = data_dir
    else:
        search_dir = rollout_dir

    npzs = sorted(glob.glob(os.path.join(search_dir, "*.npz")))
    return npzs


def load_rollout_npz(npz_path):
    """Load a rollout npz and extract joint_states, task_name, success info."""
    data = np.load(npz_path, allow_pickle=True)

    # Joint states
    if "joint_states_sim" in data:
        js = data["joint_states_sim"].astype(np.float32)
    elif "joint_pos" in data:
        js = data["joint_pos"].astype(np.float32)
    else:
        raise KeyError(f"No joint_states_sim or joint_pos in {npz_path}: {list(data.keys())}")

    # Task name
    if "task_name" in data:
        tn = data["task_name"]
        task_name = tn.item() if hasattr(tn, "item") else str(tn)
    elif "task_description" in data:
        td = data["task_description"]
        task_name = (td.item() if hasattr(td, "item") else str(td)).replace(" ", "_")
    else:
        # Try to infer from filename
        base = os.path.basename(npz_path)
        task_name = base.split("--task=")[-1].replace(".npz", "") if "--task=" in base else "unknown"

    # Success info
    success = None
    if "task_success" in data:
        sv = data["task_success"]
        success = bool(sv.item() if hasattr(sv, "item") else sv)
    elif "success" in data:
        sv = data["success"]
        success = bool(sv.item() if hasattr(sv, "item") else sv)

    success_step = None
    if "task_success_step" in data:
        ss = data["task_success_step"]
        step_val = int(ss.item()) if hasattr(ss, "item") else int(ss)
        if step_val >= 0:
            success_step = step_val

    return js, task_name, success, success_step


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Batch human-likeness evaluation on VLA rollout npz files.")
    ap.add_argument("--rollout_dir", required=True,
                    help="Path to rollout directory (containing rollout_data/*.npz)")
    ap.add_argument("--suite", default="libero_10",
                    help="LIBERO suite name (default: libero_10)")
    ap.add_argument("--control_freq", type=int, default=20,
                    help="Control frequency in Hz (default: 20)")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory. Default: <rollout_dir>/human_likeness/")
    ap.add_argument("--no_noMJE", action="store_true",
                    help="Suppress unified_noMJE output")
    args = ap.parse_args()

    # Find npz files
    npzs = find_rollout_npzs(args.rollout_dir)
    if not npzs:
        print(f"[Error] No .npz files found in {args.rollout_dir} or {args.rollout_dir}/rollout_data/")
        sys.exit(1)
    print(f"[Info] Found {len(npzs)} rollout npz files")

    # Output directory
    out_dir = args.out_dir or os.path.join(args.rollout_dir, "human_likeness")
    os.makedirs(out_dir, exist_ok=True)
    per_episode_dir = os.path.join(out_dir, "per_episode")

    dt = 1.0 / args.control_freq

    # Cache built envs per task to avoid re-creating
    env_cache = {}

    # Collect per-episode results for summary CSV
    summary_rows = []

    for i, npz_path in enumerate(npzs):
        fname = os.path.basename(npz_path)
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(npzs)}] {fname}")
        print(f"{'='*70}")

        try:
            js, task_name, success, success_step = load_rollout_npz(npz_path)
        except Exception as e:
            print(f"  [SKIP] Failed to load: {e}")
            summary_rows.append({
                "file": fname, "task": "?", "success": "?",
                "T": 0, "unified": float("nan"), "unified_noMJE": float("nan"),
                **{f"m_{k}": float("nan") for k in METRIC_KEYS},
                "error": str(e),
            })
            continue

        T = js.shape[0]
        print(f"  Task: {task_name}  T={T}  Success={success}")

        if T < 4:
            print(f"  [SKIP] Too few timesteps ({T} < 4)")
            summary_rows.append({
                "file": fname, "task": task_name, "success": success,
                "T": T, "unified": float("nan"), "unified_noMJE": float("nan"),
                **{f"m_{k}": float("nan") for k in METRIC_KEYS},
                "error": "too_few_steps",
            })
            continue

        # Build or reuse env
        if task_name not in env_cache:
            try:
                env, model, data_sim, sid, eid, wid, qadrs = build_env(task_name, args.suite)
                env_cache[task_name] = (env, model, data_sim, sid, eid, wid, qadrs)
            except Exception as e:
                print(f"  [SKIP] Failed to build env for task '{task_name}': {e}")
                summary_rows.append({
                    "file": fname, "task": task_name, "success": success,
                    "T": T, "unified": float("nan"), "unified_noMJE": float("nan"),
                    **{f"m_{k}": float("nan") for k in METRIC_KEYS},
                    "error": str(e),
                })
                continue

        env, model, data_sim, sid, eid, wid, qadrs = env_cache[task_name]

        # Evaluate
        raw, norm, uni, uni_nm, diag = evaluate_trajectory(
            js, model, data_sim, sid, eid, wid, qadrs, dt, label=fname)

        # Save per-episode results
        ep_dir = os.path.join(per_episode_dir, fname.replace(".npz", ""))
        avg, avg_nm = save_results(ep_dir, raw, norm, uni, uni_nm, diag, T,
                                   no_noMJE=args.no_noMJE)

        # Collect summary
        row = {
            "file": fname,
            "task": task_name,
            "success": success,
            "success_step": success_step,
            "T": T,
            "unified": avg,
            "unified_noMJE": avg_nm,
        }
        for k in METRIC_KEYS:
            row[f"m_{k}"] = float(np.mean(norm[k]))
            row[f"raw_{k}"] = float(np.mean(raw[k]))
        row["error"] = ""
        summary_rows.append(row)

        print(f"  Unified={avg:.4f}  noMJE={avg_nm:.4f}  Success={success}")

    # Close all cached envs
    for task_name, (env, *_) in env_cache.items():
        try:
            env.close()
        except Exception:
            pass

    # ── Write summary CSV ──
    csv_path = os.path.join(out_dir, "summary.csv")
    fieldnames = ["file", "task", "success", "success_step", "T",
                  "unified", "unified_noMJE"]
    fieldnames += [f"m_{k}" for k in METRIC_KEYS]
    fieldnames += [f"raw_{k}" for k in METRIC_KEYS]
    fieldnames += ["error"]

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print(f"\n[OK] Summary CSV: {csv_path}")

    # ── Aggregate summary ──
    valid_rows = [r for r in summary_rows if not r.get("error")]
    n_total = len(summary_rows)
    n_valid = len(valid_rows)
    n_success = sum(1 for r in valid_rows if r.get("success") is True)

    txt_path = os.path.join(out_dir, "aggregate_summary.txt")
    with open(txt_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("  VLA ROLLOUT HUMAN-LIKENESS AGGREGATE SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"  Rollout dir:    {args.rollout_dir}\n")
        f.write(f"  Suite:          {args.suite}\n")
        f.write(f"  Total episodes: {n_total}\n")
        f.write(f"  Valid episodes: {n_valid}\n")
        f.write(f"  Success rate:   {n_success}/{n_valid} ({100*n_success/n_valid:.1f}%)\n" if n_valid > 0 else "  Success rate:   N/A\n")
        f.write("\n")

        if n_valid > 0:
            avg_uni = np.mean([r["unified"] for r in valid_rows])
            avg_uni_nm = np.mean([r["unified_noMJE"] for r in valid_rows])
            f.write(f"  Avg Unified Score:      {avg_uni:.4f}\n")
            f.write(f"  Avg Unified (no MJE):   {avg_uni_nm:.4f}\n\n")

            # Success-only averages
            success_rows = [r for r in valid_rows if r.get("success") is True]
            if success_rows:
                avg_uni_s = np.mean([r["unified"] for r in success_rows])
                avg_uni_nm_s = np.mean([r["unified_noMJE"] for r in success_rows])
                f.write(f"  Avg Unified (success only):      {avg_uni_s:.4f}  (n={len(success_rows)})\n")
                f.write(f"  Avg Unified noMJE (success only): {avg_uni_nm_s:.4f}\n\n")

            f.write("  Per-metric averages (all valid episodes):\n")
            for k in METRIC_KEYS:
                vals = [r[f"m_{k}"] for r in valid_rows]
                raw_vals = [r[f"raw_{k}"] for r in valid_rows]
                f.write(f"    m_{k}:  {np.mean(vals):.4f}  (std={np.std(vals):.4f}, raw_avg={np.mean(raw_vals):.6f})\n")

            # Per-task breakdown
            tasks = sorted(set(r["task"] for r in valid_rows))
            if len(tasks) > 1:
                f.write(f"\n  Per-task breakdown ({len(tasks)} tasks):\n")
                f.write(f"  {'Task':<55s} {'N':>3s} {'SR':>6s} {'Unified':>8s} {'noMJE':>8s}\n")
                f.write("  " + "-"*82 + "\n")
                for task in tasks:
                    task_rows = [r for r in valid_rows if r["task"] == task]
                    n_t = len(task_rows)
                    n_s = sum(1 for r in task_rows if r.get("success") is True)
                    u = np.mean([r["unified"] for r in task_rows])
                    u_nm = np.mean([r["unified_noMJE"] for r in task_rows])
                    sr = f"{n_s}/{n_t}"
                    task_display = task[:54]
                    f.write(f"  {task_display:<55s} {n_t:>3d} {sr:>6s} {u:>8.4f} {u_nm:>8.4f}\n")

        f.write("\n" + "="*60 + "\n")

    # Print to stdout too
    with open(txt_path, "r") as f:
        print(f.read())

    print(f"[Done] All outputs -> {out_dir}/")


if __name__ == "__main__":
    main()
