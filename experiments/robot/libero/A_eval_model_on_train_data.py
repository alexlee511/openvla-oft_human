"""Evaluate OpenVLA action predictions on LIBERO train HDF5 demos.

This script feeds per-timestep train observations (text + agentview + wrist + proprio)
into a checkpoint and compares the first predicted action in each chunk against the
next-step supervision in the dataset.

Designed for joint-control datasets such as:
  LIBERO/libero/datasets/libero_10_humanized_no_noops

Metrics include:
- Joint target error (pred[:7] vs gt action[:7] and vs next joint state)
- Gripper sign agreement (pred sign vs gt sign)
- Gripper open/close consistency against observed width change
  (width up => opening, width down => closing)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
from experiments.robot.robot_utils import get_action, set_seed_everywhere


def parse_task_label_from_hdf5_path(hdf5_path: str) -> str:
    """Convert <TASK_NAME>_demo.hdf5 into language instruction text.

    Matches the dataset builder logic by removing the SCENE prefix tokens.
    """
    raw = os.path.basename(hdf5_path)
    if raw.endswith("_demo.hdf5"):
        raw = raw[: -len("_demo.hdf5")]
    words = raw.split("_")
    cmd = ""
    for w in words:
        if "SCENE" in w:
            cmd = ""
            continue
        cmd += w + " "
    return cmd.strip()


def rotate_180(image: np.ndarray) -> np.ndarray:
    """Apply the same 180-degree rotation used in TFDS builders/eval preprocessing."""
    return image[::-1, ::-1]


def to_sign(x: float) -> int:
    """Binarize gripper command to {-1, +1}."""
    return 1 if float(x) >= 0.0 else -1


def update_stats(stats: Dict[str, List[float]], key: str, value: float) -> None:
    stats[key].append(float(value))


def mean_or_nan(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def summarize_stats(stats: Dict[str, List[float]]) -> Dict[str, float]:
    out = {}
    for k, v in stats.items():
        if not v:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_median"] = float("nan")
            out[f"{k}_p90"] = float("nan")
            continue
        arr = np.array(v, dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(arr))
        out[f"{k}_median"] = float(np.median(arr))
        out[f"{k}_p90"] = float(np.quantile(arr, 0.9))
    return out


def format_seconds(seconds: float) -> str:
    """Format seconds into H:MM:SS."""
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def build_eval_plan(
    dataset_dir: str,
    max_demos_per_task: int,
    max_steps_per_demo: int,
) -> Tuple[List[Tuple[Path, str, List[Tuple[str, int]]]], int, int]:
    """Precompute task/demo work units and total evaluated steps for ETA reporting."""
    plan: List[Tuple[Path, str, List[Tuple[str, int]]]] = []
    total_demos = 0
    total_steps = 0

    hdf5_files = sorted(Path(dataset_dir).glob("*_demo.hdf5"))
    for hdf5_path in hdf5_files:
        task_name = hdf5_path.stem.replace("_demo", "")
        demo_plan: List[Tuple[str, int]] = []
        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
            if max_demos_per_task > 0:
                demo_keys = demo_keys[:max_demos_per_task]

            for demo_key in demo_keys:
                n = len(f["data"][demo_key]["actions"])
                if n < 2:
                    continue

                t_max = n - 1
                if max_steps_per_demo > 0:
                    t_max = min(t_max, max_steps_per_demo)

                demo_plan.append((demo_key, t_max))
                total_demos += 1
                total_steps += t_max

        plan.append((hdf5_path, task_name, demo_plan))

    return plan, total_demos, total_steps


def evaluate_dataset(
    dataset_dir: str,
    cfg: GenerateConfig,
    max_demos_per_task: int,
    max_steps_per_demo: int,
    width_delta_eps: float,
    eval_chunk_horizon: int,
) -> Dict[str, object]:
    set_seed_everywhere(cfg.seed)

    plan, planned_demos, planned_steps = build_eval_plan(
        dataset_dir=dataset_dir,
        max_demos_per_task=max_demos_per_task,
        max_steps_per_demo=max_steps_per_demo,
    )
    if not plan:
        raise FileNotFoundError(f"No *_demo.hdf5 files found in {dataset_dir}")

    print(f"[Plan] tasks={len(plan)} demos={planned_demos} eval_steps={planned_steps}")

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    global_stats: Dict[str, List[float]] = defaultdict(list)
    per_task = {}

    total_steps = 0
    total_demos = 0
    last_progress_print = 0.0
    start_time = time.time()

    def print_progress(task_name: str, demo_key: str, timestep: int, demo_steps: int, force: bool = False) -> None:
        nonlocal last_progress_print
        now = time.time()
        if not force and now - last_progress_print < 5.0:
            return

        elapsed = now - start_time
        rate = total_steps / elapsed if elapsed > 0 and total_steps > 0 else 0.0
        remaining_steps = max(planned_steps - total_steps, 0)
        eta = remaining_steps / rate if rate > 0 else float("inf")

        print(
            f"[Progress] task={task_name} demo={demo_key} step={timestep}/{demo_steps} "
            f"overall={total_steps}/{planned_steps} elapsed={format_seconds(elapsed)} "
            f"eta={format_seconds(eta)}"
        )
        last_progress_print = now

    for task_index, (hdf5_path, task_name, demo_plan) in enumerate(plan, start=1):
        task_label = parse_task_label_from_hdf5_path(str(hdf5_path))
        print(f"[Task] {task_index}/{len(plan)} {task_name} demos={len(demo_plan)}")

        task_stats: Dict[str, List[float]] = defaultdict(list)

        with h5py.File(hdf5_path, "r") as f:
            for demo_index, (demo_key, planned_demo_steps) in enumerate(demo_plan, start=1):
                print(f"[Demo] {task_name} {demo_index}/{len(demo_plan)} {demo_key} steps={planned_demo_steps}")
                demo = f["data"][demo_key]
                obs = demo["obs"]

                actions = obs_actions = demo["actions"][()].astype(np.float32)
                states = obs["state"][()].astype(np.float32)
                joint_states = obs["joint_states"][()].astype(np.float32)
                agent_imgs = obs["agentview_rgb"][()]
                wrist_imgs = obs["eye_in_hand_rgb"][()]

                n = min(
                    len(actions),
                    len(states),
                    len(joint_states),
                    len(agent_imgs),
                    len(wrist_imgs),
                )
                if n < 2:
                    continue

                t_max = n - 1
                if max_steps_per_demo > 0:
                    t_max = min(t_max, max_steps_per_demo)

                total_demos += 1
                print_progress(task_name, demo_key, 0, t_max, force=True)

                for t in range(t_max):
                    observation = {
                        "full_image": rotate_180(agent_imgs[t]),
                        "wrist_image": rotate_180(wrist_imgs[t]),
                        "state": states[t].copy(),
                    }

                    pred_chunk = get_action(
                        cfg,
                        model,
                        observation,
                        task_label,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                    max_chunk = len(pred_chunk)
                    if eval_chunk_horizon > 0:
                        max_chunk = min(max_chunk, eval_chunk_horizon)
                    max_chunk = min(max_chunk, n - 1 - t)

                    chunk_joint_maes = []
                    chunk_joint_rmses = []
                    chunk_next_joint_maes = []
                    chunk_g_sign_accs = []

                    for offset in range(max_chunk):
                        pred = np.asarray(pred_chunk[offset], dtype=np.float32)
                        gt = actions[t + offset]

                        if pred.shape[0] < 8 or gt.shape[0] < 8:
                            continue

                        next_joint = joint_states[t + offset + 1]
                        width_now = float(states[t + offset, 7])
                        width_next = float(states[t + offset + 1, 7])
                        width_delta = width_next - width_now

                        joint_mae = float(np.mean(np.abs(pred[:7] - gt[:7])))
                        joint_rmse = float(np.sqrt(np.mean((pred[:7] - gt[:7]) ** 2)))
                        next_joint_mae = float(np.mean(np.abs(pred[:7] - next_joint[:7])))

                        pred_g = to_sign(pred[7])
                        gt_g = to_sign(gt[7])
                        g_sign_acc = 1.0 if pred_g == gt_g else 0.0

                        chunk_joint_maes.append(joint_mae)
                        chunk_joint_rmses.append(joint_rmse)
                        chunk_next_joint_maes.append(next_joint_mae)
                        chunk_g_sign_accs.append(g_sign_acc)

                        if offset == 0:
                            update_stats(task_stats, "joint_mae", joint_mae)
                            update_stats(task_stats, "joint_rmse", joint_rmse)
                            update_stats(task_stats, "next_joint_mae", next_joint_mae)
                            update_stats(task_stats, "gripper_sign_acc", g_sign_acc)
                            update_stats(task_stats, "pred_gripper_value", float(pred[7]))
                            update_stats(task_stats, "gt_gripper_value", float(gt[7]))
                            update_stats(task_stats, "width_delta", width_delta)

                            # Convention used in this codebase: -1=open (width up), +1=close (width down)
                            if abs(width_delta) > width_delta_eps:
                                pred_openclose_ok = (
                                    (pred_g < 0 and width_delta > 0) or (pred_g > 0 and width_delta < 0)
                                )
                                gt_openclose_ok = (
                                    (gt_g < 0 and width_delta > 0) or (gt_g > 0 and width_delta < 0)
                                )
                                update_stats(task_stats, "pred_openclose_consistency", 1.0 if pred_openclose_ok else 0.0)
                                update_stats(task_stats, "gt_openclose_consistency", 1.0 if gt_openclose_ok else 0.0)

                            if gt_g < 0:
                                update_stats(task_stats, "p_width_up_given_gt_neg", 1.0 if width_delta > width_delta_eps else 0.0)
                            if gt_g > 0:
                                update_stats(task_stats, "p_width_down_given_gt_pos", 1.0 if width_delta < -width_delta_eps else 0.0)

                        update_stats(task_stats, f"chunk_joint_mae_offset_{offset}", joint_mae)
                        update_stats(task_stats, f"chunk_joint_rmse_offset_{offset}", joint_rmse)
                        update_stats(task_stats, f"chunk_next_joint_mae_offset_{offset}", next_joint_mae)
                        update_stats(task_stats, f"chunk_gripper_sign_acc_offset_{offset}", g_sign_acc)

                        if abs(width_delta) > width_delta_eps:
                            pred_openclose_ok = (
                                (pred_g < 0 and width_delta > 0) or (pred_g > 0 and width_delta < 0)
                            )
                            gt_openclose_ok = (
                                (gt_g < 0 and width_delta > 0) or (gt_g > 0 and width_delta < 0)
                            )
                            update_stats(task_stats, f"chunk_pred_openclose_consistency_offset_{offset}", 1.0 if pred_openclose_ok else 0.0)
                            update_stats(task_stats, f"chunk_gt_openclose_consistency_offset_{offset}", 1.0 if gt_openclose_ok else 0.0)

                    if chunk_joint_maes:
                        update_stats(task_stats, "chunk_joint_mae", float(np.mean(chunk_joint_maes)))
                        update_stats(task_stats, "chunk_joint_rmse", float(np.mean(chunk_joint_rmses)))
                        update_stats(task_stats, "chunk_next_joint_mae", float(np.mean(chunk_next_joint_maes)))
                        update_stats(task_stats, "chunk_gripper_sign_acc", float(np.mean(chunk_g_sign_accs)))
                        update_stats(task_stats, "chunk_effective_horizon", float(len(chunk_joint_maes)))

                    total_steps += 1
                    if (t + 1) % 50 == 0 or (t + 1) == t_max:
                        print_progress(task_name, demo_key, t + 1, t_max, force=(t + 1) == t_max)

        task_summary = summarize_stats(task_stats)
        task_summary["num_steps_evaluated"] = len(task_stats["joint_mae"])
        task_summary["num_demos_evaluated"] = len(demo_plan)
        per_task[task_name] = task_summary

        for k, vals in task_stats.items():
            global_stats[k].extend(vals)

    global_summary = summarize_stats(global_stats)
    global_summary["num_tasks"] = len(per_task)
    global_summary["num_demos_evaluated"] = total_demos
    global_summary["num_steps_evaluated"] = total_steps

    return {
        "dataset_dir": dataset_dir,
        "checkpoint": str(cfg.pretrained_checkpoint),
        "unnorm_key": str(cfg.unnorm_key),
        "global": global_summary,
        "per_task": per_task,
    }


def build_cfg_from_args(args: argparse.Namespace) -> GenerateConfig:
    cfg = GenerateConfig()
    cfg.model_family = "openvla"
    cfg.pretrained_checkpoint = args.pretrained_checkpoint

    cfg.use_l1_regression = args.use_l1_regression
    cfg.use_diffusion = args.use_diffusion
    cfg.num_diffusion_steps_inference = args.num_diffusion_steps_inference
    cfg.use_film = args.use_film

    cfg.num_images_in_input = args.num_images_in_input
    cfg.use_proprio = args.use_proprio
    cfg.center_crop = args.center_crop
    cfg.use_joint_pos = True

    cfg.lora_rank = args.lora_rank
    cfg.unnorm_key = args.unnorm_key

    cfg.load_in_8bit = args.load_in_8bit
    cfg.load_in_4bit = args.load_in_4bit

    cfg.task_suite_name = args.task_suite_name
    cfg.seed = args.seed
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model action outputs on train HDF5 dataset timesteps")

    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_10_humanized_no_noops",
    )
    parser.add_argument("--task_suite_name", type=str, default="libero_10")
    parser.add_argument("--unnorm_key", type=str, default="")

    parser.add_argument("--num_images_in_input", type=int, default=2)
    parser.add_argument("--use_proprio", action="store_true", default=True)
    parser.add_argument("--center_crop", action="store_true", default=True)

    parser.add_argument("--use_l1_regression", action="store_true", default=True)
    parser.add_argument("--use_diffusion", action="store_true", default=False)
    parser.add_argument("--num_diffusion_steps_inference", type=int, default=50)
    parser.add_argument("--use_film", action="store_true", default=False)

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--max_demos_per_task", type=int, default=0, help="0 means all demos")
    parser.add_argument("--max_steps_per_demo", type=int, default=0, help="0 means all valid steps")
    parser.add_argument("--width_delta_eps", type=float, default=1e-6)
    parser.add_argument(
        "--eval_chunk_horizon",
        type=int,
        default=1,
        help="How many predicted chunk actions to score per observation. 1=only first action, 8=full chunk for 8-step models, 0=use full returned chunk.",
    )

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output_json",
        type=str,
        default="./experiments/logs/train_data_action_eval.json",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_cfg_from_args(args)

    result = evaluate_dataset(
        dataset_dir=args.dataset_dir,
        cfg=cfg,
        max_demos_per_task=args.max_demos_per_task,
        max_steps_per_demo=args.max_steps_per_demo,
        width_delta_eps=args.width_delta_eps,
        eval_chunk_horizon=args.eval_chunk_horizon,
    )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print("Saved evaluation report:", args.output_json)
    print("Global summary:")
    for k, v in result["global"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
