"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import re
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_frontview_image,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_data,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Default controller output_max for JOINT_POSITION controller (Panda, per-joint per-step at 20Hz)
# The robosuite default is 0.05, which is FAR too small — the arm can't track the model's
# absolute target positions (typical step-delta ~0.05-0.15 rad), causing persistent lag
# and jerky motion. Use margin=50× typical max dq (~0.05) = 2.5, matching
# A_libero_joint_replay.py's dq_limits_from_joint_states(margin=50.0) approach.
DEFAULT_JOINT_DQ_MAX = 2.5


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                      # Number of actions to execute open-loop before requerying policy

    use_joint_pos: bool = False                       # If True, use JOINT_POSITION controller with 8D absolute actions
    joint_dq_max: float = 2.5                        # Per-joint max delta per step for JOINT_POSITION controller (margin=50x)
    joint_kp: float = 120.0                           # PD controller kp gain (optimal from A_libero_joint_replay.py sweep)
    joint_damping_ratio: float = 1.0                  # PD damping ratio (1.0=critically damped, matching replay controller)
    joint_Kp_overshoot: float = 2.5                   # Outer overshoot multiplier (matching A_libero_joint_replay.py default)
    joint_substeps: int = 1                           # Sub-steps per action for JOINT_POSITION (1=disabled, >1=auto-compute from sim_freq/control_freq=25 for 500Hz/20Hz). Higher control_freq helps PD converge.

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 512                           # Resolution for environment images (higher=better video, min 256)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Use explicit unnorm_key if provided, otherwise derive from task_suite_name
    if cfg.unnorm_key and cfg.unnorm_key in model.norm_stats:
        return  # already set and valid

    unnorm_key = cfg.unnorm_key if cfg.unnorm_key else cfg.task_suite_name

    # Try exact match first
    if unnorm_key in model.norm_stats:
        cfg.unnorm_key = unnorm_key
        return

    # Try common suffixed variants
    for suffix in ["_no_noops", "_joint_noops", "_joint_no_noops",
                   "_humanized_no_noops", "_humanized_noops", "_humanized"]:
        candidate = f"{unnorm_key}{suffix}"
        if candidate in model.norm_stats:
            cfg.unnorm_key = candidate
            return

    # Last resort: if norm_stats has exactly one key, use it
    if len(model.norm_stats) == 1:
        only_key = next(iter(model.norm_stats))
        logger.info(f"Auto-selected unnorm_key '{only_key}' (only key in norm_stats)")
        cfg.unnorm_key = only_key
        return

    assert False, (
        f"Action un-norm key '{unnorm_key}' not found in VLA `norm_stats`!\n"
        f"Available keys: {list(model.norm_stats.keys())}\n"
        f"Pass --unnorm_key explicitly."
    )


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size, use_joint_pos=False):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    if use_joint_pos:
        # Joint position proprio: [joint_pos(7), gripper_width(1)] = 8D
        gripper_width = np.mean(obs["robot0_gripper_qpos"])  # scalar
        state = np.concatenate((obs["robot0_joint_pos"], [gripper_width]))
    else:
        # EEF proprio: [pos(3), axisangle(3), gripper_qpos(2)] = 8D
        state = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        )

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": state,
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family, use_joint_pos=False, obs=None, dq_max=None, Kp_overshoot=1.8):
    """Process action before sending to environment.

    For JOINT_POSITION mode:
      - Model outputs absolute joint positions (8D: 7 joints + 1 gripper).
      - Convert joints to normalized delta with Kp overshoot:
        clip(Kp * (q_target - q_cur) / dq_max, -1, 1).
      - Kp > 1 commands overshoot so the PD controller converges faster
        (goal_qpos = q_cur + Kp * (q_target - q_cur)).
      - Gripper passes through as raw -1/+1 (no inversion needed).

    For OSC_POSE mode (original):
      - Normalize gripper [0,1] -> [-1,+1] and invert sign.
    """
    if use_joint_pos:
        assert obs is not None, "obs required for JOINT_POSITION absolute→delta conversion"
        assert dq_max is not None, "dq_max required for JOINT_POSITION conversion"

        q_cur = obs["robot0_joint_pos"]  # (7,)
        q_target = action[:7]            # absolute joint targets from model
        dq_des = q_target - q_cur        # desired delta
        # Kp > 1 commands overshoot so the PD controller converges faster.
        # Matches A_libero_joint_replay.py: u = Kp * dq_des / dq_max
        u = np.clip(Kp_overshoot * dq_des / dq_max, -1.0, 1.0).astype(np.float32)

        processed = np.zeros(8, dtype=np.float32)
        processed[:7] = u
        processed[7] = np.sign(action[7])  # binarize gripper: -1=open, +1=close
        return processed
    else:
        # Original OSC_POSE processing
        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
        action = normalize_gripper_action(action, binarize=True)

        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        if model_family == "openvla":
            action = invert_gripper_action(action)

        return action


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for per-episode object tracking and recording
# ─────────────────────────────────────────────────────────────────────────────


def _snapshot_object_positions(obs):
    """Extract all non-robot object positions from obs.

    Returns:
        dict mapping object name (e.g. 'alphabet_soup_1') to np.array([x, y, z]).
    """
    positions = {}
    for key in obs:
        if key.endswith("_pos") and not key.startswith("robot0_") and "_to_robot0_" not in key:
            obj_name = key[:-4]  # strip "_pos"
            positions[obj_name] = np.array(obs[key], dtype=np.float64).copy()
    return positions


def _update_min_eef_distances(obs, min_distances):
    """Update running minimum EEF-to-object distances using *_to_robot0_eef_pos keys."""
    for key in obs:
        if key.endswith("_to_robot0_eef_pos"):
            obj_name = key.replace("_to_robot0_eef_pos", "")
            dist = float(np.linalg.norm(obs[key]))
            if obj_name not in min_distances or dist < min_distances[obj_name]:
                min_distances[obj_name] = dist


def _parse_task_objects(task_description, available_objects):
    """Parse task description to identify manipulated and goal objects.

    Uses **progressive suffix matching**: for each object base name like
    ``akita_black_bowl``, tries "akita black bowl" → "black bowl" → "bowl"
    until a substring is found in the description.  This handles common
    LIBERO naming mismatches (e.g. ``black_book_1`` matching "the book").

    Objects are then split into *manipulated* vs *goal* by the last spatial
    preposition ("in the", "on the", …).

    Returns:
        (manipulated, goals): lists of object key names.
    """
    desc_lower = task_description.lower()

    def _match(base):
        """Try progressively shorter suffixes of *base* (space-joined).

        Returns (matched: bool, position_in_desc: int, matched_len: int).
        """
        words = base.split("_")
        for start in range(len(words)):
            substr = " ".join(words[start:])
            idx = desc_lower.find(substr)
            if idx >= 0:
                return True, idx, len(words) - start
        return False, -1, 0

    matched = []
    for obj in available_objects:
        base = re.sub(r"_\d+$", "", obj)  # strip trailing _1, _2 …
        ok, pos, n_words = _match(base)
        if ok:
            matched.append((obj, base.replace("_", " "), pos, n_words))

    if not matched:
        return [], []

    matched.sort(key=lambda x: x[2])  # sort by position in description

    # Find the last preposition that separates manipulated from goal
    split_idx = -1
    for prep in [" in the ", " on the ", " into the ", " on it", " in it",
                 " to the right of ", " to the left of "]:
        idx = desc_lower.rfind(prep)
        if idx > split_idx:
            split_idx = idx

    manipulated, goals = [], []
    if split_idx >= 0:
        for obj, display, pos, _nw in matched:
            if pos < split_idx:
                manipulated.append(obj)
            else:
                goals.append(obj)
    else:
        manipulated = [obj for obj, _, _, _ in matched]

    return manipulated, goals


def compute_episode_record(task_description, success, total_steps, max_steps,
                           initial_object_positions, final_object_positions,
                           min_eef_distances, initial_eef_pos, final_eef_pos,
                           total_eef_displacement):
    """Compute quantitative per-episode record.

    Metrics per manipulated object:
      - min_eef_distance_m: closest the EEF got (grasp proximity)
      - displacement_m: how far the object moved from start
      - progress_ratio: fraction of initial→goal distance closed (0→1)
      - final_distance_to_goal_m: remaining distance to goal object
    """
    record = {
        "task_description": task_description,
        "success": bool(success),
        "total_steps": int(total_steps),
        "max_steps": int(max_steps),
        "timed_out": int(total_steps) >= int(max_steps),
    }

    all_objects = sorted(set(list(initial_object_positions.keys())
                             + list(final_object_positions.keys())))
    manipulated, goals = _parse_task_objects(task_description, all_objects)

    # Per-object metrics
    objects_record = {}
    for obj_name in all_objects:
        init_pos = initial_object_positions.get(obj_name)
        final_pos = final_object_positions.get(obj_name)
        if init_pos is None or final_pos is None:
            continue

        displacement = float(np.linalg.norm(final_pos - init_pos))
        obj_data = {
            "initial_pos": [round(float(x), 5) for x in init_pos],
            "final_pos": [round(float(x), 5) for x in final_pos],
            "displacement_m": round(displacement, 5),
            "min_eef_distance_m": round(min_eef_distances.get(obj_name, float("inf")), 5),
            "is_manipulated": obj_name in manipulated,
            "is_goal": obj_name in goals,
        }

        # For manipulated objects with a known goal, compute progress ratio
        if obj_name in manipulated and goals:
            goal_obj = goals[0]
            goal_init = initial_object_positions.get(goal_obj)
            goal_final = final_object_positions.get(goal_obj)
            if goal_init is not None and goal_final is not None:
                init_dist = float(np.linalg.norm(init_pos - goal_init))
                final_dist = float(np.linalg.norm(final_pos - goal_final))
                progress = (init_dist - final_dist) / max(init_dist, 1e-6)
                obj_data["goal_object"] = goal_obj
                obj_data["initial_distance_to_goal_m"] = round(init_dist, 5)
                obj_data["final_distance_to_goal_m"] = round(final_dist, 5)
                obj_data["progress_ratio"] = round(progress, 4)

        objects_record[obj_name] = obj_data

    record["objects"] = objects_record

    # EEF summary
    record["eef"] = {
        "initial_pos": [round(float(x), 5) for x in initial_eef_pos] if initial_eef_pos is not None else None,
        "final_pos": [round(float(x), 5) for x in final_eef_pos] if final_eef_pos is not None else None,
        "total_displacement_m": round(total_eef_displacement, 5),
    }

    # Convenience summary for manipulated objects
    summary = []
    for obj_name in manipulated:
        if obj_name in objects_record:
            od = objects_record[obj_name]
            summary.append({
                "object": obj_name,
                "min_eef_distance_m": od["min_eef_distance_m"],
                "displacement_m": od["displacement_m"],
                "progress_ratio": od.get("progress_ratio"),
                "final_distance_to_goal_m": od.get("final_distance_to_goal_m"),
            })
    record["manipulated_objects_summary"] = summary

    return record


def save_episode_record(record, idx, success, task_description, log_file=None, rollout_dir=None):
    """Save per-episode record as JSON in the ``record/`` sub-folder."""
    if rollout_dir is None:
        rollout_dir = f"./rollouts/{DATE_TIME}"
    record_dir = os.path.join(rollout_dir, "record")
    os.makedirs(record_dir, exist_ok=True)
    processed_task = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    json_path = os.path.join(record_dir, f"episode={idx}--success={success}--task={processed_task}.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Saved episode record at {json_path}")
    if log_file:
        log_file.write(f"Saved episode record at {json_path}\n")
    return json_path


def save_aggregate_results(task_results, rollout_dir, cfg):
    """Save per-task aggregate results to ``aggregate/`` sub-folder.

    Produces: task_results.txt, task_results.csv, task_results.json.
    """
    agg_dir = os.path.join(rollout_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    total_episodes = sum(r["episodes"] for r in task_results.values())
    total_successes = sum(r["successes"] for r in task_results.values())
    overall_rate = total_successes / max(total_episodes, 1) * 100

    # --- Text table ---
    lines = [
        f"Task Suite: {cfg.task_suite_name}",
        f"Checkpoint: {cfg.pretrained_checkpoint}",
        f"Date: {DATE_TIME}",
        f"Controller: {'JOINT_POSITION' if cfg.use_joint_pos else 'OSC_POSE'}",
    ]
    if cfg.use_joint_pos:
        lines.append(f"kp={cfg.joint_kp}, damping_ratio={cfg.joint_damping_ratio}, dq_max={cfg.joint_dq_max}, Kp_overshoot={cfg.joint_Kp_overshoot}, substeps={cfg.joint_substeps}")
    lines += [
        f"Num open-loop steps: {cfg.num_open_loop_steps}",
        f"Num trials per task: {cfg.num_trials_per_task}",
        "",
        f"{'Task':<60} {'Success':>10} {'Rate':>8}",
        "-" * 82,
    ]
    for task_desc in sorted(task_results):
        r = task_results[task_desc]
        rate = r["successes"] / max(r["episodes"], 1) * 100
        short = task_desc.lower().replace(" ", "_")[:58]
        lines.append(f"{short:<60} {r['successes']:3d}/{r['episodes']:<4d} {rate:6.1f}%")
    lines += [
        "-" * 82,
        f"{'TOTAL':<60} {total_successes:3d}/{total_episodes:<4d} {overall_rate:6.1f}%",
    ]
    txt_path = os.path.join(agg_dir, "task_results.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # --- CSV ---
    csv_path = os.path.join(agg_dir, "task_results.csv")
    with open(csv_path, "w") as fh:
        fh.write("task,successes,episodes,success_rate\n")
        for task_desc in sorted(task_results):
            r = task_results[task_desc]
            short = task_desc.lower().replace(" ", "_")[:50]
            fh.write(f"{short},{r['successes']},{r['episodes']},{r['successes']/max(r['episodes'],1):.4f}\n")
        fh.write(f"TOTAL,{total_successes},{total_episodes},{total_successes/max(total_episodes,1):.4f}\n")

    # --- JSON ---
    json_path = os.path.join(agg_dir, "task_results.json")
    json_data = {
        "config": {
            "task_suite": cfg.task_suite_name,
            "checkpoint": str(cfg.pretrained_checkpoint),
            "date": DATE_TIME,
            "controller": "JOINT_POSITION" if cfg.use_joint_pos else "OSC_POSE",
            "joint_kp": cfg.joint_kp if cfg.use_joint_pos else None,
            "joint_damping_ratio": cfg.joint_damping_ratio if cfg.use_joint_pos else None,
            "joint_Kp_overshoot": cfg.joint_Kp_overshoot if cfg.use_joint_pos else None,
            "num_open_loop_steps": cfg.num_open_loop_steps,
            "num_trials_per_task": cfg.num_trials_per_task,
        },
        "per_task": {
            td: {"episodes": r["episodes"], "successes": r["successes"],
                 "success_rate": round(r["successes"] / max(r["episodes"], 1), 4)}
            for td, r in sorted(task_results.items())
        },
        "total": {
            "episodes": total_episodes,
            "successes": total_successes,
            "success_rate": round(total_successes / max(total_episodes, 1), 4),
        },
    }
    with open(json_path, "w") as fh:
        json.dump(json_data, fh, indent=2)

    logger.info(f"Saved aggregate results to {agg_dir}/")

    # --- Plots from episode records ---
    try:
        _generate_aggregate_plots(rollout_dir, agg_dir, task_results)
    except Exception as e:
        logger.warning(f"Could not generate aggregate plots: {e}")

    return agg_dir


def _generate_aggregate_plots(rollout_dir, agg_dir, task_results):
    """Read per-episode records and produce individual analysis plots.

    Creates 5 separate PNG files, each a **vertical**-bar chart (tasks on
    x-axis, full task names rotated below):
      1. success_count.png
      2. progress_ratio.png   (N/A when goal is an env fixture)
      3. goal_proximity.png   (N/A when goal is an env fixture)
      4. min_eef_distance.png (closest EEF approach to matched objects)
      5. object_displacement.png
    Each chart includes a dashed line for the overall average and per-bar
    value labels.  Tasks with no data for a metric are shown in grey with
    an "N/A" annotation.
    """
    import glob
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless servers
    import matplotlib.pyplot as plt

    record_dir = os.path.join(rollout_dir, "record")
    record_files = sorted(glob.glob(os.path.join(record_dir, "*.json")))
    if not record_files:
        logger.warning("No record JSON files found; skipping plot generation.")
        return

    # Load all records
    records = []
    for rf in record_files:
        with open(rf) as fh:
            records.append(json.load(fh))

    # ---------- Aggregate per task ----------
    from collections import defaultdict
    task_metrics = defaultdict(lambda: {
        "progress_ratios": [],
        "goal_proximities": [],
        "min_eef_dists": [],
        "displacements": [],
        "successes": 0,
        "episodes": 0,
    })

    for rec in records:
        task = rec["task_description"]
        tm = task_metrics[task]
        tm["episodes"] += 1
        if rec["success"]:
            tm["successes"] += 1

        # Re-compute manipulated / goal classification from raw object data
        # using the current (improved) parser, so old records also benefit.
        all_obj_names = list(rec.get("objects", {}).keys())
        manipulated, goals = _parse_task_objects(task, all_obj_names)

        for obj_name in manipulated:
            od = rec["objects"].get(obj_name, {})
            mef = od.get("min_eef_distance_m")
            if mef is not None and mef < 1e6:
                tm["min_eef_dists"].append(mef)
            disp = od.get("displacement_m")
            if disp is not None:
                tm["displacements"].append(disp)

            # Progress ratio / goal proximity: only when a goal object exists
            if goals:
                goal_obj = goals[0]
                goal_od = rec["objects"].get(goal_obj, {})
                init_pos = od.get("initial_pos")
                final_pos = od.get("final_pos")
                goal_init = goal_od.get("initial_pos")
                goal_final = goal_od.get("final_pos")
                if init_pos and final_pos and goal_init and goal_final:
                    init_dist = float(np.linalg.norm(
                        np.array(init_pos) - np.array(goal_init)))
                    final_dist = float(np.linalg.norm(
                        np.array(final_pos) - np.array(goal_final)))
                    progress = (init_dist - final_dist) / max(init_dist, 1e-6)
                    tm["progress_ratios"].append(progress)
                    tm["goal_proximities"].append(final_dist)

    # Sort tasks alphabetically; use full names for display
    task_names_sorted = sorted(task_metrics.keys())
    display_names = [t.lower() for t in task_names_sorted]
    n_tasks = len(task_names_sorted)

    # Per-task averages (use NaN for genuinely missing data)
    def _mean_or_nan(lst):
        return float(np.mean(lst)) if lst else float("nan")

    succ_counts   = [task_metrics[t]["successes"]                  for t in task_names_sorted]
    avg_progress  = [_mean_or_nan(task_metrics[t]["progress_ratios"])  for t in task_names_sorted]
    avg_goal_prox = [_mean_or_nan(task_metrics[t]["goal_proximities"]) for t in task_names_sorted]
    avg_eef_dist  = [_mean_or_nan(task_metrics[t]["min_eef_dists"])    for t in task_names_sorted]
    avg_disp      = [_mean_or_nan(task_metrics[t]["displacements"])    for t in task_names_sorted]

    # Overall averages (only from tasks that have data)
    def _overall(per_task_vals):
        valid = [v for v in per_task_vals if not np.isnan(v)]
        return float(np.mean(valid)) if valid else 0.0

    overall_succ = sum(succ_counts) / max(n_tasks, 1)
    overall_pr   = _overall(avg_progress)
    overall_gp   = _overall(avg_goal_prox)
    overall_eef  = _overall(avg_eef_dist)
    overall_disp = _overall(avg_disp)

    x_pos = np.arange(n_tasks)
    bar_width = 0.6
    fig_width = max(n_tasks * 1.4, 8)

    # ---------- Helper to create one vertical bar-chart ----------
    def _save_bar_plot(filename, values, color, ylabel, title, overall_avg,
                       fmt=".3f", value_labels=None):
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        # Separate valid vs N/A bars
        bar_colors = []
        plot_vals = []
        for v in values:
            if np.isnan(v):
                bar_colors.append("#CCCCCC")  # grey for N/A
                plot_vals.append(0)
            else:
                bar_colors.append(color)
                plot_vals.append(v)

        bars = ax.bar(x_pos, plot_vals, width=bar_width, color=bar_colors, alpha=0.85)
        ax.axhline(overall_avg, color="black", linestyle="--", linewidth=1.5,
                   label=f"overall avg = {overall_avg:{fmt}}")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, fontsize=8, rotation=35, ha="right")

        # Value labels above each bar
        for i, (bar, val) in enumerate(zip(bars, values)):
            if np.isnan(val):
                label = "N/A"
            elif value_labels:
                label = value_labels[i]
            else:
                label = f"{val:{fmt}}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    label, ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        path = os.path.join(agg_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot: {path}")

    # 1) Success count (always valid)
    succ_labels = [
        f"{task_metrics[task_names_sorted[i]]['successes']}/{task_metrics[task_names_sorted[i]]['episodes']}"
        for i in range(n_tasks)
    ]
    _save_bar_plot("success_count.png",
                   [float(s) for s in succ_counts], "#4CAF50",
                   "# Successes", "Success Count per Task", overall_succ,
                   fmt=".1f", value_labels=succ_labels)

    # 2) Progress ratio (N/A when goal is an env fixture like stove/cabinet)
    _save_bar_plot("progress_ratio.png", avg_progress, "#2196F3",
                   "Progress Ratio (0 → 1)",
                   "Avg Progress Ratio per Task  (grey = goal not trackable)",
                   overall_pr)

    # 3) Goal proximity (lower = better; N/A when goal not trackable)
    _save_bar_plot("goal_proximity.png", avg_goal_prox, "#FF9800",
                   "Final Distance to Goal (m)",
                   "Avg Goal Proximity per Task  (lower = better; grey = N/A)",
                   overall_gp)

    # 4) Min EEF distance (lower = better)
    _save_bar_plot("min_eef_distance.png", avg_eef_dist, "#9C27B0",
                   "Min EEF Distance (m)",
                   "Avg Min EEF Approach per Task  (lower = closer)",
                   overall_eef)

    # 5) Object displacement
    _save_bar_plot("object_displacement.png", avg_disp, "#F44336",
                   "Object Displacement (m)",
                   "Avg Object Displacement per Task",
                   overall_disp)


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment (suppress sampling-rate print warnings for sub-stepping)
    if cfg.use_joint_pos and cfg.joint_substeps > 1:
        import contextlib, io as _io
        with contextlib.redirect_stdout(_io.StringIO()):
            env.reset()
    else:
        env.reset()

    # After reset, patch observable sampling rates to match control_freq for sub-stepping.
    # env.reset() re-creates all observables at the default 20 Hz, so this must be done
    # after every reset — not just once after env creation.
    if cfg.use_joint_pos and cfg.joint_substeps > 1:
        base_env = env.env
        effective_freq = int(round(1.0 / base_env.model_timestep))  # sim_freq (500)
        for observable in base_env._observables.values():
            observable.set_sampling_rate(effective_freq)

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # --- JOINT_POSITION: patch controller output_max & kp for fast tracking ---
    # Parameters matched to A_libero_joint_replay.py:
    #   kp=120 (optimal from sweep), kd=2*sqrt(kp) (critically damped),
    #   Kp=1.8 outer overshoot (in process_action), dq_max margin=50x
    dq_max = None
    if cfg.use_joint_pos:
        try:
            ctrl = env.env.robots[0].controller
            # Patch controller output_max to allow the arm to track model targets.
            # Default robosuite value is 0.05 rad/step which is far too small.
            dq_override = np.ones(7, dtype=np.float32) * cfg.joint_dq_max
            dq_override[3] *= 2.0  # extra headroom for joint 4 (matches replay script)
            ctrl.output_max = dq_override
            ctrl.output_min = -dq_override
            # Patch kp/kd to match replay controller (A_libero_joint_replay.py).
            # kp=120 found optimal via sweep (kp=500 REGRESSED 6/10→2/10).
            # Critically damped: kd = 2*sqrt(kp), damping_ratio=1.0.
            ctrl.kp = np.ones(7) * cfg.joint_kp
            ctrl.kd = 2.0 * np.sqrt(ctrl.kp) * cfg.joint_damping_ratio
            dq_max = np.array(ctrl.output_max, dtype=np.float32).reshape(-1)[:7]
            dq_max = np.maximum(dq_max, 1e-6)
            logger.info(f"[JOINT_POS] Patched controller: output_max={dq_max}, kp={ctrl.kp[0]:.0f}, kd={ctrl.kd[0]:.2f}, damping={cfg.joint_damping_ratio}, Kp_overshoot={cfg.joint_Kp_overshoot}, substeps={cfg.joint_substeps}")
        except Exception as e:
            dq_max = np.ones(7, dtype=np.float32) * cfg.joint_dq_max
            logger.warning(f"[JOINT_POS] Could not patch controller ({e}), using dq_max={cfg.joint_dq_max}")

    # --- Object tracking for episode record ---
    initial_object_positions = _snapshot_object_positions(obs)
    min_eef_distances = {}
    initial_eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64).copy()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []             # agentview images for video
    frontview_images = []          # frontview images for video
    rollout_joint_pos = []       # (T, 7) actual joint positions
    rollout_gripper_qpos = []    # (T, 2) gripper finger qpos
    rollout_ee_pos = []          # (T, 3) end-effector position
    rollout_ee_quat = []         # (T, 4) end-effector quaternion
    rollout_actions_raw = []     # (T, 8 or 7) raw model output actions
    rollout_actions_env = []     # (T, 8 or 7) actions sent to env
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    n_sub = cfg.joint_substeps if cfg.use_joint_pos else 1

    # For sub-stepping: get the robosuite base env for low-level physics stepping
    # without camera rendering on intermediate sub-steps.
    _robo_env = None
    _robo_robot = None
    if n_sub > 1:
        _robo_env = env.env  # unwrap LIBERO wrapper → robosuite env (has .sim)
        _robo_robot = _robo_env.robots[0]
        # Auto-compute n_sub from physics params so that total physics steps per
        # model action exactly match baseline (sim_freq / base_control_freq = 25
        # for 500 Hz / 20 Hz).  This preserves the 0.05 s inter-observation time
        # the model was trained on, avoiding distribution shift.
        sim_freq = int(round(1.0 / _robo_env.model_timestep))  # 500
        base_control_freq = 20  # original LIBERO controller rate
        n_sub = sim_freq // base_control_freq  # 25
        print(f"[Sub-step] auto n_sub={n_sub} (sim_freq={sim_freq}, base_cf={base_control_freq})")

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                dummy = get_libero_dummy_action(cfg.model_family, cfg.use_joint_pos)
                if n_sub > 1:
                    # Run n_sub-1 physics steps manually (no camera rendering),
                    # then 1 full env.step().  Keeps total settling time = n_sub * dt.
                    n_physics_wait = int(round(_robo_env.control_timestep / _robo_env.model_timestep))
                    for _ in range(n_sub - 1):
                        policy_step = True
                        for _ in range(n_physics_wait):
                            _robo_env.sim.forward()
                            _robo_env._pre_action(dummy, policy_step)
                            _robo_env.sim.step()
                            policy_step = False
                obs, reward, done, info = env.step(dummy)
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size, cfg.use_joint_pos)
            replay_images.append(img)
            # Update min EEF-to-object distances for episode record
            _update_min_eef_distances(obs, min_eef_distances)
            # Collect frontview for separate video
            try:
                front_img = get_libero_frontview_image(obs)
                frontview_images.append(front_img)
            except KeyError:
                pass  # frontview not available in this env

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            raw_action = action_queue.popleft()
            rollout_actions_raw.append(np.array(raw_action, dtype=np.float32))

            # Record joint data BEFORE action execution
            rollout_joint_pos.append(np.array(obs["robot0_joint_pos"], dtype=np.float32))
            rollout_gripper_qpos.append(np.array(obs["robot0_gripper_qpos"], dtype=np.float32))
            rollout_ee_pos.append(np.array(obs["robot0_eef_pos"], dtype=np.float32))
            rollout_ee_quat.append(np.array(obs["robot0_eef_quat"], dtype=np.float32))

            # Execute action with sub-step interpolation for JOINT_POSITION
            if cfg.use_joint_pos and n_sub > 1:
                # Sub-step interpolation: run n_sub steps per model action.
                # Intermediate sub-steps use low-level physics stepping (no camera
                # rendering) for speed.  Only the final sub-step calls env.step()
                # to get full observations.
                q_cur = np.array(obs["robot0_joint_pos"], dtype=np.float32)
                q_target = np.array(raw_action[:7], dtype=np.float32)
                grip_cmd = float(np.sign(raw_action[7]))

                n_physics = int(round(_robo_env.control_timestep / _robo_env.model_timestep))

                for s_idx in range(n_sub):
                    alpha = (s_idx + 1.0) / n_sub
                    q_sub = q_cur * (1.0 - alpha) + q_target * alpha
                    # Read current joint pos directly from sim data (fast, no rendering)
                    q_now = np.array(_robo_robot._joint_positions, dtype=np.float32)
                    dq_des = q_sub - q_now
                    u = np.clip(cfg.joint_Kp_overshoot * dq_des / dq_max, -1.0, 1.0).astype(np.float32)
                    sub_action = np.zeros(8, dtype=np.float32)
                    sub_action[:7] = u
                    sub_action[7] = grip_cmd

                    if s_idx < n_sub - 1:
                        # Intermediate sub-step: step physics without camera rendering.
                        # Replicates robosuite base.step() inner loop minus _update_observables().
                        # Do NOT increment _robo_env.timestep or cur_time here — those must
                        # only advance via the final env.step() to avoid tripping the horizon
                        # limit (which would set self.done=True and crash subsequent steps).
                        policy_step = True
                        for _ in range(n_physics):
                            _robo_env.sim.forward()
                            _robo_env._pre_action(sub_action, policy_step)
                            _robo_env.sim.step()
                            policy_step = False
                    else:
                        # Final sub-step: full env.step() with camera rendering & obs
                        obs, reward, done, info = env.step(sub_action.tolist())
                        if done:
                            success = True

                rollout_actions_env.append(np.array(sub_action, dtype=np.float32))
                if success:
                    break
            else:
                # Standard single-step execution
                action = process_action(raw_action, cfg.model_family, cfg.use_joint_pos, obs=obs, dq_max=dq_max,
                                        Kp_overshoot=cfg.joint_Kp_overshoot)
                rollout_actions_env.append(np.array(action, dtype=np.float32))
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    success = True
                    break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    # --- Compute episode record (quantitative per-episode metrics) ---
    final_object_positions = _snapshot_object_positions(obs)
    final_eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64) if "robot0_eef_pos" in obs else None
    ee_arr = np.array(rollout_ee_pos) if rollout_ee_pos else np.empty((0, 3))
    total_eef_disp = float(np.sqrt(np.sum(np.diff(ee_arr, axis=0)**2, axis=1)).sum()) if len(ee_arr) > 1 else 0.0
    episode_record = compute_episode_record(
        task_description=task_description,
        success=success,
        total_steps=len(rollout_joint_pos),
        max_steps=max_steps,
        initial_object_positions=initial_object_positions,
        final_object_positions=final_object_positions,
        min_eef_distances=min_eef_distances,
        initial_eef_pos=initial_eef_pos,
        final_eef_pos=final_eef_pos,
        total_eef_displacement=total_eef_disp,
    )

    # Package rollout joint data
    joint_pos_arr = np.array(rollout_joint_pos) if rollout_joint_pos else np.empty((0, 7))
    rollout_data = {
        # Compatible with A_human_likeness_evaluate.py --sim_npz
        "joint_states_sim": joint_pos_arr,             # (T, 7) alias for human-likeness eval
        "task_name": np.array(task_description.replace(" ", "_")),
        "task_success": np.array(success),
        "task_success_step": np.array(len(rollout_joint_pos) - 1 if success else -1),
        # Full rollout data
        "joint_pos": joint_pos_arr,                    # (T, 7) actual joint positions
        "gripper_qpos": np.array(rollout_gripper_qpos) if rollout_gripper_qpos else np.empty((0, 2)),
        "ee_pos": np.array(rollout_ee_pos) if rollout_ee_pos else np.empty((0, 3)),
        "ee_quat": np.array(rollout_ee_quat) if rollout_ee_quat else np.empty((0, 4)),
        "actions_raw": np.array(rollout_actions_raw) if rollout_actions_raw else np.empty((0, 8)),  # model output (absolute joint targets + gripper)
        "actions_env": np.array(rollout_actions_env) if rollout_actions_env else np.empty((0, 8)),  # normalized delta sent to env
        "model_joint_targets": np.array(rollout_actions_raw)[:, :7] if rollout_actions_raw else np.empty((0, 7)),  # model's intended joint positions
        "success": np.array(success),
        "task_description": np.array(task_description),
    }

    return success, replay_images, frontview_images, rollout_data, episode_record


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    rollout_dir=None,
    task_results=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res,
                                           use_joint_pos=cfg.use_joint_pos,
                                           joint_substeps=cfg.joint_substeps)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, frontview_images, rollout_data, episode_record = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video and joint data
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file, rollout_dir=rollout_dir,
            frontview_images=frontview_images,
        )
        save_rollout_data(
            rollout_data, total_episodes, success=success, task_description=task_description, log_file=log_file, rollout_dir=rollout_dir
        )
        save_episode_record(
            episode_record, total_episodes, success=success, task_description=task_description, log_file=log_file, rollout_dir=rollout_dir
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Collect per-task results for aggregate
    if task_results is not None:
        task_results[task_description] = {
            "episodes": task_episodes,
            "successes": task_successes,
        }

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


def build_rollout_dir(cfg: GenerateConfig) -> str:
    """Build a descriptive rollout directory name from config."""
    # Extract checkpoint name from path (last component)
    checkpoint_path = str(cfg.pretrained_checkpoint).rstrip("/")
    checkpoint_name = os.path.basename(checkpoint_path)
    # Truncate if too long (keep first 120 chars)
    if len(checkpoint_name) > 120:
        checkpoint_name = checkpoint_name[:120]
    # Build directory: ./rollouts/<suite>/<checkpoint>/<date_time>/
    rollout_dir = os.path.join(
        "./rollouts",
        cfg.task_suite_name,
        checkpoint_name,
        DATE_TIME,
    )
    return rollout_dir


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Build rollout directory
    rollout_dir = build_rollout_dir(cfg)
    os.makedirs(rollout_dir, exist_ok=True)
    log_message(f"Rollout directory: {rollout_dir}", log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = {}
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            rollout_dir=rollout_dir,
            task_results=task_results,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Save results summary to rollout directory
    results_path = os.path.join(rollout_dir, "results.txt")
    with open(results_path, "w") as rf:
        rf.write(f"Task suite: {cfg.task_suite_name}\n")
        rf.write(f"Checkpoint: {cfg.pretrained_checkpoint}\n")
        rf.write(f"Model family: {cfg.model_family}\n")
        rf.write(f"Date: {DATE_TIME}\n")
        rf.write(f"Seed: {cfg.seed}\n")
        rf.write(f"Num trials per task: {cfg.num_trials_per_task}\n")
        rf.write(f"Num open loop steps: {cfg.num_open_loop_steps}\n")
        rf.write(f"Use joint pos: {cfg.use_joint_pos}\n")
        if cfg.use_joint_pos:
            rf.write(f"Joint kp: {cfg.joint_kp}\n")
            rf.write(f"Joint damping ratio: {cfg.joint_damping_ratio}\n")
            rf.write(f"Joint dq max: {cfg.joint_dq_max}\n")
            rf.write(f"Joint Kp overshoot: {cfg.joint_Kp_overshoot}\n")
            rf.write(f"Joint substeps: {cfg.joint_substeps}\n")
        rf.write(f"Env image resolution: {cfg.env_img_res}\n")
        rf.write(f"\nTotal episodes: {total_episodes}\n")
        rf.write(f"Total successes: {total_successes}\n")
        rf.write(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)\n")
    log_message(f"Saved results summary at: {results_path}", log_file)

    # Save per-task aggregate results
    save_aggregate_results(task_results, rollout_dir, cfg)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
