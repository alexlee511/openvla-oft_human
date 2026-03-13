"""
Packages humanized_sim.npz files into LIBERO-compatible HDF5 format.

All heavy lifting (simulation, 256x256 rendering, EEF delta action computation)
is done by A_libero_joint_replay.py --collect_obs, which saves everything to
humanized_sim.npz. This script simply reads those NPZ files and writes them
into clean HDF5 files for the LIBERO -> RLDS pipeline.

NPZ format (humanized_sim.npz, per demo — produced by --collect_obs):
    Metadata:
      - task_name               str
      - demo_idx                int
      - task_success            bool
      - task_success_step       int (-1 if not successful)

    Observations (all from the replayed simulation):
      - agentview_rgb           (T, 256, 256, 3) uint8
      - eye_in_hand_rgb         (T, 256, 256, 3) uint8
      - ee_pos                  (T, 3) float32    EEF position
      - ee_quat                 (T, 4) float32    EEF quaternion (x,y,z,w)
      - ee_states               (T, 6) float32    [pos, axisangle]
      - joint_states_obs        (T, 7) float32    joint positions from obs
      - gripper_qpos            (T, 2) float32    gripper finger qpos
      - proprio_state           (T, 8) float32    [joint_pos(7), gripper_width(1)]
      - robot_states            (T, 9) float32    [gripper_qpos, ee_pos, ee_quat]
      - sim_states              (T, ?) float32    full MuJoCo sim state

    Actions:
      - actions                 (T, 8) float32    Absolute joint pos (7) + gripper (1)
      - actions_eef_delta       (T, 7) float32    EEF delta [dx,dy,dz,drx,dry,drz,gripper] (for reference)
      - gripper_commands        (T,)   float32    raw gripper commands (-1/+1)
      - gripper_01              (T,)   float32    gripper mapped to [0=close, 1=open]

    Reference trajectories:
      - joint_states_demo       (T, 7) original demo joint angles
      - joint_states_human      (T, 7) humanized joint angles
      - joint_states_sim        (T, 7) actual simulated joint angles
      - gripper_states_sim      (T, 2) actual simulated gripper states
      - gripper_commands_human  (T,)   humanized gripper commands

Pipeline:
    1. Run A_libero_joint_replay.py --collect_obs for each demo
       → produces humanized_sim.npz with all observation + action data

    2. Run this script to package NPZ → HDF5
       → optionally filters no-ops, checks success

    3. Use rlds_dataset_builder/ to convert HDF5 → RLDS TFRecord

Usage (single task):
    python experiments/robot/libero/A_npz_to_hdf5.py \\
        --task_root /home/vsp1323/LIBERO/scripts/libero_10_humanized/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo \\
        --output_dir ./LIBERO/libero/datasets/libero_10_humanized_noops \\
        --filter_noops --require_success

Usage (all tasks in a directory):
    python experiments/robot/libero/A_npz_to_hdf5.py \\
        --task_roots_dir /home/vsp1323/LIBERO/scripts/libero_10_humanized \\
        --output_dir ./LIBERO/libero/datasets/libero_10_humanized_noops \\
        --filter_noops --require_success
"""

import argparse
import os

import h5py
import numpy as np
import tqdm


# =====================================================================
# Utilities
# =====================================================================

def is_noop(action, prev_action=None, threshold=1e-4):
    """Same no-op filter as regenerate_libero_dataset.py."""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


# =====================================================================
# Data discovery
# =====================================================================

def discover_demos_from_task_root(task_root, require_success=False):
    """
    Given a task root like:
        /home/vsp1323/LIBERO/scripts/KITCHEN_SCENE3_..._demo/
    Find all humanized_sim.npz files in:
        <task_root>/humanized_demo/demo_XX/humanized_sim.npz
    Returns list of dicts with sim_npz_path, demo_idx, task_name, success.
    """
    humanized_dir = os.path.join(task_root, "humanized_demo")
    if not os.path.isdir(humanized_dir):
        return []

    demos = []
    for demo_folder in sorted(os.listdir(humanized_dir)):
        demo_path = os.path.join(humanized_dir, demo_folder)
        sim_npz_path = os.path.join(demo_path, "humanized_sim.npz")
        if not os.path.isfile(sim_npz_path):
            continue

        sim_data = np.load(sim_npz_path, allow_pickle=True)

        # Check that --collect_obs data is present
        if "agentview_rgb" not in sim_data:
            print(f"  WARNING: {sim_npz_path} missing observation data. "
                  f"Re-run A_libero_joint_replay.py with --collect_obs. Skipping.")
            continue

        demo_idx = int(sim_data["demo_idx"])
        task_name = str(sim_data["task_name"])
        success = bool(sim_data["task_success"])

        if require_success and not success:
            continue

        demos.append({
            "sim_npz_path": sim_npz_path,
            "demo_idx": demo_idx,
            "task_name": task_name,
            "success": success,
        })
    return demos


def discover_all_task_roots(task_roots_dir):
    """
    Scan a directory for folders that contain humanized_demo/ subdirectories.
    Returns dict: task_name -> task_root_path.
    """
    task_roots = {}
    for entry in sorted(os.listdir(task_roots_dir)):
        full_path = os.path.join(task_roots_dir, entry)
        if os.path.isdir(full_path) and os.path.isdir(os.path.join(full_path, "humanized_demo")):
            task_name = entry
            if task_name.endswith("_demo"):
                task_name = task_name[:-len("_demo")]
            task_roots[task_name] = full_path
    return task_roots


# =====================================================================
# Main conversion
# =====================================================================

def process_task(task_name, task_root, output_dir, filter_noops, require_success):
    """Read humanized_sim.npz files for one task and write to HDF5."""

    demos = discover_demos_from_task_root(task_root, require_success=require_success)
    if not demos:
        print(f"  No valid humanized_sim.npz files found in {task_root}/humanized_demo/. Skipping.")
        return {"success": 0, "failed": 0, "skipped_no_obs": 0, "noops_filtered": 0}

    print(f"  Found {len(demos)} demos for task: {task_name}")

    # Create output HDF5
    os.makedirs(output_dir, exist_ok=True)
    out_hdf5_path = os.path.join(output_dir, f"{task_name}_demo.hdf5")
    out_file = h5py.File(out_hdf5_path, "w")
    out_grp = out_file.create_group("data")

    stats = {"success": 0, "failed": 0, "skipped_no_obs": 0, "noops_filtered": 0}

    for demo_info in tqdm.tqdm(demos, desc=f"    Packaging {task_name}", leave=False):
        sim_npz_path = demo_info["sim_npz_path"]
        demo_idx = demo_info["demo_idx"]
        demo_key = f"demo_{demo_idx}"

        try:
            sim_data = np.load(sim_npz_path, allow_pickle=True)
        except Exception as e:
            print(f"    WARNING: Failed to load {sim_npz_path}: {e}. Skipping.")
            stats["failed"] += 1
            continue

        # ---- Extract all data from NPZ ----
        actions = sim_data["actions"].astype(np.float32)           # (T, 8) joint pos + gripper
        agentview_rgb = sim_data["agentview_rgb"]                  # (T, 256, 256, 3) uint8
        eye_in_hand_rgb = sim_data["eye_in_hand_rgb"]              # (T, 256, 256, 3) uint8
        joint_states = sim_data["joint_states_obs"].astype(np.float32)  # (T, 7)
        gripper_states = sim_data["gripper_qpos"].astype(np.float32)    # (T, 2)
        # Proprioceptive state: [joint_pos(7), gripper_width(1)] = 8D
        if "proprio_state" in sim_data:
            proprio_state = sim_data["proprio_state"].astype(np.float32)  # (T, 8)
        else:
            # Fallback: construct from joint_states and gripper
            gripper_width = np.mean(gripper_states, axis=1, keepdims=True).astype(np.float32)
            proprio_state = np.hstack([joint_states, gripper_width]).astype(np.float32)
        # Keep ee_states for reference (not primary state for joint pos control)
        ee_states = sim_data["ee_states"].astype(np.float32) if "ee_states" in sim_data else None
        robot_states = sim_data["robot_states"].astype(np.float32) if "robot_states" in sim_data else None
        sim_states = sim_data["sim_states"].astype(np.float32) if "sim_states" in sim_data else None

        T_steps = actions.shape[0]

        # ---- Filter no-op actions ----
        if filter_noops:
            filtered_idx = []
            for t in range(T_steps):
                prev = actions[filtered_idx[-1]] if filtered_idx else None
                if not is_noop(actions[t], prev):
                    filtered_idx.append(t)
                else:
                    stats["noops_filtered"] += 1

            if len(filtered_idx) == 0:
                print(f"    WARNING: All no-ops for {demo_key}. Skipping.")
                stats["failed"] += 1
                continue
            filtered_idx = np.array(filtered_idx)
        else:
            filtered_idx = np.arange(T_steps)

        # ---- Write to HDF5 ----
        n = len(filtered_idx)

        ep_grp = out_grp.create_group(demo_key)
        obs_grp = ep_grp.create_group("obs")

        obs_grp.create_dataset("gripper_states", data=gripper_states[filtered_idx])
        obs_grp.create_dataset("joint_states", data=joint_states[filtered_idx])
        # State for RLDS: [joint_pos(7), gripper_width(1)] = 8D proprioceptive state
        obs_grp.create_dataset("state", data=proprio_state[filtered_idx])
        if ee_states is not None:
            obs_grp.create_dataset("ee_states", data=ee_states[filtered_idx])
            obs_grp.create_dataset("ee_pos", data=ee_states[filtered_idx, :3])
            obs_grp.create_dataset("ee_ori", data=ee_states[filtered_idx, 3:])
        obs_grp.create_dataset("agentview_rgb", data=agentview_rgb[filtered_idx])
        obs_grp.create_dataset("eye_in_hand_rgb", data=eye_in_hand_rgb[filtered_idx])
        ep_grp.create_dataset("actions", data=actions[filtered_idx])  # (n, 8) joint pos + gripper
        if sim_states is not None:
            ep_grp.create_dataset("states", data=sim_states[filtered_idx])
        if robot_states is not None:
            ep_grp.create_dataset("robot_states", data=robot_states[filtered_idx])

        dones = np.zeros(n, dtype=np.uint8)
        dones[-1] = 1
        rewards = np.zeros(n, dtype=np.uint8)
        rewards[-1] = 1
        ep_grp.create_dataset("rewards", data=rewards)
        ep_grp.create_dataset("dones", data=dones)

        stats["success"] += 1
        print(f"    {demo_key}: {n} steps written (from {T_steps} total)")

    out_file.close()

    print(f"  Saved: {out_hdf5_path}")
    print(f"  Stats: {stats}")
    return stats


def main(args):
    print(f"{'='*60}")
    print(f"NPZ -> HDF5 Packaging for LIBERO")
    print(f"{'='*60}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"  Filter no-ops:      {args.filter_noops}")
    print(f"  Require success:    {args.require_success}")

    # Discover task roots
    task_roots = {}
    if args.task_root:
        task_name = os.path.basename(args.task_root.rstrip("/"))
        if task_name.endswith("_demo"):
            task_name = task_name[:-len("_demo")]
        task_roots[task_name] = args.task_root
    elif args.task_roots_dir:
        task_roots = discover_all_task_roots(args.task_roots_dir)
    else:
        print("ERROR: Must specify --task_root or --task_roots_dir")
        return

    print(f"  Found {len(task_roots)} task(s): {list(task_roots.keys())}")

    total_stats = {"success": 0, "failed": 0, "skipped_no_obs": 0, "noops_filtered": 0}

    for task_name, task_root in task_roots.items():
        print(f"\n--- Processing: {task_name} ---")
        stats = process_task(
            task_name=task_name,
            task_root=task_root,
            output_dir=args.output_dir,
            filter_noops=args.filter_noops,
            require_success=args.require_success,
        )
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    print(f"\n{'='*60}")
    print(f"Packaging complete!")
    print(f"  Episodes saved:            {total_stats['success']}")
    print(f"  Episodes failed:           {total_stats['failed']}")
    print(f"  Skipped (no obs data):     {total_stats['skipped_no_obs']}")
    print(f"  No-ops filtered:           {total_stats['noops_filtered']}")
    print(f"  Output directory:          {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package humanized_sim.npz files (from A_libero_joint_replay.py --collect_obs) "
                    "into LIBERO-compatible HDF5 format."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--task_root", type=str, default=None,
        help="Single task root folder, e.g. "
             "/home/vsp1323/LIBERO/scripts/KITCHEN_SCENE3_..._demo",
    )
    source.add_argument(
        "--task_roots_dir", type=str, default=None,
        help="Directory of task root folders, e.g. /home/vsp1323/LIBERO/scripts",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for HDF5 files.",
    )
    parser.add_argument(
        "--filter_noops", action="store_true", default=True,
        help="Filter out no-op actions.",
    )
    parser.add_argument(
        "--require_success", action="store_true", default=False,
        help="Only include demos where task_success=True.",
    )

    args = parser.parse_args()
    main(args)
