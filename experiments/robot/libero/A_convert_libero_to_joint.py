"""
Convert original LIBERO HDF5 demos (EEF delta actions) → joint-position-action HDF5.

Reads the original LIBERO HDF5 files (128×128 images, 7D EEF delta actions) and
produces new HDF5 files with:
  - 256×256 images (bilinear upscale, or native from replay when --require_success)
  - 8D joint position actions: [joint_pos_target(7), gripper_command(1)]
      action[t] = concat(joint_states[t+1], gripper(t))  for t < T-1
      action[T-1] = concat(joint_states[T-1], gripper(T-1))  (hold last position)
  - 8D proprioceptive state: [joint_pos(7), gripper_width(1)]
  - No-op filtering
  - Optional replay-based success verification (--require_success)

When --require_success is set, each demo's derived joint actions are replayed in
simulation using the JOINT_POSITION controller to verify task completion.
Only successful demos are saved, using replay observations (native 256×256).

Pipeline:
    1. Run this script to convert original HDF5 → joint-action HDF5
    2. Use rlds_dataset_builder/LIBERO_10_joint/ to convert HDF5 → RLDS TFRecord
    3. Train with the new RLDS dataset (e.g. --dataset_name libero_10_joint_no_noops)

Usage (without replay verification):
    python experiments/robot/libero/A_convert_libero_to_joint.py \
        --libero_raw_data_dir /home/vsp1323/alex/LIBERO/libero/datasets/libero_10 \
        --libero_target_dir ./LIBERO/libero/datasets/libero_10_joint_no_noops \
        --filter_noops

Usage (with replay verification):
    python experiments/robot/libero/A_convert_libero_to_joint.py \
        --libero_raw_data_dir /home/vsp1323/alex/LIBERO/libero/datasets/libero_10 \
        --libero_target_dir ./LIBERO/libero/datasets/libero_10_joint_no_noops \
        --filter_noops --require_success --suite_name libero_10
"""

import argparse
import contextlib
import io
import os

import cv2
import h5py
import numpy as np
import tqdm


IMAGE_RESOLUTION = 256  # Target resolution (original LIBERO is 128×128)


def resize_image(img, target_size=IMAGE_RESOLUTION):
    """Resize image to target_size × target_size using bilinear interpolation."""
    if img.shape[0] == target_size and img.shape[1] == target_size:
        return img
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether a joint-position action is a no-op.

    For joint position actions, a no-op means the joint position target barely changed
    from the previous step (joints ~unchanged) and the gripper command didn't change.
    """
    if prev_action is None:
        # First step: check if joint changes are near zero
        return np.linalg.norm(action[:-1]) < threshold

    # Check if joint position delta from previous action is near-zero AND gripper unchanged
    joint_delta = np.linalg.norm(action[:-1] - prev_action[:-1])
    gripper_unchanged = action[-1] == prev_action[-1]
    return joint_delta < threshold and gripper_unchanged


# =====================================================================
# Replay verification helpers (only used when --require_success)
# =====================================================================

def create_env_for_task(task, resolution=256):
    """Create a JOINT_POSITION OffScreenRenderEnv for the given LIBERO task."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "controller": "JOINT_POSITION",
    }
    with contextlib.redirect_stdout(io.StringIO()):
        env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def patch_controller(env, kp, dq_max_val, damping_ratio):
    """Patch PD controller gains and return the effective dq_max array."""
    ctrl = env.env.robots[0].controller
    ctrl.kp = np.ones(7) * kp
    ctrl.kd = 2.0 * np.sqrt(ctrl.kp) * damping_ratio
    dq_override = np.ones(7, dtype=np.float32) * dq_max_val
    dq_override[3] *= 2.0  # extra headroom for joint 4
    ctrl.output_max = dq_override
    ctrl.output_min = -dq_override
    dq_max = np.maximum(np.array(ctrl.output_max, dtype=np.float32).reshape(-1)[:7], 1e-6)
    return dq_max


def _quat_to_axisangle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle (3,)."""
    q = np.array(quat, dtype=np.float64)
    sin_half = np.linalg.norm(q[:3])
    if sin_half < 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, q[3])
    axis = q[:3] / sin_half
    return (axis * angle).astype(np.float32)


def replay_demo_in_sim(env, init_state, joint_actions, Kp_overshoot, dq_max):
    """Replay joint position actions in simulation and check task success.

    Args:
        env: LIBERO OffScreenRenderEnv with JOINT_POSITION controller.
        init_state: Flattened MuJoCo state to initialize the episode.
        joint_actions: (T, 8) absolute joint targets + gripper commands.
        Kp_overshoot: Outer-loop overshoot factor for PD command.
        dq_max: (7,) per-joint output_max from controller.

    Returns:
        success: bool
        success_step: int (-1 if not successful)
        replay_data: dict with observations collected during replay.
    """
    obs = env.set_init_state(init_state)

    T = joint_actions.shape[0]
    task_success = False
    task_success_step = -1

    agentview_list = []
    wrist_list = []
    joint_states_list = []
    gripper_qpos_list = []
    ee_states_list = []

    for t in range(T):
        # Collect observation BEFORE action
        agentview_list.append(obs["agentview_image"].copy())
        wrist_list.append(obs["robot0_eye_in_hand_image"].copy())
        q_cur = np.array(obs["robot0_joint_pos"], dtype=np.float32)
        joint_states_list.append(q_cur)
        gripper_qpos_list.append(np.array(obs["robot0_gripper_qpos"], dtype=np.float32))
        ee_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
        ee_axisangle = _quat_to_axisangle(obs["robot0_eef_quat"])
        ee_states_list.append(np.concatenate([ee_pos, ee_axisangle]))

        # Compute normalized PD action: u = clip(Kp * (q_target - q_cur) / dq_max, -1, 1)
        q_target = joint_actions[t, :7].astype(np.float32)
        grip_cmd = float(joint_actions[t, 7])
        u = np.clip(Kp_overshoot * (q_target - q_cur) / dq_max, -1.0, 1.0)
        action = np.zeros(8, dtype=np.float32)
        action[:7] = u
        action[7] = grip_cmd

        obs, rew, done, info = env.step(action.tolist())

        if done and not task_success:
            task_success = True
            task_success_step = t

    replay_data = {
        "agentview_rgb": np.stack(agentview_list),       # (T, 256, 256, 3)
        "eye_in_hand_rgb": np.stack(wrist_list),          # (T, 256, 256, 3)
        "joint_states": np.stack(joint_states_list),      # (T, 7)
        "gripper_states": np.stack(gripper_qpos_list),    # (T, 2)
        "ee_states": np.stack(ee_states_list),            # (T, 6) [pos(3), axisangle(3)]
    }
    return task_success, task_success_step, replay_data


def derive_joint_actions(orig_actions, joint_states):
    """Derive absolute joint position actions from demo data.

    action[t] = (joint_states[t+1], gripper_command[t]) for t < T-1
    action[T-1] = (joint_states[T-1], gripper_command[T-1])
    """
    T = orig_actions.shape[0]
    joint_actions = np.zeros((T, 8), dtype=np.float32)
    for t in range(T - 1):
        joint_actions[t, :7] = joint_states[t + 1]
        joint_actions[t, 7] = orig_actions[t, -1]
    joint_actions[-1, :7] = joint_states[-1]
    joint_actions[-1, 7] = orig_actions[-1, -1]
    return joint_actions


def convert_task(raw_hdf5_path, output_hdf5_path, filter_noops=True,
                 env=None, Kp_overshoot=None, dq_max=None):
    """Convert one task's HDF5 from EEF delta → joint position actions.

    If env is provided, replays each demo in simulation to verify success
    and uses replay observations (native 256×256).
    Otherwise, uses original observations (upscaled 128→256).
    """
    use_replay = env is not None

    raw_file = h5py.File(raw_hdf5_path, "r")
    raw_data = raw_file["data"]

    os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
    out_file = h5py.File(output_hdf5_path, "w")
    out_grp = out_file.create_group("data")

    stats = {"demos_written": 0, "demos_total": 0, "noops_filtered": 0, "demos_skipped_fail": 0}

    demo_keys = sorted(raw_data.keys(), key=lambda k: int(k.split("_")[1]))
    stats["demos_total"] = len(demo_keys)

    for demo_key in tqdm.tqdm(demo_keys, desc=f"  Converting {os.path.basename(raw_hdf5_path)}", leave=False):
        demo = raw_data[demo_key]

        # Read original data
        orig_actions = demo["actions"][()]              # (T, 7) EEF delta
        joint_states_demo = demo["obs"]["joint_states"][()]  # (T, 7)
        T_steps = orig_actions.shape[0]

        # Derive joint actions from original demo
        joint_actions = derive_joint_actions(orig_actions, joint_states_demo)

        if use_replay:
            # ---- Replay in simulation to verify success ----
            init_state = demo["states"][0]
            success, success_step, replay_data = replay_demo_in_sim(
                env, init_state, joint_actions, Kp_overshoot, dq_max
            )
            if not success:
                stats["demos_skipped_fail"] += 1
                print(f"    {demo_key}: FAILED (skipped)")
                continue
            print(f"    {demo_key}: SUCCESS at step {success_step}/{T_steps}")

            # Use replay observations (native 256×256)
            agentview_rgb = replay_data["agentview_rgb"]
            eye_in_hand_rgb = replay_data["eye_in_hand_rgb"]
            joint_states = replay_data["joint_states"]
            gripper_states = replay_data["gripper_states"]
            ee_states = replay_data["ee_states"]

            # Re-derive actions from replay trajectory for self-consistency:
            # action[t] = [replay_joint_states[t+1], gripper_cmd[t]]
            joint_actions_out = np.zeros((T_steps, 8), dtype=np.float32)
            for t in range(T_steps - 1):
                joint_actions_out[t, :7] = joint_states[t + 1]
                joint_actions_out[t, 7] = joint_actions[t, 7]
            joint_actions_out[-1, :7] = joint_states[-1]
            joint_actions_out[-1, 7] = joint_actions[-1, 7]

            # Compute proprio from replay
            gripper_width = np.mean(gripper_states, axis=1, keepdims=True).astype(np.float32)
            proprio_state = np.concatenate([joint_states, gripper_width], axis=-1).astype(np.float32)
        else:
            # ---- No replay: use original observations (upscaled) ----
            joint_states = joint_states_demo
            gripper_states = demo["obs"]["gripper_states"][()]
            agentview_rgb = demo["obs"]["agentview_rgb"][()]
            eye_in_hand_rgb = demo["obs"]["eye_in_hand_rgb"][()] if "eye_in_hand_rgb" in demo["obs"] else None
            ee_states = demo["obs"]["ee_states"][()] if "ee_states" in demo["obs"] else None

            joint_actions_out = joint_actions
            gripper_width = np.mean(gripper_states, axis=1, keepdims=True).astype(np.float32)
            proprio_state = np.concatenate([joint_states, gripper_width], axis=-1).astype(np.float32)

        # ---- Filter no-ops ----
        if filter_noops:
            filtered_idx = []
            T_out = joint_actions_out.shape[0]
            for t in range(T_out):
                prev = joint_actions_out[filtered_idx[-1]] if filtered_idx else None
                if not is_noop(joint_actions_out[t], prev):
                    filtered_idx.append(t)
                else:
                    stats["noops_filtered"] += 1

            if len(filtered_idx) == 0:
                print(f"    WARNING: All no-ops for {demo_key}. Skipping.")
                continue
            filtered_idx = np.array(filtered_idx)
        else:
            filtered_idx = np.arange(joint_actions_out.shape[0])

        n = len(filtered_idx)

        # ---- Prepare images (replay: already 256×256, original: upscale 128→256) ----
        if use_replay:
            out_agentview = agentview_rgb[filtered_idx]
            out_wrist = eye_in_hand_rgb[filtered_idx]
        else:
            out_agentview = np.stack([resize_image(agentview_rgb[i]) for i in filtered_idx])
            out_wrist = None
            if eye_in_hand_rgb is not None:
                out_wrist = np.stack([resize_image(eye_in_hand_rgb[i]) for i in filtered_idx])

        # ---- Write to HDF5 ----
        ep_grp = out_grp.create_group(demo_key)
        obs_grp = ep_grp.create_group("obs")

        obs_grp.create_dataset("agentview_rgb", data=out_agentview)
        if out_wrist is not None:
            obs_grp.create_dataset("eye_in_hand_rgb", data=out_wrist)
        obs_grp.create_dataset("joint_states", data=joint_states[filtered_idx])
        obs_grp.create_dataset("gripper_states", data=gripper_states[filtered_idx])
        obs_grp.create_dataset("state", data=proprio_state[filtered_idx])
        if ee_states is not None:
            obs_grp.create_dataset("ee_states", data=ee_states[filtered_idx])
            obs_grp.create_dataset("ee_pos", data=ee_states[filtered_idx, :3])
            obs_grp.create_dataset("ee_ori", data=ee_states[filtered_idx, 3:])

        ep_grp.create_dataset("actions", data=joint_actions_out[filtered_idx])

        dones_out = np.zeros(n, dtype=np.uint8)
        dones_out[-1] = 1
        rewards_out = np.zeros(n, dtype=np.uint8)
        rewards_out[-1] = 1
        ep_grp.create_dataset("rewards", data=rewards_out)
        ep_grp.create_dataset("dones", data=dones_out)

        stats["demos_written"] += 1

    raw_file.close()
    out_file.close()
    return stats


def main(args):
    print(f"{'='*60}")
    print(f"Convert LIBERO HDF5: EEF delta → Joint Position Actions")
    print(f"{'='*60}")
    print(f"  Source dir:       {args.libero_raw_data_dir}")
    print(f"  Target dir:       {args.libero_target_dir}")
    print(f"  Filter no-ops:    {args.filter_noops}")
    print(f"  Require success:  {args.require_success}")
    if args.require_success:
        print(f"  Suite name:       {args.suite_name}")
        print(f"  Joint kp:         {args.joint_kp}")
        print(f"  Kp overshoot:     {args.joint_Kp_overshoot}")
        print(f"  dq_max:           {args.joint_dq_max}")
        print(f"  Damping ratio:    {args.joint_damping_ratio}")
    print()

    # Find all HDF5 files
    hdf5_files = sorted([
        f for f in os.listdir(args.libero_raw_data_dir)
        if f.endswith(".hdf5")
    ])
    if not hdf5_files:
        print(f"ERROR: No HDF5 files found in {args.libero_raw_data_dir}")
        return

    print(f"  Found {len(hdf5_files)} HDF5 file(s)")

    # Build task lookup from LIBERO benchmark (for replay verification)
    task_lookup = {}
    if args.require_success:
        from libero.libero import benchmark as libero_benchmark
        benchmark_dict = libero_benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.suite_name]()
        for i in range(task_suite.n_tasks):
            task = task_suite.get_task(i)
            task_lookup[task.name] = task
        print(f"  Loaded {len(task_lookup)} tasks from '{args.suite_name}' benchmark")

    os.makedirs(args.libero_target_dir, exist_ok=True)

    total_stats = {"demos_written": 0, "demos_total": 0, "noops_filtered": 0, "demos_skipped_fail": 0}

    for hdf5_name in hdf5_files:
        raw_path = os.path.join(args.libero_raw_data_dir, hdf5_name)
        out_path = os.path.join(args.libero_target_dir, hdf5_name)

        print(f"\n--- Converting: {hdf5_name} ---")

        # Create env for this task if replay verification is enabled
        env = None
        dq_max = None
        if args.require_success:
            task_name = os.path.splitext(hdf5_name)[0]
            if task_name.endswith("_demo"):
                task_name = task_name[:-5]
            if task_name not in task_lookup:
                print(f"  WARNING: Task '{task_name}' not found in {args.suite_name}. Skipping.")
                continue
            print(f"  Creating env for: {task_name}")
            env = create_env_for_task(task_lookup[task_name])
            env.reset()
            dq_max = patch_controller(
                env, args.joint_kp, args.joint_dq_max, args.joint_damping_ratio
            )

        stats = convert_task(
            raw_path, out_path, filter_noops=args.filter_noops,
            env=env, Kp_overshoot=args.joint_Kp_overshoot, dq_max=dq_max
        )

        if env is not None:
            env.close()

        for k in total_stats:
            total_stats[k] += stats[k]

        skip_msg = f", {stats['demos_skipped_fail']} skipped (failed)" if args.require_success else ""
        print(f"    {stats['demos_written']}/{stats['demos_total']} demos written, "
              f"{stats['noops_filtered']} no-ops filtered{skip_msg}")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Demos written:    {total_stats['demos_written']}/{total_stats['demos_total']}")
    if args.require_success:
        print(f"  Demos skipped:    {total_stats['demos_skipped_fail']} (failed)")
    print(f"  No-ops filtered:  {total_stats['noops_filtered']}")
    print(f"  Output dir:       {args.libero_target_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 demos from EEF delta actions to joint position actions."
    )
    parser.add_argument(
        "--libero_raw_data_dir", type=str, required=True,
        help="Path to directory containing original LIBERO HDF5 files. "
             "Example: /home/vsp1323/alex/LIBERO/libero/datasets/libero_10",
    )
    parser.add_argument(
        "--libero_target_dir", type=str, required=True,
        help="Path to output directory for joint-action HDF5 files. "
             "Example: ./LIBERO/libero/datasets/libero_10_joint_no_noops",
    )
    parser.add_argument(
        "--filter_noops", action="store_true", default=True,
        help="Filter out no-op actions (default: True).",
    )
    parser.add_argument(
        "--no_filter_noops", action="store_false", dest="filter_noops",
        help="Disable no-op filtering.",
    )
    parser.add_argument(
        "--require_success", action="store_true", default=False,
        help="Replay derived joint actions in simulation and only keep successful demos.",
    )
    # Controller config for replay verification (only used with --require_success)
    parser.add_argument(
        "--suite_name", type=str, default="libero_10",
        help="LIBERO benchmark suite name (required with --require_success).",
    )
    parser.add_argument(
        "--joint_kp", type=float, default=180.0,
        help="Inner PD controller kp gain (default: 180.0).",
    )
    parser.add_argument(
        "--joint_Kp_overshoot", type=float, default=3.0,
        help="Outer-loop overshoot factor for PD command (default: 3.0).",
    )
    parser.add_argument(
        "--joint_dq_max", type=float, default=2.5,
        help="Per-joint output_max for PD controller (default: 2.5).",
    )
    parser.add_argument(
        "--joint_damping_ratio", type=float, default=1.0,
        help="Damping ratio for PD controller (default: 1.0, critically damped).",
    )

    args = parser.parse_args()
    main(args)
