"""
Convert original LIBERO HDF5 demos (EEF delta actions) → joint-position-action HDF5.

Reads the original LIBERO HDF5 files (128×128 images, 7D EEF delta actions) and
produces new HDF5 files with:
  - 256×256 images (bilinear upscale from 128×128)
  - 8D joint position actions: [joint_pos_target(7), gripper_command(1)]
      action[t] = concat(joint_states[t+1], gripper(t))  for t < T-1
      action[T-1] = concat(joint_states[T-1], gripper(T-1))  (hold last position)
  - 8D proprioceptive state: [joint_pos(7), gripper_width(1)]
  - No-op filtering

This lets you train a joint-position-control model from the same demo trajectories
that were originally collected with EEF delta control.

Pipeline:
    1. Run this script to convert original HDF5 → joint-action HDF5
    2. Use rlds_dataset_builder/LIBERO_10_joint/ to convert HDF5 → RLDS TFRecord
    3. Train with the new RLDS dataset (e.g. --dataset_name libero_10_joint_noops)

Usage (single suite):
    python experiments/robot/libero/convert_libero_to_joint.py \
        --libero_raw_data_dir /home/vsp1323/alex/LIBERO/libero/datasets/libero_10 \
        --libero_target_dir ./LIBERO/libero/datasets/libero_10_joint_noops \
        --filter_noops

Usage (all four suites):
    for suite in libero_10 libero_object libero_spatial libero_goal; do
        python experiments/robot/libero/convert_libero_to_joint.py \
            --libero_raw_data_dir /home/vsp1323/alex/LIBERO/libero/datasets/${suite} \
            --libero_target_dir ./LIBERO/libero/datasets/${suite}_joint_noops \
            --filter_noops
    done
"""

import argparse
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


def convert_task(raw_hdf5_path, output_hdf5_path, filter_noops=True):
    """Convert one task's HDF5 from EEF delta → joint position actions."""

    raw_file = h5py.File(raw_hdf5_path, "r")
    raw_data = raw_file["data"]

    os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
    out_file = h5py.File(output_hdf5_path, "w")
    out_grp = out_file.create_group("data")

    stats = {"demos_written": 0, "demos_total": 0, "noops_filtered": 0}

    demo_keys = sorted(raw_data.keys(), key=lambda k: int(k.split("_")[1]))
    stats["demos_total"] = len(demo_keys)

    for demo_key in tqdm.tqdm(demo_keys, desc=f"  Converting {os.path.basename(raw_hdf5_path)}", leave=False):
        demo = raw_data[demo_key]

        # Read original data
        orig_actions = demo["actions"][()]              # (T, 7) EEF delta
        joint_states = demo["obs"]["joint_states"][()]  # (T, 7)
        gripper_states = demo["obs"]["gripper_states"][()]  # (T, 2)
        agentview_rgb = demo["obs"]["agentview_rgb"][()] # (T, H, W, 3)
        eye_in_hand_rgb = demo["obs"]["eye_in_hand_rgb"][()] if "eye_in_hand_rgb" in demo["obs"] else None
        ee_states = demo["obs"]["ee_states"][()] if "ee_states" in demo["obs"] else None

        T_steps = orig_actions.shape[0]

        # ---- Derive joint position actions ----
        # action[t] = (joint_states[t+1], gripper_command[t])
        # For the last step, hold the current joint position
        joint_actions = np.zeros((T_steps, 8), dtype=np.float32)
        for t in range(T_steps - 1):
            joint_actions[t, :7] = joint_states[t + 1]     # Target: next observed joint pos
            joint_actions[t, 7] = orig_actions[t, -1]       # Gripper command from original
        # Last step: hold current position
        joint_actions[-1, :7] = joint_states[-1]
        joint_actions[-1, 7] = orig_actions[-1, -1]

        # ---- Compute proprio state: [joint_pos(7), gripper_width(1)] ----
        gripper_width = np.mean(gripper_states, axis=1, keepdims=True).astype(np.float32)  # (T, 1)
        proprio_state = np.concatenate([joint_states, gripper_width], axis=-1).astype(np.float32)  # (T, 8)

        # ---- Filter no-ops ----
        if filter_noops:
            filtered_idx = []
            for t in range(T_steps):
                prev = joint_actions[filtered_idx[-1]] if filtered_idx else None
                if not is_noop(joint_actions[t], prev):
                    filtered_idx.append(t)
                else:
                    stats["noops_filtered"] += 1

            if len(filtered_idx) == 0:
                print(f"    WARNING: All no-ops for {demo_key} in {raw_hdf5_path}. Skipping.")
                continue
            filtered_idx = np.array(filtered_idx)
        else:
            filtered_idx = np.arange(T_steps)

        n = len(filtered_idx)

        # ---- Resize images to 256×256 ----
        resized_agentview = np.stack([resize_image(agentview_rgb[i]) for i in filtered_idx])
        resized_wrist = None
        if eye_in_hand_rgb is not None:
            resized_wrist = np.stack([resize_image(eye_in_hand_rgb[i]) for i in filtered_idx])

        # ---- Write to HDF5 ----
        ep_grp = out_grp.create_group(demo_key)
        obs_grp = ep_grp.create_group("obs")

        obs_grp.create_dataset("agentview_rgb", data=resized_agentview)
        if resized_wrist is not None:
            obs_grp.create_dataset("eye_in_hand_rgb", data=resized_wrist)
        obs_grp.create_dataset("joint_states", data=joint_states[filtered_idx])
        obs_grp.create_dataset("gripper_states", data=gripper_states[filtered_idx])
        obs_grp.create_dataset("state", data=proprio_state[filtered_idx])  # [joint(7), gripper_width(1)]
        if ee_states is not None:
            obs_grp.create_dataset("ee_states", data=ee_states[filtered_idx])
            obs_grp.create_dataset("ee_pos", data=ee_states[filtered_idx, :3])
            obs_grp.create_dataset("ee_ori", data=ee_states[filtered_idx, 3:])

        ep_grp.create_dataset("actions", data=joint_actions[filtered_idx])  # (n, 8) joint pos + gripper

        dones = np.zeros(n, dtype=np.uint8)
        dones[-1] = 1
        rewards = np.zeros(n, dtype=np.uint8)
        rewards[-1] = 1
        ep_grp.create_dataset("rewards", data=rewards)
        ep_grp.create_dataset("dones", data=dones)

        stats["demos_written"] += 1

    raw_file.close()
    out_file.close()
    return stats


def main(args):
    print(f"{'='*60}")
    print(f"Convert LIBERO HDF5: EEF delta → Joint Position Actions")
    print(f"{'='*60}")
    print(f"  Source dir:      {args.libero_raw_data_dir}")
    print(f"  Target dir:      {args.libero_target_dir}")
    print(f"  Filter no-ops:   {args.filter_noops}")
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

    os.makedirs(args.libero_target_dir, exist_ok=True)

    total_stats = {"demos_written": 0, "demos_total": 0, "noops_filtered": 0}

    for hdf5_name in hdf5_files:
        raw_path = os.path.join(args.libero_raw_data_dir, hdf5_name)
        out_path = os.path.join(args.libero_target_dir, hdf5_name)

        print(f"\n--- Converting: {hdf5_name} ---")
        stats = convert_task(raw_path, out_path, filter_noops=args.filter_noops)

        for k in total_stats:
            total_stats[k] += stats[k]

        print(f"    {stats['demos_written']}/{stats['demos_total']} demos written, "
              f"{stats['noops_filtered']} no-ops filtered")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Demos written:    {total_stats['demos_written']}/{total_stats['demos_total']}")
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
             "Example: ./LIBERO/libero/datasets/libero_10_joint_noops",
    )
    parser.add_argument(
        "--filter_noops", action="store_true", default=True,
        help="Filter out no-op actions (default: True).",
    )
    parser.add_argument(
        "--no_filter_noops", action="store_false", dest="filter_noops",
        help="Disable no-op filtering.",
    )

    args = parser.parse_args()
    main(args)
