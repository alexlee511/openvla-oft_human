"""Utils for evaluating policies in LIBERO simulation environments."""

import contextlib
import io
import math
import os

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256, use_joint_pos=False, joint_substeps=1):
    """Initializes and returns the LIBERO environment, along with the task description.

    Args:
        use_joint_pos: If True, use JOINT_POSITION controller instead of default OSC_POSE.
        joint_substeps: Number of sub-steps per action frame for JOINT_POSITION controller.
            When >1, control_freq is set to the MuJoCo sim frequency (500 Hz) so each
            sub-step corresponds to exactly 1 physics step.  The eval loop auto-computes
            n_sub = sim_freq / base_control_freq (= 25 for 500/20) to preserve the
            original 0.05 s inter-observation timing the model was trained on.
    """
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": ["agentview", "robot0_eye_in_hand", "frontview"],
    }
    if use_joint_pos:
        env_args["controller"] = "JOINT_POSITION"
        if joint_substeps > 1:
            # Set control_freq = sim_freq (500 Hz) so each sub-step = 1 physics step.
            # The eval loop auto-computes n_sub = sim_freq / 20 = 25.
            env_args["control_freq"] = 500
    if use_joint_pos and joint_substeps > 1:
        # Suppress sampling-rate warnings printed during env construction (initial
        # reset creates observables at 20 Hz which is below the 500 Hz control_freq).
        # We patch the observable rates after each env.reset() in the eval loop.
        with contextlib.redirect_stdout(io.StringIO()):
            env = OffScreenRenderEnv(**env_args)
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str, use_joint_pos: bool = False):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    if use_joint_pos:
        return [0, 0, 0, 0, 0, 0, 0, -1]  # 8D: 7 zero-delta joints + open gripper
    return [0, 0, 0, 0, 0, 0, -1]           # 7D: OSC_POSE default


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_frontview_image(obs):
    """Extracts front-view camera image from observations and preprocesses it."""
    img = obs["frontview_image"]
    img = img[::-1, ::-1]  # rotate 180 degrees to match other views
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, rollout_dir=None,
                       frontview_images=None):
    """Saves MP4 replays of an episode (agentview + optional frontview)."""
    if rollout_dir is None:
        rollout_dir = f"./rollouts/{DATE}"
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    filename = f"episode={idx}--success={success}--task={processed_task_description}.mp4"

    # Save agentview
    agentview_dir = os.path.join(rollout_dir, "video", "agentview")
    os.makedirs(agentview_dir, exist_ok=True)
    mp4_path = os.path.join(agentview_dir, filename)
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved agentview MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved agentview MP4 at path {mp4_path}\n")

    # Save frontview if provided
    if frontview_images is not None and len(frontview_images) > 0:
        frontview_dir = os.path.join(rollout_dir, "video", "frontview")
        os.makedirs(frontview_dir, exist_ok=True)
        front_mp4_path = os.path.join(frontview_dir, filename)
        video_writer = imageio.get_writer(front_mp4_path, fps=30)
        for img in frontview_images:
            video_writer.append_data(img)
        video_writer.close()
        print(f"Saved frontview MP4 at path {front_mp4_path}")
        if log_file is not None:
            log_file.write(f"Saved frontview MP4 at path {front_mp4_path}\n")

    return mp4_path


def save_rollout_data(rollout_data, idx, success, task_description, log_file=None, rollout_dir=None):
    """Saves per-timestep joint data from an episode rollout as an NPZ file."""
    if rollout_dir is None:
        rollout_dir = f"./rollouts/{DATE}"
    data_dir = os.path.join(rollout_dir, "rollout_data")
    os.makedirs(data_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    npz_path = f"{data_dir}/episode={idx}--success={success}--task={processed_task_description}.npz"
    np.savez_compressed(npz_path, **rollout_data)
    print(f"Saved rollout data at path {npz_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout data at path {npz_path}\n")
    return npz_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
