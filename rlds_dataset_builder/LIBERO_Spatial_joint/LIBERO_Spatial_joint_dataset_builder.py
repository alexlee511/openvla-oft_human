from typing import Iterator, Tuple, Any

import glob
import os
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(episode_path, demo_id):
        with h5py.File(episode_path, "r") as f:
            demo_key = f"demo_{demo_id}"
            if demo_key not in f["data"].keys():
                return None
            actions = f["data"][demo_key]["actions"][()]
            if "state" in f["data"][demo_key]["obs"]:
                states = f["data"][demo_key]["obs"]["state"][()]
            else:
                joint_states = f["data"][demo_key]["obs"]["joint_states"][()]
                gripper_states = f["data"][demo_key]["obs"]["gripper_states"][()]
                gripper_width = np.mean(gripper_states, axis=1, keepdims=True)
                states = np.concatenate([joint_states, gripper_width], axis=-1)
            joint_states = f["data"][demo_key]["obs"]["joint_states"][()]
            images = f["data"][demo_key]["obs"]["agentview_rgb"][()]
            wrist_images = f["data"][demo_key]["obs"]["eye_in_hand_rgb"][()]

        raw_file_string = os.path.basename(episode_path).split("/")[-1]
        words = raw_file_string[:-10].split("_")
        command = ""
        for word in words:
            if "SCENE" in word:
                command = ""
                continue
            command = command + word + " "
        command = command[:-1]

        episode = []
        for i in range(actions.shape[0]):
            episode.append(
                {
                    "observation": {
                        "image": images[i][::-1, ::-1],
                        "wrist_image": wrist_images[i][::-1, ::-1],
                        "state": np.asarray(states[i], np.float32),
                        "joint_state": np.asarray(joint_states[i], dtype=np.float32),
                    },
                    "action": np.asarray(actions[i], dtype=np.float32),
                    "discount": 1.0,
                    "reward": float(i == (actions.shape[0] - 1)),
                    "is_first": i == 0,
                    "is_last": i == (actions.shape[0] - 1),
                    "is_terminal": i == (actions.shape[0] - 1),
                    "language_instruction": command,
                }
            )

        sample = {"steps": episode, "episode_metadata": {"file_path": episode_path}}
        return episode_path + f"_{demo_id}", sample

    for sample in paths:
        with h5py.File(sample, "r") as f:
            demo_ids = sorted([int(k.split("_")[1]) for k in f["data"].keys()])
        for idx in demo_ids:
            ret = _parse_example(sample, idx)
            if ret is not None:
                yield ret


class LiberoSpatialJoint(MultiThreadedDatasetBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}
    N_WORKERS = 40
    MAX_PATHS_IN_MEMORY = 80
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg", doc="Main camera RGB observation."),
                                    "wrist_image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg", doc="Wrist camera RGB observation."),
                                    "state": tfds.features.Tensor(shape=(8,), dtype=np.float32, doc="Robot proprioceptive state: [joint_pos(7), gripper_width(1)]."),
                                    "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32, doc="Robot joint angles."),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(8,), dtype=np.float32, doc="Joint position targets (7) + gripper command (1). Derived from consecutive joint_states in original demos."),
                            "discount": tfds.features.Scalar(dtype=np.float32, doc="Discount if provided, default to 1."),
                            "reward": tfds.features.Scalar(dtype=np.float32, doc="Reward if provided, 1 on final step for demos."),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode if it is a terminal step, True for demos."),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict({"file_path": tfds.features.Text(doc="Path to the original data file.")}),
                }
            )
        )

    def _split_paths(self):
        return {
            "train": glob.glob("/home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_spatial_joint_no_noops/*.hdf5"),
        }