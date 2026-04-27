from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(episode_path, demo_id):
        # load raw data from joint-converted HDF5 (output of convert_libero_to_joint.py)
        with h5py.File(episode_path, "r") as F:
            if f"demo_{demo_id}" not in F['data'].keys():
                return None  # skip if demo doesn't exist
            actions = F['data'][f"demo_{demo_id}"]["actions"][()]          # (T, 8) joint pos + gripper
            # State: [joint_pos(7), gripper_width(1)] = 8D proprio
            if "state" in F['data'][f"demo_{demo_id}"]["obs"]:
                states = F['data'][f"demo_{demo_id}"]["obs"]["state"][()]  # (T, 8)
            else:
                # Fallback: construct from joint_states + mean of gripper_states
                joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]  # (T, 7)
                gripper_states = F['data'][f"demo_{demo_id}"]["obs"]["gripper_states"][()]  # (T, 2)
                gripper_width = np.mean(gripper_states, axis=1, keepdims=True)
                states = np.concatenate([joint_states, gripper_width], axis=-1)  # (T, 8)
            joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]
            images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
            wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]

        # compute language instruction from filename
        raw_file_string = os.path.basename(episode_path).split('/')[-1]
        words = raw_file_string[:-10].split("_")
        command = ''
        for w in words:
            if "SCENE" in w:
                command = ''
                continue
            command = command + w + ' '
        command = command[:-1]

        # assemble episode --> demos so reward = 1 at end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'image': images[i][::-1, ::-1],           # rotate 180 degrees
                    'wrist_image': wrist_images[i][::-1, ::-1],  # rotate 180 degrees
                    'state': np.asarray(states[i], np.float32),  # [joint_pos(7), gripper(1)] = 8D
                    'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),  # (8,) joint pos + gripper
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
            })

        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        return episode_path + f"_{demo_id}", sample

    # iterate over all HDF5 paths
    for sample in paths:
        with h5py.File(sample, "r") as F:
            demo_ids = sorted([int(k.split("_")[1]) for k in F["data"].keys()])
        for idx in demo_ids:
            ret = _parse_example(sample, idx)
            if ret is not None:
                yield ret


class Libero10Joint(MultiThreadedDatasetBuilder):
    """DatasetBuilder for LIBERO-10 joint position dataset (derived from original demos)."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40
    MAX_PATHS_IN_MEMORY = 80
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot proprioceptive state: [joint_pos(7), gripper_width(1)].',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Joint position targets (7) + gripper command (1). Derived from consecutive joint_states in original demos.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob("/home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_10_joint_no_noops/*.hdf5"),
        }
