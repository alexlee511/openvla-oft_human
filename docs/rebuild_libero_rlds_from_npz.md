# Rebuild LIBERO RLDS From NPZ

This note summarizes how the NPZ -> HDF5 -> RLDS flow relates to the original `regenerate_libero_dataset.py` flow, and provides reusable commands for rebuilding datasets from NPZ files.

## 1. Does `A_npz_to_hdf5.py` work the same way as `regenerate_libero_dataset.py`?

Not exactly, but after replay they can be made equivalent at the HDF5 semantics level used by RLDS and training.

### `regenerate_libero_dataset.py`

File: `experiments/robot/libero/regenerate_libero_dataset.py`

This is the original OpenVLA-OFT style regeneration flow for EEF-control LIBERO datasets:

1. Read raw LIBERO HDF5 demos.
2. Replay the original demo actions in the simulator.
3. Record observations and states from the environment.
4. Filter no-op transitions.
5. Keep only successful demonstrations.
6. Write regenerated HDF5 files.

Important detail:
- It keeps the original action semantics from the source demos.
- For original LIBERO datasets, those actions are EEF-space actions.

### `A_npz_to_hdf5.py`

File: `experiments/robot/libero/A_npz_to_hdf5.py`

This is a later-stage packager, not a full replay pipeline:

1. Read already-generated NPZ files.
2. Reuse the observations already stored in the NPZ.
3. Filter no-op transitions.
4. Optionally keep only successful demos.
5. Write HDF5 files for the RLDS builder.

So it is equivalent to the "packaging output after replay" stage, not the full original replay procedure.

### What is the practical difference?

`regenerate_libero_dataset.py`:
- Starts from raw HDF5.
- Replays actions itself.
- Re-collects images, states, and observations itself.

`A_npz_to_hdf5.py`:
- Starts from NPZ that already contains replayed observations and actions.
- Does not rerun the simulator.
- Just repackages the replay result into HDF5.

### Are they equivalent in the dataset-making details you care about?

Yes, if the NPZ was already produced by replaying the intended demo actions upstream.

That means:
- If the NPZ came from replaying original HDF5 demo actions, then it matches the original-demo replay source.
- If the NPZ came from replaying humanized demo actions generated upstream, then it matches the humanized replay source.

For the downstream training dataset semantics, the important checks are:

1. Are the replayed observations used? Yes.
2. Are failed replays filtered? Yes, via `--require_success` when packaging.
3. Are no-op transitions filtered? Yes, via `--filter_noops`.
4. Is the joint action label aligned as next-step target? Yes, `A_npz_to_hdf5.py` now rebuilds joint labels from `joint_states_obs[t+1]`.

So although `A_npz_to_hdf5.py` does not perform replay itself, it now preserves the same training-facing semantics you were asking about, provided the NPZ itself came from the correct replay source.

### Are they compatible for training?

Yes, as long as the HDF5 fields have the correct semantics.

For the joint-control NPZ flow, `A_npz_to_hdf5.py` now rebuilds the joint action target as:

```text
action[t, :7] = joint_states_obs[t + 1]
action[-1, :7] = joint_states_obs[-1]
```

and preserves the original gripper command.

That makes the packed HDF5 follow the same joint-target convention as `A_convert_libero_to_joint.py`.

## 2. Supported builders in this repo

### Original EEF builders

- `rlds_dataset_builder/LIBERO_10`
- `rlds_dataset_builder/LIBERO_Spatial`
- `rlds_dataset_builder/LIBERO_Goal`
- `rlds_dataset_builder/LIBERO_Object`

### Humanized joint builders

- `rlds_dataset_builder/LIBERO_10_humanized`
- `rlds_dataset_builder/LIBERO_Spatial_humanized`
- `rlds_dataset_builder/LIBERO_Goal_humanized`
- `rlds_dataset_builder/LIBERO_Object_humanized`

### Original joint builders

- `rlds_dataset_builder/LIBERO_10_joint`
- `rlds_dataset_builder/LIBERO_Spatial_joint`
- `rlds_dataset_builder/LIBERO_Goal_joint`
- `rlds_dataset_builder/LIBERO_Object_joint`

## 3. Rebuild commands from NPZ

All commands below assume the repo root is:

```bash
cd /home/vsp1323/alex/openvla-oft_human
```

### 3.1 Humanized joint datasets from NPZ

Use this for:
- `libero_10_humanized_no_noops`
- `libero_spatial_humanized_no_noops`
- `libero_goal_humanized_no_noops`
- `libero_object_humanized_no_noops`

#### LIBERO-10 Humanized

```bash
cd /home/vsp1323/alex/openvla-oft_human

python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir /home/vsp1323/alex/LIBERO/scripts/result/humanized_npz/libero_10_humanized \
  --output_dir /home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_10_humanized_no_noops \
  --filter_noops \
  --require_success

cd /home/vsp1323/alex/openvla-oft_human/rlds_dataset_builder/LIBERO_10_humanized
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
DEST=/home/vsp1323/alex/openvla-oft_human/modified_libero_rlds/libero_10_humanized_no_noops/1.0.0
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"
```

#### LIBERO-Spatial Humanized

```bash
cd /home/vsp1323/alex/openvla-oft_human

python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir /home/vsp1323/alex/LIBERO/scripts/result/humanized_npz/libero_spatial_humanized \
  --output_dir /home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_spatial_humanized_no_noops \
  --filter_noops \
  --require_success

cd /home/vsp1323/alex/openvla-oft_human/rlds_dataset_builder/LIBERO_Spatial_humanized
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
DEST=/home/vsp1323/alex/openvla-oft_human/modified_libero_rlds/libero_spatial_humanized_no_noops/1.0.0
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"
```

#### LIBERO-Goal Humanized

```bash
cd /home/vsp1323/alex/openvla-oft_human

python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir /home/vsp1323/alex/LIBERO/scripts/result/humanized_npz/libero_goal_humanized \
  --output_dir /home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_goal_humanized_no_noops \
  --filter_noops \
  --require_success

cd /home/vsp1323/alex/openvla-oft_human/rlds_dataset_builder/LIBERO_Goal_humanized
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
DEST=/home/vsp1323/alex/openvla-oft_human/modified_libero_rlds/libero_goal_humanized_no_noops/1.0.0
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"
```

#### LIBERO-Object Humanized

```bash
cd /home/vsp1323/alex/openvla-oft_human

python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir /home/vsp1323/alex/LIBERO/scripts/result/humanized_npz/libero_object_humanized \
  --output_dir /home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_object_humanized_no_noops \
  --filter_noops \
  --require_success

cd /home/vsp1323/alex/openvla-oft_human/rlds_dataset_builder/LIBERO_Object_humanized
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
DEST=/home/vsp1323/alex/openvla-oft_human/modified_libero_rlds/libero_object_humanized_no_noops/1.0.0
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"
```

### 3.2 Original joint dataset from NPZ

Supported builders in this repo:
- `libero_10_joint_no_noops`
- `libero_spatial_joint_no_noops`
- `libero_goal_joint_no_noops`
- `libero_object_joint_no_noops`

```bash
cd /home/vsp1323/alex/openvla-oft_human

python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir /home/vsp1323/alex/LIBERO/scripts/result/original_npz/libero_10 \
  --output_dir /home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_10_joint_no_noops \
  --filter_noops \
  --require_success

cd /home/vsp1323/alex/openvla-oft_human/rlds_dataset_builder/LIBERO_10_joint
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
DEST=/home/vsp1323/alex/openvla-oft_human/modified_libero_rlds/libero_10_joint_no_noops/1.0.0
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"
```

#### LIBERO-Spatial Original Joint

```bash
cd /home/vsp1323/alex/openvla-oft_human

python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir /home/vsp1323/alex/LIBERO/scripts/result/original_npz/libero_spatial \
  --output_dir /home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_spatial_joint_no_noops \
  --filter_noops \
  --require_success

cd /home/vsp1323/alex/openvla-oft_human/rlds_dataset_builder/LIBERO_Spatial_joint
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
DEST=/home/vsp1323/alex/openvla-oft_human/modified_libero_rlds/libero_spatial_joint_no_noops/1.0.0
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"
```

#### LIBERO-Goal Original Joint

```bash
cd /home/vsp1323/alex/openvla-oft_human

python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir /home/vsp1323/alex/LIBERO/scripts/result/original_npz/libero_goal \
  --output_dir /home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_goal_joint_no_noops \
  --filter_noops \
  --require_success

cd /home/vsp1323/alex/openvla-oft_human/rlds_dataset_builder/LIBERO_Goal_joint
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
DEST=/home/vsp1323/alex/openvla-oft_human/modified_libero_rlds/libero_goal_joint_no_noops/1.0.0
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"
```

#### LIBERO-Object Original Joint

```bash
cd /home/vsp1323/alex/openvla-oft_human

python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir /home/vsp1323/alex/LIBERO/scripts/result/original_npz/libero_object \
  --output_dir /home/vsp1323/alex/openvla-oft_human/LIBERO/libero/datasets/libero_object_joint_no_noops \
  --filter_noops \
  --require_success

cd /home/vsp1323/alex/openvla-oft_human/rlds_dataset_builder/LIBERO_Object_joint
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
DEST=/home/vsp1323/alex/openvla-oft_human/modified_libero_rlds/libero_object_joint_no_noops/1.0.0
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"
```

## 3.3 Bash scripts

You can run the rebuild flow directly with:

```bash
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_10
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_spatial
bash scripts/rebuild_libero_rlds_from_npz.sh original-joint libero_10
bash scripts/rebuild_libero_rlds_from_npz.sh original-joint libero_goal
```

You can run fine-tuning directly with:

```bash
bash scripts/finetune_libero_from_rlds.sh humanized libero_10
bash scripts/finetune_libero_from_rlds.sh original-joint libero_object
```

## 4. Fine-tuning commands

### Original joint LIBERO-10

```bash
cd /home/vsp1323/alex/openvla-oft_human

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /nfs/Workspace/Alex/openvla-oft_human/modified_libero_rlds \
  --dataset_name libero_10_joint_no_noops \
  --run_root_dir /nfs/Workspace/Alex/openvla-oft_human/runs/openvla-oft_joint_libero_10 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 15000 \
  --max_steps 20000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity alexlee511-national-taipei-university-of-technology \
  --wandb_project openvla-oft \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
```

### Humanized joint LIBERO-10

```bash
cd /home/vsp1323/alex/openvla-oft_human

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /nfs/Workspace/Alex/openvla-oft_human/modified_libero_rlds \
  --dataset_name libero_10_humanized_no_noops \
  --run_root_dir /nfs/Workspace/Alex/openvla-oft_human/runs/openvla-oft_humanized_libero_10 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 15000 \
  --max_steps 20000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity alexlee511-national-taipei-university-of-technology \
  --wandb_project openvla-oft_human \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
```

## 5. Important notes

### Action semantics during training

The action type used during training is determined by `dataset_name`, not by the shell command alone.

- `libero_10_no_noops` -> EEF action training
- `libero_10_joint_no_noops` -> joint-position action training
- `libero_10_humanized_no_noops` -> joint-position action training

### What `use_proprio=True` does

It adds proprio input to the model, but it does not change the action supervision type.

- EEF datasets use EEF-style proprio/state config.
- Joint datasets use joint-style proprio/state config.

### Existing checkpoints will not be fixed retroactively

If a checkpoint was trained on incorrectly aligned action labels, fixing `A_npz_to_hdf5.py` only affects future rebuilt HDF5 / RLDS datasets and future training runs.
