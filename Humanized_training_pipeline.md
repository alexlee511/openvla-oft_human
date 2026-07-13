# Humanized Training Pipeline (openvla-oft_human)

This is the formal description of the training/evaluation pipeline in this repo. Its input is the humanized dataset produced by the companion `LIBERO-humanized` repo (see `LIBERO-humanized/HUMANIZATION_PIPELINE.md` for how `humanized.npz`/`humanized_sim.npz` are generated across the five compared methods — Original, Pure-IK, Liu-IK, HRR-IK, TH-IK/ours — and their ablations). This document only covers what happens once that data lands in `openvla-oft_human`.

```text
humanized.npz / humanized_sim.npz  (from LIBERO-humanized, per method/suite/task)
    |
    v
dataset conversion → RLDS
    |
    v
LoRA fine-tuning of OpenVLA-7B (joint-control)
    |
    v
policy evaluation in LIBERO sim
    |
    v
rollout human-likeness scoring
```

## 1. Dataset Conversion

| File | Role |
|---|---|
| `experiments/robot/libero/A_npz_to_hdf5.py` | Packages replay output (`humanized_sim.npz`/`original_sim.npz` from `A_libero_joint_replay.py --collect_obs`) into HDF5. |
| `rlds_dataset_builder/LIBERO_{10,Goal,Object,Spatial}_{humanized,joint}/` | TFDS dataset builders that turn the HDF5 into RLDS TFRecord datasets (one builder dir per suite × {humanized, joint}). |
| `scripts/rebuild_libero_rlds_from_npz.sh` | End-to-end wrapper: NPZ → HDF5 → RLDS, for one `<humanized\|original-joint> <suite> [method]` at a time — `[method]` is `pure-ik`/`liu-ik`/`hrr-ik`/`th-ik` (or an ablation label like `th-ik_cs-cap`), matching the output-directory naming produced by `LIBERO-humanized/scripts/A_humanized_libero_suite.py`. See `docs/command/Make training dataset.md` for the full command set. |

Every method — including `th-ik` (ours) and the `original-joint` baseline — gets its own subdirectory under `modified_libero_rlds/`: `modified_libero_rlds/th_ik/`, `.../pure_ik/`, `.../liu_ik/`, `.../hrr_ik/`, `.../original/`. This is how `finetune.py` later distinguishes which method/baseline it's training on; datasets from different methods never collide.

## 2. Fine-Tuning

`vla-scripts/finetune.py` LoRA-finetunes OpenVLA-7B. It is close to the stock upstream openvla-oft file; original-vs-humanized and EEF-vs-joint control are **not** runtime flags — they are determined entirely by which RLDS dataset `--data_root_dir`/`--dataset_name` point at.

⚠️ **Not actually run in this environment**: the standard config below needs **at least ~80GB VRAM** (A100-80GB/H100 class). `scripts/finetune_libero_from_rlds.sh` is a reference wrapper for whenever suitable hardware is available — treat it as documented but unverified/unexercised here.

`bash scripts/finetune_libero_from_rlds.sh <humanized|original-joint> <suite> [method]` derives `--data_root_dir`/`--run_root_dir` from `[method]` using the same subdir mapping as `rebuild_libero_rlds_from_npz.sh` above, so dataset and training run always match. Direct invocation is also fine — see `docs/command/finetune command.md` for the full command set per method/suite plus a resume-from-checkpoint recipe, and `vla-scripts/patch_dtype.py` if training on a GPU without bfloat16 support (T4/V100).

## 3. Policy Evaluation

| File | Role |
|---|---|
| `experiments/robot/libero/A_generate_humanized_initial_poses.py` | Precomputes humanized initial robot poses (humanized demos may start from different poses than the original suite), writes `A_humanized_initial_poses.py`. |
| `experiments/robot/libero/run_libero_eval.py` | The core evaluator — despite lacking an `A_` prefix, this is the most heavily customized file in the fork. Loads the fine-tuned checkpoint, runs closed-loop rollouts in LIBERO sim, logs to wandb, saves per-episode rollout NPZ files. Key flags: `--use_joint_pos` (8D absolute joint controller), `--use_humanized_initial_pose` (auto-enabled for humanized joint runs), `--joint_substeps`, `--joint_Kp_overshoot`, `--unnorm_key` (fuzzy-matches dataset suffixes like `_humanized_no_noops`). |

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint runs/th_ik/openvla-oft_humanized_libero_10/<checkpoint_dir> \
  --task_suite_name libero_10 \
  --use_l1_regression true --use_proprio true --use_joint_pos true \
  --use_humanized_initial_pose true \
  --num_images_in_input 2 --center_crop true \
  --num_open_loop_steps 8 --num_trials_per_task 10 \
  --joint_substeps 1 --joint_Kp_overshoot 3.5
```

See `docs/command/libero evaluate command.md` for the full recorded command set (original EEF / original joint / each humanized method).

## 4. Rollout Human-Likeness Scoring

`experiments/robot/libero/A_vla_rollout_hl.py` scores the rollout NPZ files produced above. It does not reimplement the metrics — it adds `../LIBERO-humanized/scripts` to `sys.path` and imports `evaluate_trajectory`/`METRIC_KEYS`/`WEIGHTS` directly from `A_human_likeness_evaluate.py` in the companion repo, so the metric definitions (`HJL`, `MJE`, `SOAq`, `SOAx`, `EEA`, `UNIFIED`) are identical on both sides of the pipeline.

```bash
python experiments/robot/libero/A_vla_rollout_hl.py \
  --rollout_dir ./rollouts/libero_10/<checkpoint_name>/<datetime>/ \
  --suite libero_10
```

See `docs/command/human likeness eval command.md` for recorded examples.

## 5. Supporting Analysis Tools

Not part of the automatic pipeline above, but used post-hoc on its outputs — all under `experiments/robot/libero/analysis/`: `A_eval_model_on_train_data.py` (checks model predictions against train-set ground truth), `A_joint_distribution_diagnostics.py`, `A_rollout_controller_tracking.py`, `A_rollout_tracking_and_trainset_analysis.py`, `A_task_eval_summary.py`. `merge_lora_weights_and_save.py` merges a LoRA-only checkpoint into the base model when `--merge_lora_during_training` was not used.
