# Fine-tune Command Reference

All paths below are relative to the `openvla-oft_human/` repo root (run these from inside that folder). Prefer the wrapper script over hand-writing these:

```bash
bash scripts/finetune_libero_from_rlds.sh <humanized|original-joint> <suite> [method]
```

⚠️ **VRAM**: this config (`--batch_size 8 --grad_accumulation_steps 8`, full OpenVLA-7B LoRA) needs **at least ~80GB VRAM** (A100-80GB/H100 class). It has not been run in this environment for that reason. For a T4/V100 (no bfloat16), run `python vla-scripts/patch_dtype.py --to float16` first (`--to bfloat16` to revert for A100/H100), and expect to need a much smaller `--batch_size` with more `--grad_accumulation_steps` to fit.

## Format command

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir modified_libero_rlds/<method>          \
  --dataset_name <suite>_<humanized|joint>_no_noops \
  --run_root_dir runs/<method>/openvla-oft_<humanized|joint>_<suite> \
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
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --run_id_note <method>--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
```

`<method>` is `th_ik`/`pure_ik`/`liu_ik`/`hrr_ik` for a humanized run, or `original` for the original-joint baseline — must match the subdir `rebuild_libero_rlds_from_npz.sh` wrote the RLDS dataset into (`docs/command/Make training dataset.md`).

**Prefix `--run_id_note` with the method.** `finetune.py`'s `get_run_id()` builds the checkpoint/run-dir name from `--dataset_name` + hyperparams + `--run_id_note`, but **not** from `--run_root_dir`. So without a method prefix, every method for a given suite produces an *identically-named* checkpoint folder (e.g. `openvla-7b+libero_10_humanized_no_noops+...--parallel_dec--...`), distinguished only by the parent `runs/<method>/` directory. Prepending the method (`th-ik--...`, `pure-ik--...`, `original--...`) makes the checkpoint name itself carry the method, so directory listings and `--pretrained_checkpoint` values stay unambiguous. The wrapper `finetune_libero_from_rlds.sh` does this automatically (using the raw `[method]` arg, e.g. `th-ik`, or `original` for `original-joint`).

| Run | `--data_root_dir` | `--dataset_name` | `--run_root_dir` |
|---|---|---|---|
| Original EEF (4 tasks) | `modified_libero_rlds/original` | `libero_4_task_suites_no_noops` | `runs/original/openvla-oft_libero_4tasks` |
| Original joint (libero_10) | `modified_libero_rlds/original` | `libero_10_joint_no_noops` | `runs/original/openvla-oft_joint_libero_10` |
| TH-IK/ours (libero_goal) | `modified_libero_rlds/th_ik` | `libero_goal_humanized_no_noops` | `runs/th_ik/openvla-oft_humanized_libero_goal` |
| Pure-IK (libero_goal) | `modified_libero_rlds/pure_ik` | `libero_goal_humanized_no_noops` | `runs/pure_ik/openvla-oft_humanized_libero_goal` |
| Liu-IK (libero_object) | `modified_libero_rlds/liu_ik` | `libero_object_humanized_no_noops` | `runs/liu_ik/openvla-oft_humanized_libero_object` |
| HRR-IK (4 tasks) | `modified_libero_rlds/hrr_ik` | `libero_4_task_suites_humanized_no_noops` | `runs/hrr_ik/openvla-oft_humanized_libero_4_tasks` |

`--dataset_name` swap-ins: `libero_10_humanized_no_noops`, `libero_spatial_humanized_no_noops`, `libero_goal_humanized_no_noops`, `libero_object_humanized_no_noops`, `libero_4_task_suites_humanized_no_noops` (humanized); `libero_10_joint_no_noops` / `libero_4_task_suites_joint_no_noops` (original joint); `libero_4_task_suites_no_noops` (original EEF).

## Resuming an interrupted run

```bash
# 1. Merge LoRA into the base checkpoint (skip if merge_lora_during_training was True)
python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint openvla/openvla-7b \
  --lora_finetuned_checkpoint_dir runs/<method>/<run_dir>/<checkpoint_name>--<N>_chkpt/

# 2. Continue training — same args as the original run, but vla_path points at the
#    checkpoint and resume flags are set
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path runs/<method>/<run_dir>/<checkpoint_name>--<N>_chkpt \
  ... \
  --resume True \
  --resume_step <N>
```
