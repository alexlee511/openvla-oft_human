# Make Training Dataset (NPZ → HDF5 → RLDS → fine-tune)

Both scripts below now take a `[method]` argument for `humanized` mode (`pure-ik`｜`liu-ik`｜`hrr-ik`｜`th-ik`), matching the method-name-based NPZ directory naming produced by `LIBERO-humanized/scripts/A_humanized_libero_suite.py` (see `LIBERO-humanized/HUMANIZATION_PIPELINE.md` §3–4). No more editing commented-out lines to pick a method — every method gets its own subdirectory under `modified_libero_rlds/`, including `th-ik` (→ `modified_libero_rlds/th_ik/`) and the `original-joint` baseline (→ `modified_libero_rlds/original/`), so datasets never collide.

## 1. Rebuild RLDS from NPZ — `scripts/rebuild_libero_rlds_from_npz.sh`

```bash
bash scripts/rebuild_libero_rlds_from_npz.sh <humanized|original-joint> <suite> [method]
```

```bash
# humanized, one method at a time
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_10 th-ik
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_10 pure-ik
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_10 liu-ik
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_10 hrr-ik

# all four suites, TH-IK (ours)
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_10      th-ik
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_spatial th-ik
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_goal    th-ik
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_object  th-ik

# original joint-control baseline (no method — reads original_npz, not humanized_npz)
bash scripts/rebuild_libero_rlds_from_npz.sh original-joint libero_10
bash scripts/rebuild_libero_rlds_from_npz.sh original-joint libero_spatial
bash scripts/rebuild_libero_rlds_from_npz.sh original-joint libero_goal
bash scripts/rebuild_libero_rlds_from_npz.sh original-joint libero_object

# ablation NPZ dirs also work — method just has to match the directory
# suffix produced under LIBERO-humanized/scripts/result/humanized_npz/
bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_10 th-ik_cs-cap
```

Internally this does, per call: `A_npz_to_hdf5.py` (NPZ → HDF5) → `tfds build` in the matching `rlds_dataset_builder/LIBERO_<suite>_{humanized,joint}/` → copy the freshly built shards into `modified_libero_rlds/<method-subdir>/<suite>_{humanized,joint}_no_noops/1.0.0/` (`<method-subdir>` is `th_ik`/`pure_ik`/`liu_ik`/`hrr_ik`/`original`).

## 2. Fine-tune from the resulting RLDS — `scripts/finetune_libero_from_rlds.sh`

⚠️ Needs **at least ~80GB VRAM** — not run in this environment, treat as a reference wrapper only (see `docs/command/finetune command.md`).

```bash
bash scripts/finetune_libero_from_rlds.sh <humanized|original-joint> <suite> [method]
```

```bash
bash scripts/finetune_libero_from_rlds.sh humanized libero_goal th-ik
bash scripts/finetune_libero_from_rlds.sh humanized libero_object pure-ik
bash scripts/finetune_libero_from_rlds.sh original-joint libero_object
```

`--data_root_dir`/`--run_root_dir` are derived from `[method]` using the same subdir mapping as the rebuild script above, so the two always stay in sync. wandb logging is off by default in this wrapper — add `--use_wandb true --wandb_entity ... --wandb_project ...` yourself if you want it (see `docs/command/finetune command.md` for the full manually-written command set per method/suite, and a resume-from-checkpoint recipe).

## 3. Manual step-by-step equivalent (only if you need to run one stage in isolation)

```bash
# NPZ -> HDF5
python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir LIBERO-humanized/scripts/result/humanized_npz/libero_10_humanized_th-ik \
  --output_dir LIBERO/libero/datasets/libero_10_humanized_no_noops \
  --filter_noops --require_success

# HDF5 -> RLDS
cd rlds_dataset_builder/LIBERO_10_humanized
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

# Copy RLDS shards to the training data root
NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
DEST=modified_libero_rlds/th_ik/libero_10_humanized_no_noops/1.0.0   # swap th_ik for the method used above
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
```

## 4. Killing a stuck preprocessing run

```bash
ps -eo pid,args | grep -E 'A_run_batch_demo|A_run_pipeline|A_preprocess_human_demo' | grep -v grep | awk '{print $1}' | xargs kill -9 2>/dev/null; echo "Killed all lerobot preprocessing processes"
```

(`A_run_batch_demo.py`/`A_run_pipeline.py` now live under `LIBERO-humanized/scripts/helpers/`, but the process name match above still works unchanged.) To avoid leaving orphaned children behind next time: launch batch scripts with `setsid` so Ctrl+C propagates to the whole process group, or kill with `kill -- -<pgid>` / `pkill -P <pid>`.
