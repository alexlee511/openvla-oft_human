All paths below are relative to the openvla-oft_human/ repo root.

# ─── Step 1: generate humanized initial poses (needed once per suite before evaluating a humanized/th-ik checkpoint) ───

python experiments/robot/libero/A_generate_humanized_initial_poses.py --suite_name libero_10
python experiments/robot/libero/A_generate_humanized_initial_poses.py --suite_name libero_spatial libero_object libero_goal libero_10

# Override which humanized_npz dir to read initial poses from (default is auto-detected):
python experiments/robot/libero/A_generate_humanized_initial_poses.py \
  --suite_name libero_10 \
  --npz_root ../LIBERO-humanized/scripts/result/humanized_npz/libero_10_humanized_th-ik

# ─── Step 2: evaluate a checkpoint ───
# Format command:

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint runs/<method>/openvla-oft_<humanized|joint>_<suite>/<checkpoint_dir_name> \
  --task_suite_name <suite> \
  --use_l1_regression true \
  --use_proprio true \
  --use_joint_pos true \
  --use_humanized_initial_pose <true if method != original, else omit> \
  --num_images_in_input 2 \
  --center_crop true \
  --num_open_loop_steps 8 \
  --num_trials_per_task 10 \
  --joint_substeps 1 \
  --joint_Kp_overshoot 3.5 \
  [--use_wandb true --wandb_entity <entity> --wandb_project openvla-oft_human]

# original EEF-control checkpoint: add --use_joint_pos false and drop --joint_substeps/--joint_Kp_overshoot/--use_humanized_initial_pose

# ─── Examples ───

# humanized / pure-ik, libero_10
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint runs/pure_ik/openvla-oft_humanized_libero_10/<checkpoint_dir_name> \
  --task_suite_name libero_10 \
  --use_l1_regression true --use_proprio true --use_joint_pos true --use_humanized_initial_pose true \
  --num_images_in_input 2 --center_crop true \
  --num_open_loop_steps 8 --num_trials_per_task 10 \
  --joint_substeps 1 --joint_Kp_overshoot 3.5

# original joint-control, libero_10
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint runs/original/openvla-oft_joint_libero_10/<checkpoint_dir_name> \
  --task_suite_name libero_10 \
  --use_l1_regression true --use_proprio true --use_joint_pos true \
  --num_images_in_input 2 --center_crop true \
  --num_open_loop_steps 8 --num_trials_per_task 50 --joint_substeps 1

# original EEF-control, libero_10
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint runs/original/openvla-oft_libero_4tasks/<checkpoint_dir_name> \
  --task_suite_name libero_10 \
  --use_l1_regression true --use_proprio true --use_joint_pos false \
  --num_images_in_input 2 --center_crop true \
  --num_open_loop_steps 8 --num_trials_per_task 5

# ─── Notes ───
# --joint_substeps 2  -> code actually runs substepping at 25 internally, this is just the outer setting
# --joint_Kp_overshoot 3.5  -> outer-loop Kp gain override
# --num_steps_wait_after_humanized_pose 40  -> settle time after moving to the humanized initial pose
# --unnorm_key <dataset_name>  -> only needed if checkpoint's unnorm stats key doesn't fuzzy-match --task_suite_name automatically
# --record_settle_frames true  -> also record the settle-in frames before the episode starts
