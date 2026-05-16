# For each .npz in the rollout_data directory:
python /home/vsp1323/alex/LIBERO/scripts/A_human_likeness_evaluate.py \
  --rollout_npz <path_to_episode_npz> \
  --suite libero_10 \
  --out_dir <output_dir>
  
# After eval 1 (joint):
python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir ./rollouts/libero_10/<joint_checkpoint>/<datetime>/ \
    --suite libero_10

# After eval 2 (humanized):
python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir ./rollouts/libero_10/<humanized_checkpoint>/<datetime>/ \
    --suite libero_10
    
python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir /home/vsp1323/alex/openvla-oft_human/rollouts/libero_10/openvla-7b+libero_10_humanized_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--joint_ctrl--parallel_dec--8_acts_/2026_05_04-01_09_04 \
    --suite libero_10
    
python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir /home/vsp1323/alex/openvla-oft_human/rollouts/libero_10/openvla-7b+libero_4_task_suites_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--eef_ctrl--parallel_dec--8_acts_c/2026_05_04-04_23_08 \
    --suite libero_10
    
python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir /home/vsp1323/alex/openvla-oft_human/rollouts/libero_10/openvla-7b+libero_4_task_suites_humanized_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--joint_ctrl--parallel_d/2026_05_12-02_28_55 \
    --suite libero_10
