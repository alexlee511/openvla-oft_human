# For each .npz in the rollout_data directory:
python /home/vsp1323/Humanized-VLA/LIBERO_human/scripts/A_human_likeness_evaluate.py \
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

(humanized)
python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir /home/vsp1323/Humanized-VLA/openvla-oft_human/rollouts/libero_10/openvla-7b+libero_10_humanized_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--joint_ctrl--liu_ik_parallel_dec--/2026_05_28-14_39_35 \
    --suite libero_10

python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir /home/vsp1323/Humanized-VLA/openvla-oft_human/rollouts/libero_spatial/openvla-7b+libero_spatial_humanized_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--joint_ctrl--pure_ik_parallel/2026_06_01-14_24_19 \
    --suite libero_spatial

python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir /home/vsp1323/Humanized-VLA/openvla-oft_human/rollouts/libero_goal/openvla-7b+libero_goal_humanized_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--joint_ctrl--pure_ik_parallel_de/2026_06_01-14_40_18 \
    --suite libero_goal

python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir /home/vsp1323/Humanized-VLA/openvla-oft_human/rollouts/libero_object/openvla-7b+libero_object_humanized_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--joint_ctrl--pure_ik_parallel_/2026_06_01-15_48_13 \
    --suite libero_object
    


(joint)
python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir /home/vsp1323/Humanized-VLA/openvla-oft_human/rollouts/libero_object/openvla-7b+libero_4_task_suites_joint_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--joint_ctrl--parallel_dec--/2026_05_15-12_54_35 \
    --suite libero_object

(original EEF)
python experiments/robot/libero/A_vla_rollout_hl.py \
    --rollout_dir /home/vsp1323/Humanized-VLA/openvla-oft_human/rollouts/libero_10/openvla-7b+libero_4_task_suites_no_noops+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--eef_ctrl--parallel_dec--8_acts_c/2026_05_04-04_23_08 \
    --suite libero_10
    
    

