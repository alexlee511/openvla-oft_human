#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SUITE="${2:-}"

if [[ -z "$MODE" || -z "$SUITE" ]]; then
  echo "Usage: bash scripts/finetune_libero_from_rlds.sh <humanized|original-joint> <libero_10|libero_spatial|libero_goal|libero_object>"
  exit 1
fi

REPO_ROOT="/home/vsp1323/alex/openvla-oft_human"
cd "$REPO_ROOT"

case "$SUITE" in
  libero_10)
    SUITE_TAG="libero_10"
    RUN_TAG="libero_10"
    ;;
  libero_spatial)
    SUITE_TAG="libero_spatial"
    RUN_TAG="libero_spatial"
    ;;
  libero_goal)
    SUITE_TAG="libero_goal"
    RUN_TAG="libero_goal"
    ;;
  libero_object)
    SUITE_TAG="libero_object"
    RUN_TAG="libero_object"
    ;;
  *)
    echo "Unsupported suite: $SUITE"
    exit 1
    ;;
esac

case "$MODE" in
  humanized)
    DATASET_NAME="${SUITE_TAG}_humanized_no_noops"
    RUN_ROOT_DIR="/nfs/Workspace/Alex/openvla-oft_human/runs/openvla-oft_humanized_${RUN_TAG}"
    WANDB_PROJECT="openvla-oft_human"
    ;;
  original-joint)
    DATASET_NAME="${SUITE_TAG}_joint_no_noops"
    RUN_ROOT_DIR="/nfs/Workspace/Alex/openvla-oft_human/runs/openvla-oft_joint_${RUN_TAG}"
    WANDB_PROJECT="openvla-oft"
    ;;
  *)
    echo "Unsupported mode: $MODE"
    exit 1
    ;;
esac

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /nfs/Workspace/Alex/openvla-oft_human/modified_libero_rlds \
  --dataset_name "$DATASET_NAME" \
  --run_root_dir "$RUN_ROOT_DIR" \
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
  --wandb_project "$WANDB_PROJECT" \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state