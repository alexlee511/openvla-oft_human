#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  WARNING: full OpenVLA-7B LoRA fine-tuning (this script's config)
#  needs at least ~80GB VRAM (e.g. A100-80GB/H100). It has NOT been
#  run in this environment for that reason — treat it as a reference
#  wrapper for whenever suitable hardware is available, not a
#  verified/exercised part of the pipeline.
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

MODE="${1:-}"
SUITE="${2:-}"
METHOD="${3:-}"

if [[ -z "$MODE" || -z "$SUITE" ]]; then
  echo "Usage: bash scripts/finetune_libero_from_rlds.sh <humanized|original-joint> <libero_10|libero_spatial|libero_goal|libero_object> [method]"
  echo ""
  echo "  method (required for MODE=humanized): pure-ik | liu-ik | hrr-ik | th-ik | ..."
  echo "         (must match the subdir used by rebuild_libero_rlds_from_npz.sh for the same method)"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

case "$SUITE" in
  libero_10)      SUITE_TAG="libero_10" ;;
  libero_spatial) SUITE_TAG="libero_spatial" ;;
  libero_goal)    SUITE_TAG="libero_goal" ;;
  libero_object)  SUITE_TAG="libero_object" ;;
  *)
    echo "Unsupported suite: $SUITE"
    exit 1
    ;;
esac

case "$MODE" in
  humanized)
    if [[ -z "$METHOD" ]]; then
      echo "[ERROR] MODE=humanized requires a [method] argument (pure-ik|liu-ik|hrr-ik|th-ik|...)"
      exit 1
    fi
    # Same method -> RLDS-subdir mapping as rebuild_libero_rlds_from_npz.sh.
    RLDS_SUBDIR="${METHOD//-/_}"
    DATA_ROOT_DIR="$REPO_ROOT/modified_libero_rlds/$RLDS_SUBDIR"
    DATASET_NAME="${SUITE_TAG}_humanized_no_noops"
    RUN_ROOT_DIR="$REPO_ROOT/runs/$RLDS_SUBDIR/openvla-oft_humanized_${SUITE_TAG}"
    # Prefix run_id_note with the method so the checkpoint name itself
    # carries the method (otherwise all methods share an identical run_id
    # for a given suite, differing only by parent RUN_ROOT_DIR).
    METHOD_TAG="$METHOD"
    ;;
  original-joint)
    DATA_ROOT_DIR="$REPO_ROOT/modified_libero_rlds/original"
    DATASET_NAME="${SUITE_TAG}_joint_no_noops"
    RUN_ROOT_DIR="$REPO_ROOT/runs/original/openvla-oft_joint_${SUITE_TAG}"
    METHOD_TAG="original"
    ;;
  *)
    echo "Unsupported mode: $MODE"
    exit 1
    ;;
esac

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir "$DATA_ROOT_DIR" \
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
  --run_id_note "${METHOD_TAG}--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state"
