#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SUITE="${2:-}"

if [[ -z "$MODE" || -z "$SUITE" ]]; then
  echo "Usage: bash scripts/rebuild_libero_rlds_from_npz.sh <humanized|original-joint> <libero_10|libero_spatial|libero_goal|libero_object>"
  exit 1
fi

REPO_ROOT="/home/vsp1323/alex/openvla-oft_human"
cd "$REPO_ROOT"

case "$SUITE" in
  libero_10)
    SUITE_LABEL="10"
    BUILDER_SUFFIX="10"
    NPZ_BASE_ORIG="/home/vsp1323/alex/LIBERO/scripts/result/original_npz/libero_10"
    NPZ_BASE_HUMAN="/home/vsp1323/alex/LIBERO/scripts/result/humanized_npz/libero_10_humanized"
    ;;
  libero_spatial)
    SUITE_LABEL="spatial"
    BUILDER_SUFFIX="Spatial"
    NPZ_BASE_ORIG="/home/vsp1323/alex/LIBERO/scripts/result/original_npz/libero_spatial"
    NPZ_BASE_HUMAN="/home/vsp1323/alex/LIBERO/scripts/result/humanized_npz/libero_spatial_humanized"
    ;;
  libero_goal)
    SUITE_LABEL="goal"
    BUILDER_SUFFIX="Goal"
    NPZ_BASE_ORIG="/home/vsp1323/alex/LIBERO/scripts/result/original_npz/libero_goal"
    NPZ_BASE_HUMAN="/home/vsp1323/alex/LIBERO/scripts/result/humanized_npz/libero_goal_humanized"
    ;;
  libero_object)
    SUITE_LABEL="object"
    BUILDER_SUFFIX="Object"
    NPZ_BASE_ORIG="/home/vsp1323/alex/LIBERO/scripts/result/original_npz/libero_object"
    NPZ_BASE_HUMAN="/home/vsp1323/alex/LIBERO/scripts/result/humanized_npz/libero_object_humanized"
    ;;
  *)
    echo "Unsupported suite: $SUITE"
    exit 1
    ;;
esac

case "$MODE" in
  humanized)
    TASK_ROOTS_DIR="$NPZ_BASE_HUMAN"
    OUTPUT_NAME="${SUITE}_humanized_no_noops"
    BUILDER_DIR="rlds_dataset_builder/LIBERO_${BUILDER_SUFFIX}_humanized"
    ;;
  original-joint)
    TASK_ROOTS_DIR="$NPZ_BASE_ORIG"
    OUTPUT_NAME="${SUITE}_joint_no_noops"
    BUILDER_DIR="rlds_dataset_builder/LIBERO_${BUILDER_SUFFIX}_joint"
    ;;
  *)
    echo "Unsupported mode: $MODE"
    exit 1
    ;;
esac

OUTPUT_DIR="$REPO_ROOT/LIBERO/libero/datasets/$OUTPUT_NAME"
DEST="$REPO_ROOT/modified_libero_rlds/$OUTPUT_NAME/1.0.0"

echo "[1/3] NPZ -> HDF5"
python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir "$TASK_ROOTS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --filter_noops \
  --require_success

echo "[2/3] TFDS build"
cd "$REPO_ROOT/$BUILDER_DIR"
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

echo "[3/3] Copy latest TFDS shards"
NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"