#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SUITE="${2:-}"
METHOD="${3:-}"

if [[ -z "$MODE" || -z "$SUITE" ]]; then
  echo "Usage: bash scripts/rebuild_libero_rlds_from_npz.sh <humanized|original-joint> <libero_10|libero_spatial|libero_goal|libero_object> [method]"
  echo ""
  echo "  method (required for MODE=humanized): pure-ik | liu-ik | hrr-ik | th-ik"
  echo "         (or any other method label matching a directory under"
  echo "          <LIBERO-humanized>/scripts/result/humanized_npz/<suite>_humanized_<method>[_<ablation-suffix>],"
  echo "          e.g. 'th-ik_cs-cap' for the controller-tracking ablation)"
  echo ""
  echo "  Examples:"
  echo "    bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_10 th-ik"
  echo "    bash scripts/rebuild_libero_rlds_from_npz.sh humanized libero_10 pure-ik"
  echo "    bash scripts/rebuild_libero_rlds_from_npz.sh original-joint libero_10"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$SCRIPT_DIR"
SIBLING_LIBERO_ROOT="$(dirname "$REPO_ROOT")/LIBERO-humanized"
cd "$REPO_ROOT"

case "$SUITE" in
  libero_10)      SUITE_LABEL="10";      BUILDER_SUFFIX="10" ;;
  libero_spatial) SUITE_LABEL="spatial"; BUILDER_SUFFIX="Spatial" ;;
  libero_goal)    SUITE_LABEL="goal";    BUILDER_SUFFIX="Goal" ;;
  libero_object)  SUITE_LABEL="object";  BUILDER_SUFFIX="Object" ;;
  *)
    echo "Unsupported suite: $SUITE"
    exit 1
    ;;
esac

NPZ_BASE_ORIG="$SIBLING_LIBERO_ROOT/scripts/result/original_npz/$SUITE"

case "$MODE" in
  humanized)
    if [[ -z "$METHOD" ]]; then
      echo "[ERROR] MODE=humanized requires a [method] argument (pure-ik|liu-ik|hrr-ik|th-ik|...)"
      exit 1
    fi
    # Method label -> humanized_npz directory suffix produced by
    # A_humanized_libero_suite.py's method_label() naming (see
    # LIBERO-humanized/HUMANIZATION_PIPELINE.md §3-4). Matches the suite
    # script's naming exactly, so no manual comment-toggling needed.
    NPZ_BASE_HUMAN="$SIBLING_LIBERO_ROOT/scripts/result/humanized_npz/${SUITE}_humanized_${METHOD}"

    # Output RLDS subdir under modified_libero_rlds/, keyed by method.
    # Every method (including th-ik/ours) gets its own subdir named after
    # itself so datasets never collide.
    RLDS_SUBDIR="${METHOD//-/_}"

    TASK_ROOTS_DIR="$NPZ_BASE_HUMAN"
    OUTPUT_NAME="${SUITE}_humanized_no_noops"
    BUILDER_DIR="rlds_dataset_builder/LIBERO_${BUILDER_SUFFIX}_humanized"
    TFDS_DATASET_NAME="libero_${SUITE_LABEL}_humanized"
    RLDS_ROOT="$REPO_ROOT/modified_libero_rlds/$RLDS_SUBDIR"
    ;;
  original-joint)
    TASK_ROOTS_DIR="$NPZ_BASE_ORIG"
    OUTPUT_NAME="${SUITE}_joint_no_noops"
    BUILDER_DIR="rlds_dataset_builder/LIBERO_${BUILDER_SUFFIX}_joint"
    TFDS_DATASET_NAME="libero_${SUITE_LABEL}_joint"
    RLDS_ROOT="$REPO_ROOT/modified_libero_rlds/original"
    ;;
  *)
    echo "Unsupported mode: $MODE"
    exit 1
    ;;
esac

OUTPUT_DIR="$REPO_ROOT/LIBERO/libero/datasets/$OUTPUT_NAME"
DEST="$RLDS_ROOT/$OUTPUT_NAME/1.0.0"
TFDS_PREPARED_DIR="$HOME/tensorflow_datasets/$TFDS_DATASET_NAME/1.0.0"

echo "[0/3] Mode=$MODE Suite=$SUITE ${METHOD:+Method=$METHOD }-> $DEST"

echo "[1/3] NPZ -> HDF5"
python experiments/robot/libero/A_npz_to_hdf5.py \
  --task_roots_dir "$TASK_ROOTS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --filter_noops \
  --require_success

echo "[2/3] TFDS build"
rm -rf "$TFDS_PREPARED_DIR"
cd "$REPO_ROOT/$BUILDER_DIR"
CUDA_VISIBLE_DEVICES="" conda run -n openvla-oft tfds build --overwrite

echo "[3/3] Copy latest TFDS shards"
NEWEST=$(ls -td ~/tensorflow_datasets/*/1.0.0 | head -1)
rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$NEWEST"/* "$DEST"/
echo "Copied -> $DEST ($(ls "$DEST"/*.tfrecord* 2>/dev/null | wc -l) shards)"
