#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="reproductions/lecu/config.yaml"
CHECKPOINT_DIR="outputs/train_lecu_repro_1500"
OUTPUT_DIR="outputs/lecu_val_ckpts"

mkdir -p "$OUTPUT_DIR"

for CKPT in "$CHECKPOINT_DIR"/agent_ep*.pt
do
  NAME=$(basename "$CKPT" .pt)
  EP=${NAME#agent_ep}
  OUTPUT="$OUTPUT_DIR/ep${EP}.csv"

  if [ -f "$OUTPUT" ]; then
    echo "Skip existing $OUTPUT"
    continue
  fi

  echo "LECU checkpoint=$EP"

  env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
  python -u reproductions/lecu/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --episodes 30 \
    --device cpu \
    --output "$OUTPUT"
done
