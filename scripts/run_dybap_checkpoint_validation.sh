#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/train_online_cpu.yaml"
PROFILE="profiles/qwen_1.5b_edge_fast_2048.csv"
TRACE="data/processed/azure_2024_val.csv"
CHECKPOINT_DIR="outputs/train_dybap_repro_500"
OUTPUT_DIR="outputs/dybap_val_ckpts"

mkdir -p "$OUTPUT_DIR"

for CKPT in "$CHECKPOINT_DIR"/agent_ep*.pt
do
  NAME=$(basename "$CKPT" .pt)
  EP=${NAME#agent_ep}

  for RATE in 0.10 0.20 0.40 0.60
  do
    TAG=$(echo "$RATE" | tr -d .)
    OUTPUT="$OUTPUT_DIR/ep${EP}_r${TAG}.csv"

    if [ -f "$OUTPUT" ]; then
      echo "Skip existing $OUTPUT"
      continue
    fi

    echo "DyBAP checkpoint=$EP rate=$RATE"

    env \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
    python -u scripts/evaluate_dybap.py \
      --config "$CONFIG" \
      --profile-csv "$PROFILE" \
      --request-trace "$TRACE" \
      --checkpoint "$CKPT" \
      --episodes 20 \
      --arrival-rate-rps "$RATE" \
      --max-requests 12 \
      --max-steps 64 \
      --max-batches 32 \
      --device cpu \
      --log-every 20 \
      --output "$OUTPUT"
  done
done
