#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/train_online_cpu.yaml"
PROFILE="profiles/qwen_1.5b_edge_fast_2048.csv"
TRACE="data/processed/azure_2024_test_final.csv"
CHECKPOINT="outputs/selected_checkpoints/dybap_ep200.pt"

mkdir -p outputs/dybap_final logs

for RATE in 0.10 0.20 0.40 0.60
do
  TAG=$(echo "$RATE" | tr -d .)
  OUTPUT="outputs/dybap_final/dybap_ep200_r${TAG}_100.csv"

  if [ -f "$OUTPUT" ]; then
    echo "Skip existing: $OUTPUT"
    continue
  fi

  echo "=========================================="
  echo "DyBAP ep200, rate=$RATE"
  echo "Started: $(date)"
  echo "=========================================="

  env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
  python -u scripts/evaluate_dybap.py \
    --config "$CONFIG" \
    --profile-csv "$PROFILE" \
    --request-trace "$TRACE" \
    --checkpoint "$CHECKPOINT" \
    --episodes 100 \
    --arrival-rate-rps "$RATE" \
    --max-requests 12 \
    --max-steps 64 \
    --max-batches 32 \
    --device cpu \
    --log-every 20 \
    --output "$OUTPUT"

  echo "Finished rate=$RATE: $(date)"
done
