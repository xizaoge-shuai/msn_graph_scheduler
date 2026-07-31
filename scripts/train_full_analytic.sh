#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/train_online_cpu.yaml"
TRACE="data/processed/azure_2023_train_sample.csv"
OUTPUT="outputs/train_online_ddqn_cpu_300_analytic"

CHECKPOINT="$OUTPUT/agent_ep150.pt"

if [ -f "$CHECKPOINT" ]; then
  echo "Full checkpoint already exists:"
  echo "$CHECKPOINT"
  exit 0
fi

mkdir -p \
  "$OUTPUT" \
  logs

env \
  CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
python -u \
  scripts/train.py \
  --config "$CONFIG" \
  --request-trace "$TRACE" \
  --episodes 300 \
  --device cpu \
  --arrival-rates \
    0.10 \
    0.20 \
    0.40 \
    0.60 \
  --max-requests 12 \
  --max-batches-per-episode 32 \
  --max-steps-per-batch 64 \
  --checkpoint-every 50 \
  --log-every 10 \
  --output "$OUTPUT"

test -f "$CHECKPOINT"

echo "Full checkpoint ready:"
echo "$CHECKPOINT"
