#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/train_online_cpu.yaml"
TRACE="data/processed/azure_2024_test_sample.csv"
OUTPUT_DIR="outputs/analytic_batcher_baselines"

EPISODES=100

mkdir -p \
  "$OUTPUT_DIR" \
  logs

for RATE in \
  0.10 \
  0.20 \
  0.40 \
  0.60
do
  TAG=$(echo "$RATE" | tr -d ".")
  OUTPUT="$OUTPUT_DIR/baselines_r${TAG}_${EPISODES}.csv"

  if [ -f "$OUTPUT" ]; then
    ROWS=$(
      "$PYTHON_BIN" - "$OUTPUT" <<'PY'
import sys
import pandas as pd

try:
    print(len(pd.read_csv(sys.argv[1])))
except Exception:
    print(0)
PY
    )

    if [ "$ROWS" -eq 400 ]; then
      echo "Complete result exists, skip: $OUTPUT"
      continue
    fi

    rm -f "$OUTPUT"
  fi

  echo
  echo "=========================================="
  echo "Analytical baselines, rate=$RATE"
  echo "Started: $(date)"
  echo "=========================================="

  env \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
  "$PYTHON_BIN" -u \
    scripts/evaluate_batchers.py \
    --config "$CONFIG" \
    --request-trace "$TRACE" \
    --device cpu \
    --episodes "$EPISODES" \
    --batchers \
      no_batch \
      fixed_4 \
      sequential_greedy \
      node_conditioned_dp \
    --full-queue \
    --arrival-rate-rps "$RATE" \
    --max-requests 12 \
    --max-steps 64 \
    --max-batches 32 \
    --log-every 10 \
    --output "$OUTPUT"
done

echo
echo "All analytical baselines completed: $(date)"
