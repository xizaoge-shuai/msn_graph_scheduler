#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

TRACE="data/processed/azure_2024_test_sample.csv"
OUTPUT_DIR="outputs/three_seed_eval"

mkdir -p "$OUTPUT_DIR"

for SEED in 7 17 29
do
  if [ "$SEED" -eq 7 ]; then
    CONFIG="configs/train_online_cpu.yaml"
    CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic/agent_ep150.pt"
  else
    CONFIG="configs/train_online_cpu_seed${SEED}.yaml"
    CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic_seed${SEED}/agent_ep150.pt"
  fi

  if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
  fi

  for RATE in \
    0.10 \
    0.20 \
    0.40 \
    0.60
  do
    TAG=$(echo "$RATE" | tr -d '.')

    OUTPUT="${OUTPUT_DIR}/seed${SEED}_r${TAG}_100.csv"

    if [ -f "$OUTPUT" ]; then
      ROWS=$(
        "$PYTHON_BIN" - "$OUTPUT" <<'PY'
import sys
import pandas as pd
print(len(pd.read_csv(sys.argv[1])))
PY
      )

      if [ "$ROWS" -eq 100 ]; then
        echo "Skip complete: $OUTPUT"
        continue
      fi
    fi

    echo
    echo "=========================================="
    echo "Full seed=${SEED}, rate=${RATE}"
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
      --agent-checkpoint "$CHECKPOINT" \
      --device cpu \
      --episodes 100 \
      --batchers node_conditioned_dp \
      --full-queue \
      --arrival-rate-rps "$RATE" \
      --max-requests 12 \
      --max-steps 64 \
      --max-batches 32 \
      --log-every 10 \
      --output "$OUTPUT"
  done
done
