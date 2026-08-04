#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/train_online_cpu.yaml"
PROFILE="profiles/qwen_1.5b_edge_fast_2048.csv"
TRACE="data/processed/azure_2024_test_final.csv"

EPISODES=100

mkdir -p outputs logs

run_method() {
  local METHOD="$1"
  local BATCHER="$2"
  local PREFIX="$3"

  for RATE in 0.10 0.20 0.40 0.60
  do
    TAG=$(echo "$RATE" | tr -d ".")

    OUTPUT="outputs/${PREFIX}_r${TAG}_${EPISODES}.csv"

    if [ -f "$OUTPUT" ]; then
      echo "Existing result, skip: $OUTPUT"
      continue
    fi

    echo
    echo "=============================================="
    echo "Method=${METHOD}"
    echo "Batcher=${BATCHER}"
    echo "Rate=${RATE}"
    echo "Started=$(date)"
    echo "=============================================="

    env \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 \
    "$PYTHON_BIN" -u \
      scripts/evaluate_external_baseline.py \
      --external-mapper "$METHOD" \
      --config "$CONFIG" \
      --profile-csv "$PROFILE" \
      --request-trace "$TRACE" \
      --episodes "$EPISODES" \
      --batchers "$BATCHER" \
      --full-queue \
      --arrival-rate-rps "$RATE" \
      --max-requests 12 \
      --max-steps 64 \
      --max-batches 32 \
      --log-every 20 \
      --output "$OUTPUT"

    echo "Finished ${METHOD} rate=${RATE}: $(date)"
  done
}

run_method \
  rba \
  node_conditioned_dp \
  final_test_lecu_rba

run_method \
  dybap_core \
  sequential_greedy \
  final_test_dybap_core

echo
echo "External baseline final tests completed: $(date)"
