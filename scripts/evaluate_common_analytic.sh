#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/train_online_cpu.yaml"
TRACE="data/processed/azure_2024_test_sample.csv"
CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic/agent_ep150.pt"

OUTPUT_DIR="outputs/common_analytic"

EPISODES=100
MAX_REQUESTS=12
MAX_STEPS=64
MAX_BATCHES=32

for REQUIRED in \
  "$CONFIG" \
  "$TRACE" \
  "$CHECKPOINT"
do
  if [ ! -f "$REQUIRED" ]; then
    echo "Missing required file: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p \
  "$OUTPUT_DIR" \
  logs

row_count() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import pandas as pd

try:
    print(len(pd.read_csv(sys.argv[1])))
except Exception:
    print(0)
PY
}

run_standard_baselines() {
  local rate="$1"
  local tag="$2"
  local output="$OUTPUT_DIR/baselines_r${tag}_${EPISODES}.csv"

  if [ -f "$output" ]; then
    local rows
    rows="$(row_count "$output")"

    if [ "$rows" -eq 400 ]; then
      echo "Complete, skip: $output"
      return
    fi

    rm -f "$output"
  fi

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
    --arrival-rate-rps "$rate" \
    --max-requests "$MAX_REQUESTS" \
    --max-steps "$MAX_STEPS" \
    --max-batches "$MAX_BATCHES" \
    --log-every 10 \
    --output "$output"
}

run_lecu_rba() {
  local rate="$1"
  local tag="$2"
  local output="$OUTPUT_DIR/lecu_rba_r${tag}_${EPISODES}.csv"

  if [ -f "$output" ]; then
    local rows
    rows="$(row_count "$output")"

    if [ "$rows" -eq 100 ]; then
      echo "Complete, skip: $output"
      return
    fi

    rm -f "$output"
  fi

  env \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
  "$PYTHON_BIN" -u \
    scripts/evaluate_external_baseline.py \
    --external-mapper rba \
    --config "$CONFIG" \
    --request-trace "$TRACE" \
    --device cpu \
    --episodes "$EPISODES" \
    --batchers node_conditioned_dp \
    --full-queue \
    --arrival-rate-rps "$rate" \
    --max-requests "$MAX_REQUESTS" \
    --max-steps "$MAX_STEPS" \
    --max-batches "$MAX_BATCHES" \
    --log-every 10 \
    --output "$output"
}

run_full() {
  local rate="$1"
  local tag="$2"
  local output="$OUTPUT_DIR/full_r${tag}_${EPISODES}.csv"

  if [ -f "$output" ]; then
    local rows
    rows="$(row_count "$output")"

    if [ "$rows" -eq 100 ]; then
      echo "Complete, skip: $output"
      return
    fi

    rm -f "$output"
  fi

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
    --episodes "$EPISODES" \
    --batchers node_conditioned_dp \
    --full-queue \
    --arrival-rate-rps "$rate" \
    --max-requests "$MAX_REQUESTS" \
    --max-steps "$MAX_STEPS" \
    --max-batches "$MAX_BATCHES" \
    --log-every 10 \
    --output "$output"
}

echo "=========================================="
echo "Common analytical-profile evaluation"
echo "Trace:      $TRACE"
echo "Checkpoint: $CHECKPOINT"
echo "Episodes:   $EPISODES per rate"
echo "Started:    $(date)"
echo "=========================================="

for RATE in \
  0.10 \
  0.20 \
  0.40 \
  0.60
do
  TAG=$(echo "$RATE" | tr -d ".")

  echo
  echo "=========================================="
  echo "Arrival rate: $RATE"
  echo "Started:      $(date)"
  echo "=========================================="

  run_standard_baselines \
    "$RATE" \
    "$TAG"

  run_lecu_rba \
    "$RATE" \
    "$TAG"

  run_full \
    "$RATE" \
    "$TAG"
done

echo
echo "All common analytical tests completed"
echo "Finished: $(date)"
