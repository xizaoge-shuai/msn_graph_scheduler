#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/train_online_cpu.yaml"
PROFILE="profiles/qwen_1.5b_edge_fast_2048.csv"
TEST_TRACE="data/processed/azure_2024_test_final.csv"
BEST_CKPT="outputs/train_online_ddqn_cpu_300/agent_ep150.pt"

EPISODES=30
MAX_REQUESTS=12
MAX_STEPS=64
MAX_BATCHES=32

mkdir -p outputs logs

echo "================================================"
echo "Final test started"
echo "Python: ${PYTHON_BIN}"
echo "Time: $(date)"
echo "================================================"

for REQUIRED in \
  "$CONFIG" \
  "$PROFILE" \
  "$TEST_TRACE" \
  "$BEST_CKPT"
do
  if [ ! -f "$REQUIRED" ]; then
    echo "Missing required file: $REQUIRED" >&2
    exit 1
  fi
done

echo
echo "================================================"
echo "Stage 1: frozen baselines"
echo "================================================"

for RATE in 0.10 0.20 0.40 0.60
do
  TAG=$(echo "$RATE" | tr -d '.')

  OUTPUT="outputs/final_test_baselines_r${TAG}_${EPISODES}.csv"

  echo
  echo "------------------------------------------------"
  echo "Baselines: rate=${RATE}"
  echo "Output: ${OUTPUT}"
  echo "Start: $(date)"
  echo "------------------------------------------------"

  "$PYTHON_BIN" -u scripts/evaluate_batchers.py \
    --config "$CONFIG" \
    --profile-csv "$PROFILE" \
    --request-trace "$TEST_TRACE" \
    --episodes "$EPISODES" \
    --batchers \
      no_batch \
      fixed_4 \
      sequential_greedy \
      node_conditioned_dp \
    --full-queue \
    --arrival-rate-rps "$RATE" \
    --max-requests "$MAX_REQUESTS" \
    --max-steps "$MAX_STEPS" \
    --max-batches "$MAX_BATCHES" \
    --log-every 10 \
    --output "$OUTPUT"

  echo "Finished baselines rate=${RATE}: $(date)"
done

echo
echo "================================================"
echo "Stage 2: full method, Node-DP + DDQN ep150"
echo "================================================"

for RATE in 0.10 0.20 0.40 0.60
do
  TAG=$(echo "$RATE" | tr -d '.')

  OUTPUT="outputs/final_test_full_r${TAG}_${EPISODES}.csv"

  echo
  echo "------------------------------------------------"
  echo "Full method: rate=${RATE}"
  echo "Checkpoint: ${BEST_CKPT}"
  echo "Output: ${OUTPUT}"
  echo "Start: $(date)"
  echo "------------------------------------------------"

  "$PYTHON_BIN" -u scripts/evaluate_batchers.py \
    --config "$CONFIG" \
    --profile-csv "$PROFILE" \
    --request-trace "$TEST_TRACE" \
    --agent-checkpoint "$BEST_CKPT" \
    --device cpu \
    --episodes "$EPISODES" \
    --batchers node_conditioned_dp \
    --full-queue \
    --arrival-rate-rps "$RATE" \
    --max-requests "$MAX_REQUESTS" \
    --max-steps "$MAX_STEPS" \
    --max-batches "$MAX_BATCHES" \
    --log-every 10 \
    --output "$OUTPUT"

  echo "Finished full method rate=${RATE}: $(date)"
done

echo
echo "================================================"
echo "All final-test runs completed"
echo "Time: $(date)"
echo "================================================"
