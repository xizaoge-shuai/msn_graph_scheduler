#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/train_online_cpu.yaml"
PROFILE="profiles/qwen_1.5b_edge_fast_2048.csv"
TEST_TRACE="data/processed/azure_2024_test_final.csv"

EPISODES=100

declare -A CHECKPOINTS

CHECKPOINTS[seed2]="outputs/train_online_ddqn_cpu_seed2_300/agent_ep50.pt"
CHECKPOINTS[seed3]="outputs/train_online_ddqn_cpu_seed3_300/agent_ep100.pt"

mkdir -p outputs logs

echo "Seed2/Seed3 final test started: $(date)"
echo "Common evaluation config: ${CONFIG}"
echo "Python: ${PYTHON_BIN}"

for SEED_TAG in seed2 seed3
do
  CKPT="${CHECKPOINTS[$SEED_TAG]}"

  if [ ! -f "$CKPT" ]; then
    echo "Missing checkpoint: $CKPT" >&2
    exit 1
  fi

  for RATE in 0.10 0.20 0.40 0.60
  do
    TAG=$(echo "$RATE" | tr -d ".")

    OUTPUT="outputs/final_test_full_${SEED_TAG}_r${TAG}_${EPISODES}.csv"

    if [ -f "$OUTPUT" ]; then
      echo "Existing result, skip: $OUTPUT"
      continue
    fi

    echo
    echo "================================================"
    echo "${SEED_TAG}: rate=${RATE}, episodes=${EPISODES}"
    echo "Checkpoint: ${CKPT}"
    echo "Started: $(date)"
    echo "================================================"

    env \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 \
    "$PYTHON_BIN" -u scripts/evaluate_batchers.py \
      --config "$CONFIG" \
      --profile-csv "$PROFILE" \
      --request-trace "$TEST_TRACE" \
      --agent-checkpoint "$CKPT" \
      --device cpu \
      --episodes "$EPISODES" \
      --batchers node_conditioned_dp \
      --full-queue \
      --arrival-rate-rps "$RATE" \
      --max-requests 12 \
      --max-steps 64 \
      --max-batches 32 \
      --log-every 20 \
      --output "$OUTPUT"

    echo "Finished ${SEED_TAG} rate=${RATE}: $(date)"
  done
done

echo
echo "Seed2/Seed3 final test completed: $(date)"
