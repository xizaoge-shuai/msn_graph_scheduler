#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/msn_final_eval_common.yaml"

TRACE="data/processed/azure_2024_final_test.csv"

SELECTED="outputs/selected_checkpoints_2024val.csv"

OUTPUT_DIR="outputs/msn_final/full_three_seed"
LOG_DIR="logs/msn_final/full_three_seed"

EPISODES=100
MAX_PARALLEL=4

mkdir -p \
  "$OUTPUT_DIR" \
  "$LOG_DIR"

run_one() {
  local TRAIN_SEED="$1"
  local CHECKPOINT="$2"
  local RATE="$3"
  local TAG="$4"

  local OUTPUT
  local LOG

  OUTPUT="${OUTPUT_DIR}/seed${TRAIN_SEED}_rate${TAG}.csv"
  LOG="${LOG_DIR}/seed${TRAIN_SEED}_rate${TAG}.log"

  rm -f "$OUTPUT"

  env \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    MKL_DYNAMIC=FALSE \
  "$PYTHON_BIN" -u \
    scripts/evaluate_batchers.py \
    --config "$CONFIG" \
    --request-trace "$TRACE" \
    --device cpu \
    --episodes "$EPISODES" \
    --batchers node_conditioned_dp \
    --agent-checkpoint "$CHECKPOINT" \
    --full-queue \
    --arrival-rate-rps "$RATE" \
    --max-requests 12 \
    --max-steps 64 \
    --max-batches 32 \
    --log-every 10 \
    --output "$OUTPUT" \
    > "$LOG" 2>&1
}


ACTIVE=0

while IFS='|' read -r TRAIN_SEED CHECKPOINT
do
  for RATE_TAG in \
    "0.10|010" \
    "0.20|020" \
    "0.40|040" \
    "0.60|060"
  do
    IFS='|' read -r RATE TAG <<< "$RATE_TAG"

    run_one \
      "$TRAIN_SEED" \
      "$CHECKPOINT" \
      "$RATE" \
      "$TAG" &

    ACTIVE=$((ACTIVE + 1))

    if [ "$ACTIVE" -ge "$MAX_PARALLEL" ]; then
      wait -n
      ACTIVE=$((ACTIVE - 1))
    fi
  done

done < <(
  "$PYTHON_BIN" - "$SELECTED" <<'PY'
import sys
import pandas as pd

frame = pd.read_csv(
    sys.argv[1]
)

for _, row in frame.iterrows():
    print(
        f"{int(row['training_seed'])}"
        f"|{row['checkpoint']}"
    )
PY
)

wait

echo \
  "MSN final Full three-seed evaluation completed."
