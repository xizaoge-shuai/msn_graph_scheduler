#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

TRACE="data/processed/azure_2023_train_sample.csv"
CONFIG="configs/checkpoint_validation.yaml"

OUTPUT_DIR="outputs/checkpoint_validation"
LOG_DIR="logs/checkpoint_validation"

EPISODES=20
RATE=0.40
MAX_PARALLEL=4

mkdir -p \
  "$OUTPUT_DIR" \
  "$LOG_DIR"

find_seed_dir() {
  local SEED="$1"

  if [ "$SEED" = "7" ]; then
    echo \
      "outputs/train_online_ddqn_cpu_300_analytic"
    return
  fi

  find outputs \
    -maxdepth 1 \
    -type d \
    -iname "*seed${SEED}*" \
    | grep -E \
      "train_online|ddqn" \
    | head -n 1
}

run_one() {
  local TRAIN_SEED="$1"
  local CHECKPOINT="$2"
  local EP="$3"

  local OUTPUT
  local LOG

  OUTPUT="${OUTPUT_DIR}/seed${TRAIN_SEED}_ep${EP}.csv"
  LOG="${LOG_DIR}/seed${TRAIN_SEED}_ep${EP}.log"

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

    if [ "$ROWS" -eq "$EPISODES" ]; then
      echo "Skip completed: $OUTPUT"
      return
    fi
  fi

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
    --log-every 5 \
    --output "$OUTPUT" \
    > "$LOG" 2>&1
}

ACTIVE=0

for TRAIN_SEED in 7 17 29
do
  DIR="$(find_seed_dir "$TRAIN_SEED")"

  if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
    echo \
      "Cannot find checkpoint directory " \
      "for seed ${TRAIN_SEED}" >&2
    exit 1
  fi

  echo \
    "Seed ${TRAIN_SEED}: ${DIR}"

  for EP in 50 100 150 200 250 300
  do
    CHECKPOINT="${DIR}/agent_ep${EP}.pt"

    if [ ! -f "$CHECKPOINT" ]; then
      echo \
        "Missing checkpoint: $CHECKPOINT"
      continue
    fi

    run_one \
      "$TRAIN_SEED" \
      "$CHECKPOINT" \
      "$EP" &

    ACTIVE=$((ACTIVE + 1))

    if [ "$ACTIVE" -ge "$MAX_PARALLEL" ]; then
      wait -n
      ACTIVE=$((ACTIVE - 1))
    fi
  done
done

wait

echo \
  "Checkpoint validation completed."
