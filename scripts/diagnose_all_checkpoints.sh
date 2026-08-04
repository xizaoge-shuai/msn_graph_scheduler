#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/train_online_cpu.yaml"
TRACE="data/processed/azure_2023_train_sample.csv"
OUTPUT_DIR="outputs/checkpoint_diagnostic"

mkdir -p \
  "$OUTPUT_DIR" \
  logs

declare -A SEED_DIRS

SEED_DIRS[7]="outputs/train_online_ddqn_cpu_300_analytic"
SEED_DIRS[17]="outputs/train_online_ddqn_cpu_300_analytic_seed17"
SEED_DIRS[29]="outputs/train_online_ddqn_cpu_300_analytic_seed29"

for SEED in 7 17 29
do
  DIR="${SEED_DIRS[$SEED]}"

  mapfile -t CHECKPOINTS < <(
    find "$DIR" \
      -maxdepth 1 \
      -type f \
      -name 'agent_ep*.pt' \
      | sort -V
  )

  if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    echo "No checkpoints in $DIR" >&2
    exit 1
  fi

  for CHECKPOINT in "${CHECKPOINTS[@]}"
  do
    NAME="$(
      basename \
        "$CHECKPOINT" \
        .pt
    )"

    OUTPUT="${OUTPUT_DIR}/seed${SEED}_${NAME}_r040_20.csv"

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

      if [ "$ROWS" -eq 20 ]; then
        echo "Skip completed: $OUTPUT"
        continue
      fi

      rm -f "$OUTPUT"
    fi

    echo
    echo "=========================================="
    echo "Seed:       $SEED"
    echo "Checkpoint: $CHECKPOINT"
    echo "Rate:       0.40"
    echo "Episodes:   20"
    echo "Start:      $(date)"
    echo "=========================================="

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
      --agent-checkpoint "$CHECKPOINT" \
      --device cpu \
      --episodes 20 \
      --batchers node_conditioned_dp \
      --full-queue \
      --arrival-rate-rps 0.40 \
      --max-requests 12 \
      --max-steps 64 \
      --max-batches 32 \
      --log-every 5 \
      --output "$OUTPUT"
  done
done

echo
echo "All checkpoint diagnostic runs completed"
