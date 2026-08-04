#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

EVAL_CONFIG="configs/train_online_cpu.yaml"
TRACE="data/processed/azure_2024_test_sample.csv"
OUTPUT_DIR="outputs/three_seed_eval_common"

mkdir -p "$OUTPUT_DIR"

for TRAIN_SEED in 7 17 29
do
  case "$TRAIN_SEED" in
    7)
      CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic/agent_ep150.pt"
      ;;
    17)
      CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic_seed17/agent_ep150.pt"
      ;;
    29)
      CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic_seed29/agent_ep150.pt"
      ;;
    *)
      echo "Unsupported seed: $TRAIN_SEED" >&2
      exit 1
      ;;
  esac

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

    OUTPUT="${OUTPUT_DIR}/trainseed${TRAIN_SEED}_r${TAG}_100.csv"

    if [ -f "$OUTPUT" ]; then
      ROWS=$(
        "$PYTHON_BIN" - "$OUTPUT" <<'PY'
import sys
import pandas as pd

print(
    len(
        pd.read_csv(
            sys.argv[1]
        )
    )
)
PY
      )

      if [ "$ROWS" -eq 100 ]; then
        echo "Skip completed: $OUTPUT"
        continue
      fi

      rm -f "$OUTPUT"
    fi

    echo
    echo "=========================================="
    echo "Training seed:   $TRAIN_SEED"
    echo "Evaluation seed: common config seed"
    echo "Arrival rate:    $RATE"
    echo "Checkpoint:      $CHECKPOINT"
    echo "=========================================="

    env \
      CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" -u \
      scripts/evaluate_batchers.py \
      --config "$EVAL_CONFIG" \
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
