#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/train_online_cpu.yaml"
TRACE="data/processed/azure_2024_test_sample.csv"
CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic/agent_ep150.pt"

OUTPUT_DIR="outputs/full_mobility_policy_error"
EPISODES=100
RATE=0.40

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

error_tag() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys

value = float(sys.argv[1])

print(
    f"{int(round(value * 100)):02d}"
)
PY
}

echo "=========================================="
echo "Full mobility-policy-error experiment"
echo "Profile:    analytical"
echo "Trace:      $TRACE"
echo "Checkpoint: $CHECKPOINT"
echo "Rate:       $RATE"
echo "Episodes:   $EPISODES"
echo "Started:    $(date)"
echo "=========================================="

for MODE in \
  driving \
  high_speed
do
  for POLICY in \
    keep \
    reactive \
    prefetch
  do
    for ERROR in \
      0.0 \
      0.1 \
      0.2 \
      0.3
    do
      ERROR_TAG="$(error_tag "$ERROR")"

      OUTPUT="${OUTPUT_DIR}/full_${MODE}_${POLICY}_err${ERROR_TAG}_r040_${EPISODES}.csv"

      if [ -f "$OUTPUT" ]; then
        ROWS="$(row_count "$OUTPUT")"

        if [ "$ROWS" -eq "$EPISODES" ]; then
          echo "Complete, skip: $OUTPUT"
          continue
        fi

        echo "Remove incomplete file: $OUTPUT rows=$ROWS"
        rm -f "$OUTPUT"
      fi

      echo
      echo "------------------------------------------"
      echo "Mode:             $MODE"
      echo "Policy:           $POLICY"
      echo "Prediction error: $ERROR"
      echo "Output:           $OUTPUT"
      echo "Started:          $(date)"
      echo "------------------------------------------"

      env \
        CUDA_VISIBLE_DEVICES="" \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 \
      "$PYTHON_BIN" -u \
        scripts/evaluate_dynamic_mobility.py \
        --config "$CONFIG" \
        --request-trace "$TRACE" \
        --method full \
        --agent-checkpoint "$CHECKPOINT" \
        --mobility-mode "$MODE" \
        --migration-policy "$POLICY" \
        --prediction-error "$ERROR" \
        --arrival-rate-rps "$RATE" \
        --episodes "$EPISODES" \
        --max-requests "$MAX_REQUESTS" \
        --max-steps "$MAX_STEPS" \
        --max-batches "$MAX_BATCHES" \
        --device cpu \
        --log-every 10 \
        --output "$OUTPUT"

      echo "Finished: $(date)"
    done
  done
done

echo
echo "=========================================="
echo "All 24 settings completed"
echo "Finished: $(date)"
echo "=========================================="
