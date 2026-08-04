#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

CONFIG="configs/train_online_cpu.yaml"
TRACE="data/processed/azure_2024_test_sample.csv"
CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic/agent_ep150.pt"

OUTPUT_DIR="outputs/dynamic_mobility_methods"
EPISODES=100
RATE=0.40

mkdir -p \
  "$OUTPUT_DIR" \
  logs

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

for METHOD in \
  node_dp \
  lecu_rba \
  dybap_adapted \
  full
do
  for MODE in \
    stationary \
    walking \
    driving \
    high_speed
  do
    OUTPUT="${OUTPUT_DIR}/${METHOD}_${MODE}_r040_${EPISODES}.csv"

    if [ -f "$OUTPUT" ]; then
      ROWS="$(row_count "$OUTPUT")"

      if [ "$ROWS" -eq "$EPISODES" ]; then
        echo "Complete, skip: $OUTPUT"
        continue
      fi

      rm -f "$OUTPUT"
    fi

    EXTRA_ARGS=()

    if [ "$METHOD" = "full" ]; then
      EXTRA_ARGS+=(
        --agent-checkpoint
        "$CHECKPOINT"
      )
    fi

    echo
    echo "=========================================="
    echo "Method:   $METHOD"
    echo "Mobility: $MODE"
    echo "Rate:     $RATE"
    echo "Started:  $(date)"
    echo "=========================================="

    env \
      CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" -u \
      scripts/evaluate_dynamic_mobility.py \
      --config "$CONFIG" \
      --request-trace "$TRACE" \
      --method "$METHOD" \
      "${EXTRA_ARGS[@]}" \
      --mobility-mode "$MODE" \
      --migration-policy keep \
      --prediction-error 0.10 \
      --arrival-rate-rps "$RATE" \
      --episodes "$EPISODES" \
      --max-requests 12 \
      --max-steps 64 \
      --max-batches 32 \
      --device cpu \
      --log-every 10 \
      --output "$OUTPUT"
  done
done

echo
echo "All dynamic mobility method tests completed"
echo "Finished: $(date)"
