#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CONFIG="configs/train_online_cpu.yaml"
OUTPUT_DIR="outputs/dybap_adapted_analytic_final"

EPISODES=100
MAX_REQUESTS=12
MAX_STEPS=64
MAX_BATCHES=32

find_test_trace() {
  local candidate

  for candidate in \
    data/processed/azure_2024_test_final.csv \
    data/processed/azure_2024_test.csv \
    data/processed/azure_2024_test_sample.csv
  do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

TRACE="$(find_test_trace || true)"

if [ ! -f "$CONFIG" ]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

if [ -z "$TRACE" ] || [ ! -f "$TRACE" ]; then
  echo "Cannot locate the final test trace." >&2
  echo "Available files:" >&2
  find data/processed \
    -maxdepth 1 \
    -type f \
    -printf '%f\n' \
    | sort >&2
  exit 1
fi

mkdir -p \
  "$OUTPUT_DIR" \
  logs

echo "=========================================="
echo "DyBAP-Adapted full test"
echo "Profile: analytical model"
echo "Trace:   $TRACE"
echo "Python:  $PYTHON_BIN"
echo "Started: $(date)"
echo "=========================================="

for RATE in \
  0.10 \
  0.20 \
  0.40 \
  0.60
do
  TAG=$(echo "$RATE" | tr -d ".")

  OUTPUT="$OUTPUT_DIR/dybap_adapted_r${TAG}_${EPISODES}.csv"

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
      echo "Complete result exists, skip: $OUTPUT"
      continue
    fi

    echo "Remove incomplete result: $OUTPUT rows=$ROWS"
    rm -f "$OUTPUT"
  fi

  echo
  echo "------------------------------------------"
  echo "Rate:    $RATE"
  echo "Output:  $OUTPUT"
  echo "Started: $(date)"
  echo "------------------------------------------"

  env \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
  "$PYTHON_BIN" -u \
    scripts/evaluate_external_baseline.py \
    --external-mapper dybap_adapted \
    --config "$CONFIG" \
    --request-trace "$TRACE" \
    --device cpu \
    --episodes "$EPISODES" \
    --batchers dybap_fusion \
    --full-queue \
    --arrival-rate-rps "$RATE" \
    --max-requests "$MAX_REQUESTS" \
    --max-steps "$MAX_STEPS" \
    --max-batches "$MAX_BATCHES" \
    --log-every 10 \
    --output "$OUTPUT"

  echo "Finished rate=$RATE: $(date)"
done

echo
echo "=========================================="
echo "All DyBAP-Adapted tests completed"
echo "Finished: $(date)"
echo "=========================================="
