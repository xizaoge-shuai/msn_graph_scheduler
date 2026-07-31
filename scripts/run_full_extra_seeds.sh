#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

TRACE="data/processed/azure_2023_train_sample.csv"

for SEED in 17 29
do
  CONFIG="configs/train_online_cpu_seed${SEED}.yaml"
  OUTPUT="outputs/train_online_ddqn_cpu_300_analytic_seed${SEED}"

  if [ -f "${OUTPUT}/agent_ep150.pt" ]; then
    echo "Skip completed seed ${SEED}"
    continue
  fi

  echo
  echo "=========================================="
  echo "Training Full seed=${SEED}"
  echo "Output=${OUTPUT}"
  echo "Start=$(date)"
  echo "=========================================="

  env \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
  "$PYTHON_BIN" -u \
    scripts/train.py \
    --config "$CONFIG" \
    --request-trace "$TRACE" \
    --episodes 300 \
    --arrival-rates \
      0.10 \
      0.20 \
      0.40 \
      0.60 \
    --checkpoint-every 50 \
    --device cpu \
    --log-every 10 \
    --output "$OUTPUT"
done
