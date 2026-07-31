#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

TRACE="data/processed/azure_2024_test_sample.csv"

OUTPUT_DIR="outputs/paired_mobility_ablation_highspeed"
CONFIG_DIR="configs/paired_mobility_ablation_highspeed"
LOG_DIR="logs/paired_mobility_ablation_highspeed"

EPISODES_PER_SEED=10
RATE=0.40
MAX_PARALLEL=2

mkdir -p \
  "$OUTPUT_DIR" \
  "$CONFIG_DIR" \
  "$LOG_DIR"

create_config() {
  local SOURCE="$1"
  local OUTPUT="$2"
  local SEED="$3"

  "$PYTHON_BIN" - \
    "$SOURCE" \
    "$OUTPUT" \
    "$SEED" <<'PY'
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1])
output = Path(sys.argv[2])
seed = int(sys.argv[3])

cfg = yaml.safe_load(
    source.read_text(
        encoding="utf-8"
    )
)

cfg["seed"] = seed

output.write_text(
    yaml.safe_dump(
        cfg,
        sort_keys=False,
    ),
    encoding="utf-8",
)
PY
}

run_one() {
  local METHOD="$1"
  local EVAL_SEED="$2"

  local SOURCE_CONFIG
  local CHECKPOINT

  case "$METHOD" in
    full)
      SOURCE_CONFIG="configs/train_online_cpu.yaml"
      CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic/agent_ep150.pt"
      ;;

    wo_mobility)
      SOURCE_CONFIG="configs/ablation_wo_mobility_seed7.yaml"
      CHECKPOINT="outputs/train_ablation_wo_mobility_seed7/agent_ep150.pt"
      ;;

    *)
      echo "Unknown method: $METHOD" >&2
      exit 1
      ;;
  esac

  CONFIG="${CONFIG_DIR}/${METHOD}_seed${EVAL_SEED}.yaml"
  OUTPUT="${OUTPUT_DIR}/${METHOD}_seed${EVAL_SEED}_n${EPISODES_PER_SEED}.csv"
  LOG="${LOG_DIR}/${METHOD}_seed${EVAL_SEED}.log"

  create_config \
    "$SOURCE_CONFIG" \
    "$CONFIG" \
    "$EVAL_SEED"

  if [ -f "$OUTPUT" ]; then
    ROWS=$(
      "$PYTHON_BIN" - "$OUTPUT" <<'PY'
import sys
import pandas as pd

try:
    print(
        len(
            pd.read_csv(
                sys.argv[1]
            )
        )
    )
except Exception:
    print(0)
PY
    )

    if [ "$ROWS" -eq "$EPISODES_PER_SEED" ]; then
      echo "Skip completed: $OUTPUT"
      return
    fi

    rm -f "$OUTPUT"
  fi

  echo
  echo "=========================================="
  echo "Method:    $METHOD"
  echo "Eval seed: $EVAL_SEED"
  echo "Mode:      high_speed"
  echo "Rate:      $RATE"
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
    scripts/evaluate_dynamic_mobility.py \
    --config "$CONFIG" \
    --request-trace "$TRACE" \
    --method full \
    --agent-checkpoint "$CHECKPOINT" \
    --mobility-mode high_speed \
    --migration-policy keep \
    --prediction-error 0.10 \
    --arrival-rate-rps "$RATE" \
    --episodes "$EPISODES_PER_SEED" \
    --max-requests 12 \
    --max-steps 64 \
    --max-batches 32 \
    --device cpu \
    --log-every 2 \
    --output "$OUTPUT" \
    > "$LOG" 2>&1
}

ACTIVE=0

for EVAL_SEED in \
  8101 \
  8102 \
  8103
do
  for METHOD in \
    full \
    wo_mobility
  do
    run_one \
      "$METHOD" \
      "$EVAL_SEED" &

    ACTIVE=$((ACTIVE + 1))

    if [ "$ACTIVE" -ge "$MAX_PARALLEL" ]; then
      wait -n
      ACTIVE=$((ACTIVE - 1))
    fi
  done
done

wait

echo
echo "All mobility-ablation runs completed"
