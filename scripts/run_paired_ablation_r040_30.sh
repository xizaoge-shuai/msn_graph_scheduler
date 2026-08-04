#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

TRACE="data/processed/azure_2024_test_sample.csv"
OUTPUT_DIR="outputs/paired_ablation_r040_30"
CONFIG_DIR="configs/paired_ablation_eval"
LOG_DIR="logs/paired_ablation_r040_30"

EPISODES_PER_SEED=10
RATE=0.40
MAX_PARALLEL=4

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

print(
    f"Created {output}, seed={seed}"
)
PY
}

run_one() {
  local METHOD="$1"
  local EVAL_SEED="$2"

  local SOURCE_CONFIG
  local CONFIG
  local SCRIPT
  local CHECKPOINT
  local OUTPUT
  local LOG

  case "$METHOD" in
    full)
      SOURCE_CONFIG="configs/train_online_cpu.yaml"
      SCRIPT="scripts/evaluate_batchers.py"
      CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic/agent_ep150.pt"
      ;;

    wo_mobility)
      SOURCE_CONFIG="configs/ablation_wo_mobility_seed7.yaml"
      SCRIPT="scripts/evaluate_batchers.py"
      CHECKPOINT="outputs/train_ablation_wo_mobility_seed7/agent_ep150.pt"
      ;;

    wo_dp)
      SOURCE_CONFIG="configs/train_online_cpu.yaml"
      SCRIPT="scripts/evaluate_seqgreedy_ddqn.py"
      CHECKPOINT="outputs/train_ablation_wo_dp_seed7/agent_ep150.pt"
      ;;

    *)
      echo "Unknown method: $METHOD" >&2
      return 1
      ;;
  esac

  CONFIG="${CONFIG_DIR}/${METHOD}_evalseed${EVAL_SEED}.yaml"
  OUTPUT="${OUTPUT_DIR}/${METHOD}_evalseed${EVAL_SEED}_n${EPISODES_PER_SEED}.csv"
  LOG="${LOG_DIR}/${METHOD}_evalseed${EVAL_SEED}.log"

  create_config \
    "$SOURCE_CONFIG" \
    "$CONFIG" \
    "$EVAL_SEED"

  if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    return 1
  fi

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
      return 0
    fi

    rm -f "$OUTPUT"
  fi

  echo
  echo "=================================================="
  echo "Method:          $METHOD"
  echo "Evaluation seed: $EVAL_SEED"
  echo "Episodes:        $EPISODES_PER_SEED"
  echo "Output:          $OUTPUT"
  echo "Start:           $(date)"
  echo "=================================================="

  env \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    MKL_DYNAMIC=FALSE \
  "$PYTHON_BIN" -u \
    "$SCRIPT" \
    --config "$CONFIG" \
    --request-trace "$TRACE" \
    --agent-checkpoint "$CHECKPOINT" \
    --device cpu \
    --episodes "$EPISODES_PER_SEED" \
    --batchers node_conditioned_dp \
    --full-queue \
    --arrival-rate-rps "$RATE" \
    --max-requests 12 \
    --max-steps 64 \
    --max-batches 32 \
    --log-every 2 \
    --output "$OUTPUT" \
    > "$LOG" 2>&1

  echo "Finished: method=$METHOD seed=$EVAL_SEED time=$(date)"
}

ACTIVE=0

# 相同 eval seed 下，三个方法使用相同测试实例。
for EVAL_SEED in \
  7001 \
  7002 \
  7003
do
  for METHOD in \
    full \
    wo_mobility \
    wo_dp
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
echo "All paired ablation runs completed"
