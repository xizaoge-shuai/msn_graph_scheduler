#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

BASE_CONFIG="configs/train_online_cpu.yaml"

CHECKPOINT="outputs/train_online_ddqn_cpu_300_analytic/agent_ep150.pt"

TRACE="data/processed/azure_2024_test_sample.csv"

CONFIG_DIR="configs/msn_scalability"
OUTPUT_DIR="outputs/msn_scalability"
LOG_DIR="logs/msn_scalability"

EPISODES=10
RATE=0.60
MAX_PARALLEL=3

mkdir -p \
  "$CONFIG_DIR" \
  "$OUTPUT_DIR" \
  "$LOG_DIR"

create_config() {
  local OUTPUT="$1"
  local NODES="$2"
  local QUEUE="$3"
  local SEED="$4"

  "$PYTHON_BIN" - \
    "$BASE_CONFIG" \
    "$OUTPUT" \
    "$NODES" \
    "$QUEUE" \
    "$SEED" <<'PY'
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1])
output = Path(sys.argv[2])

nodes = int(sys.argv[3])
queue = int(sys.argv[4])
seed = int(sys.argv[5])

cfg = yaml.safe_load(
    source.read_text(
        encoding="utf-8"
    )
)

cfg["seed"] = seed

cfg.setdefault(
    "system",
    {},
)

cfg["system"][
    "num_edge_nodes"
] = nodes

cfg.setdefault(
    "requests",
    {},
)

cfg["requests"][
    "queue_size"
] = queue

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
  local NAME="$1"
  local NODES="$2"
  local QUEUE="$3"
  local MAX_REQUESTS="$4"
  local SEED="$5"

  local CONFIG
  local OUTPUT
  local LOG
  local TIME_FILE

  CONFIG="${CONFIG_DIR}/${NAME}.yaml"
  OUTPUT="${OUTPUT_DIR}/${NAME}.csv"
  LOG="${LOG_DIR}/${NAME}.log"
  TIME_FILE="${OUTPUT_DIR}/${NAME}.time"

  create_config \
    "$CONFIG" \
    "$NODES" \
    "$QUEUE" \
    "$SEED"

  rm -f \
    "$OUTPUT" \
    "$TIME_FILE"

  env \
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    MKL_DYNAMIC=FALSE \
  /usr/bin/time \
    -f "elapsed_sec=%e max_rss_kb=%M" \
    -o "$TIME_FILE" \
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
    --max-requests "$MAX_REQUESTS" \
    --max-steps 96 \
    --max-batches 64 \
    --log-every 2 \
    --output "$OUTPUT" \
    > "$LOG" 2>&1
}

ACTIVE=0

# Node scalability: queue fixed at 24.
for NODES in 4 6 8 10
do
  NAME="nodes${NODES}_queue24"

  run_one \
    "$NAME" \
    "$NODES" \
    24 \
    24 \
    $((8500 + NODES)) &

  ACTIVE=$((ACTIVE + 1))

  if [ "$ACTIVE" -ge "$MAX_PARALLEL" ]; then
    wait -n
    ACTIVE=$((ACTIVE - 1))
  fi
done

# Queue scalability: nodes fixed at 6.
for QUEUE in 8 16 24 32 48
do
  NAME="nodes6_queue${QUEUE}"

  run_one \
    "$NAME" \
    6 \
    "$QUEUE" \
    "$QUEUE" \
    $((8600 + QUEUE)) &

  ACTIVE=$((ACTIVE + 1))

  if [ "$ACTIVE" -ge "$MAX_PARALLEL" ]; then
    wait -n
    ACTIVE=$((ACTIVE - 1))
  fi
done

wait

echo \
  "MSN scalability evaluation completed."
