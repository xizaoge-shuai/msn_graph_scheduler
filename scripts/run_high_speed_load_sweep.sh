#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

TRACE="data/processed/azure_2024_test_sample.csv"
OUTPUT_DIR="outputs/high_speed_load_sweep"

mkdir -p "$OUTPUT_DIR"

for RATE in \
  0.20 \
  0.40 \
  0.60
do
  TAG=$(echo "$RATE" | tr -d '.')

  "$PYTHON_BIN" -u \
    scripts/evaluate_dynamic_mobility.py \
    --config configs/train_online_cpu.yaml \
    --request-trace "$TRACE" \
    --method full \
    --agent-checkpoint \
      outputs/train_online_ddqn_cpu_300_analytic/agent_ep150.pt \
    --mobility-mode high_speed \
    --migration-policy keep \
    --prediction-error 0.10 \
    --arrival-rate-rps "$RATE" \
    --episodes 100 \
    --max-requests 12 \
    --max-steps 64 \
    --max-batches 32 \
    --device cpu \
    --log-every 10 \
    --output \
      "${OUTPUT_DIR}/full_r${TAG}.csv"

  "$PYTHON_BIN" -u \
    scripts/evaluate_dynamic_mobility.py \
    --config configs/train_online_cpu.yaml \
    --request-trace "$TRACE" \
    --method node_dp \
    --mobility-mode high_speed \
    --migration-policy keep \
    --prediction-error 0.10 \
    --arrival-rate-rps "$RATE" \
    --episodes 100 \
    --max-requests 12 \
    --max-steps 64 \
    --max-batches 32 \
    --device cpu \
    --log-every 10 \
    --output \
      "${OUTPUT_DIR}/wo_ddqn_r${TAG}.csv"

  "$PYTHON_BIN" -u \
    scripts/evaluate_dynamic_mobility.py \
    --config \
      configs/ablation_wo_mobility_seed7.yaml \
    --request-trace "$TRACE" \
    --method full \
    --agent-checkpoint \
      outputs/train_ablation_wo_mobility_seed7/agent_ep150.pt \
    --mobility-mode high_speed \
    --migration-policy keep \
    --prediction-error 0.10 \
    --arrival-rate-rps "$RATE" \
    --episodes 100 \
    --max-requests 12 \
    --max-steps 64 \
    --max-batches 32 \
    --device cpu \
    --log-every 10 \
    --output \
      "${OUTPUT_DIR}/wo_mobility_r${TAG}.csv"

  "$PYTHON_BIN" -u \
    scripts/evaluate_dynamic_mobility.py \
    --config configs/train_online_cpu.yaml \
    --request-trace "$TRACE" \
    --method dybap_adapted \
    --mobility-mode high_speed \
    --migration-policy keep \
    --prediction-error 0.10 \
    --arrival-rate-rps "$RATE" \
    --episodes 100 \
    --max-requests 12 \
    --max-steps 64 \
    --max-batches 32 \
    --device cpu \
    --log-every 10 \
    --output \
      "${OUTPUT_DIR}/dybap_r${TAG}.csv"
done
