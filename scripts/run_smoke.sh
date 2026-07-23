#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "========================================"
echo "1. Running unit tests"
echo "========================================"
python -m pytest -q

echo
echo "========================================"
echo "2. Running environment smoke test"
echo "========================================"
python -u scripts/train.py \
  --episodes 3 \
  --device cpu \
  --no-update \
  --log-every 1 \
  --max-steps-per-episode 64 \
  --output outputs/train_smoke

echo
echo "========================================"
echo "3. Evaluating generated checkpoint"
echo "========================================"
python -u scripts/evaluate.py \
  --checkpoint outputs/train_smoke/agent_final.pt \
  --episodes 2 \
  --output outputs/eval_smoke.csv

echo
echo "========================================"
echo "4. Evaluating batching baselines"
echo "========================================"
python -u scripts/evaluate_batchers.py \
  --episodes 2 \
  --output outputs/eval_batchers_smoke.csv

echo
echo "Smoke test completed."
