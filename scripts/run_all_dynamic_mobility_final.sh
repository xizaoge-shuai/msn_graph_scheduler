#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Stage 1: mobility method comparison"
echo "Started: $(date)"
echo "=========================================="

bash scripts/run_dynamic_mobility_methods.sh

echo
echo "=========================================="
echo "Stage 2: policy and prediction error"
echo "Started: $(date)"
echo "=========================================="

bash scripts/run_full_mobility_policy_error.sh

echo
echo "=========================================="
echo "All final dynamic mobility tests completed"
echo "Finished: $(date)"
echo "=========================================="
