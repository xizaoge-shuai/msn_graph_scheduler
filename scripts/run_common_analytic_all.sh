#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/train_full_analytic.sh
bash scripts/evaluate_common_analytic.sh
