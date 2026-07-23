#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m pytest -q
python scripts/motivation_batch_node.py
python scripts/motivation_group_length.py
python scripts/train.py --episodes 8 --output outputs/train_smoke
python scripts/evaluate.py --checkpoint outputs/train_smoke/agent_final.pt --episodes 2 --output outputs/eval_smoke.csv
python scripts/evaluate_batchers.py --episodes 2 --output outputs/eval_batchers_smoke.csv
