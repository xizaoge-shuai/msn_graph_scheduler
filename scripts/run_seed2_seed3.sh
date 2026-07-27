#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

PROFILE="profiles/qwen_1.5b_edge_fast_2048.csv"
TRAIN_TRACE="data/processed/azure_2024_train.csv"
VAL_TRACE="data/processed/azure_2024_val.csv"

TRAIN_EPISODES=300
VAL_EPISODES=10

mkdir -p outputs logs

echo "================================================"
echo "Seed2/Seed3 pipeline started: $(date)"
echo "Python: ${PYTHON_BIN}"
echo "================================================"

for SEED_TAG in seed2 seed3
do
  CONFIG="configs/train_online_cpu_${SEED_TAG}.yaml"
  TRAIN_OUT="outputs/train_online_ddqn_cpu_${SEED_TAG}_300"

  echo
  echo "################################################"
  echo "Training ${SEED_TAG}"
  echo "Config: ${CONFIG}"
  echo "Output: ${TRAIN_OUT}"
  echo "Started: $(date)"
  echo "################################################"

  mkdir -p "${TRAIN_OUT}"

  if [ ! -f "${TRAIN_OUT}/agent_ep300.pt" ]; then
    "$PYTHON_BIN" -u scripts/train.py \
      --config "$CONFIG" \
      --profile-csv "$PROFILE" \
      --request-trace "$TRAIN_TRACE" \
      --episodes "$TRAIN_EPISODES" \
      --arrival-rates \
        0.10 0.20 0.40 0.60 \
      --max-requests 12 \
      --max-batches-per-episode 32 \
      --max-steps-per-batch 64 \
      --device cpu \
      --checkpoint-every 50 \
      --log-every 10 \
      --output "$TRAIN_OUT"
  else
    echo "${SEED_TAG} training checkpoint already exists; skipping training."
  fi

  echo "Training ${SEED_TAG} finished: $(date)"

  echo
  echo "################################################"
  echo "Validating checkpoints for ${SEED_TAG}"
  echo "################################################"

  for EP in 50 100 150 200 250 300
  do
    CKPT="${TRAIN_OUT}/agent_ep${EP}.pt"

    if [ ! -f "$CKPT" ]; then
      echo "Missing checkpoint: $CKPT" >&2
      exit 1
    fi

    for RATE in 0.10 0.20 0.40 0.60
    do
      TAG=$(echo "$RATE" | tr -d ".")
      RESULT="outputs/val_${SEED_TAG}_ckpt_ep${EP}_r${TAG}_${VAL_EPISODES}.csv"

      if [ -f "$RESULT" ]; then
        echo "Existing result, skip: $RESULT"
        continue
      fi

      echo
      echo "----------------------------------------------"
      echo "${SEED_TAG}: checkpoint=${EP}, rate=${RATE}"
      echo "Started: $(date)"
      echo "----------------------------------------------"

      env \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 \
      "$PYTHON_BIN" -u scripts/evaluate_batchers.py \
        --config "$CONFIG" \
        --profile-csv "$PROFILE" \
        --request-trace "$VAL_TRACE" \
        --agent-checkpoint "$CKPT" \
        --device cpu \
        --episodes "$VAL_EPISODES" \
        --batchers node_conditioned_dp \
        --full-queue \
        --arrival-rate-rps "$RATE" \
        --max-requests 12 \
        --max-steps 64 \
        --max-batches 32 \
        --log-every 10 \
        --output "$RESULT"
    done
  done

  echo
  echo "################################################"
  echo "Selecting best checkpoint for ${SEED_TAG}"
  echo "################################################"

  SEED_TAG="$SEED_TAG" \
  TRAIN_OUT="$TRAIN_OUT" \
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import re
import pandas as pd

seed_tag = os.environ["SEED_TAG"]
train_out = Path(os.environ["TRAIN_OUT"])

rows = []

pattern = re.compile(
    rf"val_{re.escape(seed_tag)}_ckpt_ep"
    r"(\d+)_r(\d+)_10"
)

for path in sorted(
    Path("outputs").glob(
        f"val_{seed_tag}_ckpt_ep*_r*_10.csv"
    )
):
    match = pattern.fullmatch(path.stem)

    if match is None:
        continue

    checkpoint_episode = int(match.group(1))
    df = pd.read_csv(path)

    rows.append({
        "checkpoint_episode":
            checkpoint_episode,
        "arrival_rate_rps":
            df["arrival_rate_rps"].mean(),
        "episodes":
            len(df),
        "completion_ratio":
            df["completion_ratio"].mean(),
        "reward":
            df["reward"].mean(),
        "mean_batch_size":
            df["mean_batch_size"].mean(),
        "avg_e2e_ms":
            df["avg_e2e_ms"].mean(),
        "p95_e2e_ms":
            df["p95_e2e_ms"].mean(),
        "slo_satisfaction":
            df["slo_satisfaction"].mean(),
        "throughput_rps":
            df["throughput_rps"].mean(),
        "goodput_rps":
            df["goodput_rps"].mean(),
    })

per_load = pd.DataFrame(rows)

if len(per_load) != 24:
    raise RuntimeError(
        f"Expected 24 validation groups, "
        f"found {len(per_load)}"
    )

overall = (
    per_load.groupby("checkpoint_episode")
    .agg(
        completion_ratio=(
            "completion_ratio",
            "min",
        ),
        mean_reward=(
            "reward",
            "mean",
        ),
        mean_batch_size=(
            "mean_batch_size",
            "mean",
        ),
        mean_e2e_ms=(
            "avg_e2e_ms",
            "mean",
        ),
        mean_p95_e2e_ms=(
            "p95_e2e_ms",
            "mean",
        ),
        mean_slo_satisfaction=(
            "slo_satisfaction",
            "mean",
        ),
        mean_throughput_rps=(
            "throughput_rps",
            "mean",
        ),
        mean_goodput_rps=(
            "goodput_rps",
            "mean",
        ),
    )
    .reset_index()
)

max_goodput = max(
    float(overall["mean_goodput_rps"].max()),
    1e-9,
)

max_throughput = max(
    float(overall["mean_throughput_rps"].max()),
    1e-9,
)

overall["selection_score"] = (
    0.45 * overall["mean_slo_satisfaction"]
    + 0.35
    * overall["mean_goodput_rps"]
    / max_goodput
    + 0.20
    * overall["mean_throughput_rps"]
    / max_throughput
)

overall = overall.sort_values(
    [
        "completion_ratio",
        "selection_score",
        "mean_reward",
    ],
    ascending=[False, False, False],
).reset_index(drop=True)

best = overall.iloc[0]
best_episode = int(best["checkpoint_episode"])
best_checkpoint = f"agent_ep{best_episode}.pt"

per_load.to_csv(
    train_out / "checkpoint_validation_per_load.csv",
    index=False,
)

overall.to_csv(
    train_out / "checkpoint_validation_ranking.csv",
    index=False,
)

record = (
    f"seed_tag={seed_tag}\n"
    f"checkpoint={best_checkpoint}\n"
    f"checkpoint_episode={best_episode}\n"
    f"selection_dataset=data/processed/azure_2024_val.csv\n"
    f"selection_rates=0.10,0.20,0.40,0.60\n"
    f"episodes_per_rate=10\n"
    f"selection_score={best['selection_score']:.8f}\n"
    f"mean_reward={best['mean_reward']:.8f}\n"
    f"mean_slo_satisfaction="
    f"{best['mean_slo_satisfaction']:.8f}\n"
    f"mean_throughput_rps="
    f"{best['mean_throughput_rps']:.8f}\n"
    f"mean_goodput_rps="
    f"{best['mean_goodput_rps']:.8f}\n"
    f"mean_e2e_ms={best['mean_e2e_ms']:.8f}\n"
    f"mean_p95_e2e_ms="
    f"{best['mean_p95_e2e_ms']:.8f}\n"
)

(train_out / "best_checkpoint.txt").write_text(
    record,
    encoding="utf-8",
)

best_link = train_out / "best_agent.pt"

if best_link.exists() or best_link.is_symlink():
    best_link.unlink()

best_link.symlink_to(best_checkpoint)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

print(f"\n{seed_tag} checkpoint ranking:")
print(overall.round(4).to_string(index=False))

print(
    f"\nSelected {seed_tag}: "
    f"{best_checkpoint}"
)
PY

  echo "${SEED_TAG} validation completed: $(date)"
done

echo
echo "================================================"
echo "Seed2/Seed3 training and validation completed"
echo "Time: $(date)"
echo "================================================"
