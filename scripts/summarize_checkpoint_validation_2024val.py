from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


INPUT_DIR = Path(
    "outputs/"
    "checkpoint_validation_2024val"
)

EXPECTED_EPISODES = 20

CHECKPOINT_DIRS = {
    7: Path(
        "outputs/"
        "train_online_ddqn_cpu_300_analytic"
    ),
    17: Path(
        "outputs/"
        "train_online_ddqn_cpu_300_analytic_seed17"
    ),
    29: Path(
        "outputs/"
        "train_online_ddqn_cpu_300_analytic_seed29"
    ),
}

METRICS = [
    "avg_e2e_ms",
    "p95_e2e_ms",
    "slo_satisfaction",
    "throughput_rps",
    "goodput_rps",
    "mean_batch_size",
]


rows = []

for path in sorted(
    INPUT_DIR.glob(
        "seed*_ep*.csv"
    )
):
    match = re.fullmatch(
        r"seed(\d+)_ep(\d+)",
        path.stem,
    )

    if match is None:
        continue

    training_seed = int(
        match.group(1)
    )

    checkpoint_episode = int(
        match.group(2)
    )

    frame = pd.read_csv(path)

    row = {
        "training_seed": training_seed,
        "checkpoint_episode":
            checkpoint_episode,
        "episodes": len(frame),
        "complete":
            len(frame)
            == EXPECTED_EPISODES,
        "checkpoint": str(
            CHECKPOINT_DIRS[
                training_seed
            ]
            / (
                f"agent_ep"
                f"{checkpoint_episode}.pt"
            )
        ),
    }

    for metric in METRICS:
        if metric in frame.columns:
            row[metric] = float(
                frame[metric].mean()
            )

    rows.append(row)


summary = pd.DataFrame(rows)

if summary.empty:
    raise RuntimeError(
        "No validation results found."
    )

summary = summary.sort_values(
    [
        "training_seed",
        "checkpoint_episode",
    ]
)

summary.to_csv(
    "outputs/"
    "checkpoint_validation_2024val_summary.csv",
    index=False,
)


selected_rows = []

for training_seed, group in summary.groupby(
    "training_seed"
):
    complete = group[
        group["complete"]
    ].copy()

    if complete.empty:
        continue

    # 固定字典序：
    # 1. 最大 SLO
    # 2. 最大 goodput
    # 3. 最小 P95
    # 4. 最小 Avg E2E
    complete = complete.sort_values(
        [
            "slo_satisfaction",
            "goodput_rps",
            "p95_e2e_ms",
            "avg_e2e_ms",
            "checkpoint_episode",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
        ],
    )

    selected_rows.append(
        complete.iloc[0]
    )


selected = pd.DataFrame(
    selected_rows
).sort_values(
    "training_seed"
)

selected.to_csv(
    "outputs/"
    "selected_checkpoints_2024val.csv",
    index=False,
)


display = [
    "training_seed",
    "checkpoint_episode",
    "episodes",
    "complete",
    "avg_e2e_ms",
    "p95_e2e_ms",
    "slo_satisfaction",
    "throughput_rps",
    "goodput_rps",
    "mean_batch_size",
]

print(
    "===== All checkpoints ====="
)

print(
    summary[display]
    .round(4)
    .to_string(index=False)
)

print(
    "\n===== Selected checkpoints ====="
)

print(
    selected[
        display
        + ["checkpoint"]
    ]
    .round(4)
    .to_string(index=False)
)

print(
    "\nSaved:"
    "\n  outputs/"
    "checkpoint_validation_2024val_summary.csv"
    "\n  outputs/"
    "selected_checkpoints_2024val.csv"
)
