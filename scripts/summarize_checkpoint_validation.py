from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(
    "outputs/checkpoint_validation"
)

EXPECTED_EPISODES = 20

METRICS = [
    "avg_e2e_ms",
    "p95_e2e_ms",
    "slo_satisfaction",
    "throughput_rps",
    "goodput_rps",
    "mean_batch_size",
]

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


rows = []

for path in sorted(
    ROOT.glob("seed*_ep*.csv")
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

    try:
        frame = pd.read_csv(path)
    except Exception as error:
        print(
            f"Cannot read {path}: {error}"
        )
        continue

    row = {
        "training_seed": training_seed,
        "checkpoint_episode":
            checkpoint_episode,
        "episodes": len(frame),
        "complete":
            len(frame) == EXPECTED_EPISODES,
        "file": str(path),
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
        "No checkpoint-validation CSV "
        "files were found."
    )

summary = summary.sort_values(
    [
        "training_seed",
        "checkpoint_episode",
    ]
).reset_index(drop=True)

summary.to_csv(
    "outputs/"
    "checkpoint_validation_summary.csv",
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
        print(
            f"Seed {training_seed}: "
            "no complete checkpoint."
        )
        continue

    # 预先固定的字典序选择规则：
    # 1. 最大化 SLO satisfaction
    # 2. 最大化 goodput
    # 3. 最小化 P95
    # 4. 最小化 Avg E2E
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
)

if not selected.empty:
    selected = selected.sort_values(
        "training_seed"
    )

    selected.to_csv(
        "outputs/"
        "selected_checkpoints.csv",
        index=False,
    )


display_columns = [
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

display_columns = [
    column
    for column in display_columns
    if column in summary.columns
]


print(
    "===== All validation checkpoints ====="
)

print(
    summary[display_columns]
    .round(4)
    .to_string(index=False)
)

print(
    "\n===== Selected checkpoints ====="
)

if selected.empty:
    print(
        "No complete checkpoints "
        "were available."
    )
else:
    print(
        selected[display_columns + [
            "checkpoint"
        ]]
        .round(4)
        .to_string(index=False)
    )

print(
    "\nSaved:"
    "\n  outputs/"
    "checkpoint_validation_summary.csv"
)

if not selected.empty:
    print(
        "  outputs/"
        "selected_checkpoints.csv"
    )
