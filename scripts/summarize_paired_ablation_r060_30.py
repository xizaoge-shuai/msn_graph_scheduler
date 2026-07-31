from __future__ import annotations

from pathlib import Path
import math

import pandas as pd


INPUT_DIR = Path(
    "outputs/paired_ablation_r060_30"
)

METHODS = [
    "full",
    "wo_dp",
    "wo_mobility",
]

EVAL_SEEDS = [
    7001,
    7002,
    7003,
]

EXPECTED_EPISODES = 10

METRICS = [
    "avg_e2e_ms",
    "p95_e2e_ms",
    "slo_satisfaction",
    "throughput_rps",
    "goodput_rps",
    "mean_batch_size",
]


completion_rows = []
raw_frames = []

for method in METHODS:
    for eval_seed in EVAL_SEEDS:
        path = INPUT_DIR / (
            f"{method}_evalseed"
            f"{eval_seed}_n10.csv"
        )

        if not path.exists():
            completion_rows.append(
                {
                    "method": method,
                    "eval_seed": eval_seed,
                    "episodes": 0,
                    "complete": False,
                    "file": str(path),
                }
            )
            continue

        frame = pd.read_csv(path)

        completion_rows.append(
            {
                "method": method,
                "eval_seed": eval_seed,
                "episodes": len(frame),
                "complete":
                    len(frame)
                    == EXPECTED_EPISODES,
                "file": str(path),
            }
        )

        frame = frame.copy()
        frame["method"] = method
        frame["eval_seed"] = eval_seed
        frame["episode_index"] = range(
            len(frame)
        )

        raw_frames.append(frame)


completion = pd.DataFrame(
    completion_rows
)

print(
    "===== Completion ====="
)

print(
    completion[
        [
            "method",
            "eval_seed",
            "episodes",
            "complete",
        ]
    ].to_string(index=False)
)

completion.to_csv(
    "outputs/"
    "paired_ablation_r060_30_completion.csv",
    index=False,
)

if not completion["complete"].all():
    raise SystemExit(
        "\nSome files are incomplete. "
        "Summary was not finalized."
    )

raw = pd.concat(
    raw_frames,
    ignore_index=True,
)

available_metrics = [
    metric
    for metric in METRICS
    if metric in raw.columns
]

raw.to_csv(
    "outputs/"
    "paired_ablation_r060_30_raw.csv",
    index=False,
)


summary_rows = []

for method in METHODS:
    group = raw[
        raw["method"] == method
    ]

    row = {
        "method": method,
        "episodes": len(group),
        "eval_seeds":
            group["eval_seed"].nunique(),
    }

    for metric in available_metrics:
        values = group[metric].astype(float)

        mean = float(values.mean())
        std = float(values.std(ddof=1))

        ci95 = (
            1.96
            * std
            / math.sqrt(len(values))
        )

        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[f"{metric}_ci95"] = ci95

    summary_rows.append(row)


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    "outputs/"
    "paired_ablation_r060_30_summary.csv",
    index=False,
)

print(
    "\n===== Aggregate summary ====="
)

print(
    summary.round(4)
    .to_string(index=False)
)


comparison_rows = []

full = raw[
    raw["method"] == "full"
]

for baseline_name in [
    "wo_dp",
    "wo_mobility",
]:
    baseline = raw[
        raw["method"] == baseline_name
    ]

    paired = full.merge(
        baseline,
        on=[
            "eval_seed",
            "episode_index",
        ],
        suffixes=(
            "_full",
            "_baseline",
        ),
    )

    if len(paired) != 30:
        raise RuntimeError(
            f"{baseline_name}: expected "
            f"30 paired episodes, "
            f"found {len(paired)}"
        )

    row = {
        "comparison":
            f"full_vs_{baseline_name}",
        "paired_episodes": len(paired),
    }

    for metric in available_metrics:
        full_values = paired[
            f"{metric}_full"
        ].astype(float)

        baseline_values = paired[
            f"{metric}_baseline"
        ].astype(float)

        full_mean = float(
            full_values.mean()
        )

        baseline_mean = float(
            baseline_values.mean()
        )

        if metric in [
            "avg_e2e_ms",
            "p95_e2e_ms",
        ]:
            row[
                f"{metric}_improvement_pct"
            ] = (
                (
                    baseline_mean
                    - full_mean
                )
                / baseline_mean
                * 100.0
            )

        elif metric == "slo_satisfaction":
            row[
                "slo_improvement_pp"
            ] = (
                full_mean
                - baseline_mean
            ) * 100.0

        elif metric in [
            "throughput_rps",
            "goodput_rps",
        ]:
            row[
                f"{metric}_improvement_pct"
            ] = (
                (
                    full_mean
                    - baseline_mean
                )
                / baseline_mean
                * 100.0
            )

        elif metric == "mean_batch_size":
            row[
                "mean_batch_size_difference"
            ] = (
                full_mean
                - baseline_mean
            )

    comparison_rows.append(row)


comparison = pd.DataFrame(
    comparison_rows
)

comparison.to_csv(
    "outputs/"
    "paired_ablation_r060_30_improvement.csv",
    index=False,
)

print(
    "\n===== Paired improvements ====="
)

print(
    comparison.round(4)
    .to_string(index=False)
)

print(
    "\nSaved:"
    "\n  outputs/"
    "paired_ablation_r060_30_summary.csv"
    "\n  outputs/"
    "paired_ablation_r060_30_improvement.csv"
    "\n  outputs/"
    "paired_ablation_r060_30_completion.csv"
)
