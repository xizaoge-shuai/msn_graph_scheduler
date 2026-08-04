#!/usr/bin/env bash

set -u

cd "$(dirname "$0")/.."

echo "============================================================"
echo "MSN scheduler experiment dashboard"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Logical CPU cores: $(nproc)"
echo "============================================================"

echo
echo "===== TMUX SESSIONS ====="

tmux ls 2>/dev/null || echo "No tmux sessions"

echo
echo "===== ACTIVE TRAINING / EVALUATION PROCESSES ====="

ps -eo \
  pid,ppid,etime,%cpu,%mem,cmd \
  --sort=-%cpu \
  | grep -E \
  'scripts/(train|evaluate_batchers|evaluate_dynamic_mobility|evaluate_external_baseline|evaluate_seqgreedy_ddqn)' \
  | grep -v grep \
  || echo "No active experiment processes"

echo
echo "===== RESULT FILE PROGRESS ====="

python - <<'PY'
from pathlib import Path
import pandas as pd


def count_rows(path: Path) -> int:
    try:
        return len(pd.read_csv(path))
    except Exception:
        return -1


groups = [
    {
        "name": "Dynamic method comparison",
        "root": Path(
            "outputs/dynamic_mobility_methods"
        ),
        "pattern": "*_r040_100.csv",
        "expected_files": 16,
        "expected_rows": 100,
    },
    {
        "name": "Mobility policy/error",
        "root": Path(
            "outputs/full_mobility_policy_error"
        ),
        "pattern": "*_r040_100.csv",
        "expected_files": 24,
        "expected_rows": 100,
    },
    {
        "name": "Three-seed common evaluation",
        "root": Path(
            "outputs/three_seed_eval_common"
        ),
        "pattern": "trainseed*_r*_100.csv",
        "expected_files": 12,
        "expected_rows": 100,
    },
    {
        "name": "Ablation w/o mobility",
        "root": Path(
            "outputs/ablation_wo_mobility"
        ),
        "pattern": "*.csv",
        "expected_files": 3,
        "expected_rows": 100,
    },
    {
        "name": "Ablation w/o node-conditioned DP",
        "root": Path(
            "outputs/ablation_wo_dp"
        ),
        "pattern": "*.csv",
        "expected_files": 2,
        "expected_rows": 100,
    },
    {
        "name": "High-speed load sweep",
        "root": Path(
            "outputs/high_speed_load_sweep"
        ),
        "pattern": "*.csv",
        "expected_files": 12,
        "expected_rows": 100,
    },
]

for group in groups:
    root = group["root"]
    paths = sorted(
        root.glob(group["pattern"])
    )

    completed = 0

    print(
        f"\n--- {group['name']} ---"
    )

    if not paths:
        print(
            f"No files yet: {root}"
        )
        print(
            f"Progress: 0/"
            f"{group['expected_files']}"
        )
        continue

    for path in paths:
        rows = count_rows(path)

        if rows == group["expected_rows"]:
            completed += 1
            status = "DONE"
        elif rows < 0:
            status = "BAD"
        else:
            status = "RUN"

        print(
            f"{status:<4} "
            f"{rows:>4}/"
            f"{group['expected_rows']}  "
            f"{path.name}"
        )

    print(
        f"Progress: {completed}/"
        f"{group['expected_files']} complete; "
        f"{len(paths)}/"
        f"{group['expected_files']} files exist"
    )
PY

echo
echo "===== AVAILABLE CHECKPOINTS ====="

for DIR in \
  outputs/train_online_ddqn_cpu_300_analytic \
  outputs/train_online_ddqn_cpu_300_analytic_seed17 \
  outputs/train_online_ddqn_cpu_300_analytic_seed29 \
  outputs/train_ablation_wo_mobility_seed7 \
  outputs/train_ablation_wo_dp_seed7
do
  echo
  echo "--- $DIR ---"

  if [ -d "$DIR" ]; then
    ls -lh \
      "$DIR"/agent_ep*.pt \
      2>/dev/null \
      || echo "No checkpoint yet"
  else
    echo "Directory does not exist"
  fi
done

echo
echo "===== MAIN ANALYTICAL RESULTS ====="

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path(
    "outputs/common_analytic_summary.csv"
)

if not path.exists():
    print("Not available:", path)
else:
    df = pd.read_csv(path)

    columns = [
        "method",
        "arrival_rate_rps",
        "avg_e2e_ms",
        "p95_e2e_ms",
        "slo_satisfaction",
        "throughput_rps",
        "goodput_rps",
        "mean_batch_size",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df[columns]
        .round(4)
        .to_string(index=False)
    )
PY

echo
echo "===== FULL VS EACH BASELINE ====="

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path(
    "outputs/plot_tables/"
    "common_ours_vs_each_baseline_wide.csv"
)

if not path.exists():
    print("Not available:", path)
else:
    df = pd.read_csv(path)

    df = df[
        df["method"] == "Full"
    ]

    print(
        df.round(3)
        .to_string(index=False)
    )
PY

echo
echo "===== COMMON ABLATION: FULL VS W/O DDQN ====="

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path(
    "outputs/"
    "common_analytic_full_vs_nodedp_ablation.csv"
)

alternative = Path(
    "outputs/plot_tables/"
    "common_full_vs_without_ddqn.csv"
)

if path.exists():
    df = pd.read_csv(path)
elif alternative.exists():
    df = pd.read_csv(alternative)
else:
    print(
        "No common DDQN ablation summary"
    )
    raise SystemExit(0)

print(
    df.round(3)
    .to_string(index=False)
)
PY

echo
echo "===== DYNAMIC MOBILITY METHOD RESULTS ====="

python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path(
    "outputs/"
    "dynamic_mobility_methods_summary_final.csv"
)

if not path.exists():
    print("Not available:", path)
else:
    df = pd.read_csv(path)

    columns = [
        "method",
        "mobility_mode",
        "avg_e2e_ms",
        "p95_e2e_ms",
        "slo_satisfaction",
        "throughput_rps",
        "goodput_rps",
        "handovers",
        "handover_overhead_ms",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df[columns]
        .round(4)
        .to_string(index=False)
    )
PY

echo
echo "===== MOBILITY POLICY / PREDICTION ERROR ====="

python - <<'PY'
from pathlib import Path
import pandas as pd

paths = [
    Path(
        "outputs/"
        "full_mobility_policy_error_summary_final.csv"
    ),
    Path(
        "outputs/"
        "full_mobility_policy_error_summary.csv"
    ),
]

path = next(
    (
        candidate
        for candidate in paths
        if candidate.exists()
    ),
    None,
)

if path is None:
    print(
        "Mobility policy/error summary "
        "not available"
    )
else:
    df = pd.read_csv(path)

    columns = [
        "mobility_mode",
        "migration_policy",
        "prediction_error",
        "avg_e2e_ms",
        "p95_e2e_ms",
        "slo",
        "goodput",
        "handovers",
        "handover_slo",
        "migration_mb",
        "overhead_ms",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df[columns]
        .round(4)
        .to_string(index=False)
    )
PY

echo
echo "===== THREE TRAINING SEEDS ====="

python - <<'PY'
from pathlib import Path
import pandas as pd

paths = [
    Path(
        "outputs/"
        "full_three_seed_common_eval_per_seed.csv"
    ),
    Path(
        "outputs/"
        "full_three_seed_per_seed.csv"
    ),
]

path = next(
    (
        candidate
        for candidate in paths
        if candidate.exists()
    ),
    None,
)

if path is None:
    print(
        "Three-seed summary not available"
    )
else:
    df = pd.read_csv(path)

    print(
        df.sort_values(
            [
                column
                for column in [
                    "arrival_rate_rps",
                    "train_seed",
                    "seed",
                ]
                if column in df.columns
            ]
        )
        .round(4)
        .to_string(index=False)
    )
PY

echo
echo "===== RAW ABLATION RESULTS ====="

python - <<'PY'
from pathlib import Path
import pandas as pd


def show_directory(
    title: str,
    root: Path,
) -> None:
    print(f"\n--- {title} ---")

    paths = sorted(
        root.glob("*.csv")
    )

    if not paths:
        print(
            f"No result files: {root}"
        )
        return

    rows = []

    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(
                f"Cannot read {path}: {exc}"
            )
            continue

        row = {
            "file": path.name,
            "episodes": len(df),
        }

        for column in [
            "avg_e2e_ms",
            "p95_e2e_ms",
            "slo_satisfaction",
            "throughput_rps",
            "goodput_rps",
            "handover_events",
            "handover_overhead_ms",
        ]:
            if column in df.columns:
                row[column] = (
                    df[column].mean()
                )

        rows.append(row)

    if rows:
        result = pd.DataFrame(rows)

        print(
            result.round(4)
            .to_string(index=False)
        )


show_directory(
    "Ablation w/o mobility",
    Path(
        "outputs/ablation_wo_mobility"
    ),
)

show_directory(
    "Ablation w/o node-conditioned DP",
    Path(
        "outputs/ablation_wo_dp"
    ),
)

show_directory(
    "High-speed load sweep",
    Path(
        "outputs/high_speed_load_sweep"
    ),
)
PY

echo
echo "===== LATEST LOG OUTPUT ====="

for LOG in \
  logs/full_extra_seeds.log \
  logs/full_three_seed_common_eval.log \
  logs/train_ablation_wo_mobility_seed7.log \
  logs/train_ablation_wo_dp_seed7.log \
  logs/high_speed_load_sweep.log \
  logs/all_dynamic_mobility_final.log
do
  if [ -f "$LOG" ]; then
    echo
    echo "--- $LOG ---"
    tail -n 8 "$LOG"
  fi
done

echo
echo "============================================================"
echo "Dashboard completed"
echo "============================================================"
