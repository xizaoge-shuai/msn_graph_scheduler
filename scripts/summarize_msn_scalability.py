from pathlib import Path
import re

import pandas as pd

root = Path(
    "outputs/msn_scalability"
)

rows = []

for path in sorted(
    root.glob(
        "nodes*_queue*.csv"
    )
):
    match = re.fullmatch(
        r"nodes(\d+)_queue(\d+)",
        path.stem,
    )

    if match is None:
        continue

    nodes = int(
        match.group(1)
    )

    queue = int(
        match.group(2)
    )

    frame = pd.read_csv(path)

    time_path = path.with_suffix(
        ".time"
    )

    elapsed = None
    max_rss = None

    if time_path.exists():
        text = time_path.read_text(
            encoding="utf-8"
        )

        elapsed_match = re.search(
            r"elapsed_sec=([\d.]+)",
            text,
        )

        rss_match = re.search(
            r"max_rss_kb=(\d+)",
            text,
        )

        if elapsed_match:
            elapsed = float(
                elapsed_match.group(1)
            )

        if rss_match:
            max_rss = int(
                rss_match.group(1)
            )

    rows.append(
        {
            "num_edge_nodes": nodes,
            "queue_size": queue,
            "episodes": len(frame),
            "avg_e2e_ms": (
                frame[
                    "avg_e2e_ms"
                ].mean()
            ),
            "p95_e2e_ms": (
                frame[
                    "p95_e2e_ms"
                ].mean()
            ),
            "slo_satisfaction": (
                frame[
                    "slo_satisfaction"
                ].mean()
            ),
            "throughput_rps": (
                frame[
                    "throughput_rps"
                ].mean()
            ),
            "goodput_rps": (
                frame[
                    "goodput_rps"
                ].mean()
            ),
            "mean_batch_size": (
                frame[
                    "mean_batch_size"
                ].mean()
            ),
            "wall_time_sec": elapsed,
            "wall_time_per_episode_sec": (
                elapsed / len(frame)
                if elapsed is not None
                else None
            ),
            "max_rss_mb": (
                max_rss / 1024
                if max_rss is not None
                else None
            ),
        }
    )

summary = pd.DataFrame(rows)

summary.to_csv(
    "outputs/"
    "msn_scalability_summary.csv",
    index=False,
)

print(
    summary.round(4)
    .to_string(index=False)
)

print(
    "\nSaved outputs/"
    "msn_scalability_summary.csv"
)
