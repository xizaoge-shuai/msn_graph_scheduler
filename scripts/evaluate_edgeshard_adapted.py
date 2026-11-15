from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)


import evaluate_batchers as base

from msn_scheduler.edgeshard import (
    choose_edgeshard_adapted,
)


def run_one_mapping_edgeshard(
    env,
    infra,
    queue,
    now_ms: float,
    max_steps: int,
    agent=None,
):
    if agent is not None:
        raise ValueError(
            "EdgeShard-Adapted is "
            "a non-learning baseline"
        )

    # EdgeShard should see all feasible nodes.
    #
    # This disables NODEGRAPH-specific top-N
    # pruning only; it does NOT modify block
    # deployment or resource feasibility.
    env.cfg[
        "scheduler"
    ][
        "top_n_nodes"
    ] = len(
        infra.nodes
    )

    env.reset(
        infra,
        queue,
        now_ms=now_ms,
    )

    result = None

    for _ in range(
        max_steps
    ):

        action_idx = (
            choose_edgeshard_adapted(
                env
            )
        )

        (
            next_obs,
            _,
            done,
            result,
        ) = env.step(
            action_idx
        )

        if done:

            if result is None:
                raise RuntimeError(
                    "EdgeShard-Adapted "
                    "ended without result"
                )

            return result

        if next_obs is None:
            raise RuntimeError(
                "EdgeShard-Adapted "
                "returned no observation"
            )

    raise RuntimeError(
        "EdgeShard-Adapted exceeded "
        f"{max_steps} mapping steps"
    )


def _output_path():

    if "--output" not in sys.argv:
        return None

    idx = sys.argv.index(
        "--output"
    )

    if idx + 1 >= len(
        sys.argv
    ):
        return None

    return Path(
        sys.argv[
            idx + 1
        ]
    )


def main():

    if (
        "--agent-checkpoint"
        in sys.argv
    ):
        raise ValueError(
            "Do not pass a DDQN "
            "checkpoint to "
            "EdgeShard-Adapted"
        )

    # EdgeShard does not contain NODEGRAPH's
    # request batching mechanism.
    if "--batchers" not in sys.argv:

        sys.argv.extend(
            [
                "--batchers",
                "no_batch",
            ]
        )

    base.run_one_mapping = (
        run_one_mapping_edgeshard
    )

    output = _output_path()

    base.main()

    if (
        output is not None
        and output.exists()
    ):

        df = pd.read_csv(
            output
        )

        if "batcher" in df.columns:

            df["batcher"] = (
                "edgeshard_adapted"
            )

        df.to_csv(
            output,
            index=False,
        )


if __name__ == "__main__":
    main()
