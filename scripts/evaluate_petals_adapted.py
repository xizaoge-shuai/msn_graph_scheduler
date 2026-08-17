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

from msn_scheduler.petals import (
    get_petals_chain,
    choose_petals_action,
)


def run_one_mapping_petals(
    env,
    infra,
    queue,
    now_ms: float,
    max_steps: int,
    agent=None,
):
    if agent is not None:
        raise ValueError(
            "PETALS-Adapted is "
            "a non-learning baseline"
        )

    # PETALS routing should see all servers
    # capable of hosting the selected blocks;
    # do not inherit NODEGRAPH's top-N pruning.
    env.cfg[
        "scheduler"
    ][
        "top_n_nodes"
    ] = len(
        infra.nodes
    )

    # Compute and cache a routing chain.
    # Does NOT modify deployed_blocks.
    get_petals_chain(
        env.cfg,
        infra,
        env.profile,
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
            choose_petals_action(
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
                    "PETALS mapping "
                    "ended without result"
                )

            return result

        if next_obs is None:
            raise RuntimeError(
                "PETALS mapping returned "
                "no observation"
            )

    raise RuntimeError(
        "PETALS mapping exceeded "
        f"{max_steps} steps"
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
        sys.argv[idx + 1]
    )


def main():

    if (
        "--agent-checkpoint"
        in sys.argv
    ):
        raise ValueError(
            "Do not pass a DDQN "
            "checkpoint to PETALS"
        )

    # PETALS sequential autoregressive
    # inference is naturally batch size 1.
    if "--batchers" not in sys.argv:
        sys.argv.extend(
            [
                "--batchers",
                "no_batch",
            ]
        )

    base.run_one_mapping = (
        run_one_mapping_petals
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
                "petals_adapted"
            )

        df.to_csv(
            output,
            index=False,
        )


if __name__ == "__main__":
    main()
