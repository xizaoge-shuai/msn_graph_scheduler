from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reproductions.lecu.core import (
    LECUEnvironment,
    SACAgent,
)


def baseline_action(
    env: LECUEnvironment,
    method: str,
):
    n = env.num_nodes

    if method in {
        "RU",
        "RULS",
    }:
        rho = 0.25

        priorities = np.linspace(
            1.0,
            -1.0,
            n,
            dtype=np.float32,
        )

    elif method == "LS":
        rho = 0.25

        estimated = (
            env.estimate_update_times(
                sharing=True,
            )
        )

        priorities = -estimated

        scale = max(
            float(
                np.max(
                    np.abs(
                        priorities
                    )
                )
            ),
            1.0,
        )

        priorities = (
            priorities / scale
        )

    elif method == "FB":
        resource = (
            env.cpu_free
            / np.maximum(
                env.cpu_total,
                1e-6,
            )
            + env.memory_free
            / np.maximum(
                env.memory_total,
                1e-6,
            )
        )

        rho = float(
            np.clip(
                np.mean(resource) / 2.0,
                0.25,
                0.75,
            )
        )

        priorities = (
            resource
            - resource.mean()
        )

    else:
        raise ValueError(method)

    first = (
        2.0 * rho - 1.0
    )

    return np.concatenate(
        [
            np.asarray(
                [first],
                dtype=np.float32,
            ),
            np.asarray(
                priorities,
                dtype=np.float32,
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="reproductions/lecu/config.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--output",
        default="outputs/lecu_repro_eval.csv",
    )

    args = parser.parse_args()

    cfg = yaml.safe_load(
        Path(args.config)
        .read_text(
            encoding="utf-8"
        )
    )

    probe = LECUEnvironment(
        cfg,
        int(cfg["seed"]),
    )

    state = probe.reset()

    agent = SACAgent(
        cfg,
        state_dim=len(state),
        action_dim=probe.action_dim,
        device=args.device,
    )

    agent.load(
        args.checkpoint
    )

    rows = []

    methods = [
        "LECU",
        "RU",
        "RULS",
        "LS",
        "FB",
    ]

    for episode in range(
        args.episodes
    ):
        seed = (
            int(cfg["seed"])
            + 10000
            + episode
        )

        for method in methods:
            env = LECUEnvironment(
                cfg,
                seed,
            )

            state = env.reset()

            done = False

            results = []

            while not done:
                if method == "LECU":
                    action = agent.act(
                        state,
                        deterministic=True,
                    )

                    sharing = True
                else:
                    action = baseline_action(
                        env,
                        method,
                    )

                    sharing = (
                        method != "RU"
                    )

                (
                    state,
                    reward,
                    done,
                    result,
                ) = env.step(
                    action,
                    sharing=sharing,
                )

                results.append(
                    result
                )

            rows.append(
                {
                    "episode": episode,
                    "method": method,
                    "update_cost": float(
                        np.mean(
                            [
                                result.update_cost
                                for result
                                in results
                            ]
                        )
                    ),
                    "scheduling_cost": float(
                        np.mean(
                            [
                                result.scheduling_cost
                                for result
                                in results
                            ]
                        )
                    ),
                    "total_cost": float(
                        np.mean(
                            [
                                result.total_cost
                                for result
                                in results
                            ]
                        )
                    ),
                    "interrupted_tasks": float(
                        np.mean(
                            [
                                result.interrupted_tasks
                                for result
                                in results
                            ]
                        )
                    ),
                    "cloud_tasks": float(
                        np.mean(
                            [
                                result.cloud_tasks
                                for result
                                in results
                            ]
                        )
                    ),
                    "task_latency_ms": float(
                        np.mean(
                            [
                                result.mean_task_latency_ms
                                for result
                                in results
                            ]
                        )
                    ),
                }
            )

    df = pd.DataFrame(rows)

    df.to_csv(
        args.output,
        index=False,
    )

    print(
        df.groupby("method")
        .mean(
            numeric_only=True
        )
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
