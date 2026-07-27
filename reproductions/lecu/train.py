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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="reproductions/lecu/config.yaml",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--output",
        default="outputs/train_lecu",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    cfg = yaml.safe_load(
        Path(args.config)
        .read_text(
            encoding="utf-8"
        )
    )

    episodes = (
        args.episodes
        or int(
            cfg["training"][
                "episodes"
            ]
        )
    )

    env = LECUEnvironment(
        cfg,
        int(cfg["seed"]),
    )

    state = env.reset()

    agent = SACAgent(
        cfg,
        state_dim=len(state),
        action_dim=env.action_dim,
        device=args.device,
    )

    output = Path(args.output)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    global_step = 0

    for episode in range(
        1,
        episodes + 1,
    ):
        state = env.reset()

        reward_sum = 0.0
        update_costs = []
        scheduling_costs = []
        losses = []

        done = False

        while not done:
            global_step += 1

            if global_step < int(
                cfg["sac"][
                    "warmup_steps"
                ]
            ):
                action = np.random.uniform(
                    -1.0,
                    1.0,
                    size=env.action_dim,
                ).astype(np.float32)
            else:
                action = agent.act(
                    state
                )

            (
                next_state,
                reward,
                done,
                result,
            ) = env.step(
                action,
                sharing=True,
            )

            agent.replay.add(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state
            reward_sum += reward

            update_costs.append(
                result.update_cost
            )

            scheduling_costs.append(
                result.scheduling_cost
            )

            for _ in range(
                int(
                    cfg["sac"][
                        "updates_per_step"
                    ]
                )
            ):
                loss = agent.update()

                if loss is not None:
                    losses.append(loss)

        rows.append(
            {
                "episode": episode,
                "reward": reward_sum,
                "update_cost": float(
                    np.mean(
                        update_costs
                    )
                ),
                "scheduling_cost": float(
                    np.mean(
                        scheduling_costs
                    )
                ),
                "actor_loss": (
                    float(
                        np.mean(losses)
                    )
                    if losses
                    else np.nan
                ),
                "replay_size":
                    len(agent.replay),
            }
        )

        if (
            episode == 1
            or episode
            % args.log_every
            == 0
        ):
            print(
                f"episode={episode:4d}/"
                f"{episodes} "
                f"reward={reward_sum:9.3f} "
                f"update="
                f"{rows[-1]['update_cost']:.3f} "
                f"scheduling="
                f"{rows[-1]['scheduling_cost']:.3f}",
                flush=True,
            )

            pd.DataFrame(
                rows
            ).to_csv(
                output
                / "train_metrics.csv",
                index=False,
            )

        if (
            episode
            % int(
                cfg["training"][
                    "checkpoint_every"
                ]
            )
            == 0
        ):
            agent.save(
                output
                / f"agent_ep{episode}.pt"
            )

    agent.save(
        output
        / "agent_final.pt"
    )


if __name__ == "__main__":
    main()
