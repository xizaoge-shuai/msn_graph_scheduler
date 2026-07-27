from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from evaluate_batchers import (
    assign_poisson_arrivals,
    run_full_queue,
)

from msn_scheduler.config import load_config
from msn_scheduler.dybap import (
    DyBAPFusionBatcher,
    DyBAPPPOAgent,
)
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import (
    ModelProfile,
    ProfileTable,
)
from msn_scheduler.synthetic import (
    make_request_queue,
    make_synthetic_infrastructure,
)


class PPOOnlineAdapter:
    def __init__(
        self,
        agent: DyBAPPPOAgent,
    ):
        self.agent = agent

    def __call__(
        self,
        obs,
        device,
    ):
        return self.agent.logits(
            obs
        )


class PPOAgentAdapter:
    def __init__(
        self,
        agent: DyBAPPPOAgent,
    ):
        self.device = agent.device
        self.online = PPOOnlineAdapter(
            agent
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/train_online_cpu.yaml",
    )

    parser.add_argument(
        "--profile-csv",
        required=True,
    )

    parser.add_argument(
        "--request-trace",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--arrival-rate-rps",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--max-requests",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    request_pool = pd.read_csv(
        args.request_trace,
        usecols=[
            "input_tokens",
            "output_tokens",
            "request_type",
            "source",
        ],
    )

    model = ModelProfile(
        num_blocks=int(
            cfg["model"]["num_blocks"]
        ),
        hidden_size=int(
            cfg["model"]["hidden_size"]
        ),
        bytes_per_element=int(
            cfg["model"][
                "bytes_per_element"
            ]
        ),
        block_parameter_gb=float(
            cfg["model"][
                "block_parameter_gb"
            ]
        ),
        kv_bytes_per_token_per_block=float(
            cfg["model"][
                "kv_bytes_per_token_per_block"
            ]
        ),
    )

    profile = ProfileTable.from_csv(
        model,
        args.profile_csv,
    )

    probe_rng = np.random.default_rng(
        int(cfg["seed"]) + 987654
    )

    probe_infra = (
        make_synthetic_infrastructure(
            cfg,
            probe_rng,
        )
    )

    probe_queue = make_request_queue(
        cfg,
        probe_rng,
        request_pool=request_pool,
    )[:1]

    probe_batcher = DyBAPFusionBatcher(
        cfg,
        profile,
    )

    probe_env = SchedulingEnv(
        cfg,
        profile,
        probe_batcher,
    )

    probe_obs = probe_env.reset(
        probe_infra,
        probe_queue,
        now_ms=0.0,
    )

    agent = DyBAPPPOAgent(
        cfg,
        node_dim=int(
            probe_obs.node_features.shape[1]
        ),
        batch_dim=int(
            probe_obs.batch_features.shape[0]
        ),
        device=args.device,
    )

    agent.load(
        args.checkpoint
    )

    agent.network.eval()

    adapter = PPOAgentAdapter(
        agent
    )

    rows = []
    start = time.perf_counter()

    for episode in range(
        args.episodes
    ):
        seed = (
            int(cfg["seed"])
            + 20000
            + episode
        )

        rng = np.random.default_rng(
            seed
        )

        infra = (
            make_synthetic_infrastructure(
                cfg,
                rng,
            )
        )

        queue = make_request_queue(
            cfg,
            rng,
            request_pool=request_pool,
        )[:args.max_requests]

        queue = assign_poisson_arrivals(
            queue,
            rng,
            args.arrival_rate_rps,
        )

        metrics = run_full_queue(
            cfg=cfg,
            profile=profile,
            batcher=DyBAPFusionBatcher(
                cfg,
                profile,
            ),
            infra=infra,
            queue=queue,
            max_steps=args.max_steps,
            max_batches=args.max_batches,
            agent=adapter,
        )

        rows.append(
            {
                "episode": episode,
                "batcher": "dybap_repro",
                "arrival_rate_rps":
                    args.arrival_rate_rps,
                "max_requests":
                    len(queue),
                "agent_checkpoint":
                    args.checkpoint,
                **metrics,
            }
        )

        if (
            episode == 0
            or (
                episode + 1
            )
            % args.log_every
            == 0
            or episode + 1
            == args.episodes
        ):
            elapsed = (
                time.perf_counter()
                - start
            )

            print(
                f"episode={episode + 1}/"
                f"{args.episodes} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

            pd.DataFrame(
                rows
            ).to_csv(
                args.output,
                index=False,
            )

    df = pd.DataFrame(rows)
    df.to_csv(
        args.output,
        index=False,
    )

    print(
        df.groupby("batcher")
        .agg(
            episodes=("episode", "count"),
            completion_ratio=(
                "completion_ratio",
                "mean",
            ),
            mean_batch_size=(
                "mean_batch_size",
                "mean",
            ),
            avg_e2e_ms=(
                "avg_e2e_ms",
                "mean",
            ),
            p95_e2e_ms=(
                "p95_e2e_ms",
                "mean",
            ),
            slo_satisfaction=(
                "slo_satisfaction",
                "mean",
            ),
            throughput_rps=(
                "throughput_rps",
                "mean",
            ),
            goodput_rps=(
                "goodput_rps",
                "mean",
            ),
        )
        .round(4)
    )


if __name__ == "__main__":
    main()
