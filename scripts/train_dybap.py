from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "src"),
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


def model_profile(
    cfg: dict,
) -> ModelProfile:
    return ModelProfile(
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


def poisson_arrivals(
    queue,
    rng,
    rate_rps,
):
    current_ms = 0.0
    result = []

    for index, request in enumerate(
        queue
    ):
        if index > 0:
            current_ms += float(
                rng.exponential(
                    1000.0 / rate_rps
                )
            )

        result.append(
            replace(
                request,
                arrival_ms=current_ms,
            )
        )

    return result


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
        "--episodes",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--output",
        default="outputs/train_dybap",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--arrival-rates",
        nargs="+",
        type=float,
        default=[
            0.1,
            0.2,
            0.4,
            0.6,
        ],
    )

    parser.add_argument(
        "--max-requests",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    seed = int(
        cfg["seed"]
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rng = np.random.default_rng(
        seed
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

    profile = ProfileTable.from_csv(
        model_profile(cfg),
        args.profile_csv,
    )

    probe_infra = (
        make_synthetic_infrastructure(
            cfg,
            rng,
        )
    )

    probe_queue = make_request_queue(
        cfg,
        rng,
        request_pool=request_pool,
    )[:1]

    probe_queue = poisson_arrivals(
        probe_queue,
        rng,
        args.arrival_rates[0],
    )

    probe_batcher = (
        DyBAPFusionBatcher(
            cfg,
            profile,
        )
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

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    start_time = time.perf_counter()

    for episode in range(
        1,
        args.episodes + 1,
    ):
        rate = float(
            rng.choice(
                args.arrival_rates
            )
        )

        infra = (
            make_synthetic_infrastructure(
                cfg,
                rng,
            )
        )

        anchor_index = int(
            rng.integers(
                0,
                int(
                    cfg["system"][
                        "num_edge_nodes"
                    ]
                ),
            )
        )

        anchor = (
            f"edge_{anchor_index}"
        )

        infra.anchor_node = anchor

        queue = make_request_queue(
            cfg,
            rng,
            request_pool=request_pool,
        )[:args.max_requests]

        queue = [
            replace(
                request,
                anchor_node=anchor,
                next_anchor_node=(
                    request.next_anchor_node
                    if request.next_anchor_node
                    != anchor
                    else (
                        f"edge_"
                        f"{(anchor_index + 1) % int(cfg['system']['num_edge_nodes'])}"
                    )
                ),
            )
            for request in queue
        ]

        future = poisson_arrivals(
            queue,
            rng,
            rate,
        )

        future.sort(
            key=lambda request:
                request.arrival_ms
        )

        pending = []
        cursor = 0
        now_ms = 0.0
        batches = 0
        episode_reward = 0.0
        served = 0
        violations = 0
        batch_sizes = []

        while batches < args.max_batches:
            while (
                cursor < len(future)
                and future[cursor].arrival_ms
                <= now_ms + 1e-9
            ):
                pending.append(
                    future[cursor]
                )
                cursor += 1

            if not pending:
                if cursor >= len(future):
                    break

                now_ms = float(
                    future[cursor]
                    .arrival_ms
                )
                continue

            batcher = DyBAPFusionBatcher(
                cfg,
                profile,
            )

            env = SchedulingEnv(
                cfg,
                profile,
                batcher,
            )

            try:
                obs = env.reset(
                    infra,
                    pending,
                    now_ms=now_ms,
                )
            except RuntimeError:
                pending.pop(0)
                violations += 1
                continue

            cumulative_ms = 0.0
            result = None

            for _ in range(
                args.max_steps
            ):
                (
                    action,
                    log_probability,
                    value,
                ) = agent.act(obs)

                (
                    next_obs,
                    _environment_reward,
                    done,
                    result,
                ) = env.step(action)

                step = env.mapping[-1]

                immediate_cost = (
                    step.prefill_ms
                    + step.transfer_ms
                    + step.handover_ms
                )

                cumulative_ms += (
                    immediate_cost
                )

                deadline_reference = max(
                    float(
                        env.batch
                        .min_remaining_deadline_ms
                    ),
                    1.0,
                )

                long_cost = cumulative_ms

                if done and result is not None:
                    long_cost += float(
                        result.expected_decode_ms
                    )

                block_progress = (
                    float(env.current_block)
                    / max(
                        float(
                            cfg["model"][
                                "num_blocks"
                            ]
                        ),
                        1.0,
                    )
                )

                progressive_gamma = float(
                    np.exp(
                        block_progress
                    )
                )

                reward = (
                    (
                        deadline_reference
                        / float(
                            cfg["model"][
                                "num_blocks"
                            ]
                        )
                        - immediate_cost
                    )
                    + progressive_gamma
                    * (
                        deadline_reference
                        - long_cost
                    )
                ) / 1000.0

                if done and result is not None:
                    reward -= (
                        float(
                            cfg["dybap"][
                                "slo_penalty"
                            ]
                        )
                        * result.slo_violations
                    )

                agent.store(
                    observation=obs,
                    action=action,
                    old_log_prob=log_probability,
                    reward=reward,
                    value=value,
                    done=done,
                )

                episode_reward += (
                    reward
                )

                if done:
                    break

                if next_obs is None:
                    raise RuntimeError(
                        "DyBAP non-terminal "
                        "step returned no state"
                    )

                obs = next_obs

            if (
                result is None
                or env.batch is None
            ):
                raise RuntimeError(
                    "DyBAP mapping failed"
                )

            selected_ids = {
                request.request_id
                for request
                in env.batch.requests
            }

            pending = [
                request
                for request in pending
                if request.request_id
                not in selected_ids
            ]

            per_request_decode = (
                env._per_request_decode_ms()
            )

            service_ms = (
                result.total_prefill_ms
                + result.handover_ms
                + max(
                    per_request_decode,
                    default=(
                        result.expected_decode_ms
                    ),
                )
            )

            now_ms += service_ms
            served += result.batch_size
            violations += result.slo_violations
            batches += 1

            batch_sizes.append(
                result.batch_size
            )

        update_metrics = (
            agent.update()
        )

        rows.append(
            {
                "episode": episode,
                "arrival_rate_rps": rate,
                "reward": episode_reward,
                "served_requests": served,
                "slo_violations": violations,
                "slo_satisfaction": (
                    1.0
                    - violations
                    / max(
                        len(future),
                        1,
                    )
                ),
                "mean_batch_size": (
                    float(
                        np.mean(
                            batch_sizes
                        )
                    )
                    if batch_sizes
                    else 0.0
                ),
                **update_metrics,
            }
        )

        if (
            episode == 1
            or episode
            % args.log_every
            == 0
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"episode={episode:4d}/"
                f"{args.episodes} "
                f"reward={episode_reward:9.3f} "
                f"batch="
                f"{rows[-1]['mean_batch_size']:.3f} "
                f"slo="
                f"{rows[-1]['slo_satisfaction']:.3f} "
                f"loss="
                f"{update_metrics['loss']:.4f} "
                f"elapsed={elapsed:.1f}s",
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
            % args.checkpoint_every
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

    pd.DataFrame(
        rows
    ).to_csv(
        output
        / "train_metrics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
