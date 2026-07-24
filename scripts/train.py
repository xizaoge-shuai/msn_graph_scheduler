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
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.agent import DDQNAgent
from msn_scheduler.batching import (
    NodeConditionedDPBatcher,
)
from msn_scheduler.config import load_config
from msn_scheduler.datatypes import Transition
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import (
    ModelProfile,
    ProfileTable,
)
from msn_scheduler.synthetic import (
    make_request_queue,
    make_synthetic_infrastructure,
)


def assign_poisson_arrivals(
    queue,
    rng: np.random.Generator,
    rate_rps: float,
):
    if rate_rps <= 0:
        raise ValueError(
            "arrival rate must be positive"
        )

    current_ms = 0.0
    result = []

    for index, request in enumerate(queue):
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


def request_decode_times(
    env: SchedulingEnv,
    expected_ms: float,
    batch_size: int,
) -> list[float]:
    function = getattr(
        env,
        "_per_request_decode_ms",
        None,
    )

    if callable(function):
        values = [
            float(value)
            for value in function()
        ]

        if len(values) == batch_size:
            return values

    return [
        float(expected_ms)
        for _ in range(batch_size)
    ]


def build_model_profile(
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs/default.yaml"
        ),
    )
    parser.add_argument(
        "--profile-csv",
        default=None,
    )
    parser.add_argument(
        "--request-trace",
        required=True,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "outputs/train_online"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
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
        "--max-batches-per-episode",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--max-steps-per-batch",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["seed"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rng = np.random.default_rng(seed)

    request_pool = pd.read_csv(
        args.request_trace,
        usecols=[
            "input_tokens",
            "output_tokens",
            "request_type",
            "source",
        ],
    )

    print(
        f"Using request trace: "
        f"{args.request_trace} "
        f"rows={len(request_pool):,}",
        flush=True,
    )

    model = build_model_profile(cfg)

    profile = (
        ProfileTable.from_csv(
            model,
            args.profile_csv,
        )
        if args.profile_csv
        else ProfileTable(model)
    )

    batcher = NodeConditionedDPBatcher(
        cfg,
        profile,
    )

    # Probe graph and batch feature dimensions.
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

    probe_queue = assign_poisson_arrivals(
        probe_queue,
        rng,
        args.arrival_rates[0],
    )

    probe_env = SchedulingEnv(
        cfg,
        profile,
        batcher,
    )

    probe_obs = probe_env.reset(
        probe_infra,
        probe_queue,
        now_ms=0.0,
    )

    agent = DDQNAgent(
        cfg,
        node_dim=(
            probe_obs.node_features.shape[1]
        ),
        edge_dim=(
            probe_obs.edge_features.shape[1]
        ),
        batch_dim=(
            probe_obs.batch_features.shape[0]
        ),
        device=args.device,
    )

    episodes = (
        args.episodes
        or int(
            cfg["training"]["episodes"]
        )
    )

    warmup_steps = int(
        cfg["training"].get(
            "warmup_steps",
            0,
        )
    )

    updates_per_episode = int(
        cfg["training"].get(
            "updates_per_episode",
            1,
        )
    )

    checkpoint_every = (
        args.checkpoint_every
        or int(
            cfg["training"][
                "checkpoint_every"
            ]
        )
    )

    out_dir = Path(args.output)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []
    start_time = time.perf_counter()

    print(
        "Starting online DDQN training: "
        f"episodes={episodes} "
        f"device={agent.device} "
        f"arrival_rates={args.arrival_rates} "
        f"max_requests={args.max_requests} "
        f"warmup_steps={warmup_steps} "
        f"updates_per_episode="
        f"{updates_per_episode} "
        f"no_update={args.no_update}",
        flush=True,
    )

    for episode in range(
        1,
        episodes + 1,
    ):
        arrival_rate = float(
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

        queue = make_request_queue(
            cfg,
            rng,
            request_pool=request_pool,
        )

        queue = queue[
            :args.max_requests
        ]

        future = assign_poisson_arrivals(
            queue,
            rng,
            arrival_rate,
        )

        future.sort(
            key=lambda request:
                request.arrival_ms
        )

        pending = []
        cursor = 0

        now_ms = (
            float(
                future[0].arrival_ms
            )
            if future
            else 0.0
        )

        batch_count = 0
        served = 0
        dropped = 0
        slo_violations = 0
        ep_reward = 0.0
        ep_steps = 0

        batch_sizes: list[int] = []
        request_e2e: list[float] = []

        # The last mapping transition of a batch is
        # connected to the first state of the next
        # batch, so the DDQN can learn queue-level
        # consequences across batches.
        delayed_terminal = None

        while (
            batch_count
            < args.max_batches_per_episode
        ):
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

                now_ms = max(
                    now_ms,
                    float(
                        future[cursor]
                        .arrival_ms
                    ),
                )
                continue

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
                drop_index = min(
                    range(len(pending)),
                    key=lambda index: (
                        pending[index]
                        .remaining_deadline_ms(
                            now_ms
                        ),
                        pending[index]
                        .arrival_ms,
                    ),
                )

                pending.pop(drop_index)

                dropped += 1
                slo_violations += 1

                ep_reward -= float(
                    cfg["scheduler"][
                        "reward_slo"
                    ]
                )
                continue

            if delayed_terminal is not None:
                (
                    previous_obs,
                    previous_action,
                    previous_reward,
                ) = delayed_terminal

                agent.add_transition(
                    Transition(
                        previous_obs,
                        previous_action,
                        previous_reward,
                        obs,
                        False,
                    )
                )

                delayed_terminal = None

            batch_start_ms = now_ms
            result = None
            steps_this_batch = 0

            while True:
                steps_this_batch += 1
                ep_steps += 1

                if (
                    steps_this_batch
                    > args.max_steps_per_batch
                ):
                    raise RuntimeError(
                        f"Episode {episode} "
                        f"batch {batch_count + 1} "
                        f"exceeded "
                        f"{args.max_steps_per_batch} "
                        "mapping steps"
                    )

                action = agent.act(obs)

                (
                    next_obs,
                    reward,
                    done,
                    result,
                ) = env.step(action)

                if done:
                    delayed_terminal = (
                        obs,
                        action,
                        reward,
                    )
                    break

                if next_obs is None:
                    raise RuntimeError(
                        "Non-terminal step "
                        "returned no observation"
                    )

                agent.add_transition(
                    Transition(
                        obs,
                        action,
                        reward,
                        next_obs,
                        False,
                    )
                )

                obs = next_obs

            if (
                result is None
                or env.batch is None
            ):
                raise RuntimeError(
                    "Batch mapping ended "
                    "without a result"
                )

            selected = list(
                env.batch.requests
            )

            selected_ids = {
                request.request_id
                for request in selected
            }

            pending = [
                request
                for request in pending
                if request.request_id
                not in selected_ids
            ]

            decode_times = (
                request_decode_times(
                    env,
                    result.expected_decode_ms,
                    result.batch_size,
                )
            )

            service_decode_ms = max(
                decode_times,
                default=(
                    result.expected_decode_ms
                ),
            )

            fixed_ms = (
                result.total_prefill_ms
                + result.handover_ms
            )

            for request, decode_ms in zip(
                selected,
                decode_times,
            ):
                completion_ms = (
                    batch_start_ms
                    + fixed_ms
                    + decode_ms
                )

                request_e2e.append(
                    max(
                        0.0,
                        completion_ms
                        - request.arrival_ms,
                    )
                )

            now_ms += (
                fixed_ms
                + service_decode_ms
            )

            batch_count += 1
            served += result.batch_size

            slo_violations += (
                result.slo_violations
            )

            ep_reward += (
                result.total_reward
            )

            batch_sizes.append(
                result.batch_size
            )

        unfinished = (
            len(pending)
            + len(future)
            - cursor
        )

        if unfinished > 0:
            dropped += unfinished
            slo_violations += unfinished

            ep_reward -= (
                float(
                    cfg["scheduler"][
                        "reward_slo"
                    ]
                )
                * unfinished
            )

        if delayed_terminal is not None:
            (
                previous_obs,
                previous_action,
                previous_reward,
            ) = delayed_terminal

            agent.add_transition(
                Transition(
                    previous_obs,
                    previous_action,
                    previous_reward,
                    None,
                    True,
                )
            )

        losses = []

        if (
            not args.no_update
            and agent.steps >= warmup_steps
        ):
            for _ in range(
                updates_per_episode
            ):
                loss = agent.update()

                if loss is not None:
                    losses.append(loss)

        total_requests = len(future)

        served_violations = max(
            slo_violations - dropped,
            0,
        )

        successful = max(
            served - served_violations,
            0,
        )

        makespan_seconds = (
            now_ms / 1000.0
        )

        rows.append(
            {
                "episode": episode,
                "arrival_rate_rps":
                    arrival_rate,
                "reward": ep_reward,
                "served_requests": served,
                "dropped_requests": dropped,
                "completion_ratio": (
                    served
                    / max(
                        total_requests,
                        1,
                    )
                ),
                "batches": batch_count,
                "mean_batch_size": (
                    float(
                        np.mean(batch_sizes)
                    )
                    if batch_sizes
                    else 0.0
                ),
                "makespan_ms": now_ms,
                "avg_e2e_ms": (
                    float(
                        np.mean(request_e2e)
                    )
                    if request_e2e
                    else np.nan
                ),
                "p95_e2e_ms": (
                    float(
                        np.quantile(
                            request_e2e,
                            0.95,
                        )
                    )
                    if request_e2e
                    else np.nan
                ),
                "slo_satisfaction": (
                    1.0
                    - slo_violations
                    / max(
                        total_requests,
                        1,
                    )
                ),
                "throughput_rps": (
                    served
                    / makespan_seconds
                    if makespan_seconds > 0
                    else 0.0
                ),
                "goodput_rps": (
                    successful
                    / makespan_seconds
                    if makespan_seconds > 0
                    else 0.0
                ),
                "mapping_steps": ep_steps,
                "replay_size": (
                    len(agent.replay)
                ),
                "loss": (
                    float(
                        np.mean(losses)
                    )
                    if losses
                    else np.nan
                ),
                "epsilon": (
                    agent.epsilon()
                ),
            }
        )

        if (
            episode == 1
            or episode
            % max(
                args.log_every,
                1,
            )
            == 0
        ):
            recent = rows[
                -min(
                    args.log_every,
                    len(rows),
                ):
            ]

            finite_losses = [
                row["loss"]
                for row in recent
                if np.isfinite(
                    row["loss"]
                )
            ]

            elapsed = (
                time.perf_counter()
                - start_time
            )

            eta = (
                elapsed
                / episode
                * (
                    episodes
                    - episode
                )
            )

            print(
                f"episode={episode:5d}/"
                f"{episodes} "
                f"reward="
                f"{np.mean([row['reward'] for row in recent]):9.3f} "
                f"batch="
                f"{np.mean([row['mean_batch_size'] for row in recent]):5.2f} "
                f"slo_sat="
                f"{np.mean([row['slo_satisfaction'] for row in recent]):6.3f} "
                f"goodput="
                f"{np.mean([row['goodput_rps'] for row in recent]):7.4f} "
                f"loss="
                f"{np.mean(finite_losses) if finite_losses else float('nan'):.4f} "
                f"eps={agent.epsilon():.3f} "
                f"replay={len(agent.replay):5d} "
                f"elapsed={elapsed:7.1f}s "
                f"eta={eta:7.1f}s",
                flush=True,
            )

            pd.DataFrame(rows).to_csv(
                out_dir
                / "train_metrics.csv",
                index=False,
            )

        if (
            episode
            % checkpoint_every
            == 0
        ):
            agent.save(
                str(
                    out_dir
                    / f"agent_ep{episode}.pt"
                )
            )

            pd.DataFrame(rows).to_csv(
                out_dir
                / "train_metrics.csv",
                index=False,
            )

    agent.save(
        str(
            out_dir
            / "agent_final.pt"
        )
    )

    pd.DataFrame(rows).to_csv(
        out_dir
        / "train_metrics.csv",
        index=False,
    )

    print(
        f"Saved results to {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
