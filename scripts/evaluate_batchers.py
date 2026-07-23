from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.baselines import (
    choose_min_lower_bound,
)
from msn_scheduler.batching import (
    FixedSizeBatcher,
    NodeConditionedDPBatcher,
    SequentialGreedyBatcher,
    SingleRequestBatcher,
)
from msn_scheduler.config import load_config
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import (
    ModelProfile,
    ProfileTable,
)
from msn_scheduler.synthetic import (
    make_request_queue,
    make_synthetic_infrastructure,
)


def run_one_mapping(
    env: SchedulingEnv,
    infra,
    queue,
    now_ms: float,
    max_steps: int,
):
    obs = env.reset(
        infra,
        queue,
        now_ms=now_ms,
    )

    result = None

    for _ in range(max_steps):
        action = choose_min_lower_bound(obs)

        next_obs, _, done, result = env.step(
            action
        )

        if done:
            if result is None:
                raise RuntimeError(
                    "Mapping ended without a result"
                )
            return result

        if next_obs is None:
            raise RuntimeError(
                "Missing next observation"
            )

        obs = next_obs

    raise RuntimeError(
        f"Mapping exceeded {max_steps} steps"
    )


def get_per_request_decode(
    env: SchedulingEnv,
    expected_decode_ms: float,
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
        float(expected_decode_ms)
        for _ in range(batch_size)
    ]


def assign_poisson_arrivals(
    queue,
    rng: np.random.Generator,
    arrival_rate_rps: float,
):
    """Assign online Poisson arrivals to one request trace."""

    if arrival_rate_rps <= 0:
        raise ValueError(
            "arrival_rate_rps must be positive"
        )

    updated = []
    current_ms = 0.0

    for index, request in enumerate(queue):
        if index > 0:
            current_ms += float(
                rng.exponential(
                    1000.0 / arrival_rate_rps
                )
            )

        try:
            new_request = replace(
                request,
                arrival_ms=current_ms,
            )
        except TypeError:
            # Fallback for a non-dataclass request object.
            new_request = request
            new_request.arrival_ms = current_ms

        updated.append(new_request)

    return updated


def run_full_queue(
    cfg: dict,
    profile: ProfileTable,
    batcher,
    infra,
    queue,
    max_steps: int,
    max_batches: int,
) -> dict:
    # Requests that have not arrived yet.
    future = sorted(
        list(queue),
        key=lambda req: req.arrival_ms,
    )

    total_requests = len(future)

    if total_requests == 0:
        raise ValueError(
            "The request queue is empty"
        )

    # Requests visible to the scheduler.
    pending = []
    cursor = 0
    now_ms = float(future[0].arrival_ms)

    total_reward = 0.0
    total_prefill_ms = 0.0
    total_decode_service_ms = 0.0
    total_transfer_mb = 0.0
    total_handover_ms = 0.0
    total_slo_violations = 0

    served_requests = 0
    dropped_requests = 0
    mapping_steps = 0

    batch_sizes: list[int] = []
    pending_sizes: list[int] = []
    request_e2e_ms: list[float] = []
    failure_reason = ""

    while len(batch_sizes) < max_batches:
        # Release requests that have arrived by now.
        while (
            cursor < len(future)
            and float(
                future[cursor].arrival_ms
            )
            <= now_ms + 1e-9
        ):
            pending.append(future[cursor])
            cursor += 1

        # No available request: jump directly to the
        # next arrival event instead of busy waiting.
        if not pending:
            if cursor >= len(future):
                break

            now_ms = max(
                now_ms,
                float(
                    future[cursor].arrival_ms
                ),
            )
            continue

        # Record how many requests were available
        # when the batching decision was made.
        pending_sizes.append(
            len(pending)
        )

        env = SchedulingEnv(
            cfg,
            profile,
            batcher,
        )

        batch_start_ms = now_ms

        try:
            result = run_one_mapping(
                env=env,
                infra=infra,
                queue=pending,
                now_ms=now_ms,
                max_steps=max_steps,
            )
        except RuntimeError as exc:
            # Drop only one physically infeasible request;
            # never terminate the entire episode.
            drop_index = min(
                range(len(pending)),
                key=lambda idx: (
                    pending[idx]
                    .remaining_deadline_ms(now_ms),
                    pending[idx].arrival_ms,
                ),
            )

            dropped = pending.pop(
                drop_index
            )

            dropped_requests += 1
            total_slo_violations += 1

            total_reward -= float(
                cfg["scheduler"]["reward_slo"]
            )

            message = (
                f"{dropped.request_id}: {exc}"
            )

            failure_reason = (
                message
                if not failure_reason
                else failure_reason
                + " | "
                + message
            )
            continue

        if env.batch is None:
            raise RuntimeError(
                "Environment finished without "
                "selecting a batch"
            )

        selected = list(
            env.batch.requests
        )

        if not selected:
            raise RuntimeError(
                "Selected batch is empty"
            )

        selected_ids = {
            req.request_id
            for req in selected
        }

        previous_pending_size = len(pending)

        pending = [
            req
            for req in pending
            if req.request_id
            not in selected_ids
        ]

        if (
            len(pending)
            >= previous_pending_size
        ):
            raise RuntimeError(
                "The evaluator made no "
                "request-level progress"
            )

        per_request_decode = (
            get_per_request_decode(
                env=env,
                expected_decode_ms=(
                    result.expected_decode_ms
                ),
                batch_size=result.batch_size,
            )
        )

        # A batch occupies the serving pipeline until
        # the slowest request finishes.
        service_decode_ms = max(
            per_request_decode,
            default=float(
                result.expected_decode_ms
            ),
        )

        fixed_batch_ms = (
            result.total_prefill_ms
            + result.handover_ms
        )

        batch_service_ms = (
            fixed_batch_ms
            + service_decode_ms
        )

        for request, decode_ms in zip(
            selected,
            per_request_decode,
        ):
            completion_ms = (
                batch_start_ms
                + fixed_batch_ms
                + decode_ms
            )

            request_e2e_ms.append(
                max(
                    0.0,
                    completion_ms
                    - float(
                        request.arrival_ms
                    ),
                )
            )

        now_ms += batch_service_ms

        served_requests += result.batch_size
        total_slo_violations += (
            result.slo_violations
        )
        total_reward += result.total_reward
        total_prefill_ms += (
            result.total_prefill_ms
        )
        total_decode_service_ms += (
            service_decode_ms
        )
        total_transfer_mb += (
            result.transfer_mb
        )
        total_handover_ms += (
            result.handover_ms
        )
        mapping_steps += len(
            result.mapping
        )
        batch_sizes.append(
            result.batch_size
        )

    unfinished_requests = (
        len(pending)
        + len(future)
        - cursor
    )

    if unfinished_requests > 0:
        dropped_requests += (
            unfinished_requests
        )
        total_slo_violations += (
            unfinished_requests
        )

    makespan_ms = now_ms
    makespan_seconds = (
        makespan_ms / 1000.0
    )

    served_violations = max(
        total_slo_violations
        - dropped_requests,
        0,
    )

    successful_requests = max(
        served_requests
        - served_violations,
        0,
    )

    return {
        "total_requests": total_requests,
        "served_requests": served_requests,
        "dropped_requests": dropped_requests,
        "completion_ratio": (
            served_requests
            / max(total_requests, 1)
        ),
        "batches": len(batch_sizes),
        "mean_pending_size": (
            float(np.mean(pending_sizes))
            if pending_sizes
            else 0.0
        ),
        "max_pending_size": (
            int(max(pending_sizes))
            if pending_sizes
            else 0
        ),
        "batch_opportunity_rate": (
            float(
                np.mean(
                    [
                        size >= 2
                        for size in pending_sizes
                    ]
                )
            )
            if pending_sizes
            else 0.0
        ),
        "mean_batch_size": (
            float(np.mean(batch_sizes))
            if batch_sizes
            else 0.0
        ),
        "makespan_ms": makespan_ms,
        "avg_e2e_ms": (
            float(np.mean(request_e2e_ms))
            if request_e2e_ms
            else np.nan
        ),
        "p95_e2e_ms": (
            float(
                np.quantile(
                    request_e2e_ms,
                    0.95,
                )
            )
            if request_e2e_ms
            else np.nan
        ),
        "prefill_ms": total_prefill_ms,
        "decode_service_ms": (
            total_decode_service_ms
        ),
        "transfer_mb": total_transfer_mb,
        "handover_ms": total_handover_ms,
        "slo_violations": (
            total_slo_violations
        ),
        "slo_rate": (
            total_slo_violations
            / max(total_requests, 1)
        ),
        "slo_satisfaction": (
            1.0
            - total_slo_violations
            / max(total_requests, 1)
        ),
        "throughput_rps": (
            served_requests
            / makespan_seconds
            if makespan_seconds > 0
            else 0.0
        ),
        "goodput_rps": (
            successful_requests
            / makespan_seconds
            if makespan_seconds > 0
            else 0.0
        ),
        "mapping_steps": mapping_steps,
        "reward": total_reward,
        "failure_reason": failure_reason,
    }


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
        default=None,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--batchers",
        nargs="+",
        default=None,
        choices=[
            "no_batch",
            "fixed_4",
            "sequential_greedy",
            "node_conditioned_dp",
        ],
    )
    parser.add_argument(
        "--arrival-rate-rps",
        type=float,
        default=0.20,
        help=(
            "Poisson request arrival rate "
            "in requests per second."
        ),
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help=(
            "Limit requests per episode for "
            "fast simulator validation."
        ),
    )
    parser.add_argument(
        "--full-queue",
        action="store_true",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "outputs/eval_batchers.csv"
        ),
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    request_pool = None

    if args.request_trace:
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

    profile = (
        ProfileTable.from_csv(
            model,
            args.profile_csv,
        )
        if args.profile_csv
        else ProfileTable(model)
    )

    factories = {
        "no_batch": lambda: (
            SingleRequestBatcher(
                cfg,
                profile,
            )
        ),
        "fixed_4": lambda: (
            FixedSizeBatcher(
                cfg,
                profile,
                args.fixed_batch_size,
            )
        ),
        "sequential_greedy": lambda: (
            SequentialGreedyBatcher(
                cfg,
                profile,
            )
        ),
        "node_conditioned_dp": lambda: (
            NodeConditionedDPBatcher(
                cfg,
                profile,
            )
        ),
    }

    selected_names = (
        args.batchers
        if args.batchers
        else list(factories)
    )

    rows: list[dict] = []
    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_runs = (
        args.episodes
        * len(selected_names)
    )
    completed = 0
    start_time = time.perf_counter()

    print(
        f"Starting evaluation: "
        f"episodes={args.episodes}, "
        f"batchers={selected_names}, "
        f"full_queue={args.full_queue}, "
        f"total_runs={total_runs}",
        flush=True,
    )

    for episode in range(args.episodes):
        seed = (
            int(cfg["seed"])
            + 20000
            + episode
        )

        for name in selected_names:
            rng = np.random.default_rng(seed)

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

            if args.max_requests is not None:
                if args.max_requests <= 0:
                    raise ValueError(
                        "--max-requests must "
                        "be positive"
                    )

                queue = queue[
                    :args.max_requests
                ]

            queue = assign_poisson_arrivals(
                queue=queue,
                rng=rng,
                arrival_rate_rps=(
                    args.arrival_rate_rps
                ),
            )

            batcher = factories[name]()

            if not args.full_queue:
                raise RuntimeError(
                    "This calibrated evaluator now "
                    "expects --full-queue."
                )

            metrics = run_full_queue(
                cfg=cfg,
                profile=profile,
                batcher=batcher,
                infra=infra,
                queue=queue,
                max_steps=args.max_steps,
                max_batches=args.max_batches,
            )

            rows.append(
                {
                    "episode": episode,
                    "batcher": name,
                    "arrival_rate_rps": (
                        args.arrival_rate_rps
                    ),
                    "max_requests": (
                        len(queue)
                    ),
                    **metrics,
                }
            )
            completed += 1

        if (
            episode == 0
            or (
                episode + 1
            )
            % max(args.log_every, 1)
            == 0
            or episode + 1
            == args.episodes
        ):
            pd.DataFrame(rows).to_csv(
                output,
                index=False,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )
            rate = (
                completed / elapsed
                if elapsed > 0
                else 0.0
            )
            eta = (
                (total_runs - completed)
                / rate
                if rate > 0
                else float("inf")
            )

            print(
                f"episode={episode + 1:4d}/"
                f"{args.episodes} "
                f"runs={completed:4d}/"
                f"{total_runs} "
                f"elapsed={elapsed:8.1f}s "
                f"eta={eta:8.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    print(
        df.groupby("batcher")
        .agg(
            episodes=("episode", "count"),
            completion_ratio=(
                "completion_ratio",
                "mean",
            ),
            batches=("batches", "mean"),
            mean_batch_size=(
                "mean_batch_size",
                "mean",
            ),
            makespan_ms=(
                "makespan_ms",
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
            reward=("reward", "mean"),
        )
        .round(4),
        flush=True,
    )

    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
