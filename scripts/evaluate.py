from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.agent import DDQNAgent
from msn_scheduler.baselines import (
    choose_anchor,
    choose_cloud,
    choose_min_lower_bound,
)
from msn_scheduler.batching import NodeConditionedDPBatcher
from msn_scheduler.config import load_config
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import ModelProfile, ProfileTable
from msn_scheduler.synthetic import (
    make_request_queue,
    make_synthetic_infrastructure,
)


AVAILABLE_METHODS = {
    "cloud_only",
    "anchor_first",
    "latency_greedy",
    "gat_dueling_ddqn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scheduling policies."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/default.yaml"),
    )
    parser.add_argument(
        "--profile-csv",
        default=None,
        help=(
            "Measured profile CSV. "
            "Analytic profile is used when omitted."
        ),
    )
    parser.add_argument(
        "--request-trace",
        default=None,
        help=(
            "CSV request-shape pool containing "
            "input_tokens and output_tokens."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Trained DDQN checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs/eval.csv"),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the DDQN policy: cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "cloud_only",
            "anchor_first",
            "latency_greedy",
        ],
        help=(
            "Methods to evaluate. Available: "
            "cloud_only anchor_first latency_greedy "
            "gat_dueling_ddqn"
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Print and save progress every N episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
        help="Maximum mapping steps in one episode.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
    )
    return parser.parse_args()


def run_policy(
    env: SchedulingEnv,
    infra,
    queue,
    policy,
    max_steps: int,
):
    obs = env.reset(infra, queue)
    result = None

    for _ in range(max_steps):
        node_ids = [
            payload.node_id
            for payload in obs.candidate_payloads
        ]
        action_index = policy(obs, node_ids)

        next_obs, _, done, result = env.step(
            action_index
        )

        if done:
            if result is None:
                raise RuntimeError(
                    "Episode finished without a result."
                )
            return result

        if next_obs is None:
            raise RuntimeError(
                "Environment returned no observation "
                "before termination."
            )

        obs = next_obs

    raise RuntimeError(
        f"Episode exceeded {max_steps} mapping steps."
    )


def save_rows(
    rows: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    args = parse_args()

    unknown = set(args.methods) - AVAILABLE_METHODS
    if unknown:
        raise ValueError(
            f"Unknown methods: {sorted(unknown)}"
        )

    if (
        "gat_dueling_ddqn" in args.methods
        and not args.checkpoint
    ):
        raise ValueError(
            "gat_dueling_ddqn requires --checkpoint."
        )

    torch.set_num_threads(
        max(int(args.cpu_threads), 1)
    )

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

        if request_pool.empty:
            raise RuntimeError(
                "Request trace is empty: "
                f"{args.request_trace}"
            )

        print(
            f"Using request trace: "
            f"{args.request_trace} "
            f"rows={len(request_pool):,}",
            flush=True,
        )

    model = ModelProfile(
        num_blocks=int(cfg["model"]["num_blocks"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        bytes_per_element=int(
            cfg["model"]["bytes_per_element"]
        ),
        block_parameter_gb=float(
            cfg["model"]["block_parameter_gb"]
        ),
        kv_bytes_per_token_per_block=float(
            cfg["model"]["kv_bytes_per_token_per_block"]
        ),
    )

    if args.profile_csv:
        profile = ProfileTable.from_csv(
            model,
            args.profile_csv,
        )
        print(
            f"Using measured profile: {args.profile_csv}",
            flush=True,
        )
    else:
        profile = ProfileTable(model)
        print(
            "Warning: using analytic profile fallback.",
            flush=True,
        )

    batcher = NodeConditionedDPBatcher(
        cfg,
        profile,
    )
    env = SchedulingEnv(
        cfg,
        profile,
        batcher,
    )

    probe_rng = np.random.default_rng(
        int(cfg["seed"]) + 1000
    )
    probe = env.reset(
        make_synthetic_infrastructure(
            cfg,
            probe_rng,
        ),
        make_request_queue(
            cfg,
            probe_rng,
        ),
    )

    agent = None
    if "gat_dueling_ddqn" in args.methods:
        agent = DDQNAgent(
            cfg,
            node_dim=probe.node_features.shape[1],
            edge_dim=probe.edge_features.shape[1],
            batch_dim=probe.batch_features.shape[0],
            device=args.device,
        )
        agent.load(args.checkpoint)
        agent.online.eval()

        print(
            f"Loaded checkpoint: {args.checkpoint}",
            flush=True,
        )
        print(
            f"Agent device: {agent.device}",
            flush=True,
        )

    policies = {
        "cloud_only": (
            lambda obs, nodes:
            choose_cloud(obs, nodes, "cloud_0")
        ),
        "anchor_first": (
            lambda obs, nodes:
            choose_anchor(obs, nodes, "edge_0")
        ),
        "latency_greedy": (
            lambda obs, nodes:
            choose_min_lower_bound(obs)
        ),
    }

    if agent is not None:
        policies["gat_dueling_ddqn"] = (
            lambda obs, nodes:
            agent.act(obs, deterministic=True)
        )

    selected_policies = {
        name: policies[name]
        for name in args.methods
    }

    output_path = Path(args.output)
    rows: list[dict] = []
    failures: list[dict] = []

    total_runs = (
        int(args.episodes)
        * len(selected_policies)
    )
    completed_runs = 0
    start_time = time.perf_counter()

    print(
        "Starting evaluation: "
        f"episodes={args.episodes}, "
        f"methods={list(selected_policies)}, "
        f"total_runs={total_runs}",
        flush=True,
    )

    for episode in range(args.episodes):
        for name, policy in selected_policies.items():
            episode_seed = (
                int(cfg["seed"])
                + 10000
                + episode
            )
            local_rng = np.random.default_rng(
                episode_seed
            )

            infra = make_synthetic_infrastructure(
                cfg,
                local_rng,
            )
            queue = make_request_queue(
                cfg,
                local_rng,
            )

            run_start = time.perf_counter()

            try:
                result = run_policy(
                    env=env,
                    infra=infra,
                    queue=queue,
                    policy=policy,
                    max_steps=args.max_steps,
                )
            except RuntimeError as exc:
                failures.append(
                    {
                        "episode": episode,
                        "method": name,
                        "error": str(exc),
                    }
                )
                print(
                    f"[failed] episode={episode} "
                    f"method={name}: {exc}",
                    flush=True,
                )
                continue

            run_seconds = (
                time.perf_counter() - run_start
            )

            rows.append(
                {
                    "episode": episode,
                    "method": name,
                    "reward": result.total_reward,
                    "prefill_ms": result.total_prefill_ms,
                    "decode_ms": result.expected_decode_ms,
                    "e2e_est_ms": (
                        result.total_prefill_ms
                        + result.expected_decode_ms
                        + result.handover_ms
                    ),
                    "transfer_mb": result.transfer_mb,
                    "handover_ms": result.handover_ms,
                    "slo_violations": (
                        result.slo_violations
                    ),
                    "batch_size": result.batch_size,
                    "mapping_steps": len(result.mapping),
                    "runtime_seconds": run_seconds,
                }
            )

            completed_runs += 1

        log_every = max(
            int(args.log_every),
            1,
        )

        if (
            episode == 0
            or (episode + 1) % log_every == 0
            or episode + 1 == args.episodes
        ):
            elapsed = (
                time.perf_counter() - start_time
            )
            rate = (
                completed_runs / elapsed
                if elapsed > 0
                else 0.0
            )
            remaining = total_runs - completed_runs
            eta = (
                remaining / rate
                if rate > 0
                else float("inf")
            )

            save_rows(
                rows,
                output_path,
            )

            print(
                f"episode={episode + 1:4d}/"
                f"{args.episodes} "
                f"runs={completed_runs:4d}/"
                f"{total_runs} "
                f"elapsed={elapsed:8.1f}s "
                f"eta={eta:8.1f}s "
                f"failures={len(failures)}",
                flush=True,
            )

    if not rows:
        raise RuntimeError(
            "No evaluation run completed."
        )

    save_rows(rows, output_path)

    if failures:
        failure_path = output_path.with_name(
            output_path.stem
            + "_failures.csv"
        )
        pd.DataFrame(failures).to_csv(
            failure_path,
            index=False,
        )
        print(
            f"Saved failures to {failure_path}",
            flush=True,
        )

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("method")
        .agg(
            episodes=("episode", "count"),
            reward=("reward", "mean"),
            e2e_est_ms=("e2e_est_ms", "mean"),
            prefill_ms=("prefill_ms", "mean"),
            decode_ms=("decode_ms", "mean"),
            transfer_mb=("transfer_mb", "mean"),
            handover_ms=("handover_ms", "mean"),
            slo_violations=(
                "slo_violations",
                "mean",
            ),
            batch_size=("batch_size", "mean"),
            mapping_steps=(
                "mapping_steps",
                "mean",
            ),
            runtime_seconds=(
                "runtime_seconds",
                "mean",
            ),
        )
        .round(3)
    )

    print("", flush=True)
    print(summary, flush=True)
    print(
        f"Saved {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
