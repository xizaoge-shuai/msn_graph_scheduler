from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(
        ROOT / "src"
    ),
)


from msn_scheduler.agent import DDQNAgent
from msn_scheduler.batching import (
    NodeConditionedDPBatcher,
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


def build_profile(
    cfg: dict,
) -> ProfileTable:
    model_cfg = cfg["model"]

    model = ModelProfile(
        num_blocks=int(
            model_cfg["num_blocks"]
        ),
        hidden_size=int(
            model_cfg["hidden_size"]
        ),
        bytes_per_element=int(
            model_cfg[
                "bytes_per_element"
            ]
        ),
        block_parameter_gb=float(
            model_cfg[
                "block_parameter_gb"
            ]
        ),
        kv_bytes_per_token_per_block=float(
            model_cfg[
                "kv_bytes_per_token_per_block"
            ]
        ),
    )

    return ProfileTable(
        model
    )


def locate_state_dict(
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    for key in [
        "online",
        "online_state_dict",
        "model_state_dict",
        "state_dict",
        "model",
    ]:
        value = checkpoint.get(
            key
        )

        if (
            isinstance(
                value,
                dict,
            )
            and any(
                torch.is_tensor(
                    item
                )
                for item in value.values()
            )
        ):
            return value

    if any(
        torch.is_tensor(
            item
        )
        for item in checkpoint.values()
    ):
        return checkpoint

    raise KeyError(
        "Cannot locate state_dict. "
        f"Keys={list(checkpoint.keys())}"
    )


def build_agent(
    cfg: dict,
    observation,
    checkpoint_path: Path,
    device: str,
) -> DDQNAgent:
    agent = DDQNAgent(
        cfg,
        node_dim=int(
            observation
            .node_features.shape[1]
        ),
        edge_dim=int(
            observation
            .edge_features.shape[1]
        ),
        batch_dim=int(
            observation
            .batch_features.shape[0]
        ),
        device=device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=agent.device,
        weights_only=False,
    )

    state = locate_state_dict(
        checkpoint
    )

    agent.online.load_state_dict(
        state,
        strict=True,
    )

    agent.target.load_state_dict(
        agent.online.state_dict()
    )

    agent.online.eval()
    agent.target.eval()

    return agent


def rank_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = np.asarray(
        first,
        dtype=np.float64,
    )

    second = np.asarray(
        second,
        dtype=np.float64,
    )

    if first.shape != second.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{first.shape} vs {second.shape}"
        )

    if first.size <= 1:
        return 1.0

    first_rank = (
        pd.Series(first)
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    second_rank = (
        pd.Series(second)
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    first_constant = bool(
        np.allclose(
            first_rank,
            first_rank[0],
        )
    )

    second_constant = bool(
        np.allclose(
            second_rank,
            second_rank[0],
        )
    )

    if (
        first_constant
        or second_constant
    ):
        return (
            1.0
            if np.allclose(
                first_rank,
                second_rank,
            )
            else 0.0
        )

    value = float(
        np.corrcoef(
            first_rank,
            second_rank,
        )[0, 1]
    )

    return (
        value
        if np.isfinite(
            value
        )
        else 1.0
    )


def evaluate_pair(
    method: str,
    agent: DDQNAgent,
    stable,
    risky,
    target_node: int,
) -> dict:
    with torch.no_grad():
        stable_q = (
            agent.online(
                stable,
                agent.device,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

        risky_q = (
            agent.online(
                risky,
                agent.device,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float64
            )
        )

    stable_action = int(
        np.argmax(
            stable_q
        )
    )

    risky_action = int(
        np.argmax(
            risky_q
        )
    )

    candidate_nodes = np.asarray(
        stable.candidate_node_indices,
        dtype=np.int64,
    )

    candidate_groups = np.asarray(
        stable.candidate_group_sizes,
        dtype=np.int64,
    )

    target_mask = (
        candidate_nodes
        == target_node
    )

    other_mask = (
        ~target_mask
    )

    if (
        np.any(
            target_mask
        )
        and np.any(
            other_mask
        )
    ):
        stable_margin = float(
            stable_q[
                target_mask
            ].mean()
            - stable_q[
                other_mask
            ].mean()
        )

        risky_margin = float(
            risky_q[
                target_mask
            ].mean()
            - risky_q[
                other_mask
            ].mean()
        )
    else:
        stable_margin = 0.0
        risky_margin = 0.0

    return {
        "method": method,
        "max_abs_q_difference": float(
            np.max(
                np.abs(
                    risky_q
                    - stable_q
                )
            )
        ),
        "mean_abs_q_difference": float(
            np.mean(
                np.abs(
                    risky_q
                    - stable_q
                )
            )
        ),
        "rank_correlation": (
            rank_correlation(
                stable_q,
                risky_q,
            )
        ),
        "stable_action": stable_action,
        "risky_action": risky_action,
        "action_changed": int(
            stable_action
            != risky_action
        ),
        "stable_node": int(
            candidate_nodes[
                stable_action
            ]
        ),
        "risky_node": int(
            candidate_nodes[
                risky_action
            ]
        ),
        "node_changed": int(
            candidate_nodes[
                stable_action
            ]
            != candidate_nodes[
                risky_action
            ]
        ),
        "stable_group": int(
            candidate_groups[
                stable_action
            ]
        ),
        "risky_group": int(
            candidate_groups[
                risky_action
            ]
        ),
        "group_changed": int(
            candidate_groups[
                stable_action
            ]
            != candidate_groups[
                risky_action
            ]
        ),
        "target_node_index": int(
            target_node
        ),
        "stable_target_margin": (
            stable_margin
        ),
        "risky_target_margin": (
            risky_margin
        ),
        "target_preference_shift": float(
            risky_margin
            - stable_margin
        ),
        "num_actions": int(
            len(
                candidate_nodes
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--msn-config",
        required=True,
    )

    parser.add_argument(
        "--tmc-config",
        required=True,
    )

    parser.add_argument(
        "--msn-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--tmc-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--max-requests",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=91001,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    msn_cfg = load_config(
        args.msn_config
    )

    tmc_cfg = load_config(
        args.tmc_config
    )

    profile = build_profile(
        tmc_cfg
    )

    probe_rng = (
        np.random.default_rng(
            args.seed
        )
    )

    probe_infra = (
        make_synthetic_infrastructure(
            tmc_cfg,
            probe_rng,
        )
    )

    probe_queue = make_request_queue(
        tmc_cfg,
        probe_rng,
    )[
        : args.max_requests
    ]

    probe_batcher = (
        NodeConditionedDPBatcher(
            tmc_cfg,
            profile,
        )
    )

    probe_env = SchedulingEnv(
        tmc_cfg,
        profile,
        probe_batcher,
    )

    probe_obs = probe_env.reset(
        probe_infra,
        probe_queue,
        now_ms=float(
            probe_queue[0].arrival_ms
        ),
    )

    msn_agent = build_agent(
        msn_cfg,
        probe_obs,
        Path(
            args.msn_checkpoint
        ),
        args.device,
    )

    tmc_agent = build_agent(
        tmc_cfg,
        probe_obs,
        Path(
            args.tmc_checkpoint
        ),
        args.device,
    )

    rows = []
    skipped = 0

    for sample_index in range(
        args.samples
    ):
        sample_seed = (
            args.seed
            + sample_index
        )

        rng = np.random.default_rng(
            sample_seed
        )

        try:
            infra = (
                make_synthetic_infrastructure(
                    tmc_cfg,
                    rng,
                )
            )

            queue = make_request_queue(
                tmc_cfg,
                rng,
            )[
                : args.max_requests
            ]

            batcher = (
                NodeConditionedDPBatcher(
                    tmc_cfg,
                    profile,
                )
            )

            env = SchedulingEnv(
                tmc_cfg,
                profile,
                batcher,
            )

            observation = env.reset(
                infra,
                queue,
                now_ms=float(
                    queue[0].arrival_ms
                ),
            )
        except RuntimeError:
            skipped += 1
            continue

        stable = copy.deepcopy(
            observation
        )

        risky = copy.deepcopy(
            observation
        )

        stable.mobility_features = (
            np.asarray(
                stable.mobility_features,
                dtype=np.float32,
            ).copy()
        )

        risky.mobility_features = (
            np.asarray(
                risky.mobility_features,
                dtype=np.float32,
            ).copy()
        )

        num_nodes = int(
            observation
            .node_features.shape[0]
        )

        node_scale = max(
            num_nodes - 1,
            1,
        )

        current_anchor = int(
            observation.agent_id
        )

        unique_candidates = np.unique(
            np.asarray(
                observation
                .candidate_node_indices,
                dtype=np.int64,
            )
        )

        alternatives = [
            int(
                node
            )
            for node in unique_candidates
            if int(
                node
            )
            != current_anchor
        ]

        target_node = (
            alternatives[
                sample_index
                % len(
                    alternatives
                )
            ]
            if alternatives
            else int(
                unique_candidates[-1]
            )
        )

        stable.mobility_features[
            :,
            1,
        ] = (
            stable.mobility_features[
                :,
                0,
            ]
        )

        stable.mobility_features[
            :,
            2,
        ] = 0.05

        stable.mobility_features[
            :,
            3,
        ] = 20.0

        risky.mobility_features[
            :,
            1,
        ] = (
            target_node
            / node_scale
        )

        risky.mobility_features[
            :,
            2,
        ] = 0.95

        risky.mobility_features[
            :,
            3,
        ] = 1.0

        for method, agent in [
            (
                "MSN",
                msn_agent,
            ),
            (
                "TMC-Stage1",
                tmc_agent,
            ),
        ]:
            result = evaluate_pair(
                method,
                agent,
                stable,
                risky,
                target_node,
            )

            result["sample"] = (
                sample_index
            )

            result["sample_seed"] = (
                sample_seed
            )

            rows.append(
                result
            )

        completed = (
            sample_index
            + 1
        )

        if (
            completed % 20 == 0
            or completed
            == args.samples
        ):
            print(
                f"processed={completed}/"
                f"{args.samples} "
                f"skipped={skipped}",
                flush=True,
            )

    frame = pd.DataFrame(
        rows
    )

    if frame.empty:
        raise RuntimeError(
            "No valid evaluation states."
        )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output,
        index=False,
    )

    summary = (
        frame.groupby(
            "method"
        )
        .agg(
            samples=(
                "sample",
                "count",
            ),
            mean_max_abs_q_difference=(
                "max_abs_q_difference",
                "mean",
            ),
            median_max_abs_q_difference=(
                "max_abs_q_difference",
                "median",
            ),
            mean_rank_correlation=(
                "rank_correlation",
                "mean",
            ),
            action_change_rate=(
                "action_changed",
                "mean",
            ),
            node_change_rate=(
                "node_changed",
                "mean",
            ),
            group_change_rate=(
                "group_changed",
                "mean",
            ),
            mean_target_preference_shift=(
                "target_preference_shift",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        output.with_name(
            output.stem
            + "_summary.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n===== Action-sensitivity summary ====="
    )

    print(
        summary.round(
            10
        ).to_string(
            index=False
        )
    )

    print(
        f"\nRaw output: {output}"
    )

    print(
        f"Summary output: {summary_path}"
    )


if __name__ == "__main__":
    main()
