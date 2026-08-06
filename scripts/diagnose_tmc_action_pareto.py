from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

sys.path.insert(
    0,
    str(ROOT / "src"),
)

import scripts.evaluate_dynamic_mobility as evaluator

from msn_scheduler.profiles import (
    shortest_path_latency_ms,
)


OUTPUT = Path(
    os.environ["TMC_ACTION_PARETO_OUTPUT"]
)

ORIGINAL_CHOOSE_ACTION = (
    evaluator.choose_action
)

ORIGINAL_RUN_ONE_MAPPING = (
    evaluator.run_one_mapping
)

ORIGINAL_RUN_DYNAMIC_EPISODE = (
    evaluator.run_dynamic_episode
)


STATE = {
    "episode_seed": None,
    "mapping_id": -1,
    "decision_id": -1,
}

ROWS: list[dict] = []


def finite_path_latency_ms(
    infra,
    source: str,
    destination: str,
) -> float:
    value = float(
        shortest_path_latency_ms(
            infra,
            source,
            destination,
        )
    )

    return (
        value
        if np.isfinite(value)
        else 1e6
    )


def incremental_mobility_costs_ms(
    infra,
    actions,
    fallback_requests,
) -> np.ndarray:
    """Pure mobility-induced incremental cost.

    This deliberately excludes current-access latency,
    which exists even when the user is stationary.
    """

    costs: list[float] = []

    for action in actions:
        requests = (
            list(action.batch.requests)
            if action.batch is not None
            else list(fallback_requests)
        )

        if not requests:
            costs.append(0.0)
            continue

        execution_ms = max(
            float(
                action.latency_lower_bound_ms
            ),
            1.0,
        )

        request_costs = []

        for request in requests:
            dwell_ms = max(
                float(
                    request.residual_dwell_ms
                ),
                1.0,
            )

            effective_probability = float(
                np.clip(
                    float(
                        request.handover_probability
                    )
                    * execution_ms
                    / dwell_ms,
                    0.0,
                    1.0,
                )
            )

            current_access_ms = (
                finite_path_latency_ms(
                    infra,
                    request.anchor_node,
                    action.node_id,
                )
            )

            future_access_ms = (
                finite_path_latency_ms(
                    infra,
                    request.next_anchor_node,
                    action.node_id,
                )
            )

            incremental_handover_ms = (
                12.0
                + max(
                    0.0,
                    future_access_ms
                    - current_access_ms,
                )
            )

            request_costs.append(
                effective_probability
                * incremental_handover_ms
            )

        costs.append(
            float(
                np.mean(
                    request_costs
                )
            )
        )

    return np.asarray(
        costs,
        dtype=np.float64,
    )


def batch_ids(
    action,
    fallback_requests,
) -> tuple[str, ...]:
    requests = (
        list(action.batch.requests)
        if action.batch is not None
        else list(fallback_requests)
    )

    return tuple(
        sorted(
            request.request_id
            for request in requests
        )
    )


def traced_choose_action(
    observation,
    method,
    cfg,
    agent,
    env,
    mobility_prior_beta,
):
    selected_index = (
        ORIGINAL_CHOOSE_ACTION(
            observation=observation,
            method=method,
            cfg=cfg,
            agent=agent,
            env=env,
            mobility_prior_beta=(
                mobility_prior_beta
            ),
        )
    )

    if (
        method != "full"
        or agent is None
        or STATE["episode_seed"] is None
    ):
        return selected_index

    if mobility_prior_beta != 0.0:
        raise RuntimeError(
            "Action Pareto diagnostic requires "
            "--mobility-prior-beta 0."
        )

    STATE["decision_id"] += 1

    actions = list(
        observation.candidate_payloads
    )

    if not actions:
        return selected_index

    if not (
        0 <= selected_index < len(actions)
    ):
        raise RuntimeError(
            "Selected action index is invalid."
        )

    with torch.no_grad():
        q_values = (
            agent.online(
                observation,
                agent.device,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    if q_values.shape != (len(actions),):
        raise RuntimeError(
            "Unexpected Q-value shape: "
            f"{q_values.shape}, "
            f"actions={len(actions)}"
        )

    expected_index = int(
        np.argmax(q_values)
    )

    if expected_index != selected_index:
        raise RuntimeError(
            "Baseline action differs from Q argmax: "
            f"selected={selected_index}, "
            f"argmax={expected_index}"
        )

    fallback_requests = (
        list(env.batch.requests)
        if env.batch is not None
        else list(
            env.queue[
                : env.batcher.window_size
            ]
        )
    )

    mobility_costs = (
        incremental_mobility_costs_ms(
            infra=env.infra,
            actions=actions,
            fallback_requests=(
                fallback_requests
            ),
        )
    )

    selected = actions[selected_index]

    selected_ids = batch_ids(
        selected,
        fallback_requests,
    )

    selected_q = float(
        q_values[selected_index]
    )

    selected_latency = max(
        float(
            selected.latency_lower_bound_ms
        ),
        1e-6,
    )

    selected_mobility = float(
        mobility_costs[selected_index]
    )

    q_scale = max(
        float(np.std(q_values)),
        1e-6,
    )

    stage = (
        "initial"
        if env.batch is None
        else "continuation"
    )

    for index, candidate in enumerate(
        actions
    ):
        if index == selected_index:
            continue

        candidate_ids = batch_ids(
            candidate,
            fallback_requests,
        )

        same_batch = (
            candidate_ids == selected_ids
        )

        same_group = (
            int(candidate.group_size)
            == int(selected.group_size)
        )

        candidate_q = float(
            q_values[index]
        )

        q_regret_std = (
            selected_q
            - candidate_q
        ) / q_scale

        if q_regret_std < -1e-8:
            raise RuntimeError(
                "Alternative Q exceeds selected "
                "argmax Q."
            )

        q_regret_std = max(
            0.0,
            q_regret_std,
        )

        candidate_latency = float(
            candidate.latency_lower_bound_ms
        )

        latency_delta_ms = (
            candidate_latency
            - selected_latency
        )

        latency_regret_rel = (
            latency_delta_ms
            / selected_latency
        )

        candidate_mobility = float(
            mobility_costs[index]
        )

        mobility_reduction = (
            selected_mobility
            - candidate_mobility
        )

        ROWS.append(
            {
                "episode_seed": int(
                    STATE["episode_seed"]
                ),
                "mapping_id": int(
                    STATE["mapping_id"]
                ),
                "decision_id": int(
                    STATE["decision_id"]
                ),
                "stage": stage,
                "current_block": int(
                    env.current_block
                ),
                "candidate_count": len(
                    actions
                ),
                "same_batch": int(
                    same_batch
                ),
                "same_group": int(
                    same_group
                ),
                "selected_node": str(
                    selected.node_id
                ),
                "candidate_node": str(
                    candidate.node_id
                ),
                "selected_group_size": int(
                    selected.group_size
                ),
                "candidate_group_size": int(
                    candidate.group_size
                ),
                "selected_batch_ids": "|".join(
                    selected_ids
                ),
                "candidate_batch_ids": "|".join(
                    candidate_ids
                ),
                "selected_q": selected_q,
                "candidate_q": candidate_q,
                "q_regret_std": float(
                    q_regret_std
                ),
                "selected_latency_lb_ms": (
                    selected_latency
                ),
                "candidate_latency_lb_ms": (
                    candidate_latency
                ),
                "latency_delta_ms": float(
                    latency_delta_ms
                ),
                "latency_regret_rel": float(
                    latency_regret_rel
                ),
                "selected_mobility_cost_ms": (
                    selected_mobility
                ),
                "candidate_mobility_cost_ms": (
                    candidate_mobility
                ),
                "mobility_reduction_ms": float(
                    mobility_reduction
                ),
                "lower_mobility": int(
                    mobility_reduction
                    > 1e-12
                ),
            }
        )

    return selected_index


def traced_run_one_mapping(
    *args,
    **kwargs,
):
    STATE["mapping_id"] += 1
    STATE["decision_id"] = -1

    return ORIGINAL_RUN_ONE_MAPPING(
        *args,
        **kwargs,
    )


def traced_run_dynamic_episode(
    *args,
    **kwargs,
):
    STATE["episode_seed"] = int(
        kwargs["episode_seed"]
    )

    STATE["mapping_id"] = -1
    STATE["decision_id"] = -1

    try:
        return (
            ORIGINAL_RUN_DYNAMIC_EPISODE(
                *args,
                **kwargs,
            )
        )
    finally:
        STATE["episode_seed"] = None


def main() -> None:
    evaluator.choose_action = (
        traced_choose_action
    )

    evaluator.run_one_mapping = (
        traced_run_one_mapping
    )

    evaluator.run_dynamic_episode = (
        traced_run_dynamic_episode
    )

    try:
        evaluator.main()
    finally:
        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(ROWS).to_csv(
            OUTPUT,
            index=False,
        )

        print(
            f"Saved action Pareto audit: "
            f"{OUTPUT}",
            flush=True,
        )


if __name__ == "__main__":
    main()
