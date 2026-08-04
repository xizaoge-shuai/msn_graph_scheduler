from __future__ import annotations

import numpy as np

from .candidates import ActionCandidate
from .datatypes import Infrastructure, Request
from .profiles import shortest_path_latency_ms


def _finite_path_latency_ms(
    infra: Infrastructure,
    src: str,
    dst: str,
) -> float:
    value = float(
        shortest_path_latency_ms(
            infra,
            src,
            dst,
        )
    )

    return (
        value
        if np.isfinite(value)
        else 1e6
    )


def candidate_mobility_costs_ms(
    infra: Infrastructure,
    actions: list[ActionCandidate],
    fallback_requests: list[Request],
) -> np.ndarray:
    """Compute an analytical mobility cost for each action.

    For an initial action, use the action-specific selected batch.
    For continuation actions, use the already selected batch supplied
    through fallback_requests.
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
                    request.handover_probability
                    * execution_ms
                    / dwell_ms,
                    0.0,
                    1.0,
                )
            )

            current_access_ms = (
                _finite_path_latency_ms(
                    infra,
                    request.anchor_node,
                    action.node_id,
                )
            )

            future_access_ms = (
                _finite_path_latency_ms(
                    infra,
                    request.next_anchor_node,
                    action.node_id,
                )
            )

            expected_access_ms = (
                (
                    1.0
                    - effective_probability
                )
                * current_access_ms
                + effective_probability
                * future_access_ms
            )

            interruption_ms = (
                12.0
                * effective_probability
            )

            dwell_overrun_ratio = max(
                execution_ms
                - dwell_ms,
                0.0,
            ) / dwell_ms

            dwell_overrun_penalty_ms = (
                12.0
                * min(
                    dwell_overrun_ratio,
                    4.0,
                )
            )

            request_costs.append(
                expected_access_ms
                + interruption_ms
                + dwell_overrun_penalty_ms
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


def adjust_q_values_with_mobility_prior(
    q_values: np.ndarray,
    mobility_costs_ms: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Combine frozen MSN Q-values with normalized mobility costs."""
    q_values = np.asarray(
        q_values,
        dtype=np.float64,
    )

    mobility_costs_ms = np.asarray(
        mobility_costs_ms,
        dtype=np.float64,
    )

    if q_values.shape != mobility_costs_ms.shape:
        raise ValueError(
            "Q/cost shape mismatch: "
            f"{q_values.shape} vs "
            f"{mobility_costs_ms.shape}"
        )

    if beta <= 0.0:
        return q_values.copy()

    q_scale = max(
        float(
            np.std(
                q_values
            )
        ),
        1e-6,
    )

    cost_scale = max(
        float(
            np.std(
                mobility_costs_ms
            )
        ),
        1e-6,
    )

    normalized_q = (
        q_values
        - float(
            np.mean(
                q_values
            )
        )
    ) / q_scale

    normalized_cost = (
        mobility_costs_ms
        - float(
            np.mean(
                mobility_costs_ms
            )
        )
    ) / cost_scale

    return (
        normalized_q
        - float(beta)
        * normalized_cost
    )
