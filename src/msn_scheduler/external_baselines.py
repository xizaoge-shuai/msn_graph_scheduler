from __future__ import annotations

import heapq
from typing import Any

import numpy as np


def _section(
    cfg: dict,
    name: str,
) -> dict:
    return (
        cfg
        .get("external_baselines", {})
        .get(name, {})
    )


def _array(
    value: Any,
    dtype=float,
) -> np.ndarray:
    return np.asarray(
        value,
        dtype=dtype,
    )


def _normalize(
    values: np.ndarray,
) -> np.ndarray:
    values = _array(values, float)

    finite = np.isfinite(values)

    if not finite.any():
        return np.zeros_like(
            values,
            dtype=float,
        )

    result = values.copy()

    fill = float(
        np.nanmedian(
            result[finite]
        )
    )

    result[~finite] = fill

    minimum = float(result.min())
    maximum = float(result.max())

    if maximum - minimum <= 1e-12:
        return np.zeros_like(
            result,
            dtype=float,
        )

    return (
        result - minimum
    ) / (
        maximum - minimum
    )


def _candidate_nodes(
    obs: Any,
) -> np.ndarray:
    values = getattr(
        obs,
        "candidate_node_indices",
        None,
    )

    if values is None:
        raise AttributeError(
            "Observation does not provide "
            "candidate_node_indices"
        )

    return _array(
        values,
        int,
    ).reshape(-1)


def _candidate_groups(
    obs: Any,
    count: int,
) -> np.ndarray:
    values = getattr(
        obs,
        "candidate_group_sizes",
        None,
    )

    if values is None:
        return np.ones(
            count,
            dtype=float,
        )

    result = _array(
        values,
        float,
    ).reshape(-1)

    if len(result) != count:
        raise ValueError(
            "candidate_group_sizes and "
            "candidate_node_indices differ "
            "in length"
        )

    return result


def _candidate_lower_bounds(
    obs: Any,
    count: int,
) -> np.ndarray:
    names = [
        "candidate_lower_bounds",
        "action_lower_bounds",
        "candidate_costs",
        "lower_bounds",
        "candidate_estimated_costs",
        "candidate_estimated_latency",
    ]

    for name in names:
        values = getattr(
            obs,
            name,
            None,
        )

        if values is None:
            continue

        result = _array(
            values,
            float,
        ).reshape(-1)

        if len(result) == count:
            return result

    # The current evaluator already enforces
    # action feasibility. Missing analytical
    # lower bounds therefore fall back to zero.
    return np.zeros(
        count,
        dtype=float,
    )


def _feature_index(
    obs: Any,
    configured: int,
    aliases: list[str],
    fallback: int,
) -> int:
    node_features = _array(
        obs.node_features,
        float,
    )

    width = int(
        node_features.shape[1]
    )

    if 0 <= configured < width:
        return configured

    feature_names = getattr(
        obs,
        "node_feature_names",
        None,
    )

    if feature_names is not None:
        normalized_names = [
            str(name).lower()
            for name in feature_names
        ]

        for alias in aliases:
            alias = alias.lower()

            for index, name in enumerate(
                normalized_names
            ):
                if alias in name:
                    return index

    return min(
        max(fallback, 0),
        width - 1,
    )


def _anchor_index(
    obs: Any,
) -> int | None:
    names = [
        "predicted_anchor_node_index",
        "next_anchor_node_index",
        "anchor_node_index",
        "source_node_index",
        "current_node_index",
        "request_node_index",
    ]

    for name in names:
        value = getattr(
            obs,
            name,
            None,
        )

        if value is None:
            continue

        array = np.asarray(
            value
        ).reshape(-1)

        if array.size:
            return int(array[0])

    return None


def _edge_costs_from_anchor(
    obs: Any,
    candidates: np.ndarray,
    feature_index: int,
    inverse: bool,
) -> np.ndarray:
    anchor = _anchor_index(obs)

    if anchor is None:
        return np.zeros(
            len(candidates),
            dtype=float,
        )

    edge_index = _array(
        obs.edge_index,
        int,
    )

    if (
        edge_index.ndim != 2
        or edge_index.size == 0
    ):
        return np.zeros(
            len(candidates),
            dtype=float,
        )

    if edge_index.shape[0] != 2:
        edge_index = edge_index.T

    edge_features = _array(
        obs.edge_features,
        float,
    )

    if edge_features.ndim == 1:
        edge_features = (
            edge_features
            .reshape(-1, 1)
        )

    if (
        edge_features.ndim != 2
        or len(edge_features)
        != edge_index.shape[1]
    ):
        return np.zeros(
            len(candidates),
            dtype=float,
        )

    feature_index = min(
        max(feature_index, 0),
        edge_features.shape[1] - 1,
    )

    adjacency: dict[
        int,
        list[tuple[int, float]],
    ] = {}

    for edge_id in range(
        edge_index.shape[1]
    ):
        source = int(
            edge_index[0, edge_id]
        )

        target = int(
            edge_index[1, edge_id]
        )

        raw = float(
            edge_features[
                edge_id,
                feature_index,
            ]
        )

        if not np.isfinite(raw):
            continue

        if inverse:
            cost = 1.0 / max(
                abs(raw),
                1e-6,
            )
        else:
            cost = max(
                abs(raw),
                1e-6,
            )

        adjacency.setdefault(
            source,
            [],
        ).append(
            (
                target,
                cost,
            )
        )

    distances = {
        anchor: 0.0,
    }

    queue = [
        (
            0.0,
            anchor,
        )
    ]

    while queue:
        distance, node = heapq.heappop(
            queue
        )

        if distance > distances.get(
            node,
            float("inf"),
        ):
            continue

        for target, cost in adjacency.get(
            node,
            [],
        ):
            candidate_distance = (
                distance + cost
            )

            if candidate_distance < distances.get(
                target,
                float("inf"),
            ):
                distances[target] = (
                    candidate_distance
                )

                heapq.heappush(
                    queue,
                    (
                        candidate_distance,
                        target,
                    ),
                )

    finite_distances = [
        value
        for value in distances.values()
        if np.isfinite(value)
    ]

    fallback = (
        max(finite_distances) * 2.0
        if finite_distances
        else 1.0
    )

    return np.asarray(
        [
            distances.get(
                int(node),
                fallback,
            )
            for node in candidates
        ],
        dtype=float,
    )


def choose_rba_action(
    obs: Any,
    cfg: dict,
) -> int:
    """Adapted LECU-RBA resource-balanced mapper.

    Candidate actions are ranked by residual compute and
    memory resources. Analytical lower bound is only used
    as a small tie breaker.
    """
    section = _section(
        cfg,
        "rba",
    )

    candidates = _candidate_nodes(
        obs
    )

    node_features = _array(
        obs.node_features,
        float,
    )

    compute_index = _feature_index(
        obs,
        int(
            section.get(
                "compute_feature_index",
                -1,
            )
        ),
        [
            "compute_free",
            "compute_remaining",
            "cpu_free",
            "cpu_remaining",
            "available_compute",
        ],
        0,
    )

    memory_index = _feature_index(
        obs,
        int(
            section.get(
                "memory_feature_index",
                -1,
            )
        ),
        [
            "memory_free",
            "memory_remaining",
            "mem_free",
            "mem_remaining",
            "available_memory",
        ],
        1,
    )

    compute = _normalize(
        node_features[
            candidates,
            compute_index,
        ]
    )

    memory = _normalize(
        node_features[
            candidates,
            memory_index,
        ]
    )

    lower_bounds = _normalize(
        _candidate_lower_bounds(
            obs,
            len(candidates),
        )
    )

    compute_weight = float(
        section.get(
            "compute_weight",
            0.5,
        )
    )

    memory_weight = float(
        section.get(
            "memory_weight",
            0.5,
        )
    )

    lower_bound_weight = float(
        section.get(
            "lower_bound_tiebreak_weight",
            0.05,
        )
    )

    resource_score = (
        compute_weight * compute
        + memory_weight * memory
    )

    score = (
        resource_score
        - lower_bound_weight
        * lower_bounds
    )

    return int(
        np.argmax(score)
    )


def choose_dybap_core_action(
    obs: Any,
    cfg: dict,
) -> int:
    """Compressed DyBAP block-partition mapper.

    This is an adapted core implementation rather than the
    original MARL/PPO implementation. It combines analytical
    latency, residual resources, graph/mobility cost, and block
    group size in every block-allocation decision.
    """
    section = _section(
        cfg,
        "dybap_core",
    )

    candidates = _candidate_nodes(
        obs
    )

    groups = _candidate_groups(
        obs,
        len(candidates),
    )

    lower_bounds = _normalize(
        _candidate_lower_bounds(
            obs,
            len(candidates),
        )
    )

    node_features = _array(
        obs.node_features,
        float,
    )

    compute_index = _feature_index(
        obs,
        int(
            section.get(
                "compute_feature_index",
                -1,
            )
        ),
        [
            "compute_free",
            "compute_remaining",
            "cpu_free",
            "available_compute",
        ],
        0,
    )

    memory_index = _feature_index(
        obs,
        int(
            section.get(
                "memory_feature_index",
                -1,
            )
        ),
        [
            "memory_free",
            "memory_remaining",
            "mem_free",
            "available_memory",
        ],
        1,
    )

    compute = _normalize(
        node_features[
            candidates,
            compute_index,
        ]
    )

    memory = _normalize(
        node_features[
            candidates,
            memory_index,
        ]
    )

    resource_penalty = (
        1.0
        - 0.5 * (
            compute + memory
        )
    )

    mobility_cost = _normalize(
        _edge_costs_from_anchor(
            obs,
            candidates,
            feature_index=int(
                section.get(
                    "edge_cost_feature_index",
                    0,
                )
            ),
            inverse=bool(
                section.get(
                    "edge_feature_is_bandwidth",
                    False,
                )
            ),
        )
    )

    normalized_groups = _normalize(
        groups
    )

    if bool(
        section.get(
            "prefer_larger_groups",
            True,
        )
    ):
        group_penalty = (
            1.0
            - normalized_groups
        )
    else:
        group_penalty = (
            normalized_groups
        )

    score = (
        float(
            section.get(
                "lower_bound_weight",
                1.0,
            )
        )
        * lower_bounds
        + float(
            section.get(
                "resource_weight",
                0.25,
            )
        )
        * resource_penalty
        + float(
            section.get(
                "mobility_weight",
                0.20,
            )
        )
        * mobility_cost
        + float(
            section.get(
                "group_weight",
                0.10,
            )
        )
        * group_penalty
    )

    return int(
        np.argmin(score)
    )


def choose_external_action(
    obs: Any,
    mapper: str,
    cfg: dict,
) -> int:
    if mapper == "rba":
        return choose_rba_action(
            obs,
            cfg,
        )

    if mapper == "dybap_core":
        return choose_dybap_core_action(
            obs,
            cfg,
        )

    raise ValueError(
        f"Unknown external mapper: {mapper}"
    )
