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
    """Read the analytical lower bound of each action."""
    payloads = getattr(
        obs,
        "candidate_payloads",
        None,
    )

    if (
        payloads is not None
        and len(payloads) == count
    ):
        result = np.asarray(
            [
                float(
                    getattr(
                        payload,
                        "latency_lower_bound_ms",
                        float("inf"),
                    )
                )
                for payload in payloads
            ],
            dtype=float,
        )

        finite = np.isfinite(result)

        if finite.any():
            maximum = float(
                np.max(result[finite])
            )

            fallback = max(
                maximum * 2.0,
                1.0,
            )

            result[~finite] = fallback
            return result

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

        result = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)

        if len(result) == count:
            return result

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
    """Idea-adapted LECU resource-balanced allocation.

    Select the edge node with the highest residual-resource
    score, then select the minimum-lower-bound action on that
    node. Cloud actions are considered only when no edge action
    is feasible.
    """
    section = _section(
        cfg,
        "rba",
    )

    candidates = _candidate_nodes(
        obs
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "RBA received no feasible action"
        )

    groups = _candidate_groups(
        obs,
        len(candidates),
    )

    node_features = np.asarray(
        obs.node_features,
        dtype=float,
    )

    if node_features.ndim != 2:
        raise ValueError(
            "node_features must be a matrix"
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
            "compute_scale",
            "compute",
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
            "available_memory",
        ],
        1,
    )

    background_index = _feature_index(
        obs,
        int(
            section.get(
                "background_load_feature_index",
                -1,
            )
        ),
        [
            "background_load",
            "load",
        ],
        3,
    )

    compute_scale = np.maximum(
        node_features[
            candidates,
            compute_index,
        ],
        0.0,
    )

    background_load = np.clip(
        node_features[
            candidates,
            background_index,
        ],
        0.0,
        1.0,
    )

    residual_compute_raw = (
        compute_scale
        * (
            1.0
            - background_load
        )
    )

    memory_raw = np.maximum(
        node_features[
            candidates,
            memory_index,
        ],
        0.0,
    )

    residual_compute = _normalize(
        residual_compute_raw
    )

    memory = _normalize(
        memory_raw
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

    resource_score = (
        compute_weight
        * residual_compute
        + memory_weight
        * memory
    )

    edge_first = bool(
        section.get(
            "edge_first",
            True,
        )
    )

    if (
        edge_first
        and node_features.shape[1] >= 2
    ):
        # The final two features are edge/cloud one-hot flags.
        edge_mask = (
            node_features[
                candidates,
                -2,
            ]
            > node_features[
                candidates,
                -1,
            ]
        )
    else:
        edge_mask = np.zeros(
            len(candidates),
            dtype=bool,
        )

    eligible_indices = np.flatnonzero(
        edge_mask
    )

    if len(eligible_indices) == 0:
        eligible_indices = np.arange(
            len(candidates),
            dtype=int,
        )

    unique_nodes = sorted(
        {
            int(candidates[index])
            for index in eligible_indices
        }
    )

    def node_key(
        node_index: int,
    ) -> tuple:
        action_indices = [
            int(index)
            for index in eligible_indices
            if int(candidates[index])
            == node_index
        ]

        representative = (
            action_indices[0]
        )

        return (
            float(
                resource_score[
                    representative
                ]
            ),
            float(
                residual_compute_raw[
                    representative
                ]
            ),
            float(
                memory_raw[
                    representative
                ]
            ),
            -int(node_index),
        )

    selected_node = max(
        unique_nodes,
        key=node_key,
    )

    node_actions = [
        int(index)
        for index in range(
            len(candidates)
        )
        if int(candidates[index])
        == selected_node
    ]

    lower_bounds = (
        _candidate_lower_bounds(
            obs,
            len(candidates),
        )
    )

    selected_action = min(
        node_actions,
        key=lambda index: (
            float(
                lower_bounds[index]
            ),
            -float(groups[index]),
            int(index),
        ),
    )

    return int(selected_action)

def choose_dybap_core_action(
    obs: Any,
    cfg: dict,
) -> int:
    """Compatibility wrapper for DyBAP-Adapted."""
    from .dybap import (
        choose_dybap_action,
    )

    return choose_dybap_action(
        obs,
        cfg,
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

    if mapper in {
        "dybap_core",
        "dybap_adapted",
    }:
        return choose_dybap_core_action(
            obs,
            cfg,
        )

    raise ValueError(
        f"Unknown external mapper: {mapper}"
    )
