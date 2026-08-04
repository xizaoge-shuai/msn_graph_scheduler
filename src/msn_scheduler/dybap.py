from __future__ import annotations

import heapq
from typing import Any

import numpy as np

from .batching import HeuristicBatcherBase
from .datatypes import (
    BatchCandidate,
    Infrastructure,
    Observation,
    Request,
)
from .profiles import ProfileTable


def _normalize_cost(
    values: np.ndarray,
) -> np.ndarray:
    """Normalize a non-negative cost array to [0, 1]."""
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if values.size == 0:
        return values

    finite = np.isfinite(values)

    if not finite.any():
        return np.ones_like(values)

    result = values.copy()
    finite_values = result[finite]

    maximum = float(
        np.max(finite_values)
    )

    minimum = float(
        np.min(finite_values)
    )

    result[~finite] = maximum

    if maximum - minimum <= 1e-12:
        return np.zeros_like(result)

    return (
        result - minimum
    ) / (
        maximum - minimum
    )


def _normalize_benefit(
    values: np.ndarray,
) -> np.ndarray:
    """Normalize a benefit array to [0, 1]."""
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if values.size == 0:
        return values

    finite = np.isfinite(values)

    if not finite.any():
        return np.zeros_like(values)

    result = values.copy()
    finite_values = result[finite]

    minimum = float(
        np.min(finite_values)
    )

    maximum = float(
        np.max(finite_values)
    )

    result[~finite] = minimum

    if maximum - minimum <= 1e-12:
        return np.zeros_like(result)

    return (
        result - minimum
    ) / (
        maximum - minimum
    )


class DyBAPFusionBatcher(
    HeuristicBatcherBase
):
    """Idea-adapted DyBAP dynamic batch fusion.

    Requests are considered in arrival order. A request
    is fused into the current batch only when fusion:

    1. reduces predicted deadline violations; or
    2. keeps the same violation count and reduces the
       aggregate request completion time.

    Model blocks are already deployed in the common
    simulator, so block-loading benefit is zero.
    """

    def __init__(
        self,
        cfg: dict,
        profile: ProfileTable,
    ):
        super().__init__(
            cfg,
            profile,
        )

        dcfg = cfg.get(
            "dybap_adapted",
            {},
        )

        self.fusion_min_gain_ms = float(
            dcfg.get(
                "fusion_min_gain_ms",
                0.0,
            )
        )

    def _service_estimate_ms(
        self,
        requests: list[Request],
        now_ms: float,
        infra: Infrastructure,
        node_id: str,
        group_size: int,
    ) -> tuple[
        BatchCandidate,
        float,
    ] | None:
        candidate = (
            self._candidate_from_requests(
                requests=requests,
                now_ms=now_ms,
                infra=infra,
                node_id=node_id,
                group_size=group_size,
            )
        )

        if candidate is None:
            return None

        decode_times = [
            self._estimate_request_decode_ms(
                infra=infra,
                node_id=node_id,
                batch_size=len(requests),
                max_len=(
                    candidate.max_input_tokens
                ),
                group_size=group_size,
                req=request,
            )
            for request in requests
        ]

        service_ms = float(
            candidate.estimated_prefill_ms
            + max(
                decode_times,
                default=0.0,
            )
        )

        return candidate, service_ms

    @staticmethod
    def _violation_count(
        requests: list[Request],
        completion_times_ms: list[float],
        now_ms: float,
    ) -> int:
        return sum(
            int(
                completion_ms
                > request.remaining_deadline_ms(
                    now_ms
                )
            )
            for request, completion_ms
            in zip(
                requests,
                completion_times_ms,
            )
        )

    def best_batch(
        self,
        queue: list[Request],
        now_ms: float,
        infra: Infrastructure,
        node_id: str,
        group_size: int,
    ) -> BatchCandidate | None:
        if not queue:
            return None

        compatible = [
            request
            for request
            in queue[:self.window_size]
            if request.model_name
            == queue[0].model_name
        ]

        compatible.sort(
            key=lambda request: (
                request.arrival_ms,
                request.request_id,
            )
        )

        selected: list[Request] = []

        current_candidate: (
            BatchCandidate | None
        ) = None

        current_service_ms = 0.0

        for request in compatible:
            if (
                len(selected)
                >= self.max_batch_size
            ):
                break

            if not selected:
                first = (
                    self._service_estimate_ms(
                        requests=[request],
                        now_ms=now_ms,
                        infra=infra,
                        node_id=node_id,
                        group_size=group_size,
                    )
                )

                if first is None:
                    continue

                (
                    current_candidate,
                    current_service_ms,
                ) = first

                selected = [request]
                continue

            single = (
                self._service_estimate_ms(
                    requests=[request],
                    now_ms=now_ms,
                    infra=infra,
                    node_id=node_id,
                    group_size=group_size,
                )
            )

            merged = (
                self._service_estimate_ms(
                    requests=(
                        selected + [request]
                    ),
                    now_ms=now_ms,
                    infra=infra,
                    node_id=node_id,
                    group_size=group_size,
                )
            )

            if (
                single is None
                or merged is None
            ):
                continue

            (
                _single_candidate,
                single_service_ms,
            ) = single

            (
                merged_candidate,
                merged_service_ms,
            ) = merged

            requests_after_addition = (
                selected + [request]
            )

            separate_completion = (
                [current_service_ms]
                * len(selected)
                + [
                    current_service_ms
                    + single_service_ms
                ]
            )

            fused_completion = (
                [merged_service_ms]
                * len(
                    requests_after_addition
                )
            )

            separate_violations = (
                self._violation_count(
                    requests=(
                        requests_after_addition
                    ),
                    completion_times_ms=(
                        separate_completion
                    ),
                    now_ms=now_ms,
                )
            )

            fused_violations = (
                self._violation_count(
                    requests=(
                        requests_after_addition
                    ),
                    completion_times_ms=(
                        fused_completion
                    ),
                    now_ms=now_ms,
                )
            )

            completion_gain_ms = float(
                np.sum(
                    separate_completion
                )
                - np.sum(
                    fused_completion
                )
            )

            accept = (
                fused_violations
                < separate_violations
                or (
                    fused_violations
                    == separate_violations
                    and completion_gain_ms
                    > self.fusion_min_gain_ms
                )
            )

            if accept:
                selected.append(
                    request
                )

                current_candidate = (
                    merged_candidate
                )

                current_service_ms = (
                    merged_service_ms
                )

        return current_candidate


def _candidate_lower_bounds(
    obs: Observation,
) -> np.ndarray:
    values = []

    for payload in obs.candidate_payloads:
        values.append(
            float(
                getattr(
                    payload,
                    "latency_lower_bound_ms",
                    float("inf"),
                )
            )
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def _graph_adjacency(
    obs: Observation,
) -> dict[
    int,
    list[tuple[int, float]],
]:
    edge_index = np.asarray(
        obs.edge_index,
        dtype=np.int64,
    )

    if edge_index.ndim != 2:
        return {}

    if edge_index.shape[0] != 2:
        edge_index = edge_index.T

    edge_features = np.asarray(
        obs.edge_features,
        dtype=np.float64,
    )

    if edge_features.ndim == 1:
        edge_features = (
            edge_features.reshape(-1, 1)
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

        # Graph feature 1 stores latency / 50.
        if (
            edge_features.ndim == 2
            and edge_id
            < edge_features.shape[0]
            and edge_features.shape[1]
            >= 2
        ):
            latency_ms = max(
                float(
                    edge_features[
                        edge_id,
                        1,
                    ]
                )
                * 50.0,
                1e-6,
            )
        else:
            latency_ms = 1.0

        adjacency.setdefault(
            source,
            [],
        ).append(
            (
                target,
                latency_ms,
            )
        )

    return adjacency


def _shortest_paths(
    adjacency: dict[
        int,
        list[tuple[int, float]],
    ],
    source: int,
) -> dict[int, float]:
    distances = {
        int(source): 0.0,
    }

    queue = [
        (
            0.0,
            int(source),
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
            proposal = distance + cost

            if proposal < distances.get(
                target,
                float("inf"),
            ):
                distances[target] = proposal

                heapq.heappush(
                    queue,
                    (
                        proposal,
                        target,
                    ),
                )

    return distances


def _mobility_costs(
    obs: Observation,
    candidates: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
]:
    mobility = np.asarray(
        obs.mobility_features,
        dtype=np.float64,
    )

    if mobility.ndim == 1:
        mobility = mobility.reshape(1, -1)

    if (
        mobility.size == 0
        or mobility.shape[1] < 4
    ):
        return (
            np.zeros(
                len(candidates),
                dtype=np.float64,
            ),
            0.0,
        )

    num_nodes = int(
        obs.node_features.shape[0]
    )

    node_scale = max(
        num_nodes - 1,
        1,
    )

    adjacency = _graph_adjacency(
        obs
    )

    path_cache: dict[
        int,
        dict[int, float],
    ] = {}

    def distance(
        source: int,
        target: int,
    ) -> float:
        source = int(
            np.clip(
                source,
                0,
                num_nodes - 1,
            )
        )

        target = int(
            np.clip(
                target,
                0,
                num_nodes - 1,
            )
        )

        if source not in path_cache:
            path_cache[source] = (
                _shortest_paths(
                    adjacency,
                    source,
                )
            )

        distances = path_cache[source]

        if target in distances:
            return float(
                distances[target]
            )

        finite = [
            value
            for value in distances.values()
            if np.isfinite(value)
        ]

        return (
            max(finite) * 2.0 + 50.0
            if finite
            else 1000.0
        )

    request_rows = []

    for row in mobility:
        anchor = int(
            round(
                np.clip(
                    row[0],
                    0.0,
                    1.0,
                )
                * node_scale
            )
        )

        next_anchor = int(
            round(
                np.clip(
                    row[1],
                    0.0,
                    1.0,
                )
                * node_scale
            )
        )

        probability = float(
            np.clip(
                row[2],
                0.0,
                1.0,
            )
        )

        dwell_normalized = max(
            float(row[3]),
            0.0,
        )

        request_rows.append(
            (
                anchor,
                next_anchor,
                probability,
                dwell_normalized,
            )
        )

    mobility_pressure = float(
        np.mean(
            [
                probability
                / (
                    1.0
                    + dwell_normalized
                )
                for (
                    _anchor,
                    _next_anchor,
                    probability,
                    dwell_normalized,
                )
                in request_rows
            ]
        )
    )

    costs = []

    for candidate in candidates:
        request_costs = []

        for (
            anchor,
            next_anchor,
            probability,
            _dwell_normalized,
        ) in request_rows:
            old_path = distance(
                anchor,
                int(candidate),
            )

            new_path = distance(
                next_anchor,
                int(candidate),
            )

            handover_ms = (
                12.0
                + max(
                    0.0,
                    new_path - old_path,
                )
            )

            request_costs.append(
                probability
                * handover_ms
            )

        costs.append(
            float(
                np.mean(
                    request_costs
                )
            )
            if request_costs
            else 0.0
        )

    return (
        np.asarray(
            costs,
            dtype=np.float64,
        ),
        mobility_pressure,
    )


def choose_dybap_action(
    obs: Observation,
    cfg: dict,
) -> int:
    """DyBAP-adapted mobility-aware block allocation.

    The action cost combines:

    - measured/analytical candidate lower bound;
    - residual compute and memory;
    - predicted mobility/handover cost;
    - mobility-sensitive block-group risk;
    - excessive block fragmentation.
    """
    candidates = np.asarray(
        obs.candidate_node_indices,
        dtype=np.int64,
    ).reshape(-1)

    groups = np.asarray(
        obs.candidate_group_sizes,
        dtype=np.float64,
    ).reshape(-1)

    if len(candidates) == 0:
        raise RuntimeError(
            "DyBAP received no feasible action"
        )

    if len(groups) != len(candidates):
        raise ValueError(
            "Candidate nodes and group sizes "
            "have different lengths"
        )

    lower_bounds_raw = (
        _candidate_lower_bounds(
            obs
        )
    )

    if (
        len(lower_bounds_raw)
        != len(candidates)
    ):
        raise ValueError(
            "Candidate payload count does not "
            "match candidate action count"
        )

    node_features = np.asarray(
        obs.node_features,
        dtype=np.float64,
    )

    if node_features.shape[1] < 4:
        raise ValueError(
            "DyBAP requires standard MSN "
            "node features"
        )

    compute_scale = node_features[
        candidates,
        0,
    ]

    free_memory_ratio = np.clip(
        node_features[
            candidates,
            1,
        ],
        0.0,
        1.0,
    )

    background_load = np.clip(
        node_features[
            candidates,
            3,
        ],
        0.0,
        1.0,
    )

    residual_compute = (
        compute_scale
        * (
            1.0 - background_load
        )
    )

    compute_benefit = (
        _normalize_benefit(
            residual_compute
        )
    )

    memory_benefit = (
        _normalize_benefit(
            free_memory_ratio
        )
    )

    resource_penalty = (
        1.0
        - 0.5
        * (
            compute_benefit
            + memory_benefit
        )
    )

    (
        mobility_raw,
        mobility_pressure,
    ) = _mobility_costs(
        obs,
        candidates,
    )

    maximum_group = max(
        float(np.max(groups)),
        1.0,
    )

    group_ratio = (
        groups / maximum_group
    )

    # Large groups are more exposed when the
    # predicted handover pressure is high.
    mobility_group_risk = (
        group_ratio
        * mobility_pressure
    )

    # Very small groups create excessive
    # partition and transfer decisions.
    fragmentation_penalty = (
        1.0 - group_ratio
    )

    dcfg = cfg.get(
        "dybap_adapted",
        cfg.get(
            "external_baselines",
            {},
        ).get(
            "dybap_core",
            {},
        ),
    )

    lower_bound_weight = float(
        dcfg.get(
            "lower_bound_weight",
            1.0,
        )
    )

    resource_weight = float(
        dcfg.get(
            "resource_weight",
            0.25,
        )
    )

    mobility_weight = float(
        dcfg.get(
            "mobility_weight",
            0.20,
        )
    )

    group_risk_weight = float(
        dcfg.get(
            "group_risk_weight",
            dcfg.get(
                "group_weight",
                0.10,
            ),
        )
    )

    fragmentation_weight = float(
        dcfg.get(
            "fragmentation_weight",
            0.05,
        )
    )

    score = (
        lower_bound_weight
        * _normalize_cost(
            lower_bounds_raw
        )
        + resource_weight
        * resource_penalty
        + mobility_weight
        * _normalize_cost(
            mobility_raw
        )
        + group_risk_weight
        * mobility_group_risk
        + fragmentation_weight
        * fragmentation_penalty
    )

    best = min(
        range(len(candidates)),
        key=lambda index: (
            float(score[index]),
            float(
                lower_bounds_raw[index]
            ),
            float(
                mobility_raw[index]
            ),
            -float(groups[index]),
            int(index),
        ),
    )

    return int(best)


# Compatibility alias for old experiment scripts.
choose_dybap_core_action = (
    choose_dybap_action
)
