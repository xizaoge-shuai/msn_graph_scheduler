from __future__ import annotations

from typing import Callable

import numpy as np

from .datatypes import Observation


def choose_cloud(obs: Observation, payload_node_ids: list[str], cloud_node: str) -> int:
    for idx, node_id in enumerate(payload_node_ids):
        if node_id == cloud_node:
            return idx
    return int(np.argmin(obs.candidate_group_sizes))


def choose_anchor(obs: Observation, payload_node_ids: list[str], anchor_node: str) -> int:
    anchor_actions = [i for i, n in enumerate(payload_node_ids) if n == anchor_node]
    if anchor_actions:
        return max(anchor_actions, key=lambda i: int(obs.candidate_group_sizes[i]))
    return 0


def choose_min_lower_bound(obs: Observation) -> int:
    # Candidate payloads are ActionCandidate instances in this codebase.
    return int(np.argmin([p.latency_lower_bound_ms for p in obs.candidate_payloads]))


def choose_fixed_group(obs: Observation, target_group: int) -> int:
    matches = [i for i, g in enumerate(obs.candidate_group_sizes) if int(g) == target_group]
    if matches:
        return min(matches, key=lambda i: obs.candidate_payloads[i].latency_lower_bound_ms)
    return choose_min_lower_bound(obs)
