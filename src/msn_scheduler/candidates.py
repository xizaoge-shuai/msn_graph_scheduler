from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .batching import NodeConditionedDPBatcher
from .datatypes import BatchCandidate, Infrastructure, Request
from .profiles import ProfileTable, shortest_path_latency_ms


@dataclass(frozen=True)
class ActionCandidate:
    node_id: str
    group_size: int
    batch: BatchCandidate | None
    latency_lower_bound_ms: float


def feasible_nodes_for_group(
    infra: Infrastructure,
    profile: ProfileTable,
    start_block: int,
    group_size: int,
    batch_size: int,
    max_input_tokens: int,
    expected_output_tokens: int,
) -> list[str]:
    feasible: list[str] = []
    for node_id, node in infra.nodes.items():
        if not node.supports(start_block, group_size):
            continue
        memory = profile.memory_gb(batch_size, max_input_tokens, expected_output_tokens, group_size)
        if memory > node.free_memory_gb:
            continue
        if not np.isfinite(shortest_path_latency_ms(infra, infra.anchor_node, node_id)):
            continue
        feasible.append(node_id)
    return feasible


def initial_action_candidates(
    cfg: dict,
    queue: list[Request],
    now_ms: float,
    infra: Infrastructure,
    profile: ProfileTable,
    batcher: NodeConditionedDPBatcher,
) -> list[ActionCandidate]:
    actions: list[ActionCandidate] = []
    groups = [g for g in cfg["scheduler"]["group_sizes"] if g <= profile.model.num_blocks]
    top_n = int(cfg["scheduler"]["top_n_nodes"])
    for g in groups:
        scored: list[ActionCandidate] = []
        for node_id, node in infra.nodes.items():
            if not node.supports(0, g):
                continue
            batch = batcher.best_batch(queue, now_ms, infra, node_id, g)
            if batch is None:
                continue
            lower = (
                node.queue_delay_ms
                + shortest_path_latency_ms(infra, infra.anchor_node, node_id)
                + profile.prefill_ms(node, batch.batch_size, batch.max_input_tokens, g)
            )
            scored.append(ActionCandidate(node_id, g, batch, float(lower)))
        scored.sort(key=lambda x: x.latency_lower_bound_ms)
        keep: dict[tuple[str, int], ActionCandidate] = {}
        for item in scored[:top_n]:
            keep[(item.node_id, item.group_size)] = item
        for item in scored:
            if item.node_id in {infra.anchor_node, infra.cloud_node}:
                keep[(item.node_id, item.group_size)] = item
        actions.extend(keep.values())
    return actions


def continuation_action_candidates(
    cfg: dict,
    infra: Infrastructure,
    profile: ProfileTable,
    start_block: int,
    batch: BatchCandidate,
    current_node: str,
) -> list[ActionCandidate]:
    actions: list[ActionCandidate] = []
    remaining = profile.model.num_blocks - start_block
    top_n = int(cfg["scheduler"]["top_n_nodes"])
    for g in cfg["scheduler"]["group_sizes"]:
        g = int(g)
        if g > remaining:
            continue
        feasible = feasible_nodes_for_group(
            infra,
            profile,
            start_block,
            g,
            batch.batch_size,
            batch.max_input_tokens,
            int(round(batch.expected_output_mean)),
        )
        scored: list[ActionCandidate] = []
        for node_id in feasible:
            node = infra.nodes[node_id]
            lower = (
                node.queue_delay_ms
                + shortest_path_latency_ms(infra, current_node, node_id)
                + profile.prefill_ms(node, batch.batch_size, batch.max_input_tokens, g)
            )
            scored.append(ActionCandidate(node_id, g, None, float(lower)))
        scored.sort(key=lambda x: x.latency_lower_bound_ms)
        keep: dict[tuple[str, int], ActionCandidate] = {}
        for item in scored[:top_n]:
            keep[(item.node_id, item.group_size)] = item
        for item in scored:
            if item.node_id in {infra.anchor_node, infra.cloud_node}:
                keep[(item.node_id, item.group_size)] = item
        actions.extend(keep.values())
    return actions
