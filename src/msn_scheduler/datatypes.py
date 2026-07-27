from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_ms: float
    input_tokens: int
    expected_output_tokens: int
    deadline_ms: float
    anchor_node: str
    next_anchor_node: str
    handover_probability: float
    residual_dwell_ms: float
    model_name: str = "qwen-1.5b-sim"
    service_class: str = "default"

    def remaining_deadline_ms(self, now_ms: float) -> float:
        return self.deadline_ms - max(0.0, now_ms - self.arrival_ms)


@dataclass
class ComputeNode:
    node_id: str
    node_type: str
    total_memory_gb: float
    free_memory_gb: float
    compute_scale: float
    queue_delay_ms: float
    background_load: float
    kv_cache_gb: float
    deployed_blocks: np.ndarray

    def supports(self, start_block: int, group_size: int) -> bool:
        end = start_block + group_size
        if start_block < 0 or end > len(self.deployed_blocks):
            return False
        return bool(np.all(self.deployed_blocks[start_block:end] > 0))


@dataclass(frozen=True)
class Link:
    src: str
    dst: str
    bandwidth_mbps: float
    latency_ms: float
    reliability: float = 1.0


@dataclass
class Infrastructure:
    nodes: dict[str, ComputeNode]
    links: dict[tuple[str, str], Link]
    cloud_node: str
    anchor_node: str


@dataclass
class BatchCandidate:
    requests: list[Request]
    node_id: str
    group_size: int
    utility: float
    estimated_prefill_ms: float
    estimated_memory_gb: float
    estimated_handover_ms: float
    token_sum: int
    max_input_tokens: int
    expected_output_mean: float
    min_remaining_deadline_ms: float

    @property
    def batch_size(self) -> int:
        return len(self.requests)


@dataclass(frozen=True)
class MappingStep:
    node_id: str
    start_block: int
    group_size: int
    prefill_ms: float
    transfer_ms: float
    handover_ms: float


@dataclass
class Observation:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    batch_features: np.ndarray
    current_block: int
    candidate_node_indices: np.ndarray
    candidate_group_sizes: np.ndarray
    candidate_payloads: list[object] = field(default_factory=list)
    mobility_features: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (1, 4),
            dtype=np.float32,
        )
    )
    agent_id: int = 0
    mobility_features: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (1, 4),
            dtype=np.float32,
        )
    )
    agent_id: int = 0


@dataclass
class Transition:
    observation: Observation
    action_index: int
    reward: float
    next_observation: Optional[Observation]
    done: bool
