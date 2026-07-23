from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .datatypes import ComputeNode, Infrastructure, Link


@dataclass
class ModelProfile:
    num_blocks: int
    hidden_size: int
    bytes_per_element: int
    block_parameter_gb: float
    kv_bytes_per_token_per_block: float


class ProfileTable:
    """Latency/memory oracle.

    It can use measured CSV rows, while retaining an analytic fallback so the
    simulator is immediately runnable before local GPU profiling is complete.
    """

    def __init__(self, model: ModelProfile, measured: pd.DataFrame | None = None):
        self.model = model
        self.measured = measured

    @classmethod
    def from_csv(cls, model: ModelProfile, path: str | Path) -> "ProfileTable":
        df = pd.read_csv(path)
        required = {"node_type", "phase", "batch_size", "tokens", "group_size", "latency_ms"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Profile CSV missing columns: {sorted(missing)}")
        return cls(model=model, measured=df)

    def _nearest_measured(
        self, node_type: str, phase: str, batch_size: int, tokens: int, group_size: int
    ) -> float | None:
        if self.measured is None or self.measured.empty:
            return None
        subset = self.measured[
            (self.measured["node_type"] == node_type)
            & (self.measured["phase"] == phase)
        ]
        if subset.empty:
            return None
        distance = (
            np.abs(subset["batch_size"].to_numpy() - batch_size) / max(batch_size, 1)
            + np.abs(subset["tokens"].to_numpy() - tokens) / max(tokens, 1)
            + np.abs(subset["group_size"].to_numpy() - group_size) / max(group_size, 1)
        )
        row = subset.iloc[int(np.argmin(distance))]
        measured_group = max(float(row["group_size"]), 1.0)
        # Full-model profiling can be used before direct block-group profiling
        # is available. Scale by g/L as an explicit first-stage approximation.
        return float(row["latency_ms"]) * float(group_size) / measured_group

    def prefill_ms(self, node: ComputeNode, batch_size: int, max_input_tokens: int, group_size: int) -> float:
        measured = self._nearest_measured(
            node.node_type, "prefill", batch_size, max_input_tokens, group_size
        )
        if measured is not None:
            return measured * (1.0 + 0.85 * node.background_load) + node.queue_delay_ms
        # Sublinear batch scaling models GPU parallelism, quadratic token term
        # approximates attention cost, and group_size scales the layer segment.
        token_factor = max_input_tokens / 128.0
        batch_factor = 0.55 + 0.45 * (batch_size ** 0.72)
        layer_factor = group_size / max(self.model.num_blocks, 1)
        base_full_model_ms = 32.0 / max(node.compute_scale, 0.1)
        return (
            base_full_model_ms
            * layer_factor
            * batch_factor
            * (0.60 * token_factor + 0.40 * token_factor**1.55)
            * (1.0 + 1.1 * node.background_load)
            + node.queue_delay_ms
        )

    def decode_per_token_ms(self, node: ComputeNode, batch_size: int, context_tokens: int, group_size: int) -> float:
        measured = self._nearest_measured(
            node.node_type, "decode", batch_size, context_tokens, group_size
        )
        if measured is not None:
            return measured * (1.0 + 0.70 * node.background_load) + 0.02 * node.queue_delay_ms
        layer_factor = group_size / max(self.model.num_blocks, 1)
        context_factor = 1.0 + 0.18 * np.log2(max(context_tokens, 1) / 128.0 + 1.0)
        batch_factor = 0.72 + 0.28 * (batch_size ** 0.55)
        return (
            3.4
            * layer_factor
            * context_factor
            * batch_factor
            / max(node.compute_scale, 0.1)
            * (1.0 + 0.8 * node.background_load)
        )

    def activation_gb(self, batch_size: int, max_input_tokens: int) -> float:
        return (
            batch_size
            * max_input_tokens
            * self.model.hidden_size
            * self.model.bytes_per_element
            / (1024**3)
        )

    def kv_cache_gb(self, batch_size: int, context_tokens: int, group_size: int) -> float:
        return (
            batch_size
            * context_tokens
            * group_size
            * self.model.kv_bytes_per_token_per_block
            / (1024**3)
        )

    def memory_gb(self, batch_size: int, max_input_tokens: int, expected_output_tokens: int, group_size: int) -> float:
        weights = group_size * self.model.block_parameter_gb
        activations = self.activation_gb(batch_size, max_input_tokens)
        kv = self.kv_cache_gb(batch_size, max_input_tokens + expected_output_tokens, group_size)
        return weights + activations + kv

    def intermediate_mb(self, batch_size: int, tokens: int, decode: bool = False) -> float:
        effective_tokens = 1 if decode else tokens
        return (
            batch_size
            * effective_tokens
            * self.model.hidden_size
            * self.model.bytes_per_element
            / (1024**2)
        )


def shortest_path_latency_ms(infra: Infrastructure, src: str, dst: str) -> float:
    if src == dst:
        return 0.0
    import heapq

    pq: list[tuple[float, str]] = [(0.0, src)]
    dist = {src: 0.0}
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            return d
        if d > dist.get(u, float("inf")):
            continue
        for (a, b), link in infra.links.items():
            if a != u:
                continue
            nd = d + link.latency_ms
            if nd < dist.get(b, float("inf")):
                dist[b] = nd
                heapq.heappush(pq, (nd, b))
    return float("inf")


def transfer_time_ms(infra: Infrastructure, src: str, dst: str, data_mb: float) -> float:
    if src == dst:
        return 0.0
    # Dijkstra with serialization + propagation cost per link.
    import heapq

    pq: list[tuple[float, str]] = [(0.0, src)]
    dist = {src: 0.0}
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            return d
        if d > dist.get(u, float("inf")):
            continue
        for (a, b), link in infra.links.items():
            if a != u or link.bandwidth_mbps <= 0:
                continue
            serialization_ms = data_mb * 8.0 / link.bandwidth_mbps * 1000.0
            reliability_penalty = 1.0 / max(link.reliability, 1e-3)
            nd = d + (link.latency_ms + serialization_ms) * reliability_penalty
            if nd < dist.get(b, float("inf")):
                dist[b] = nd
                heapq.heappush(pq, (nd, b))
    return float("inf")
