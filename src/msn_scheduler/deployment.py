from __future__ import annotations

import numpy as np


def generate_block_layout(
    node_ids: list[str],
    num_blocks: int,
    cloud_node: str,
    coverage: float,
    mode: str,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate fixed model-block replicas shared by all compared methods."""
    result: dict[str, np.ndarray] = {}
    edge_ids = [n for n in node_ids if n != cloud_node]
    for rank, node_id in enumerate(edge_ids):
        mask = np.zeros(num_blocks, dtype=np.float32)
        if mode == "uniform":
            span = max(1, int(round(num_blocks * coverage)))
            start = int(round(rank * max(num_blocks - span, 0) / max(len(edge_ids) - 1, 1)))
            mask[start : start + span] = 1.0
        elif mode == "heterogeneous":
            local_cov = np.clip(coverage * (0.65 + 0.7 * (rank + 1) / len(edge_ids)), 0.15, 0.95)
            span = max(1, int(round(num_blocks * local_cov)))
            start = int(rng.integers(0, max(num_blocks - span + 1, 1)))
            mask[start : start + span] = 1.0
            # Add a small second replica segment for overlap diversity.
            extra = max(1, span // 5)
            start2 = int(rng.integers(0, max(num_blocks - extra + 1, 1)))
            mask[start2 : start2 + extra] = 1.0
        elif mode == "sparse":
            span = max(1, int(round(num_blocks * coverage * 0.55)))
            start = int(rng.integers(0, max(num_blocks - span + 1, 1)))
            mask[start : start + span] = 1.0
        else:
            raise ValueError(f"Unknown block layout mode: {mode}")
        result[node_id] = mask
    result[cloud_node] = np.ones(num_blocks, dtype=np.float32)
    return result
