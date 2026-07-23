from __future__ import annotations

import numpy as np

from .datatypes import Infrastructure


def infrastructure_to_tensors(infra: Infrastructure) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    node_ids = list(infra.nodes.keys())
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    max_blocks = max(len(n.deployed_blocks) for n in infra.nodes.values())
    node_features: list[np.ndarray] = []
    for node_id in node_ids:
        node = infra.nodes[node_id]
        block_vec = node.deployed_blocks
        if len(block_vec) < max_blocks:
            block_vec = np.pad(block_vec, (0, max_blocks - len(block_vec)))
        type_one_hot = np.array([1.0, 0.0]) if node.node_type.startswith("edge") else np.array([0.0, 1.0])
        scalars = np.array(
            [
                node.compute_scale,
                node.free_memory_gb / max(node.total_memory_gb, 1e-6),
                node.queue_delay_ms / 100.0,
                node.background_load,
                node.kv_cache_gb / max(node.total_memory_gb, 1e-6),
            ],
            dtype=np.float32,
        )
        node_features.append(np.concatenate([scalars, block_vec.astype(np.float32), type_one_hot]))

    edges: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for (src, dst), link in infra.links.items():
        if src not in node_to_idx or dst not in node_to_idx:
            continue
        edges.append((node_to_idx[src], node_to_idx[dst]))
        edge_features.append(
            [
                np.log1p(link.bandwidth_mbps) / 8.0,
                link.latency_ms / 50.0,
                link.reliability,
            ]
        )
    if not edges:
        raise ValueError("Infrastructure graph contains no edges")
    edge_index = np.array(edges, dtype=np.int64).T
    return (
        np.stack(node_features).astype(np.float32),
        edge_index,
        np.asarray(edge_features, dtype=np.float32),
        node_ids,
    )
