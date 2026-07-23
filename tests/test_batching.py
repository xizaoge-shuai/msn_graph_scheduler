from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.batching import NodeConditionedDPBatcher
from msn_scheduler.config import load_config
from msn_scheduler.profiles import ModelProfile, ProfileTable
from msn_scheduler.synthetic import make_request_queue, make_synthetic_infrastructure


def test_dp_batch_is_feasible():
    cfg = load_config(ROOT / "configs/default.yaml")
    rng = np.random.default_rng(123)
    model = ModelProfile(
        num_blocks=cfg["model"]["num_blocks"],
        hidden_size=cfg["model"]["hidden_size"],
        bytes_per_element=cfg["model"]["bytes_per_element"],
        block_parameter_gb=cfg["model"]["block_parameter_gb"],
        kv_bytes_per_token_per_block=cfg["model"]["kv_bytes_per_token_per_block"],
    )
    profile = ProfileTable(model)
    infra = make_synthetic_infrastructure(cfg, rng)
    queue = make_request_queue(cfg, rng)
    batcher = NodeConditionedDPBatcher(cfg, profile)
    candidate = batcher.best_batch(queue, 0.0, infra, infra.cloud_node, 4)
    assert candidate is not None
    assert candidate.batch_size <= cfg["batching"]["max_batch_size"]
    assert candidate.token_sum <= cfg["batching"]["token_capacity"]
    assert candidate.estimated_memory_gb <= infra.nodes[infra.cloud_node].free_memory_gb
