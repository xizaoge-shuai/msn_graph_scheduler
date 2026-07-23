from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.batching import NodeConditionedDPBatcher
from msn_scheduler.config import load_config
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import ModelProfile, ProfileTable
from msn_scheduler.synthetic import make_request_queue, make_synthetic_infrastructure


def test_cloud_greedy_completes_episode():
    cfg = load_config(ROOT / "configs/default.yaml")
    rng = np.random.default_rng(321)
    model = ModelProfile(
        num_blocks=cfg["model"]["num_blocks"],
        hidden_size=cfg["model"]["hidden_size"],
        bytes_per_element=cfg["model"]["bytes_per_element"],
        block_parameter_gb=cfg["model"]["block_parameter_gb"],
        kv_bytes_per_token_per_block=cfg["model"]["kv_bytes_per_token_per_block"],
    )
    profile = ProfileTable(model)
    batcher = NodeConditionedDPBatcher(cfg, profile)
    env = SchedulingEnv(cfg, profile, batcher)
    infra = make_synthetic_infrastructure(cfg, rng)
    obs = env.reset(infra, make_request_queue(cfg, rng))
    result = None
    for _ in range(100):
        cloud_indices = [i for i, p in enumerate(obs.candidate_payloads) if p.node_id == infra.cloud_node]
        action = max(cloud_indices, key=lambda i: int(obs.candidate_group_sizes[i])) if cloud_indices else 0
        obs, _, done, result = env.step(action)
        if done:
            break
    assert result is not None
    assert sum(step.group_size for step in result.mapping) == model.num_blocks
