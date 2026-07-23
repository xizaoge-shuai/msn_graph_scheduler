from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.agent import DDQNAgent
from msn_scheduler.baselines import choose_anchor, choose_cloud, choose_min_lower_bound
from msn_scheduler.batching import NodeConditionedDPBatcher
from msn_scheduler.config import load_config
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import ModelProfile, ProfileTable
from msn_scheduler.synthetic import make_request_queue, make_synthetic_infrastructure


def run_policy(env, infra, queue, policy):
    obs = env.reset(infra, queue)
    result = None
    while True:
        node_ids = [p.node_id for p in obs.candidate_payloads]
        action = policy(obs, node_ids)
        obs_next, _, done, result = env.step(action)
        if done:
            return result
        obs = obs_next


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--profile-csv", default=None, help="Measured profile table; analytic fallback when omitted")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default=str(ROOT / "outputs/eval.csv"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(int(cfg["seed"]) + 1000)
    model = ModelProfile(
        num_blocks=int(cfg["model"]["num_blocks"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        bytes_per_element=int(cfg["model"]["bytes_per_element"]),
        block_parameter_gb=float(cfg["model"]["block_parameter_gb"]),
        kv_bytes_per_token_per_block=float(cfg["model"]["kv_bytes_per_token_per_block"]),
    )
    profile = ProfileTable.from_csv(model, args.profile_csv) if args.profile_csv else ProfileTable(model)
    batcher = NodeConditionedDPBatcher(cfg, profile)
    env = SchedulingEnv(cfg, profile, batcher)
    probe = env.reset(make_synthetic_infrastructure(cfg, rng), make_request_queue(cfg, rng))
    agent = None
    if args.checkpoint:
        agent = DDQNAgent(
            cfg,
            probe.node_features.shape[1],
            probe.edge_features.shape[1],
            probe.batch_features.shape[0],
        )
        agent.load(args.checkpoint)

    policies = {
        "cloud_only": lambda obs, nodes, infra=None: choose_cloud(obs, nodes, "cloud_0"),
        "anchor_first": lambda obs, nodes, infra=None: choose_anchor(obs, nodes, "edge_0"),
        "latency_greedy": lambda obs, nodes, infra=None: choose_min_lower_bound(obs),
    }
    if agent is not None:
        policies["gat_dueling_ddqn"] = lambda obs, nodes, infra=None: agent.act(obs, deterministic=True)

    rows = []
    for ep in range(args.episodes):
        for name, policy in policies.items():
            # Generate the same episode independently for every method by reusing a seed.
            episode_seed = int(cfg["seed"]) + 10000 + ep
            local_rng = np.random.default_rng(episode_seed)
            infra = make_synthetic_infrastructure(cfg, local_rng)
            queue = make_request_queue(cfg, local_rng)
            try:
                result = run_policy(env, infra, queue, policy)
            except RuntimeError:
                continue
            rows.append(
                {
                    "episode": ep,
                    "method": name,
                    "reward": result.total_reward,
                    "prefill_ms": result.total_prefill_ms,
                    "decode_ms": result.expected_decode_ms,
                    "e2e_est_ms": result.total_prefill_ms + result.expected_decode_ms + result.handover_ms,
                    "transfer_mb": result.transfer_mb,
                    "handover_ms": result.handover_ms,
                    "slo_violations": result.slo_violations,
                    "batch_size": result.batch_size,
                }
            )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby("method").mean(numeric_only=True).round(3))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
