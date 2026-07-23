from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.agent import DDQNAgent
from msn_scheduler.batching import NodeConditionedDPBatcher
from msn_scheduler.config import load_config
from msn_scheduler.datatypes import Transition
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import ModelProfile, ProfileTable
from msn_scheduler.synthetic import make_request_queue, make_synthetic_infrastructure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--profile-csv", default=None, help="Measured profile table; analytic fallback when omitted")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", default=str(ROOT / "outputs/train"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
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

    # Probe dimensions from one environment instance.
    obs = env.reset(make_synthetic_infrastructure(cfg, rng), make_request_queue(cfg, rng))
    agent = DDQNAgent(
        cfg,
        node_dim=obs.node_features.shape[1],
        edge_dim=obs.edge_features.shape[1],
        batch_dim=obs.batch_features.shape[0],
        device=args.device,
    )
    episodes = args.episodes or int(cfg["training"]["episodes"])
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode in range(1, episodes + 1):
        infra = make_synthetic_infrastructure(cfg, rng)
        queue = make_request_queue(cfg, rng)
        try:
            obs = env.reset(infra, queue)
        except RuntimeError:
            continue
        done = False
        ep_loss = []
        result = None
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, result = env.step(action)
            agent.add_transition(Transition(obs, action, reward, next_obs, done))
            obs = next_obs if next_obs is not None else obs
            loss = agent.update()
            if loss is not None:
                ep_loss.append(loss)
        assert result is not None
        rows.append(
            {
                "episode": episode,
                "reward": result.total_reward,
                "prefill_ms": result.total_prefill_ms,
                "decode_ms": result.expected_decode_ms,
                "transfer_mb": result.transfer_mb,
                "handover_ms": result.handover_ms,
                "slo_violations": result.slo_violations,
                "batch_size": result.batch_size,
                "loss": float(np.mean(ep_loss)) if ep_loss else np.nan,
                "epsilon": agent.epsilon(),
            }
        )
        if episode % 20 == 0:
            recent = rows[-20:]
            print(
                f"episode={episode:5d} reward={np.mean([x['reward'] for x in recent]):8.3f} "
                f"prefill={np.mean([x['prefill_ms'] for x in recent]):7.2f}ms "
                f"slo={np.mean([x['slo_violations'] for x in recent]):5.2f} eps={agent.epsilon():.3f}"
            )
        if episode % int(cfg["training"]["checkpoint_every"]) == 0:
            agent.save(str(out_dir / f"agent_ep{episode}.pt"))
            pd.DataFrame(rows).to_csv(out_dir / "train_metrics.csv", index=False)
    agent.save(str(out_dir / "agent_final.pt"))
    pd.DataFrame(rows).to_csv(out_dir / "train_metrics.csv", index=False)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
