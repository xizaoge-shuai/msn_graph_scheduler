from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.baselines import choose_min_lower_bound
from msn_scheduler.batching import (
    FixedSizeBatcher,
    NodeConditionedDPBatcher,
    SequentialGreedyBatcher,
    SingleRequestBatcher,
)
from msn_scheduler.config import load_config
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import ModelProfile, ProfileTable
from msn_scheduler.synthetic import make_request_queue, make_synthetic_infrastructure


def run_episode(env, infra, queue):
    obs = env.reset(infra, queue)
    result = None
    while True:
        action = choose_min_lower_bound(obs)
        obs, _, done, result = env.step(action)
        if done:
            return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--profile-csv", default=None, help="Measured profile table; analytic fallback when omitted")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--fixed-batch-size", type=int, default=4)
    parser.add_argument("--output", default=str(ROOT / "outputs/eval_batchers.csv"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    model = ModelProfile(
        num_blocks=int(cfg["model"]["num_blocks"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        bytes_per_element=int(cfg["model"]["bytes_per_element"]),
        block_parameter_gb=float(cfg["model"]["block_parameter_gb"]),
        kv_bytes_per_token_per_block=float(cfg["model"]["kv_bytes_per_token_per_block"]),
    )
    profile = ProfileTable.from_csv(model, args.profile_csv) if args.profile_csv else ProfileTable(model)
    batchers = {
        "no_batch": SingleRequestBatcher(cfg, profile),
        f"fixed_{args.fixed_batch_size}": FixedSizeBatcher(cfg, profile, args.fixed_batch_size),
        "sequential_greedy": SequentialGreedyBatcher(cfg, profile),
        "node_conditioned_dp": NodeConditionedDPBatcher(cfg, profile),
    }
    rows = []
    for episode in range(args.episodes):
        seed = int(cfg["seed"]) + 20000 + episode
        for name, batcher in batchers.items():
            rng = np.random.default_rng(seed)
            infra = make_synthetic_infrastructure(cfg, rng)
            queue = make_request_queue(cfg, rng)
            env = SchedulingEnv(cfg, profile, batcher)
            try:
                result = run_episode(env, infra, queue)
            except RuntimeError:
                continue
            rows.append(
                {
                    "episode": episode,
                    "batcher": name,
                    "e2e_est_ms": result.total_prefill_ms + result.expected_decode_ms + result.handover_ms,
                    "prefill_ms": result.total_prefill_ms,
                    "decode_ms": result.expected_decode_ms,
                    "transfer_mb": result.transfer_mb,
                    "handover_ms": result.handover_ms,
                    "slo_violations": result.slo_violations,
                    "batch_size": result.batch_size,
                    "reward": result.total_reward,
                }
            )
    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("batcher").mean(numeric_only=True).round(3))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
