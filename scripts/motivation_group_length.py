from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.config import load_config
from msn_scheduler.profiles import ModelProfile, ProfileTable, transfer_time_ms
from msn_scheduler.synthetic import make_synthetic_infrastructure


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml")
    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    infra = make_synthetic_infrastructure(cfg, rng)
    model = ModelProfile(
        num_blocks=int(cfg["model"]["num_blocks"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        bytes_per_element=int(cfg["model"]["bytes_per_element"]),
        block_parameter_gb=float(cfg["model"]["block_parameter_gb"]),
        kv_bytes_per_token_per_block=float(cfg["model"]["kv_bytes_per_token_per_block"]),
    )
    profile = ProfileTable(model)
    rows = []
    for g in [1, 2, 4, 7, 14, 28]:
        current = infra.anchor_node
        block = 0
        total = 0.0
        transfers = 0
        decisions = 0
        while block < model.num_blocks:
            actual_g = min(g, model.num_blocks - block)
            feasible = [
                n for n, node in infra.nodes.items()
                if node.supports(block, actual_g)
                and profile.memory_gb(4, 512, 128, actual_g) <= node.free_memory_gb
            ]
            if not feasible:
                feasible = [infra.cloud_node]
            best = min(
                feasible,
                key=lambda n: infra.nodes[n].queue_delay_ms
                + profile.prefill_ms(infra.nodes[n], 4, 512, actual_g)
                + transfer_time_ms(infra, current, n, profile.intermediate_mb(4, 512)),
            )
            if best != current:
                total += transfer_time_ms(infra, current, best, profile.intermediate_mb(4, 512))
                transfers += 1
            total += profile.prefill_ms(infra.nodes[best], 4, 512, actual_g)
            decisions += 1
            block += actual_g
            current = best
        rows.append(
            {
                "group_size": g,
                "prefill_path_ms": total,
                "transfers": transfers,
                "decisions": decisions,
            }
        )
    out_dir = ROOT / "outputs/motivation_group_length"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    for metric in ["prefill_path_ms", "transfers", "decisions"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["group_size"], df[metric], marker="o")
        ax.set_xlabel("Continuous block group size")
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=180)
        plt.close(fig)
    print(df.round(3).to_string(index=False))
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
