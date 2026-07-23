from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.config import load_config
from msn_scheduler.profiles import ModelProfile, ProfileTable
from msn_scheduler.synthetic import make_synthetic_infrastructure


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml")
    rng = np.random.default_rng(int(cfg["seed"]))
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
    for node_id, node in infra.nodes.items():
        for batch in [1, 2, 4, 8]:
            latency = profile.prefill_ms(node, batch, 512, 4)
            memory = profile.memory_gb(batch, 512, 128, 4)
            rows.append(
                {
                    "node": node_id,
                    "node_type": node.node_type,
                    "batch_size": batch,
                    "latency_ms": latency,
                    "memory_gb": memory,
                    "throughput_req_s": batch / latency * 1000.0,
                    "feasible": memory <= node.free_memory_gb,
                }
            )
    out_dir = ROOT / "outputs/motivation_batch_node"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    for metric in ["latency_ms", "throughput_req_s", "memory_gb"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for node, group in df.groupby("node"):
            ax.plot(group["batch_size"], group[metric], marker="o", label=node)
        ax.set_xlabel("Batch size")
        ax.set_ylabel(metric)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=180)
        plt.close(fig)
    print(df.round(3).to_string(index=False))
    print(f"Saved {out_dir}")


if __name__ == "__main__":
    main()
