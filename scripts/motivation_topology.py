from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msn_scheduler.config import load_config
from msn_scheduler.graph import infrastructure_to_tensors
from msn_scheduler.models import EdgeAwareGATLayer
from msn_scheduler.profiles import ModelProfile, ProfileTable, transfer_time_ms
from msn_scheduler.synthetic import make_synthetic_infrastructure


@dataclass
class Sample:
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    label: int


class GATNodeScorer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 96):
        super().__init__()
        self.gat1 = EdgeAwareGATLayer(node_dim, edge_dim, hidden)
        self.gat2 = EdgeAwareGATLayer(hidden, edge_dim, hidden)
        self.score = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, sample: Sample, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(sample.x, dtype=torch.float32, device=device)
        ei = torch.as_tensor(sample.edge_index, dtype=torch.long, device=device)
        ea = torch.as_tensor(sample.edge_attr, dtype=torch.float32, device=device)
        h = self.gat2(self.gat1(x, ei, ea), ei, ea)
        global_h = h.mean(dim=0, keepdim=True).expand(h.shape[0], -1)
        return self.score(torch.cat([h, global_h], dim=-1)).squeeze(-1)


class FixedVectorMLP(nn.Module):
    def __init__(self, node_dim: int, max_nodes: int, hidden: int = 128):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.net = nn.Sequential(
            nn.Linear(node_dim * max_nodes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max_nodes),
        )

    def forward(self, x: np.ndarray, device: torch.device) -> torch.Tensor:
        padded = np.zeros((self.max_nodes, self.node_dim), dtype=np.float32)
        count = min(self.max_nodes, x.shape[0])
        padded[:count] = x[:count]
        return self.net(torch.as_tensor(padded.reshape(-1), device=device))


def make_sample(cfg: dict, profile: ProfileTable, rng: np.random.Generator, n_edge: int) -> Sample:
    local_cfg = copy.deepcopy(cfg)
    local_cfg["system"]["num_edge_nodes"] = n_edge
    local_cfg["system"]["anchor_node"] = "edge_0"
    infra = make_synthetic_infrastructure(local_cfg, rng)
    x, edge_index, edge_attr, node_ids = infrastructure_to_tensors(infra)
    costs = []
    data_mb = profile.intermediate_mb(batch_size=4, tokens=512)
    for node_id in node_ids:
        node = infra.nodes[node_id]
        costs.append(
            node.queue_delay_ms
            + profile.prefill_ms(node, 4, 512, 4)
            + transfer_time_ms(infra, infra.anchor_node, node_id, data_mb)
        )
    label = int(np.argmin(costs))
    return Sample(x=x, edge_index=edge_index, edge_attr=edge_attr, label=label)


def accuracy_gat(model, samples, device):
    correct = 0
    with torch.no_grad():
        for s in samples:
            pred = int(torch.argmax(model(s, device)).item())
            correct += int(pred == s.label)
    return correct / max(len(samples), 1)


def accuracy_mlp(model, samples, device):
    correct = 0
    with torch.no_grad():
        for s in samples:
            pred = int(torch.argmax(model(s.x, device)).item())
            correct += int(pred == s.label)
    return correct / max(len(samples), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=300)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output", default=str(ROOT / "outputs/motivation_topology/results.csv"))
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/default.yaml")
    rng = np.random.default_rng(int(cfg["seed"]) + 33)
    model_profile = ModelProfile(
        num_blocks=int(cfg["model"]["num_blocks"]),
        hidden_size=int(cfg["model"]["hidden_size"]),
        bytes_per_element=int(cfg["model"]["bytes_per_element"]),
        block_parameter_gb=float(cfg["model"]["block_parameter_gb"]),
        kv_bytes_per_token_per_block=float(cfg["model"]["kv_bytes_per_token_per_block"]),
    )
    profile = ProfileTable(model_profile)
    train = [make_sample(cfg, profile, rng, int(rng.choice([4, 5]))) for _ in range(args.train_samples)]
    tests = {
        4: [make_sample(cfg, profile, rng, 4) for _ in range(args.test_samples)],
        6: [make_sample(cfg, profile, rng, 6) for _ in range(args.test_samples)],
        8: [make_sample(cfg, profile, rng, 8) for _ in range(args.test_samples)],
    }
    node_dim = train[0].x.shape[1]
    edge_dim = train[0].edge_attr.shape[1]
    # Training graphs have at most 6 total nodes: five edges plus cloud.
    max_train_nodes = max(s.x.shape[0] for s in train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gat = GATNodeScorer(node_dim, edge_dim).to(device)
    mlp = FixedVectorMLP(node_dim, max_train_nodes).to(device)
    opt_g = torch.optim.Adam(gat.parameters(), lr=3e-4)
    opt_m = torch.optim.Adam(mlp.parameters(), lr=3e-4)
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        for idx in order:
            sample = train[int(idx)]
            logits_g = gat(sample, device)
            loss_g = F.cross_entropy(logits_g.unsqueeze(0), torch.tensor([sample.label], device=device))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            logits_m = mlp(sample.x, device)
            # All training labels fit the fixed output size by construction.
            loss_m = F.cross_entropy(logits_m.unsqueeze(0), torch.tensor([sample.label], device=device))
            opt_m.zero_grad(set_to_none=True)
            loss_m.backward()
            opt_m.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch+1} train_gat={accuracy_gat(gat, train[:80], device):.3f} train_mlp={accuracy_mlp(mlp, train[:80], device):.3f}")
    rows = []
    for n_edge, samples in tests.items():
        rows.append({"model": "edge_aware_gat", "num_edge_nodes": n_edge, "accuracy": accuracy_gat(gat, samples, device)})
        rows.append({"model": "fixed_vector_mlp", "num_edge_nodes": n_edge, "accuracy": accuracy_mlp(mlp, samples, device)})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
