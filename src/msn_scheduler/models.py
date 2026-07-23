from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .datatypes import Observation


class EdgeAwareGATLayer(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False)
        self.attn = nn.Linear(out_dim * 3, 1, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.node_proj(x)
        src, dst = edge_index[0], edge_index[1]
        e = self.edge_proj(edge_attr)
        score = F.leaky_relu(self.attn(torch.cat([h[src], h[dst], e], dim=-1)).squeeze(-1), 0.2)
        # Segment softmax over incoming edges. Graphs are small, so this explicit
        # implementation avoids requiring torch-geometric.
        alpha = torch.zeros_like(score)
        for node_idx in range(h.shape[0]):
            mask = dst == node_idx
            if torch.any(mask):
                alpha[mask] = torch.softmax(score[mask], dim=0)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * h[src])
        return self.norm(F.elu(out + h))


class GraphDuelingQNetwork(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        batch_dim: int,
        num_blocks: int,
        hidden_dim: int = 128,
        num_gat_layers: int = 2,
    ):
        super().__init__()
        layers = []
        current = node_dim
        for _ in range(num_gat_layers):
            layers.append(EdgeAwareGATLayer(current, edge_dim, hidden_dim))
            current = hidden_dim
        self.gat_layers = nn.ModuleList(layers)
        self.batch_encoder = nn.Sequential(
            nn.Linear(batch_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.group_encoder = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32))
        self.block_encoder = nn.Embedding(num_blocks + 1, 32)
        state_dim = hidden_dim + hidden_dim + 32
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(state_dim + hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.num_blocks = num_blocks

    def forward(self, obs: Observation, device: torch.device) -> torch.Tensor:
        x = torch.as_tensor(obs.node_features, dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(obs.edge_index, dtype=torch.long, device=device)
        edge_attr = torch.as_tensor(obs.edge_features, dtype=torch.float32, device=device)
        for layer in self.gat_layers:
            x = layer(x, edge_index, edge_attr)
        global_h = torch.mean(x, dim=0)
        batch_h = self.batch_encoder(
            torch.as_tensor(obs.batch_features, dtype=torch.float32, device=device)
        )
        block_idx = min(max(int(obs.current_block), 0), self.num_blocks)
        block_h = self.block_encoder(torch.tensor(block_idx, device=device))
        state = torch.cat([global_h, batch_h, block_h], dim=-1)
        value = self.value_head(state).squeeze(-1)
        candidate_nodes = torch.as_tensor(obs.candidate_node_indices, dtype=torch.long, device=device)
        groups = torch.as_tensor(obs.candidate_group_sizes, dtype=torch.float32, device=device).unsqueeze(-1)
        group_h = self.group_encoder(groups / max(float(self.num_blocks), 1.0))
        state_expand = state.unsqueeze(0).expand(len(candidate_nodes), -1)
        advantages = self.advantage_head(
            torch.cat([state_expand, x[candidate_nodes], group_h], dim=-1)
        ).squeeze(-1)
        return value + advantages - advantages.mean()
