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
        mobility_dim: int = 4,
        mobility_hidden_dim: int = 64,
        mobility_residual_scale: float = 1.0,
        use_mobility: bool = False,
    ):
        super().__init__()

        layers = []
        current = node_dim

        for _ in range(
            num_gat_layers
        ):
            layers.append(
                EdgeAwareGATLayer(
                    current,
                    edge_dim,
                    hidden_dim,
                )
            )

            current = hidden_dim

        self.gat_layers = nn.ModuleList(
            layers
        )

        self.batch_encoder = nn.Sequential(
            nn.Linear(
                batch_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.group_encoder = nn.Sequential(
            nn.Linear(
                1,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                32,
            ),
        )

        self.block_encoder = nn.Embedding(
            num_blocks + 1,
            32,
        )

        self.use_mobility = bool(
            use_mobility
        )

        self.mobility_dim = int(
            mobility_dim
        )

        self.mobility_hidden_dim = int(
            mobility_hidden_dim
        )

        self.mobility_residual_scale = float(
            mobility_residual_scale
        )

        if self.use_mobility:
            self.mobility_scalar_encoder = (
                nn.Sequential(
                    nn.Linear(
                        2,
                        32,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        32,
                        32,
                    ),
                    nn.ReLU(),
                )
            )

            self.mobility_encoder = (
                nn.Sequential(
                    nn.Linear(
                        hidden_dim
                        + hidden_dim
                        + 32,
                        mobility_hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        mobility_hidden_dim,
                        mobility_hidden_dim,
                    ),
                    nn.ReLU(),
                )
            )

            self.mobility_query = nn.Linear(
                hidden_dim,
                mobility_hidden_dim,
                bias=False,
            )

            self.mobility_key = nn.Linear(
                mobility_hidden_dim,
                mobility_hidden_dim,
                bias=False,
            )

            self.mobility_value = nn.Linear(
                mobility_hidden_dim,
                mobility_hidden_dim,
                bias=False,
            )

            self.relative_mobility_encoder = (
                nn.Sequential(
                    nn.Linear(
                        6,
                        mobility_hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        mobility_hidden_dim,
                        mobility_hidden_dim,
                    ),
                    nn.ReLU(),
                )
            )

            self.mobility_context_norm = (
                nn.LayerNorm(
                    mobility_hidden_dim
                )
            )

        state_dim = (
            hidden_dim
            + hidden_dim
            + 32
        )

        self.value_head = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        advantage_dim = (
            state_dim
            + hidden_dim
            + 32
        )

        if self.use_mobility:
            advantage_dim += (
                mobility_hidden_dim
            )

        self.advantage_head = nn.Sequential(
            nn.Linear(
                advantage_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        self.num_blocks = int(
            num_blocks
        )

    def _candidate_mobility_context(
        self,
        obs: Observation,
        graph_x: torch.Tensor,
        candidate_node_h: torch.Tensor,
        candidate_node_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        mobility = torch.as_tensor(
            obs.mobility_features,
            dtype=torch.float32,
            device=device,
        )

        if mobility.ndim == 1:
            mobility = mobility.unsqueeze(
                0
            )

        if (
            mobility.shape[0] == 0
            or mobility.shape[1]
            < self.mobility_dim
        ):
            mobility = torch.zeros(
                (
                    1,
                    self.mobility_dim,
                ),
                dtype=torch.float32,
                device=device,
            )

        mobility = mobility[
            :,
            : self.mobility_dim,
        ]

        num_nodes = int(
            graph_x.shape[0]
        )

        node_scale = max(
            num_nodes - 1,
            1,
        )

        current_anchor_indices = (
            torch.round(
                mobility[:, 0]
                * node_scale
            )
            .long()
            .clamp(
                0,
                num_nodes - 1,
            )
        )

        next_anchor_indices = (
            torch.round(
                mobility[:, 1]
                * node_scale
            )
            .long()
            .clamp(
                0,
                num_nodes - 1,
            )
        )

        scalar_h = (
            self.mobility_scalar_encoder(
                mobility[:, 2:4]
            )
        )

        request_input = torch.cat(
            [
                graph_x[
                    current_anchor_indices
                ],
                graph_x[
                    next_anchor_indices
                ],
                scalar_h,
            ],
            dim=-1,
        )

        request_h = self.mobility_encoder(
            request_input
        )

        query = self.mobility_query(
            candidate_node_h
        )

        key = self.mobility_key(
            request_h
        )

        value = self.mobility_value(
            request_h
        )

        scores = (
            query
            @ key.transpose(
                0,
                1,
            )
            / math.sqrt(
                float(
                    self.mobility_hidden_dim
                )
            )
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        attention_context = (
            weights
            @ value
        )

        candidate_indices = (
            candidate_node_indices
            .reshape(
                -1,
                1,
            )
        )

        current_indices = (
            current_anchor_indices
            .reshape(
                1,
                -1,
            )
        )

        next_indices = (
            next_anchor_indices
            .reshape(
                1,
                -1,
            )
        )

        current_match = (
            candidate_indices
            == current_indices
        ).float()

        next_match = (
            candidate_indices
            == next_indices
        ).float()

        probability = (
            mobility[:, 2]
            .clamp(
                0.0,
                1.0,
            )
            .reshape(
                1,
                -1,
            )
        )

        dwell = (
            mobility[:, 3]
            .clamp_min(
                1e-3
            )
            .reshape(
                1,
                -1,
            )
        )

        urgency = (
            probability
            / dwell
        ).clamp(
            0.0,
            1.0,
        )

        candidate_normalized = F.normalize(
            candidate_node_h,
            dim=-1,
        )

        current_normalized = F.normalize(
            graph_x[
                current_anchor_indices
            ],
            dim=-1,
        )

        next_normalized = F.normalize(
            graph_x[
                next_anchor_indices
            ],
            dim=-1,
        )

        current_similarity = (
            candidate_normalized
            @ current_normalized.transpose(
                0,
                1,
            )
        )

        next_similarity = (
            candidate_normalized
            @ next_normalized.transpose(
                0,
                1,
            )
        )

        relative_features = torch.stack(
            [
                current_match.mean(
                    dim=-1
                ),
                next_match.mean(
                    dim=-1
                ),
                (
                    (
                        1.0
                        - probability
                    )
                    * current_match
                ).mean(
                    dim=-1
                ),
                (
                    probability
                    * next_match
                ).mean(
                    dim=-1
                ),
                (
                    urgency
                    * next_match
                ).mean(
                    dim=-1
                ),
                (
                    probability
                    * (
                        next_similarity
                        - current_similarity
                    )
                ).mean(
                    dim=-1
                ),
            ],
            dim=-1,
        )

        relative_context = (
            self.relative_mobility_encoder(
                relative_features
            )
        )

        return self.mobility_context_norm(
            attention_context
            + self.mobility_residual_scale
            * relative_context
        )

    def forward_batch(
        self,
        observations: list[Observation],
        device: torch.device,
    ) -> list[torch.Tensor]:
        if not observations:
            return []

        node_tensors = []
        edge_index_tensors = []
        edge_attr_tensors = []

        node_ranges: list[
            tuple[int, int]
        ] = []

        node_offset = 0

        for obs in observations:
            node_x = torch.as_tensor(
                obs.node_features,
                dtype=torch.float32,
                device=device,
            )

            edge_index = torch.as_tensor(
                obs.edge_index,
                dtype=torch.long,
                device=device,
            )

            edge_attr = torch.as_tensor(
                obs.edge_features,
                dtype=torch.float32,
                device=device,
            )

            node_count = int(
                node_x.shape[0]
            )

            node_tensors.append(
                node_x
            )

            if edge_index.numel() > 0:
                edge_index_tensors.append(
                    edge_index
                    + node_offset
                )

                edge_attr_tensors.append(
                    edge_attr
                )

            node_ranges.append(
                (
                    node_offset,
                    node_offset
                    + node_count,
                )
            )

            node_offset += (
                node_count
            )

        x = torch.cat(
            node_tensors,
            dim=0,
        )

        if edge_index_tensors:
            edge_index = torch.cat(
                edge_index_tensors,
                dim=1,
            )

            edge_attr = torch.cat(
                edge_attr_tensors,
                dim=0,
            )
        else:
            edge_dim = int(
                observations[0]
                .edge_features.shape[-1]
            )

            edge_index = torch.empty(
                (
                    2,
                    0,
                ),
                dtype=torch.long,
                device=device,
            )

            edge_attr = torch.empty(
                (
                    0,
                    edge_dim,
                ),
                dtype=torch.float32,
                device=device,
            )

        for layer in self.gat_layers:
            x = layer(
                x,
                edge_index,
                edge_attr,
            )

        outputs: list[
            torch.Tensor
        ] = []

        for (
            obs,
            (
                node_start,
                node_end,
            ),
        ) in zip(
            observations,
            node_ranges,
        ):
            graph_x = x[
                node_start:node_end
            ]

            global_h = torch.mean(
                graph_x,
                dim=0,
            )

            batch_h = self.batch_encoder(
                torch.as_tensor(
                    obs.batch_features,
                    dtype=torch.float32,
                    device=device,
                )
            )

            block_idx = min(
                max(
                    int(
                        obs.current_block
                    ),
                    0,
                ),
                self.num_blocks,
            )

            block_h = self.block_encoder(
                torch.tensor(
                    block_idx,
                    dtype=torch.long,
                    device=device,
                )
            )

            state = torch.cat(
                [
                    global_h,
                    batch_h,
                    block_h,
                ],
                dim=-1,
            )

            value = self.value_head(
                state
            ).squeeze(
                -1
            )

            candidate_nodes = torch.as_tensor(
                obs.candidate_node_indices,
                dtype=torch.long,
                device=device,
            )

            groups = torch.as_tensor(
                obs.candidate_group_sizes,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(
                -1
            )

            group_h = self.group_encoder(
                groups
                / max(
                    float(
                        self.num_blocks
                    ),
                    1.0,
                )
            )

            candidate_node_h = graph_x[
                candidate_nodes
            ]

            state_expand = (
                state.unsqueeze(
                    0
                )
                .expand(
                    len(
                        candidate_nodes
                    ),
                    -1,
                )
            )

            advantage_parts = [
                state_expand,
                candidate_node_h,
                group_h,
            ]

            if self.use_mobility:
                mobility_context = (
                    self
                    ._candidate_mobility_context(
                        obs=obs,
                        graph_x=graph_x,
                        candidate_node_h=(
                            candidate_node_h
                        ),
                        candidate_node_indices=(
                            candidate_nodes
                        ),
                        device=device,
                    )
                )

                advantage_parts.append(
                    mobility_context
                )

            advantages = (
                self.advantage_head(
                    torch.cat(
                        advantage_parts,
                        dim=-1,
                    )
                )
                .squeeze(
                    -1
                )
            )

            outputs.append(
                value
                + advantages
                - advantages.mean()
            )

        return outputs

    def forward(
        self,
        obs: Observation,
        device: torch.device,
    ) -> torch.Tensor:
        return self.forward_batch(
            [
                obs
            ],
            device,
        )[0]

