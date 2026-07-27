from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .batching import HeuristicBatcherBase
from .datatypes import (
    BatchCandidate,
    Infrastructure,
    Observation,
    Request,
)
from .profiles import ProfileTable


class DyBAPFusionBatcher(
    HeuristicBatcherBase
):
    """Dynamic Batch Fusion from DyBAP.

    The decision follows the paper's rule

        Benefit(Y) > Cost(Y)

    using the measured profile as the computation
    latency oracle. Since the unified simulator uses
    pre-deployed model blocks, T_load is zero here.
    """

    def __init__(
        self,
        cfg: dict,
        profile: ProfileTable,
    ):
        super().__init__(
            cfg,
            profile,
        )

        dcfg = cfg.get(
            "dybap",
            {},
        )

        self.v_wait = float(
            dcfg.get(
                "v_wait",
                1.0,
            )
        )

        self.v_load = float(
            dcfg.get(
                "v_load",
                1.0,
            )
        )

        self.v_com = float(
            dcfg.get(
                "v_com",
                1.0,
            )
        )

    def best_batch(
        self,
        queue: list[Request],
        now_ms: float,
        infra: Infrastructure,
        node_id: str,
        group_size: int,
    ) -> BatchCandidate | None:
        if not queue:
            return None

        compatible = [
            request
            for request
            in queue[:self.window_size]
            if request.model_name
            == queue[0].model_name
        ]

        if not compatible:
            return None

        compatible.sort(
            key=lambda request: (
                request.arrival_ms,
                request.input_tokens,
            )
        )

        selected: list[Request] = []
        current: BatchCandidate | None = None

        for request in compatible:
            if len(selected) >= self.max_batch_size:
                break

            single = self._candidate_from_requests(
                [request],
                now_ms,
                infra,
                node_id,
                group_size,
            )

            if single is None:
                continue

            if current is None:
                selected = [request]
                current = single
                continue

            merged = self._candidate_from_requests(
                selected + [request],
                now_ms,
                infra,
                node_id,
                group_size,
            )

            if merged is None:
                continue

            t_batch = float(
                current.estimated_prefill_ms
            )

            t_request = float(
                single.estimated_prefill_ms
            )

            t_merged = float(
                merged.estimated_prefill_ms
            )

            # The new request waits behind the host
            # batch if it is not fused.
            t_wait_request = t_batch

            # Additional waiting introduced inside
            # the fused batch.
            t_wait_merged = max(
                0.0,
                t_merged - t_batch,
            )

            # Blocks are already deployed in the
            # unified simulator.
            t_load_request = 0.0

            benefit = (
                self.v_wait
                * (
                    t_wait_request
                    - t_wait_merged
                )
                + self.v_load
                * t_load_request
            )

            xi = float(
                len(selected)
            )

            cost = (
                self.v_com
                * (
                    (
                        t_merged
                        - t_request
                    )
                    + xi
                    * (
                        t_merged
                        - t_batch
                    )
                )
            )

            if benefit > cost:
                selected.append(request)
                current = merged

        return current


def _candidate_scalars(
    obs: Observation,
    num_blocks: int,
) -> np.ndarray:
    lower_bounds = np.asarray(
        [
            float(
                getattr(
                    payload,
                    "latency_lower_bound_ms",
                    0.0,
                )
            )
            for payload
            in obs.candidate_payloads
        ],
        dtype=np.float32,
    )

    if lower_bounds.size:
        scale = max(
            float(
                np.nanmax(
                    np.abs(
                        lower_bounds
                    )
                )
            ),
            1.0,
        )

        lower_bounds = (
            lower_bounds / scale
        )

    groups = (
        np.asarray(
            obs.candidate_group_sizes,
            dtype=np.float32,
        )
        / max(
            float(num_blocks),
            1.0,
        )
    )

    return np.stack(
        [
            lower_bounds,
            groups,
        ],
        axis=1,
    ).astype(np.float32)


class DyBAPActorCritic(nn.Module):
    """Mobility-aware distributed actors and centralized critic."""

    def __init__(
        self,
        node_dim: int,
        batch_dim: int,
        num_blocks: int,
        num_agents: int,
        hidden_dim: int = 128,
        attention_heads: int = 4,
    ):
        super().__init__()

        self.num_blocks = int(
            num_blocks
        )

        self.num_agents = int(
            num_agents
        )

        self.node_encoder = nn.Sequential(
            nn.Linear(
                node_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
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
            nn.ReLU(),
        )

        self.mobility_projection = nn.Linear(
            4,
            hidden_dim,
        )

        self.mobility_attention = (
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                batch_first=True,
            )
        )

        self.block_encoder = nn.Embedding(
            self.num_blocks + 1,
            32,
        )

        state_dim = (
            hidden_dim * 3
            + 32
        )

        candidate_dim = (
            hidden_dim
            + 2
        )

        self.actor_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        state_dim
                        + candidate_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_dim,
                        1,
                    ),
                )
                for _ in range(
                    self.num_agents
                )
            ]
        )

        self.critic = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        obs: Observation,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        node_x = torch.as_tensor(
            obs.node_features,
            dtype=torch.float32,
            device=device,
        )

        node_h = self.node_encoder(
            node_x
        )

        global_h = node_h.mean(
            dim=0
        )

        batch_h = self.batch_encoder(
            torch.as_tensor(
                obs.batch_features,
                dtype=torch.float32,
                device=device,
            )
        )

        mobility_x = torch.as_tensor(
            obs.mobility_features,
            dtype=torch.float32,
            device=device,
        )

        if mobility_x.ndim == 1:
            mobility_x = (
                mobility_x.unsqueeze(0)
            )

        mobility_h = self.mobility_projection(
            mobility_x
        ).unsqueeze(0)

        attended, _ = (
            self.mobility_attention(
                mobility_h,
                mobility_h,
                mobility_h,
                need_weights=False,
            )
        )

        mobility_pool = attended.mean(
            dim=1
        ).squeeze(0)

        block_index = min(
            max(
                int(obs.current_block),
                0,
            ),
            self.num_blocks,
        )

        block_h = self.block_encoder(
            torch.tensor(
                block_index,
                dtype=torch.long,
                device=device,
            )
        )

        state_h = torch.cat(
            [
                global_h,
                batch_h,
                mobility_pool,
                block_h,
            ],
            dim=-1,
        )

        candidate_nodes = torch.as_tensor(
            obs.candidate_node_indices,
            dtype=torch.long,
            device=device,
        )

        scalar_features = torch.as_tensor(
            _candidate_scalars(
                obs,
                self.num_blocks,
            ),
            dtype=torch.float32,
            device=device,
        )

        candidate_h = torch.cat(
            [
                node_h[candidate_nodes],
                scalar_features,
            ],
            dim=-1,
        )

        expanded_state = (
            state_h.unsqueeze(0)
            .expand(
                candidate_h.shape[0],
                -1,
            )
        )

        actor_id = int(
            obs.agent_id
        ) % self.num_agents

        logits = self.actor_heads[
            actor_id
        ](
            torch.cat(
                [
                    expanded_state,
                    candidate_h,
                ],
                dim=-1,
            )
        ).squeeze(-1)

        value = self.critic(
            state_h
        ).squeeze(-1)

        return logits, value


@dataclass
class PPORecord:
    observation: Observation
    action: int
    old_log_prob: float
    reward: float
    value: float
    done: bool


class DyBAPPPOAgent:
    def __init__(
        self,
        cfg: dict,
        node_dim: int,
        batch_dim: int,
        device: str | None = None,
    ):
        dcfg = cfg.get(
            "dybap",
            {},
        )

        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.network = DyBAPActorCritic(
            node_dim=node_dim,
            batch_dim=batch_dim,
            num_blocks=int(
                cfg["model"]["num_blocks"]
            ),
            num_agents=int(
                cfg["system"][
                    "num_edge_nodes"
                ]
            )
            + int(
                cfg["system"].get(
                    "num_cloud_nodes",
                    1,
                )
            ),
            hidden_dim=int(
                dcfg.get(
                    "hidden_dim",
                    128,
                )
            ),
            attention_heads=int(
                dcfg.get(
                    "attention_heads",
                    4,
                )
            ),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(
                dcfg.get(
                    "learning_rate",
                    3e-4,
                )
            ),
        )

        self.gamma = float(
            dcfg.get(
                "gamma",
                0.99,
            )
        )

        self.gae_lambda = float(
            dcfg.get(
                "gae_lambda",
                0.95,
            )
        )

        self.clip_ratio = float(
            dcfg.get(
                "clip_ratio",
                0.2,
            )
        )

        self.entropy_coef = float(
            dcfg.get(
                "entropy_coef",
                0.01,
            )
        )

        self.value_coef = float(
            dcfg.get(
                "value_coef",
                0.5,
            )
        )

        self.update_epochs = int(
            dcfg.get(
                "update_epochs",
                4,
            )
        )

        self.minibatch_size = int(
            dcfg.get(
                "minibatch_size",
                32,
            )
        )

        self.records: list[PPORecord] = []

    @torch.no_grad()
    def act(
        self,
        obs: Observation,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        logits, value = self.network(
            obs,
            self.device,
        )

        distribution = Categorical(
            logits=logits
        )

        if deterministic:
            action = torch.argmax(
                logits
            )
        else:
            action = (
                distribution.sample()
            )

        log_prob = distribution.log_prob(
            action
        )

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def logits(
        self,
        obs: Observation,
    ) -> torch.Tensor:
        logits, _ = self.network(
            obs,
            self.device,
        )

        return logits

    def store(
        self,
        observation: Observation,
        action: int,
        old_log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.records.append(
            PPORecord(
                observation=observation,
                action=int(action),
                old_log_prob=float(
                    old_log_prob
                ),
                reward=float(reward),
                value=float(value),
                done=bool(done),
            )
        )

    def update(
        self,
    ) -> dict[str, float]:
        if not self.records:
            return {
                "loss": float("nan"),
                "actor_loss": float("nan"),
                "critic_loss": float("nan"),
            }

        rewards = np.asarray(
            [
                record.reward
                for record in self.records
            ],
            dtype=np.float32,
        )

        values = np.asarray(
            [
                record.value
                for record in self.records
            ],
            dtype=np.float32,
        )

        dones = np.asarray(
            [
                record.done
                for record in self.records
            ],
            dtype=np.float32,
        )

        advantages = np.zeros_like(
            rewards
        )

        gae = 0.0
        next_value = 0.0

        for index in reversed(
            range(len(self.records))
        ):
            nonterminal = (
                1.0 - dones[index]
            )

            delta = (
                rewards[index]
                + self.gamma
                * next_value
                * nonterminal
                - values[index]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * nonterminal
                * gae
            )

            advantages[index] = gae
            next_value = values[index]

        returns = (
            advantages + values
        )

        advantages = (
            advantages
            - advantages.mean()
        ) / (
            advantages.std()
            + 1e-8
        )

        indices = np.arange(
            len(self.records)
        )

        losses = []
        actor_losses = []
        critic_losses = []

        for _ in range(
            self.update_epochs
        ):
            np.random.shuffle(
                indices
            )

            for start in range(
                0,
                len(indices),
                self.minibatch_size,
            ):
                batch_indices = indices[
                    start:
                    start
                    + self.minibatch_size
                ]

                new_log_probs = []
                entropies = []
                predicted_values = []

                for index in batch_indices:
                    record = self.records[
                        int(index)
                    ]

                    logits, value = (
                        self.network(
                            record.observation,
                            self.device,
                        )
                    )

                    distribution = Categorical(
                        logits=logits
                    )

                    action_tensor = torch.tensor(
                        record.action,
                        dtype=torch.long,
                        device=self.device,
                    )

                    new_log_probs.append(
                        distribution.log_prob(
                            action_tensor
                        )
                    )

                    entropies.append(
                        distribution.entropy()
                    )

                    predicted_values.append(
                        value
                    )

                new_log_probs_tensor = (
                    torch.stack(
                        new_log_probs
                    )
                )

                entropy_tensor = torch.stack(
                    entropies
                ).mean()

                predicted_values_tensor = (
                    torch.stack(
                        predicted_values
                    )
                )

                old_log_probs_tensor = (
                    torch.as_tensor(
                        [
                            self.records[
                                int(index)
                            ].old_log_prob
                            for index
                            in batch_indices
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

                advantages_tensor = (
                    torch.as_tensor(
                        advantages[
                            batch_indices
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

                returns_tensor = (
                    torch.as_tensor(
                        returns[
                            batch_indices
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

                ratio = torch.exp(
                    new_log_probs_tensor
                    - old_log_probs_tensor
                )

                unclipped = (
                    ratio
                    * advantages_tensor
                )

                clipped = (
                    torch.clamp(
                        ratio,
                        1.0
                        - self.clip_ratio,
                        1.0
                        + self.clip_ratio,
                    )
                    * advantages_tensor
                )

                actor_loss = -torch.min(
                    unclipped,
                    clipped,
                ).mean()

                critic_loss = (
                    nn.functional.mse_loss(
                        predicted_values_tensor,
                        returns_tensor,
                    )
                )

                loss = (
                    actor_loss
                    + self.value_coef
                    * critic_loss
                    - self.entropy_coef
                    * entropy_tensor
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    5.0,
                )

                self.optimizer.step()

                losses.append(
                    float(loss.item())
                )

                actor_losses.append(
                    float(
                        actor_loss.item()
                    )
                )

                critic_losses.append(
                    float(
                        critic_loss.item()
                    )
                )

        self.records.clear()

        return {
            "loss": float(
                np.mean(losses)
            ),
            "actor_loss": float(
                np.mean(actor_losses)
            ),
            "critic_loss": float(
                np.mean(critic_losses)
            ),
        }

    def save(
        self,
        path: str | Path,
    ) -> None:
        torch.save(
            {
                "network":
                    self.network.state_dict(),
                "optimizer":
                    self.optimizer.state_dict(),
            },
            str(path),
        )

    def load(
        self,
        path: str | Path,
    ) -> None:
        checkpoint = torch.load(
            str(path),
            map_location=self.device,
            weights_only=False,
        )

        self.network.load_state_dict(
            checkpoint["network"]
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )
