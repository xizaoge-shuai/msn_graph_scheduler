from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
from torch import nn

from .datatypes import Observation, Transition
from .models import GraphDuelingQNetwork
from .replay import ReplayBuffer


class DDQNAgent:
    def __init__(
        self,
        cfg: dict,
        node_dim: int,
        edge_dim: int,
        batch_dim: int,
        device: str | None = None,
    ):
        scfg = cfg["scheduler"]

        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        tmc_cfg = cfg.get(
            "tmc_mobility",
            {},
        )

        self.online = GraphDuelingQNetwork(
            node_dim=node_dim,
            edge_dim=edge_dim,
            batch_dim=batch_dim,
            num_blocks=int(
                cfg["model"]["num_blocks"]
            ),
            mobility_dim=int(
                tmc_cfg.get(
                    "mobility_input_dim",
                    4,
                )
            ),
            mobility_hidden_dim=int(
                tmc_cfg.get(
                    "mobility_hidden_dim",
                    64,
                )
            ),
            mobility_residual_scale=float(
                tmc_cfg.get(
                    "mobility_residual_scale",
                    1.0,
                )
            ),
            use_mobility=bool(
                tmc_cfg.get(
                    "enabled",
                    False,
                )
            ),
        ).to(
            self.device
        )

        self.target = copy.deepcopy(
            self.online
        ).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=float(
                scfg["learning_rate"]
            ),
        )

        self.gamma = float(
            scfg["gamma"]
        )

        self.batch_size = int(
            scfg["batch_size"]
        )

        self.target_interval = int(
            scfg["target_update_interval"]
        )

        self.eps_start = float(
            scfg["epsilon_start"]
        )

        self.eps_end = float(
            scfg["epsilon_end"]
        )

        self.eps_decay = int(
            scfg["epsilon_decay_steps"]
        )

        self.replay = ReplayBuffer(
            int(
                scfg["replay_capacity"]
            )
        )

        self.steps = 0
        self.updates = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.steps / max(self.eps_decay, 1))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs: Observation, deterministic: bool = False) -> int:
        self.steps += 1
        if not deterministic and random.random() < self.epsilon():
            return random.randrange(len(obs.candidate_node_indices))
        q = self.online(obs, self.device)
        return int(torch.argmax(q).item())

    def add_transition(self, transition: Transition) -> None:
        self.replay.add(transition)

    def update(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        samples = self.replay.sample(
            self.batch_size
        )

        observations = [
            transition.observation
            for transition in samples
        ]

        # One disjoint-graph GAT pass replaces one
        # separate GAT pass per replay sample.
        q_batches = self.online.forward_batch(
            observations,
            self.device,
        )

        q_selected = torch.stack(
            [
                q_values[
                    transition.action_index
                ]
                for q_values, transition
                in zip(
                    q_batches,
                    samples,
                )
            ]
        )

        targets = torch.as_tensor(
            [
                transition.reward
                for transition in samples
            ],
            dtype=torch.float32,
            device=self.device,
        ).clone()

        nonterminal_positions = [
            index
            for index, transition
            in enumerate(samples)
            if (
                not transition.done
                and transition.next_observation
                is not None
            )
        ]

        with torch.no_grad():
            if nonterminal_positions:
                next_observations = [
                    samples[index]
                    .next_observation
                    for index
                    in nonterminal_positions
                ]

                next_online_batches = (
                    self.online.forward_batch(
                        next_observations,
                        self.device,
                    )
                )

                next_target_batches = (
                    self.target.forward_batch(
                        next_observations,
                        self.device,
                    )
                )

                for local_index, sample_index in enumerate(
                    nonterminal_positions
                ):
                    best_action = int(
                        torch.argmax(
                            next_online_batches[
                                local_index
                            ]
                        ).item()
                    )

                    next_value = (
                        next_target_batches[
                            local_index
                        ][best_action]
                    )

                    targets[sample_index] += (
                        self.gamma
                        * next_value
                    )

        loss = nn.functional.smooth_l1_loss(
            q_selected,
            targets,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online.parameters(),
            5.0,
        )

        self.optimizer.step()

        self.updates += 1

        if (
            self.updates
            % self.target_interval
            == 0
        ):
            self.target.load_state_dict(
                self.online.state_dict()
            )

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "updates": self.updates,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt.get("target", ckpt["online"]))
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = int(ckpt.get("steps", 0))
        self.updates = int(ckpt.get("updates", 0))
