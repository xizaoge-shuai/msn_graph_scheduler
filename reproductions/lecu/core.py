from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class LECUStepResult:
    update_cost: float
    scheduling_cost: float
    total_cost: float
    interrupted_tasks: int
    cloud_tasks: int
    mean_task_latency_ms: float


class LECUEnvironment:
    """Layer-aware edge-cloud collaborative update environment."""

    def __init__(
        self,
        cfg: dict,
        seed: int,
    ):
        self.cfg = cfg
        self.ecfg = cfg["environment"]
        self.rng = np.random.default_rng(
            seed
        )

        self.num_nodes = int(
            self.ecfg["num_edge_nodes"]
        )

        self.num_layers = int(
            self.ecfg["num_layers"]
        )

        self.action_dim = (
            self.num_nodes + 1
        )

        self.steps = 0

        self._build_static_state()

    def _build_static_state(
        self,
    ) -> None:
        self.cpu_total = self.rng.uniform(
            self.ecfg["cpu_min"],
            self.ecfg["cpu_max"],
            size=self.num_nodes,
        )

        self.memory_total = self.rng.uniform(
            self.ecfg["memory_min_gb"],
            self.ecfg["memory_max_gb"],
            size=self.num_nodes,
        )

        self.layer_sizes_mb = (
            self.rng.uniform(
                self.ecfg[
                    "layer_size_min_mb"
                ],
                self.ecfg[
                    "layer_size_max_mb"
                ],
                size=self.num_layers,
            )
        )

        self.bandwidth = self.rng.uniform(
            self.ecfg[
                "edge_bandwidth_min_mbps"
            ],
            self.ecfg[
                "edge_bandwidth_max_mbps"
            ],
            size=(
                self.num_nodes,
                self.num_nodes,
            ),
        )

        np.fill_diagonal(
            self.bandwidth,
            0.0,
        )

    def reset(
        self,
    ) -> np.ndarray:
        self.steps = 0

        self.cpu_free = (
            self.cpu_total
            * self.rng.uniform(
                0.45,
                1.0,
                size=self.num_nodes,
            )
        )

        self.memory_free = (
            self.memory_total
            * self.rng.uniform(
                0.45,
                1.0,
                size=self.num_nodes,
            )
        )

        self.layer_state = (
            self.rng.random(
                (
                    self.num_nodes,
                    self.num_layers,
                )
            )
            < 0.35
        ).astype(np.float32)

        self.required_layers = (
            self._sample_required_layers()
        )

        self.active_tasks = [
            []
            for _ in range(
                self.num_nodes
            )
        ]

        # Container updates can begin while
        # workload is already running. Without
        # these initial tasks, updating every
        # node concurrently causes no interruption
        # and destroys LECU's central trade-off.
        self._seed_initial_tasks()

        return self.state()

    def _seed_initial_tasks(
        self,
    ) -> None:
        load_min = float(
            self.ecfg[
                "initial_task_load_min"
            ]
        )

        load_max = float(
            self.ecfg[
                "initial_task_load_max"
            ]
        )

        max_tasks = int(
            self.ecfg[
                "initial_task_max_per_node"
            ]
        )

        for node in range(
            self.num_nodes
        ):
            target_cpu = (
                self.cpu_capacity[node]
                * float(
                    self.rng.uniform(
                        load_min,
                        load_max,
                    )
                )
            )

            used_cpu = 0.0

            for _ in range(max_tasks):
                if used_cpu >= target_cpu:
                    break

                cpu_need = float(
                    self.rng.uniform(
                        self.ecfg[
                            "task_cpu_min"
                        ],
                        self.ecfg[
                            "task_cpu_max"
                        ],
                    )
                )

                memory_need = float(
                    self.rng.uniform(
                        self.ecfg[
                            "task_memory_min_gb"
                        ],
                        self.ecfg[
                            "task_memory_max_gb"
                        ],
                    )
                )

                if (
                    cpu_need
                    > self.cpu_free[node]
                    or memory_need
                    > self.memory_free[node]
                ):
                    continue

                task = {
                    "cpu": cpu_need,
                    "memory": memory_need,
                    "remaining_ms": float(
                        self.rng.uniform(
                            self.ecfg[
                                "task_duration_min_ms"
                            ],
                            self.ecfg[
                                "task_duration_max_ms"
                            ],
                        )
                    ),
                }

                self.cpu_free[node] -= cpu_need
                self.memory_free[node] -= (
                    memory_need
                )

                used_cpu += cpu_need

                self.active_tasks[node].append(
                    task
                )

    def _sample_required_layers(
        self,
    ) -> np.ndarray:
        count = int(
            self.rng.integers(
                self.ecfg[
                    "changed_layers_min"
                ],
                self.ecfg[
                    "changed_layers_max"
                ]
                + 1,
            )
        )

        selected = self.rng.choice(
            self.num_layers,
            size=count,
            replace=False,
        )

        vector = np.zeros(
            self.num_layers,
            dtype=np.float32,
        )

        vector[selected] = 1.0

        return vector

    def state(
        self,
    ) -> np.ndarray:
        resource_state = np.concatenate(
            [
                self.cpu_free
                / np.maximum(
                    self.cpu_total,
                    1e-6,
                ),
                self.memory_free
                / np.maximum(
                    self.memory_total,
                    1e-6,
                ),
                self.cpu_total
                / max(
                    float(
                        self.ecfg[
                            "cpu_max"
                        ]
                    ),
                    1.0,
                ),
                self.memory_total
                / max(
                    float(
                        self.ecfg[
                            "memory_max_gb"
                        ]
                    ),
                    1.0,
                ),
            ]
        )

        bandwidth_state = (
            self.bandwidth.reshape(-1)
            / max(
                float(
                    self.ecfg[
                        "edge_bandwidth_max_mbps"
                    ]
                ),
                1.0,
            )
        )

        return np.concatenate(
            [
                resource_state,
                bandwidth_state,
                self.required_layers,
                self.layer_state.reshape(
                    -1
                ),
                np.asarray(
                    [
                        self.steps
                        / max(
                            int(
                                self.ecfg[
                                    "versions_per_episode"
                                ]
                            ),
                            1,
                        )
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

    def _update_time_ms(
        self,
        node_index: int,
        sharing: bool,
    ) -> float:
        required = np.where(
            self.required_layers > 0
        )[0]

        total_ms = 0.0
        downloaded_mb = 0.0

        cloud_bandwidth = float(
            self.ecfg[
                "cloud_bandwidth_mbps"
            ]
        )

        for layer in required:
            if (
                self.layer_state[
                    node_index,
                    layer,
                ]
                > 0
            ):
                continue

            size_mb = float(
                self.layer_sizes_mb[
                    layer
                ]
            )

            source_bandwidth = (
                cloud_bandwidth
            )

            if sharing:
                for source in range(
                    self.num_nodes
                ):
                    if (
                        source != node_index
                        and self.layer_state[
                            source,
                            layer,
                        ]
                        > 0
                    ):
                        source_bandwidth = max(
                            source_bandwidth,
                            float(
                                self.bandwidth[
                                    source,
                                    node_index,
                                ]
                            ),
                        )

            total_ms += (
                size_mb
                * 8.0
                / max(
                    source_bandwidth,
                    1e-6,
                )
                * 1000.0
            )

            downloaded_mb += size_mb

        total_ms += (
            float(
                self.ecfg[
                    "initialization_delta"
                ]
            )
            * downloaded_mb
            / max(
                float(
                    self.cpu_total[
                        node_index
                    ]
                ),
                1e-6,
            )
            * 1000.0
        )

        return total_ms

    def estimate_update_times(
        self,
        sharing: bool = True,
    ) -> np.ndarray:
        return np.asarray(
            [
                self._update_time_ms(
                    node,
                    sharing,
                )
                for node in range(
                    self.num_nodes
                )
            ],
            dtype=np.float32,
        )

    def _schedule_tasks(
        self,
        unavailable: set[int],
        duration_ms: float,
    ) -> tuple[int, int, list[float]]:
        interruptions = 0
        cloud_tasks = 0
        latencies = []

        for node in unavailable:
            interruptions += len(
                self.active_tasks[node]
            )

            self.active_tasks[node] = []

        expected_tasks = (
            float(
                self.ecfg[
                    "task_arrival_rate_rps"
                ]
            )
            * duration_ms
            / 1000.0
        )

        task_count = int(
            self.rng.poisson(
                expected_tasks
            )
        )

        for _ in range(task_count):
            cpu_need = float(
                self.rng.uniform(
                    self.ecfg[
                        "task_cpu_min"
                    ],
                    self.ecfg[
                        "task_cpu_max"
                    ],
                )
            )

            memory_need = float(
                self.rng.uniform(
                    self.ecfg[
                        "task_memory_min_gb"
                    ],
                    self.ecfg[
                        "task_memory_max_gb"
                    ],
                )
            )

            feasible = []

            for node in range(
                self.num_nodes
            ):
                if node in unavailable:
                    continue

                if (
                    self.cpu_free[node]
                    >= cpu_need
                    and self.memory_free[node]
                    >= memory_need
                ):
                    score = (
                        self.cpu_free[node]
                        / max(
                            self.cpu_total[node],
                            1e-6,
                        )
                        + self.memory_free[node]
                        / max(
                            self.memory_total[node],
                            1e-6,
                        )
                    )

                    feasible.append(
                        (
                            score,
                            node,
                        )
                    )

            if not feasible:
                cloud_tasks += 1

                cloud_service_ms = float(
                    self.rng.uniform(
                        self.ecfg[
                            "cloud_task_duration_min_ms"
                        ],
                        self.ecfg[
                            "cloud_task_duration_max_ms"
                        ],
                    )
                )

                cloud_communication_ms = float(
                    self.rng.uniform(
                        self.ecfg[
                            "cloud_task_communication_min_ms"
                        ],
                        self.ecfg[
                            "cloud_task_communication_max_ms"
                        ],
                    )
                )

                latencies.append(
                    cloud_service_ms
                    + cloud_communication_ms
                )

                continue

            _, selected = max(
                feasible
            )

            compute_latency = (
                1000.0
                * cpu_need
                / max(
                    self.cpu_total[
                        selected
                    ],
                    1e-6,
                )
            )

            latencies.append(
                compute_latency
            )

            self.active_tasks[
                selected
            ].append(
                (
                    cpu_need,
                    memory_need,
                )
            )

        return (
            interruptions,
            cloud_tasks,
            latencies,
        )

    def step(
        self,
        action: np.ndarray,
        sharing: bool = True,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        LECUStepResult,
    ]:
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if len(action) != self.action_dim:
            raise ValueError(
                "Invalid LECU action dimension"
            )

        rho = float(
            np.clip(
                (
                    action[0] + 1.0
                )
                / 2.0,
                1.0 / self.num_nodes,
                1.0,
            )
        )

        concurrency = max(
            1,
            int(
                np.ceil(
                    rho
                    * self.num_nodes
                )
            ),
        )

        priority = action[1:]

        update_order = list(
            np.argsort(
                -priority
            )
        )

        total_update_cost = 0.0
        interruptions = 0
        cloud_tasks = 0
        task_latencies = []

        for start in range(
            0,
            self.num_nodes,
            concurrency,
        ):
            group = update_order[
                start:
                start + concurrency
            ]

            update_times = [
                self._update_time_ms(
                    node,
                    sharing,
                )
                for node in group
            ]

            duration_ms = max(
                update_times,
                default=0.0,
            )

            (
                group_interruptions,
                group_cloud_tasks,
                group_latencies,
            ) = self._schedule_tasks(
                unavailable=set(group),
                duration_ms=duration_ms,
            )

            interruptions += (
                group_interruptions
            )

            cloud_tasks += (
                group_cloud_tasks
            )

            task_latencies.extend(
                group_latencies
            )

            total_update_cost += (
                duration_ms
            )

            required = np.where(
                self.required_layers > 0
            )[0]

            for node in group:
                self.layer_state[
                    node,
                    required,
                ] = 1.0

        # LECU defines scheduling cost as
        # the number of tasks interrupted when
        # their hosting container begins updating.
        # Cloud fallback is reported separately.
        scheduling_cost = float(
            interruptions
        )

        lambda_update = float(
            self.ecfg[
                "lambda_update"
            ]
        )

        total_cost = (
            lambda_update
            * (
                total_update_cost
                / float(
                    self.ecfg[
                        "update_cost_scale"
                    ]
                )
            )
            + (
                1.0 - lambda_update
            )
            * scheduling_cost
        )

        reward = -float(
            total_cost
        )

        self.steps += 1

        done = (
            self.steps
            >= int(
                self.ecfg[
                    "versions_per_episode"
                ]
            )
        )

        self.required_layers = (
            self._sample_required_layers()
        )

        result = LECUStepResult(
            update_cost=float(
                total_update_cost
            ),
            scheduling_cost=float(
                scheduling_cost
            ),
            total_cost=float(
                total_cost
            ),
            interrupted_tasks=int(
                interruptions
            ),
            cloud_tasks=int(
                cloud_tasks
            ),
            mean_task_latency_ms=(
                float(
                    np.mean(
                        task_latencies
                    )
                )
                if task_latencies
                else 0.0
            ),
        )

        return (
            self.state(),
            reward,
            done,
            result,
        )


class GaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.body = nn.Sequential(
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
        )

        self.mean = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.log_std = nn.Linear(
            hidden_dim,
            action_dim,
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = self.body(
            state
        )

        mean = self.mean(
            hidden
        )

        log_std = torch.clamp(
            self.log_std(hidden),
            -5.0,
            2.0,
        )

        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
    ):
        mean, log_std = self(
            state
        )

        std = log_std.exp()

        distribution = Normal(
            mean,
            std,
        )

        raw_action = distribution.rsample()
        action = torch.tanh(
            raw_action
        )

        log_prob = (
            distribution.log_prob(
                raw_action
            )
            - torch.log(
                1.0
                - action.pow(2)
                + 1e-6
            )
        ).sum(
            dim=-1,
            keepdim=True,
        )

        deterministic = torch.tanh(
            mean
        )

        return (
            action,
            log_prob,
            deterministic,
        )


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                state_dim
                + action_dim,
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
        state,
        action,
    ):
        return self.network(
            torch.cat(
                [
                    state,
                    action,
                ],
                dim=-1,
            )
        )


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
    ):
        self.buffer = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.buffer.append(
            (
                np.asarray(
                    state,
                    dtype=np.float32,
                ),
                np.asarray(
                    action,
                    dtype=np.float32,
                ),
                float(reward),
                np.asarray(
                    next_state,
                    dtype=np.float32,
                ),
                float(done),
            )
        )

    def sample(
        self,
        size: int,
    ):
        samples = random.sample(
            self.buffer,
            size,
        )

        return tuple(
            np.asarray(values)
            for values in zip(
                *samples
            )
        )

    def __len__(
        self,
    ):
        return len(
            self.buffer
        )


class SACAgent:
    def __init__(
        self,
        cfg: dict,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        scfg = cfg["sac"]

        self.device = torch.device(
            device
        )

        hidden = int(
            scfg["hidden_dim"]
        )

        self.actor = GaussianActor(
            state_dim,
            action_dim,
            hidden,
        ).to(self.device)

        self.critic1 = Critic(
            state_dim,
            action_dim,
            hidden,
        ).to(self.device)

        self.critic2 = Critic(
            state_dim,
            action_dim,
            hidden,
        ).to(self.device)

        self.target1 = Critic(
            state_dim,
            action_dim,
            hidden,
        ).to(self.device)

        self.target2 = Critic(
            state_dim,
            action_dim,
            hidden,
        ).to(self.device)

        self.target1.load_state_dict(
            self.critic1.state_dict()
        )

        self.target2.load_state_dict(
            self.critic2.state_dict()
        )

        learning_rate = float(
            scfg["learning_rate"]
        )

        self.actor_optimizer = (
            torch.optim.Adam(
                self.actor.parameters(),
                lr=learning_rate,
            )
        )

        self.critic1_optimizer = (
            torch.optim.Adam(
                self.critic1.parameters(),
                lr=learning_rate,
            )
        )

        self.critic2_optimizer = (
            torch.optim.Adam(
                self.critic2.parameters(),
                lr=learning_rate,
            )
        )

        self.gamma = float(
            scfg["gamma"]
        )

        self.tau = float(
            scfg["tau"]
        )

        self.alpha = float(
            scfg["alpha"]
        )

        self.batch_size = int(
            scfg["batch_size"]
        )

        self.replay = ReplayBuffer(
            int(
                scfg[
                    "replay_capacity"
                ]
            )
        )

    @torch.no_grad()
    def act(
        self,
        state,
        deterministic=False,
    ):
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        (
            sampled,
            _,
            mean,
        ) = self.actor.sample(
            state_tensor
        )

        selected = (
            mean
            if deterministic
            else sampled
        )

        return (
            selected.squeeze(0)
            .cpu()
            .numpy()
        )

    def update(
        self,
    ):
        if len(self.replay) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay.sample(
            self.batch_size
        )

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(-1)

        with torch.no_grad():
            (
                next_actions,
                next_log_probs,
                _,
            ) = self.actor.sample(
                next_states
            )

            target_q = torch.min(
                self.target1(
                    next_states,
                    next_actions,
                ),
                self.target2(
                    next_states,
                    next_actions,
                ),
            ) - self.alpha * next_log_probs

            target = (
                rewards
                + self.gamma
                * (
                    1.0 - dones
                )
                * target_q
            )

        q1 = self.critic1(
            states,
            actions,
        )

        q2 = self.critic2(
            states,
            actions,
        )

        loss1 = (
            nn.functional.mse_loss(
                q1,
                target,
            )
        )

        loss2 = (
            nn.functional.mse_loss(
                q2,
                target,
            )
        )

        self.critic1_optimizer.zero_grad(
            set_to_none=True
        )

        loss1.backward()

        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad(
            set_to_none=True
        )

        loss2.backward()

        self.critic2_optimizer.step()

        (
            new_actions,
            log_probs,
            _,
        ) = self.actor.sample(
            states
        )

        actor_loss = (
            self.alpha
            * log_probs
            - torch.min(
                self.critic1(
                    states,
                    new_actions,
                ),
                self.critic2(
                    states,
                    new_actions,
                ),
            )
        ).mean()

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()

        self.actor_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target1.parameters(),
                self.critic1.parameters(),
            ):
                target_parameter.mul_(
                    1.0 - self.tau
                ).add_(
                    parameter,
                    alpha=self.tau,
                )

            for target_parameter, parameter in zip(
                self.target2.parameters(),
                self.critic2.parameters(),
            ):
                target_parameter.mul_(
                    1.0 - self.tau
                ).add_(
                    parameter,
                    alpha=self.tau,
                )

        return float(
            actor_loss.item()
        )

    def save(
        self,
        path,
    ):
        torch.save(
            {
                "actor":
                    self.actor.state_dict(),
                "critic1":
                    self.critic1.state_dict(),
                "critic2":
                    self.critic2.state_dict(),
            },
            str(path),
        )

    def load(
        self,
        path,
    ):
        checkpoint = torch.load(
            str(path),
            map_location=self.device,
            weights_only=False,
        )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.critic1.load_state_dict(
            checkpoint["critic1"]
        )

        self.critic2.load_state_dict(
            checkpoint["critic2"]
        )
