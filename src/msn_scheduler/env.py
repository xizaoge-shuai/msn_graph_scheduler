from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .batching import NodeConditionedDPBatcher
from .candidates import ActionCandidate, continuation_action_candidates, initial_action_candidates
from .datatypes import BatchCandidate, Infrastructure, MappingStep, Observation, Request
from .graph import infrastructure_to_tensors
from .profiles import ProfileTable, shortest_path_latency_ms, transfer_time_ms


@dataclass
class EpisodeResult:
    total_reward: float
    total_prefill_ms: float
    expected_decode_ms: float
    transfer_mb: float
    handover_ms: float
    slo_violations: int
    batch_size: int
    mapping: list[MappingStep]


class SchedulingEnv:
    """One episode maps one prefill batch across all Transformer blocks."""

    def __init__(self, cfg: dict, profile: ProfileTable, batcher: NodeConditionedDPBatcher):
        self.cfg = cfg
        self.profile = profile
        self.batcher = batcher
        scfg = cfg["scheduler"]
        self.w_prefill = float(scfg["reward_prefill"])
        self.w_transfer = float(scfg["reward_transfer"])
        self.w_handover = float(scfg["reward_handover"])
        self.w_decode = float(scfg["reward_decode"])
        self.w_slo = float(scfg["reward_slo"])
        self.infra: Infrastructure | None = None
        self.queue: list[Request] = []
        self.now_ms = 0.0
        self.batch: BatchCandidate | None = None
        self.current_block = 0
        self.current_node = ""
        self.mapping: list[MappingStep] = []
        self.actions: list[ActionCandidate] = []
        self.total_reward = 0.0
        self.total_prefill_ms = 0.0
        self.total_transfer_mb = 0.0
        self.total_handover_ms = 0.0
        self.node_ids: list[str] = []

    def reset(self, infra: Infrastructure, queue: list[Request], now_ms: float = 0.0) -> Observation:
        self.infra = infra
        self.queue = list(queue)
        self.now_ms = now_ms
        self.batch = None
        self.current_block = 0
        self.current_node = infra.anchor_node
        self.mapping = []
        self.total_reward = 0.0
        self.total_prefill_ms = 0.0
        self.total_transfer_mb = 0.0
        self.total_handover_ms = 0.0
        self.actions = initial_action_candidates(
            self.cfg, self.queue, self.now_ms, infra, self.profile, self.batcher
        )
        if not self.actions:
            raise RuntimeError("No feasible initial batch/action. Increase deadlines or free capacity.")
        return self._observation()

    def _batch_feature_vector(
        self,
        batch: BatchCandidate | None,
    ) -> np.ndarray:
        mask_mobility = bool(
            self.cfg.get(
                "ablation",
                {},
            ).get(
                "mask_mobility_features",
                False,
            )
        )

        if batch is None:
            reqs = self.queue[
                : self.batcher.window_size
            ]

            if not reqs:
                return np.zeros(
                    7,
                    dtype=np.float32,
                )

            mobility_probability = (
                0.0
                if mask_mobility
                else float(
                    np.mean(
                        [
                            request.handover_probability
                            for request in reqs
                        ]
                    )
                )
            )

            residual_dwell = (
                0.0
                if mask_mobility
                else float(
                    np.mean(
                        [
                            np.clip(
                                request.residual_dwell_ms,
                                1.0,
                                2500.0,
                            )
                            for request in reqs
                        ]
                    )
                    / 3000.0
                )
            )

            return np.array(
                [
                    0.0,
                    np.mean(
                        [
                            request.input_tokens
                            for request in reqs
                        ]
                    )
                    / 1024.0,
                    sum(
                        request.input_tokens
                        for request in reqs
                    )
                    / 8192.0,
                    np.mean(
                        [
                            request.expected_output_tokens
                            for request in reqs
                        ]
                    )
                    / 512.0,
                    min(
                        request.remaining_deadline_ms(
                            self.now_ms
                        )
                        for request in reqs
                    )
                    / 5000.0,
                    mobility_probability,
                    residual_dwell,
                ],
                dtype=np.float32,
            )

        mobility_probability = (
            0.0
            if mask_mobility
            else float(
                np.mean(
                    [
                        request.handover_probability
                        for request in batch.requests
                    ]
                )
            )
        )

        residual_dwell = (
            0.0
            if mask_mobility
            else float(
                np.mean(
                    [
                        np.clip(
                                request.residual_dwell_ms,
                                1.0,
                                2500.0,
                            )
                        for request in batch.requests
                    ]
                )
                / 3000.0
            )
        )

        return np.array(
            [
                batch.batch_size
                / max(
                    self.batcher.max_batch_size,
                    1,
                ),
                batch.max_input_tokens
                / 1024.0,
                batch.token_sum
                / 8192.0,
                batch.expected_output_mean
                / 512.0,
                batch.min_remaining_deadline_ms
                / 5000.0,
                mobility_probability,
                residual_dwell,
            ],
            dtype=np.float32,
        )

    def _observation(
        self,
    ) -> Observation:
        assert self.infra is not None

        (
            node_features,
            edge_index,
            edge_features,
            node_ids,
        ) = infrastructure_to_tensors(
            self.infra
        )

        self.node_ids = node_ids

        node_to_idx = {
            node_id: index
            for index, node_id
            in enumerate(node_ids)
        }

        active_requests = (
            self.batch.requests
            if self.batch is not None
            else self.queue[
                : self.batcher.window_size
            ]
        )

        node_scale = max(
            len(node_ids) - 1,
            1,
        )

        mask_mobility = bool(
            self.cfg.get(
                "ablation",
                {},
            ).get(
                "mask_mobility_features",
                False,
            )
        )

        mobility_features = np.asarray(
            [
                [
                    node_to_idx.get(
                        request.anchor_node,
                        0,
                    )
                    / node_scale,
                    (
                        0.0
                        if mask_mobility
                        else node_to_idx.get(
                            request.next_anchor_node,
                            0,
                        )
                        / node_scale
                    ),
                    (
                        0.0
                        if mask_mobility
                        else float(
                            request.handover_probability
                        )
                    ),
                    (
                        0.0
                        if mask_mobility
                        else float(
                            np.clip(
                                request.residual_dwell_ms,
                                1.0,
                                2500.0,
                            )
                        )
                        / 3000.0
                    ),
                ]
                for request in active_requests
            ],
            dtype=np.float32,
        )

        if mobility_features.size == 0:
            mobility_features = np.zeros(
                (1, 4),
                dtype=np.float32,
            )

        return Observation(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            batch_features=(
                self._batch_feature_vector(
                    self.batch
                )
            ),
            current_block=self.current_block,
            candidate_node_indices=np.array(
                [
                    node_to_idx[
                        action.node_id
                    ]
                    for action in self.actions
                ],
                dtype=np.int64,
            ),
            candidate_group_sizes=np.array(
                [
                    action.group_size
                    for action in self.actions
                ],
                dtype=np.int64,
            ),
            candidate_payloads=list(
                self.actions
            ),
            mobility_features=(
                mobility_features
            ),
            agent_id=int(
                node_to_idx.get(
                    self.infra.anchor_node,
                    0,
                )
            ),
        )

    def _step_costs(self, action: ActionCandidate) -> tuple[float, float, float]:
        assert self.infra is not None and self.batch is not None
        node = self.infra.nodes[action.node_id]
        prefill = self.profile.prefill_ms(
            node,
            self.batch.batch_size,
            self.batch.max_input_tokens,
            action.group_size,
        )
        data_mb = self.profile.intermediate_mb(
            self.batch.batch_size,
            self.batch.max_input_tokens,
            decode=False,
        )
        transfer = transfer_time_ms(self.infra, self.current_node, action.node_id, data_mb)
        handover = float(
            np.mean(
                [
                    req.handover_probability
                    * (
                        12.0
                        + max(
                            0.0,
                            shortest_path_latency_ms(self.infra, req.next_anchor_node, action.node_id)
                            - shortest_path_latency_ms(self.infra, req.anchor_node, action.node_id),
                        )
                    )
                    for req in self.batch.requests
                ]
            )
        )
        return float(prefill), float(transfer), float(handover)


    def _per_request_decode_ms(
        self,
    ) -> list[float]:
        assert (
            self.infra is not None
            and self.batch is not None
        )

        if not self.mapping:
            return [
                0.0
                for _ in self.batch.requests
            ]

        token_mb = (
            self.profile.intermediate_mb(
                self.batch.batch_size,
                1,
                decode=True,
            )
        )

        values: list[float] = []

        for req in self.batch.requests:
            total_per_token = 0.0
            previous = req.anchor_node

            context = int(
                req.input_tokens
                + req.expected_output_tokens
                / 2
            )

            for step in self.mapping:
                node = self.infra.nodes[
                    step.node_id
                ]

                total_per_token += (
                    self.profile
                    .decode_per_token_ms(
                        node,
                        self.batch.batch_size,
                        context,
                        step.group_size,
                    )
                )

                total_per_token += (
                    transfer_time_ms(
                        self.infra,
                        previous,
                        step.node_id,
                        token_mb,
                    )
                )

                previous = step.node_id

            values.append(
                float(
                    total_per_token
                    * req.expected_output_tokens
                )
            )

        return values

    def _expected_decode_ms(
        self,
    ) -> float:
        values = (
            self._per_request_decode_ms()
        )

        if not values:
            return 0.0

        return float(np.mean(values))

    def step(self, action_index: int) -> tuple[Observation | None, float, bool, EpisodeResult | None]:
        if action_index < 0 or action_index >= len(self.actions):
            raise IndexError("Action index out of bounds")
        assert self.infra is not None
        action = self.actions[action_index]
        if self.batch is None:
            if action.batch is None:
                raise RuntimeError("Initial action must contain a DP-generated batch")
            self.batch = action.batch
        prefill, transfer, handover = self._step_costs(action)
        self.mapping.append(
            MappingStep(
                node_id=action.node_id,
                start_block=self.current_block,
                group_size=action.group_size,
                prefill_ms=prefill,
                transfer_ms=transfer,
                handover_ms=handover,
            )
        )
        mb = self.profile.intermediate_mb(
            self.batch.batch_size,
            self.batch.max_input_tokens,
            decode=False,
        ) if action.node_id != self.current_node else 0.0
        step_reward = -(
            self.w_prefill * prefill
            + self.w_transfer * mb
            + self.w_handover * handover
        )
        self.total_prefill_ms += prefill + transfer
        self.total_transfer_mb += mb
        self.total_handover_ms += handover
        self.total_reward += step_reward
        self.current_node = action.node_id
        self.current_block += action.group_size
        done = self.current_block >= self.profile.model.num_blocks
        if done:
            per_request_decode = (
                self._per_request_decode_ms()
            )

            expected_decode = float(
                np.mean(per_request_decode)
            )

            terminal = (
                -self.w_decode
                * expected_decode
            )

            violations = sum(
                int(
                    (
                        self.total_prefill_ms
                        + decode_ms
                        + self.total_handover_ms
                    )
                    > req.remaining_deadline_ms(
                        self.now_ms
                    )
                )
                for req, decode_ms
                in zip(
                    self.batch.requests,
                    per_request_decode,
                )
            )

            terminal -= (
                self.w_slo
                * violations
            )
            self.total_reward += terminal
            result = EpisodeResult(
                total_reward=self.total_reward,
                total_prefill_ms=self.total_prefill_ms,
                expected_decode_ms=expected_decode,
                transfer_mb=self.total_transfer_mb,
                handover_ms=self.total_handover_ms,
                slo_violations=violations,
                batch_size=self.batch.batch_size,
                mapping=list(self.mapping),
            )
            return None, float(step_reward + terminal), True, result
        self.actions = continuation_action_candidates(
            self.cfg,
            self.infra,
            self.profile,
            self.current_block,
            self.batch,
            self.current_node,
        )
        if not self.actions:
            raise RuntimeError("No continuation action despite cloud fallback guarantee")
        return self._observation(), float(step_reward), False, None
