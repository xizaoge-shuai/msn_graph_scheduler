from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .datatypes import BatchCandidate, Infrastructure, Request
from .profiles import ProfileTable, shortest_path_latency_ms, transfer_time_ms


@dataclass
class BatchingWeights:
    urgency: float
    batch_gain: float
    padding: float
    mobility: float
    decode: float
    deadline_risk: float
    output_spread: float


class NodeConditionedDPBatcher:
    """Bounded-window rolling DP from the paper design.

    For fixed (node, group size, max-length bucket, target batch size), the
    nonlinear batch terms become constants and the request-selection part is
    additive. The request dimension is rolled away, leaving DP[k, token_sum].
    """

    def __init__(self, cfg: dict, profile: ProfileTable):
        bcfg = cfg["batching"]
        self.window_size = int(cfg["requests"]["window_size"])
        self.max_batch_size = int(bcfg["max_batch_size"])
        self.token_capacity = int(bcfg["token_capacity"])
        self.token_quantum = int(bcfg["token_quantum"])
        self.length_buckets = [int(x) for x in bcfg["length_buckets"]]
        self.slack_ms = float(bcfg["deadline_slack_ms"])
        self.hard_deadline_filter = bool(
            bcfg.get(
                "hard_deadline_filter",
                False,
            )
        )
        # Frozen weights used only by heuristic baselines.
        self.weights = BatchingWeights(
            urgency=float(
                bcfg.get(
                    "baseline_alpha_urgency",
                    1.0,
                )
            ),
            batch_gain=float(
                bcfg.get(
                    "baseline_alpha_batch_gain",
                    0.025,
                )
            ),
            padding=float(
                bcfg.get(
                    "baseline_alpha_padding",
                    0.0005,
                )
            ),
            mobility=float(
                bcfg.get(
                    "baseline_alpha_mobility",
                    0.003,
                )
            ),
            decode=0.0,
            deadline_risk=0.0,
            output_spread=0.0,
        )

        # Tunable weights used only by NodeConditionedDPBatcher.
        self.dp_weights = BatchingWeights(
            urgency=float(
                bcfg.get(
                    "dp_alpha_urgency",
                    1.0,
                )
            ),
            batch_gain=float(
                bcfg.get(
                    "dp_alpha_batch_gain",
                    4.0,
                )
            ),
            padding=float(
                bcfg.get(
                    "dp_alpha_padding",
                    0.0005,
                )
            ),
            mobility=float(
                bcfg.get(
                    "dp_alpha_mobility",
                    0.003,
                )
            ),
            decode=float(
                bcfg.get(
                    "dp_alpha_decode",
                    0.00010,
                )
            ),
            deadline_risk=float(
                bcfg.get(
                    "dp_alpha_deadline_risk",
                    1.0,
                )
            ),
            output_spread=float(
                bcfg.get(
                    "dp_alpha_output_spread",
                    0.0020,
                )
            ),
        )

        self.deadline_risk_start = float(
            bcfg.get(
                "deadline_risk_start",
                0.80,
            )
        )
        self.profile = profile
        self.cloud_reserve_gb = float(cfg["system"]["cloud_reserved_memory_gb"])

        tmc_cfg = cfg.get(
            "tmc_mobility",
            {},
        )

        self.mobility_risk_horizon = str(
            tmc_cfg.get(
                "batch_risk_horizon",
                "prefill",
            )
        )

        if self.mobility_risk_horizon not in {
            "prefill",
            "service",
        }:
            raise ValueError(
                "tmc_mobility.batch_risk_horizon "
                "must be either 'prefill' or 'service'"
            )

        self.guarded_mobility_batching = bool(
            tmc_cfg.get(
                "guarded_batching",
                False,
            )
        )

        self.guard_base_weight = float(
            tmc_cfg.get(
                "guard_base_weight",
                self.dp_weights.mobility,
            )
        )

        self.guard_proposal_weight = float(
            tmc_cfg.get(
                "guard_proposal_weight",
                0.6,
            )
        )

        self.guard_max_utility_regret = float(
            tmc_cfg.get(
                "guard_max_utility_regret",
                0.0,
            )
        )

        self.guard_min_risk_reduction = float(
            tmc_cfg.get(
                "guard_min_risk_reduction",
                0.0,
            )
        )

        if self.guard_max_utility_regret < 0.0:
            raise ValueError(
                "guard_max_utility_regret "
                "must be non-negative"
            )

        if self.guard_min_risk_reduction < 0.0:
            raise ValueError(
                "guard_min_risk_reduction "
                "must be non-negative"
            )

    def _handover_cost_ms(self, req: Request, infra: Infrastructure, node_id: str) -> float:
        reroute_ms = 12.0
        old_path = shortest_path_latency_ms(infra, req.anchor_node, node_id)
        new_path = shortest_path_latency_ms(infra, req.next_anchor_node, node_id)
        if not np.isfinite(new_path):
            return 1e6
        stretch_ms = max(0.0, new_path - old_path)
        return reroute_ms + stretch_ms

    def _estimate_completion_reference_ms(
        self,
        infra: Infrastructure,
        node_id: str,
        batch_size: int,
        max_len: int,
        group_size: int,
        expected_out: int,
    ) -> float:
        """Run the first group locally and
        remaining blocks in the cloud."""
        node = infra.nodes[node_id]

        cloud = infra.nodes[
            infra.cloud_node
        ]

        activation_mb = (
            self.profile.intermediate_mb(
                batch_size,
                max_len,
                decode=False,
            )
        )

        token_mb = (
            self.profile.intermediate_mb(
                batch_size,
                1,
                decode=True,
            )
        )

        anchor_transfer = (
            transfer_time_ms(
                infra,
                infra.anchor_node,
                node_id,
                activation_mb,
            )
        )

        local_prefill = (
            self.profile.prefill_ms(
                node,
                batch_size,
                max_len,
                group_size,
            )
        )

        local_decode = (
            self.profile.decode_per_token_ms(
                node,
                batch_size,
                (
                    max_len
                    + expected_out // 2
                ),
                group_size,
            )
            * expected_out
        )

        remaining_blocks = (
            self.profile.model.num_blocks
            - group_size
        )

        if remaining_blocks <= 0:
            return float(
                anchor_transfer
                + local_prefill
                + local_decode
            )

        cloud_prefill_transfer = (
            transfer_time_ms(
                infra,
                node_id,
                infra.cloud_node,
                activation_mb,
            )
        )

        cloud_decode_transfer = (
            transfer_time_ms(
                infra,
                node_id,
                infra.cloud_node,
                token_mb,
            )
            * expected_out
        )

        cloud_prefill = (
            self.profile.prefill_ms(
                cloud,
                batch_size,
                max_len,
                remaining_blocks,
            )
        )

        cloud_decode = (
            self.profile.decode_per_token_ms(
                cloud,
                batch_size,
                (
                    max_len
                    + expected_out // 2
                ),
                remaining_blocks,
            )
            * expected_out
        )

        return float(
            anchor_transfer
            + local_prefill
            + local_decode
            + cloud_prefill_transfer
            + cloud_decode_transfer
            + cloud_prefill
            + cloud_decode
        )

    def _estimate_request_decode_ms(
        self,
        infra: Infrastructure,
        node_id: str,
        batch_size: int,
        max_len: int,
        group_size: int,
        req: Request,
    ) -> float:
        """Estimate request-specific decode cost.

        The selected node executes the current block group.
        Remaining blocks use the cloud reference path.
        """
        node = infra.nodes[node_id]
        cloud = infra.nodes[
            infra.cloud_node
        ]

        output_tokens = max(
            int(req.expected_output_tokens),
            1,
        )

        context_tokens = int(
            max_len
            + output_tokens / 2
        )

        local_decode = (
            self.profile.decode_per_token_ms(
                node,
                batch_size,
                context_tokens,
                group_size,
            )
            * output_tokens
        )

        remaining_blocks = (
            self.profile.model.num_blocks
            - group_size
        )

        if remaining_blocks <= 0:
            return float(local_decode)

        cloud_decode = (
            self.profile.decode_per_token_ms(
                cloud,
                batch_size,
                context_tokens,
                remaining_blocks,
            )
            * output_tokens
        )

        token_mb = (
            self.profile.intermediate_mb(
                batch_size,
                1,
                decode=True,
            )
        )

        decode_transfer = (
            transfer_time_ms(
                infra,
                node_id,
                infra.cloud_node,
                token_mb,
            )
            * output_tokens
        )

        return float(
            local_decode
            + cloud_decode
            + decode_transfer
        )

    def _cache_key(
        self, queue: list[Request], now_ms: float, infra: Infrastructure, node_id: str, group_size: int
    ) -> tuple:
        node = infra.nodes[node_id]
        req_key = tuple(
            (
                r.request_id,
                r.input_tokens,
                r.expected_output_tokens,
                round(r.deadline_ms, 3),
                round(r.arrival_ms, 3),
                round(r.handover_probability, 4),
                round(r.residual_dwell_ms, 3),
                r.anchor_node,
                r.next_anchor_node,
            )
            for r in queue[: self.window_size]
        )
        cloud = infra.nodes[infra.cloud_node]
        link_signature = tuple(
            sorted(
                (
                    src,
                    dst,
                    round(link.bandwidth_mbps, 2),
                    round(link.latency_ms, 3),
                    round(link.reliability, 4),
                )
                for (src, dst), link in infra.links.items()
            )
        )
        return (
            req_key,
            round(now_ms, 3),
            node_id,
            group_size,
            round(node.free_memory_gb, 3),
            round(node.background_load, 3),
            round(node.queue_delay_ms, 3),
            round(cloud.free_memory_gb, 3),
            link_signature,
        )

    def _best_batch_once(
        self,
        queue: list[Request],
        now_ms: float,
        infra: Infrastructure,
        node_id: str,
        group_size: int,
        mobility_weight: float,
        fixed_batch_size: int | None = None,
    ) -> BatchCandidate | None:
        if not queue:
            return None
        if not hasattr(self, "_cache"):
            self._cache = {}
        key = (
            self._cache_key(
                queue,
                now_ms,
                infra,
                node_id,
                group_size,
            )
            + (
                round(
                    float(mobility_weight),
                    6,
                ),
                fixed_batch_size,
            )
        )
        if key in self._cache:
            return self._cache[key]

        node = infra.nodes[node_id]
        cloud = infra.nodes[infra.cloud_node]
        candidates = [r for r in queue[: self.window_size] if r.model_name == queue[0].model_name]
        if not candidates:
            return None
        best: BatchCandidate | None = None
        max_s = self.token_capacity // self.token_quantum
        for max_len in self.length_buckets:
            eligible = [r for r in candidates if r.input_tokens <= max_len]
            if not eligible:
                continue
            max_target = min(self.max_batch_size, len(eligible))
            expected_out_pool = int(round(np.mean([r.expected_output_tokens for r in eligible])))
            for target_b in range(1, max_target + 1):
                if (
                    fixed_batch_size is not None
                    and target_b != fixed_batch_size
                ):
                    continue

                batch_prefill = self.profile.prefill_ms(
                    node,
                    target_b,
                    max_len,
                    group_size,
                )
                batch_gain = (
                    target_b * self.profile.prefill_ms(node, 1, max_len, group_size) - batch_prefill
                )
                mem = self.profile.memory_gb(target_b, max_len, expected_out_pool, group_size)
                cloud_mem = self.profile.memory_gb(
                    target_b, max_len, expected_out_pool, self.profile.model.num_blocks
                )
                if mem > node.free_memory_gb:
                    continue
                if cloud_mem > cloud.free_memory_gb - self.cloud_reserve_gb:
                    continue
                completion_reference = self._estimate_completion_reference_ms(
                    infra=infra,
                    node_id=node_id,
                    batch_size=target_b,
                    max_len=max_len,
                    group_size=group_size,
                    expected_out=expected_out_pool,
                )

                # Sparse rolling DP. Each state stores (value, selected bitmask).
                # With K<=16 this is substantially faster than scanning a dense
                # tensor for every unreachable token state.
                states: dict[tuple[int, int], tuple[float, int]] = {(0, 0): (0.0, 0)}
                for idx, req in enumerate(eligible):
                    units = int(np.ceil(req.input_tokens / self.token_quantum))
                    remaining_raw = float(
                        req.remaining_deadline_ms(
                            now_ms
                        )
                    )

                    deadline_scale_ms = max(
                        float(req.deadline_ms),
                        1.0,
                    )

                    waiting_ratio = min(
                        max(
                            (
                                now_ms
                                - req.arrival_ms
                            )
                            / deadline_scale_ms,
                            0.0,
                        ),
                        2.0,
                    )

                    handover = self._handover_cost_ms(
                        req,
                        infra,
                        node_id,
                    )

                    estimated_decode_ms = (
                        self._estimate_request_decode_ms(
                            infra=infra,
                            node_id=node_id,
                            batch_size=target_b,
                            max_len=max_len,
                            group_size=group_size,
                            req=req,
                        )
                    )

                    predicted_total_ms = (
                        batch_prefill
                        + estimated_decode_ms
                    )

                    mobility_horizon_ms = (
                        predicted_total_ms
                        if self.mobility_risk_horizon
                        == "service"
                        else batch_prefill
                    )

                    p_ho = min(
                        1.0,
                        req.handover_probability
                        * mobility_horizon_ms
                        / max(
                            req.residual_dwell_ms,
                            1.0,
                        ),
                    )

                    # Soft and bounded SLO risk.
                    #
                    # Using predicted_total / max(remaining, 1)
                    # with a squared penalty makes the value explode
                    # after a request becomes overdue. Normalize the
                    # predicted miss by the request's original SLO
                    # budget and cap the result instead.
                    predicted_miss_ms = max(
                        0.0,
                        predicted_total_ms
                        - max(
                            remaining_raw,
                            0.0,
                        ),
                    )

                    deadline_risk = min(
                        predicted_miss_ms
                        / deadline_scale_ms,
                        2.0,
                    )

                    value = (
                        self.dp_weights.urgency
                        * waiting_ratio
                        - self.dp_weights.padding
                        * (
                            max_len
                            - req.input_tokens
                        )
                        - mobility_weight
                        * p_ho
                        * handover
                        - self.dp_weights.decode
                        * estimated_decode_ms
                        - self.dp_weights.deadline_risk
                        * deadline_risk
                    )
                    updates: dict[tuple[int, int], tuple[float, int]] = {}
                    for (k, s), (old_value, mask) in states.items():
                        if k >= target_b or s + units > max_s:
                            continue
                        state_key = (k + 1, s + units)
                        proposal = (old_value + value, mask | (1 << idx))
                        incumbent = states.get(state_key)
                        pending = updates.get(state_key)
                        best_existing = incumbent if pending is None else (
                            pending if incumbent is None or pending[0] >= incumbent[0] else incumbent
                        )
                        if best_existing is None or proposal[0] > best_existing[0]:
                            updates[state_key] = proposal
                    for state_key, proposal in updates.items():
                        incumbent = states.get(state_key)
                        if incumbent is None or proposal[0] > incumbent[0]:
                            states[state_key] = proposal

                for (k, _s), (base, mask) in states.items():
                    if k != target_b:
                        continue
                    chosen = [req for idx, req in enumerate(eligible) if mask & (1 << idx)]
                    if len(chosen) != target_b:
                        continue
                    token_sum = sum(r.input_tokens for r in chosen)
                    if token_sum > self.token_capacity:
                        continue
                    min_deadline = min(r.remaining_deadline_ms(now_ms) for r in chosen)
                    if (
                        self.hard_deadline_filter
                        and completion_reference
                        > min_deadline + self.slack_ms
                    ):
                        continue
                    mean_handover = float(
                        np.mean(
                            [
                                self._handover_cost_ms(r, infra, node_id)
                                * r.handover_probability
                                for r in chosen
                            ]
                        )
                    )
                    output_lengths = [
                        int(
                            r.expected_output_tokens
                        )
                        for r in chosen
                    ]

                    output_spread = (
                        max(output_lengths)
                        - min(output_lengths)
                    )

                    # Normalize prefill saving into a dimensionless
                    # relative batching gain. Using raw milliseconds
                    # makes the utility depend excessively on hardware
                    # speed, sequence length, and batch size.
                    independent_prefill_ms = (
                        target_b
                        * self.profile.prefill_ms(
                            node,
                            1,
                            max_len,
                            group_size,
                        )
                    )

                    relative_batch_gain = (
                        max(
                            0.0,
                            batch_gain,
                        )
                        / max(
                            independent_prefill_ms,
                            1e-6,
                        )
                    )

                    # `base` is the sum of request-level
                    # values. Comparing this raw sum across
                    # different batch sizes structurally favors
                    # small batches because every added request
                    # contributes another decode/deadline cost.
                    # Normalize it before comparing candidates.
                    mean_request_value = (
                        base
                        / max(
                            float(len(chosen)),
                            1.0,
                        )
                    )

                    # Reward progressively larger batches.
                    #
                    # Because len(chosen) == target_b here, dividing
                    # by target_b - 1 would make every multi-request
                    # candidate receive exactly the same value 1.0.
                    # Normalize by the largest feasible target instead.
                    coverage_gain = (
                        max(
                            float(target_b - 1),
                            0.0,
                        )
                        / max(
                            float(max_target - 1),
                            1.0,
                        )
                    )

                    utility = float(
                        mean_request_value
                        + self.dp_weights.batch_gain
                        * (
                            relative_batch_gain
                            + coverage_gain
                        )
                        - self.dp_weights.output_spread
                        * output_spread
                    )
                    candidate = BatchCandidate(
                        requests=chosen,
                        node_id=node_id,
                        group_size=group_size,
                        utility=utility,
                        estimated_prefill_ms=batch_prefill,
                        estimated_memory_gb=mem,
                        estimated_handover_ms=mean_handover,
                        token_sum=token_sum,
                        max_input_tokens=max(r.input_tokens for r in chosen),
                        expected_output_mean=float(
                            np.mean([r.expected_output_tokens for r in chosen])
                        ),
                        min_remaining_deadline_ms=min_deadline,
                    )
                    if best is None or candidate.utility > best.utility:
                        best = candidate
        if len(self._cache) > 2048:
            self._cache.clear()
        self._cache[key] = best
        return best


    def _candidate_prefill_mobility_risk(
        self,
        candidate: BatchCandidate,
        infra: Infrastructure,
        node_id: str,
    ) -> float:
        if not candidate.requests:
            return 0.0

        risks = []

        for request in candidate.requests:
            exposure_probability = min(
                1.0,
                request.handover_probability
                * candidate.estimated_prefill_ms
                / max(
                    request.residual_dwell_ms,
                    1.0,
                ),
            )

            risks.append(
                exposure_probability
                * self._handover_cost_ms(
                    request,
                    infra,
                    node_id,
                )
            )

        return float(
            np.mean(risks)
        )

    def best_batch(
        self,
        queue: list[Request],
        now_ms: float,
        infra: Infrastructure,
        node_id: str,
        group_size: int,
    ) -> BatchCandidate | None:
        if not self.guarded_mobility_batching:
            return self._best_batch_once(
                queue=queue,
                now_ms=now_ms,
                infra=infra,
                node_id=node_id,
                group_size=group_size,
                mobility_weight=(
                    self.dp_weights.mobility
                ),
            )

        baseline = self._best_batch_once(
            queue=queue,
            now_ms=now_ms,
            infra=infra,
            node_id=node_id,
            group_size=group_size,
            mobility_weight=(
                self.guard_base_weight
            ),
        )

        if baseline is None:
            return None

        proposal = self._best_batch_once(
            queue=queue,
            now_ms=now_ms,
            infra=infra,
            node_id=node_id,
            group_size=group_size,
            mobility_weight=(
                self.guard_proposal_weight
            ),
            fixed_batch_size=(
                baseline.batch_size
            ),
        )

        if proposal is None:
            return baseline

        baseline_ids = tuple(
            request.request_id
            for request in baseline.requests
        )

        proposal_ids = tuple(
            request.request_id
            for request in proposal.requests
        )

        if proposal_ids == baseline_ids:
            return baseline

        baseline_risk = (
            self._candidate_prefill_mobility_risk(
                baseline,
                infra,
                node_id,
            )
        )

        proposal_risk = (
            self._candidate_prefill_mobility_risk(
                proposal,
                infra,
                node_id,
            )
        )

        risk_reduction = (
            baseline_risk
            - proposal_risk
        )

        if (
            risk_reduction
            <= self.guard_min_risk_reduction
        ):
            return baseline

        # The proposal utility was evaluated using the
        # stronger mobility coefficient. Convert it back
        # to the reference coefficient before applying
        # the utility-regret guard.
        proposal_reference_utility = (
            proposal.utility
            + (
                self.guard_proposal_weight
                - self.guard_base_weight
            )
            * proposal_risk
        )

        utility_regret = (
            baseline.utility
            - proposal_reference_utility
        )

        if (
            utility_regret
            > self.guard_max_utility_regret
            + 1e-12
        ):
            return baseline

        return replace(
            proposal,
            utility=float(
                proposal_reference_utility
            ),
        )

class HeuristicBatcherBase(NodeConditionedDPBatcher):
    """Shared feasibility/cost logic for batching baselines."""

    def _candidate_from_requests(
        self,
        requests: list[Request],
        now_ms: float,
        infra: Infrastructure,
        node_id: str,
        group_size: int,
    ) -> BatchCandidate | None:
        if not requests:
            return None
        node = infra.nodes[node_id]
        cloud = infra.nodes[infra.cloud_node]
        batch_size = len(requests)
        max_len = max(r.input_tokens for r in requests)
        expected_out = int(round(np.mean([r.expected_output_tokens for r in requests])))
        token_sum = sum(r.input_tokens for r in requests)
        if batch_size > self.max_batch_size or token_sum > self.token_capacity:
            return None
        prefill = self.profile.prefill_ms(node, batch_size, max_len, group_size)
        mem = self.profile.memory_gb(batch_size, max_len, expected_out, group_size)
        cloud_mem = self.profile.memory_gb(
            batch_size, max_len, expected_out, self.profile.model.num_blocks
        )
        if mem > node.free_memory_gb:
            return None
        if cloud_mem > cloud.free_memory_gb - self.cloud_reserve_gb:
            return None
        completion_reference = self._estimate_completion_reference_ms(
            infra=infra,
            node_id=node_id,
            batch_size=batch_size,
            max_len=max_len,
            group_size=group_size,
            expected_out=expected_out,
        )
        min_deadline = min(r.remaining_deadline_ms(now_ms) for r in requests)
        if (
                        self.hard_deadline_filter
                        and completion_reference
                        > min_deadline + self.slack_ms
                    ):
            return None
        padding = batch_size * max_len - token_sum
        gain = (
            sum(self.profile.prefill_ms(node, 1, r.input_tokens, group_size) for r in requests)
            - prefill
        )
        mobility = sum(
            r.handover_probability * self._handover_cost_ms(r, infra, node_id)
            for r in requests
        )
        urgency = sum(
            max(0.0, now_ms - r.arrival_ms)
            / max(r.remaining_deadline_ms(now_ms), 1.0)
            for r in requests
        )
        utility = (
            self.weights.urgency * urgency
            + self.weights.batch_gain * gain
            - self.weights.padding * padding
            - self.weights.mobility * mobility
        )
        return BatchCandidate(
            requests=list(requests),
            node_id=node_id,
            group_size=group_size,
            utility=float(utility),
            estimated_prefill_ms=float(prefill),
            estimated_memory_gb=float(mem),
            estimated_handover_ms=float(mobility / batch_size),
            token_sum=token_sum,
            max_input_tokens=max_len,
            expected_output_mean=float(expected_out),
            min_remaining_deadline_ms=float(min_deadline),
        )


class SingleRequestBatcher(HeuristicBatcherBase):
    """No batching: schedule the oldest feasible request only."""

    def best_batch(
        self,
        queue: list[Request],
        now_ms: float,
        infra: Infrastructure,
        node_id: str,
        group_size: int,
    ) -> BatchCandidate | None:
        for req in queue[: self.window_size]:
            candidate = self._candidate_from_requests(
                [req], now_ms, infra, node_id, group_size
            )
            if candidate is not None:
                return candidate
        return None


class FixedSizeBatcher(HeuristicBatcherBase):
    """FCFS fixed batching with feasibility-based shrinking."""

    def __init__(self, cfg: dict, profile: ProfileTable, fixed_size: int = 4):
        super().__init__(cfg, profile)
        self.fixed_size = min(int(fixed_size), self.max_batch_size)

    def best_batch(
        self,
        queue: list[Request],
        now_ms: float,
        infra: Infrastructure,
        node_id: str,
        group_size: int,
    ) -> BatchCandidate | None:
        compatible = [r for r in queue[: self.window_size] if r.model_name == queue[0].model_name]
        for size in range(min(self.fixed_size, len(compatible)), 0, -1):
            candidate = self._candidate_from_requests(
                compatible[:size], now_ms, infra, node_id, group_size
            )
            if candidate is not None:
                return candidate
        return None


class SequentialGreedyBatcher(HeuristicBatcherBase):
    """Reference-style request-by-request profit/cost greedy baseline."""

    def best_batch(
        self,
        queue: list[Request],
        now_ms: float,
        infra: Infrastructure,
        node_id: str,
        group_size: int,
    ) -> BatchCandidate | None:
        compatible = [r for r in queue[: self.window_size] if r.model_name == queue[0].model_name]
        if not compatible:
            return None
        selected: list[Request] = []
        current: BatchCandidate | None = None
        # Highest urgency first, then greedily accept only positive marginal gain.
        compatible.sort(
            key=lambda r: (
                -max(0.0, now_ms - r.arrival_ms)
                / max(r.remaining_deadline_ms(now_ms), 1.0),
                r.arrival_ms,
            )
        )
        for req in compatible:
            if len(selected) >= self.max_batch_size:
                break
            proposal = self._candidate_from_requests(
                selected + [req], now_ms, infra, node_id, group_size
            )
            if proposal is None:
                continue
            if current is None or proposal.utility > current.utility:
                selected.append(req)
                current = proposal
        return current
