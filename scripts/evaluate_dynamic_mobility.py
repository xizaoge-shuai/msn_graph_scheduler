from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from msn_scheduler.agent import DDQNAgent
from msn_scheduler.baselines import (
    choose_min_lower_bound,
)
from msn_scheduler.batching import (
    NodeConditionedDPBatcher,
)
from msn_scheduler.config import load_config
from msn_scheduler.dybap import (
    DyBAPFusionBatcher,
)
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.mobility import (
    adjust_q_values_with_mobility_prior,
    candidate_mobility_costs_ms,
)
from msn_scheduler.external_baselines import (
    choose_external_action,
)
from msn_scheduler.profiles import (
    ModelProfile,
    ProfileTable,
    shortest_path_latency_ms,
    transfer_time_ms,
)
from msn_scheduler.synthetic import (
    make_request_queue,
    make_synthetic_infrastructure,
)


MOBILITY_PROFILES = {
    "stationary": {
        "dwell_min_ms": float("inf"),
        "dwell_max_ms": float("inf"),
        "handover_probability": 0.0,
    },
    "walking": {
        "dwell_min_ms": 20000.0,
        "dwell_max_ms": 60000.0,
        "handover_probability": 0.15,
    },
    "driving": {
        "dwell_min_ms": 8000.0,
        "dwell_max_ms": 25000.0,
        "handover_probability": 0.45,
    },
    "high_speed": {
        "dwell_min_ms": 3000.0,
        "dwell_max_ms": 10000.0,
        "handover_probability": 0.75,
    },
}


@dataclass
class MobilityState:
    current_anchor: str
    predicted_next_anchor: str
    actual_next_anchor: str
    next_handover_ms: float
    transition_index: int
    handover_probability: float


def build_model_profile(
    cfg: dict,
) -> ModelProfile:
    return ModelProfile(
        num_blocks=int(
            cfg["model"]["num_blocks"]
        ),
        hidden_size=int(
            cfg["model"]["hidden_size"]
        ),
        bytes_per_element=int(
            cfg["model"][
                "bytes_per_element"
            ]
        ),
        block_parameter_gb=float(
            cfg["model"][
                "block_parameter_gb"
            ]
        ),
        kv_bytes_per_token_per_block=float(
            cfg["model"][
                "kv_bytes_per_token_per_block"
            ]
        ),
    )


def assign_poisson_arrivals(
    queue,
    rng: np.random.Generator,
    rate_rps: float,
):
    if rate_rps <= 0:
        raise ValueError(
            "arrival_rate_rps must be positive"
        )

    result = []
    current_ms = 0.0

    for index, request in enumerate(queue):
        if index > 0:
            current_ms += float(
                rng.exponential(
                    1000.0 / rate_rps
                )
            )

        result.append(
            replace(
                request,
                arrival_ms=current_ms,
            )
        )

    return result


def deterministic_rng(
    episode_seed: int,
    request_id: str,
    transition_index: int,
    stream: str = "default",
) -> np.random.Generator:
    """Create an independent deterministic RNG stream."""
    value = (
        f"{episode_seed}:"
        f"{request_id}:"
        f"{transition_index}:"
        f"{stream}"
    )

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )

    return np.random.default_rng(seed)

def choose_next_anchor(
    current_anchor: str,
    edge_ids: list[str],
    rng: np.random.Generator,
    excluded_anchor: str | None = None,
) -> str:
    candidates = [
        node_id
        for node_id in edge_ids
        if node_id != current_anchor
        and node_id != excluded_anchor
    ]

    if not candidates:
        candidates = [
            node_id
            for node_id in edge_ids
            if node_id != current_anchor
        ]

    if not candidates:
        return current_anchor

    return str(
        rng.choice(candidates)
    )


def create_mobility_state(
    request_id: str,
    current_anchor: str,
    now_ms: float,
    transition_index: int,
    episode_seed: int,
    edge_ids: list[str],
    mode: str,
    prediction_error: float,
    preferred_prediction: str | None = None,
) -> MobilityState:
    """Generate a fixed actual trajectory and a noisy prediction.

    The actual next anchor and dwell time are independent of
    prediction_error. Changing prediction_error therefore changes
    only the predicted anchor, not the underlying user trajectory.
    """
    profile = MOBILITY_PROFILES[mode]

    if mode == "stationary":
        return MobilityState(
            current_anchor=current_anchor,
            predicted_next_anchor=(
                current_anchor
            ),
            actual_next_anchor=current_anchor,
            next_handover_ms=float("inf"),
            transition_index=(
                transition_index
            ),
            handover_probability=0.0,
        )

    actual_rng = deterministic_rng(
        episode_seed=episode_seed,
        request_id=request_id,
        transition_index=(
            transition_index
        ),
        stream="actual_anchor",
    )

    dwell_rng = deterministic_rng(
        episode_seed=episode_seed,
        request_id=request_id,
        transition_index=(
            transition_index
        ),
        stream="dwell_time",
    )

    error_rng = deterministic_rng(
        episode_seed=episode_seed,
        request_id=request_id,
        transition_index=(
            transition_index
        ),
        stream="prediction_error",
    )

    wrong_prediction_rng = (
        deterministic_rng(
            episode_seed=episode_seed,
            request_id=request_id,
            transition_index=(
                transition_index
            ),
            stream="wrong_prediction",
        )
    )

    # The initial synthetic next anchor is treated as the
    # ground-truth first transition. Subsequent transitions
    # use an independent deterministic actual-trajectory stream.
    if (
        preferred_prediction in edge_ids
        and preferred_prediction
        != current_anchor
    ):
        actual_next = str(
            preferred_prediction
        )
    else:
        actual_next = (
            choose_next_anchor(
                current_anchor=current_anchor,
                edge_ids=edge_ids,
                rng=actual_rng,
            )
        )

    prediction_is_wrong = (
        error_rng.random()
        < prediction_error
    )

    if prediction_is_wrong:
        predicted_next = (
            choose_next_anchor(
                current_anchor=current_anchor,
                edge_ids=edge_ids,
                rng=wrong_prediction_rng,
                excluded_anchor=actual_next,
            )
        )
    else:
        predicted_next = actual_next

    dwell_ms = float(
        dwell_rng.uniform(
            profile["dwell_min_ms"],
            profile["dwell_max_ms"],
        )
    )

    return MobilityState(
        current_anchor=current_anchor,
        predicted_next_anchor=(
            predicted_next
        ),
        actual_next_anchor=actual_next,
        next_handover_ms=(
            now_ms + dwell_ms
        ),
        transition_index=(
            transition_index
        ),
        handover_probability=float(
            profile[
                "handover_probability"
            ]
        ),
    )

def refresh_request(
    request,
    state: MobilityState,
    now_ms: float,
):
    # Keep the uncapped residual dwell in Request for
    # analytical mobility decisions. SchedulingEnv clips
    # it only when constructing frozen-network features.
    mobility_feature_dwell_cap_ms = 2500.0

    if np.isfinite(
        state.next_handover_ms
    ):
        raw_residual_dwell_ms = max(
            state.next_handover_ms
            - now_ms,
            1.0,
        )
    else:
        raw_residual_dwell_ms = (
            mobility_feature_dwell_cap_ms
        )

    residual_dwell_ms = float(
        max(
            raw_residual_dwell_ms,
            1.0,
        )
    )

    return replace(
        request,
        anchor_node=state.current_anchor,
        next_anchor_node=(
            state.predicted_next_anchor
        ),
        handover_probability=(
            state.handover_probability
        ),
        residual_dwell_ms=(
            residual_dwell_ms
        ),
    )


def advance_state_to_time(
    request,
    state: MobilityState,
    target_ms: float,
    episode_seed: int,
    edge_ids: list[str],
    mode: str,
    prediction_error: float,
) -> MobilityState:
    transitions = 0

    while (
        np.isfinite(
            state.next_handover_ms
        )
        and state.next_handover_ms
        <= target_ms
    ):
        event_ms = (
            state.next_handover_ms
        )

        state = create_mobility_state(
            request_id=request.request_id,
            current_anchor=(
                state.actual_next_anchor
            ),
            now_ms=event_ms,
            transition_index=(
                state.transition_index + 1
            ),
            episode_seed=episode_seed,
            edge_ids=edge_ids,
            mode=mode,
            prediction_error=(
                prediction_error
            ),
        )

        transitions += 1

        if transitions >= 128:
            raise RuntimeError(
                "Mobility transition limit "
                "exceeded"
            )

    return state


def initialize_mobility(
    queue,
    infra,
    episode_seed: int,
    mode: str,
    prediction_error: float,
):
    edge_ids = [
        node_id
        for node_id, node
        in infra.nodes.items()
        if node.node_type.startswith(
            "edge"
        )
    ]

    states = {}
    updated = []

    for request in queue:
        current_anchor = (
            request.anchor_node
            if request.anchor_node
            in edge_ids
            else infra.anchor_node
        )

        state = create_mobility_state(
            request_id=request.request_id,
            current_anchor=current_anchor,
            now_ms=float(
                request.arrival_ms
            ),
            transition_index=0,
            episode_seed=episode_seed,
            edge_ids=edge_ids,
            mode=mode,
            prediction_error=(
                prediction_error
            ),
            preferred_prediction=(
                request.next_anchor_node
            ),
        )

        states[
            request.request_id
        ] = state

        updated.append(
            refresh_request(
                request=request,
                state=state,
                now_ms=float(
                    request.arrival_ms
                ),
            )
        )

    return updated, states, edge_ids


def refresh_pending_requests(
    pending,
    states: dict[str, MobilityState],
    now_ms: float,
    episode_seed: int,
    edge_ids: list[str],
    mode: str,
    prediction_error: float,
):
    updated = []

    for request in pending:
        state = states[
            request.request_id
        ]

        state = advance_state_to_time(
            request=request,
            state=state,
            target_ms=now_ms,
            episode_seed=episode_seed,
            edge_ids=edge_ids,
            mode=mode,
            prediction_error=(
                prediction_error
            ),
        )

        states[
            request.request_id
        ] = state

        updated.append(
            refresh_request(
                request=request,
                state=state,
                now_ms=now_ms,
            )
        )

    return updated


def estimate_kv_cache_mb(
    request,
    profile: ProfileTable,
    progress_fraction: float,
) -> float:
    """Estimate KV state present at a handover event.

    Input-token KV is already available after prefill. Generated-token
    KV grows with decode progress, so early handovers migrate less state
    than late handovers.
    """
    progress_fraction = float(
        np.clip(
            progress_fraction,
            0.0,
            1.0,
        )
    )

    generated_tokens = int(
        round(
            request.expected_output_tokens
            * progress_fraction
        )
    )

    context_tokens = (
        int(request.input_tokens)
        + max(
            generated_tokens,
            0,
        )
    )

    kv_bytes = (
        context_tokens
        * profile.model.num_blocks
        * profile.model
        .kv_bytes_per_token_per_block
    )

    return float(
        kv_bytes
        / (
            1024.0
            * 1024.0
        )
    )


def simulate_handover_overhead(
    request,
    state: MobilityState,
    service_start_ms: float,
    base_finish_ms: float,
    execution_node: str,
    infra,
    profile: ProfileTable,
    episode_seed: int,
    edge_ids: list[str],
    mode: str,
    prediction_error: float,
    migration_policy: str,
    handover_setup_ms: float,
):
    """Evaluate handovers occurring during the base inference window.

    Migration interruption is added to request completion time, but
    migration delay itself does not recursively create further handover
    events. This avoids an unstable positive-feedback loop in which each
    migration prolongs the service window and triggers more migrations.
    """
    service_window_end_ms = float(
        base_finish_ms
    )

    total_overhead_ms = 0.0
    total_migration_mb = 0.0
    handover_events = 0

    # A next-anchor prediction becomes available when
    # the current service interval starts. After a real
    # handover, the next prediction becomes available
    # at that handover event.
    prediction_available_ms = float(
        service_start_ms
    )

    while (
        np.isfinite(
            state.next_handover_ms
        )
        and state.next_handover_ms
        <= service_window_end_ms
    ):
        event_ms = float(
            state.next_handover_ms
        )

        old_anchor = str(
            state.current_anchor
        )

        new_anchor = str(
            state.actual_next_anchor
        )

        prediction_correct = (
            state.predicted_next_anchor
            == new_anchor
        )

        service_duration_ms = max(
            service_window_end_ms
            - service_start_ms,
            1.0,
        )

        progress_fraction = float(
            np.clip(
                (
                    event_ms
                    - service_start_ms
                )
                / service_duration_ms,
                0.0,
                1.0,
            )
        )

        kv_cache_mb = (
            estimate_kv_cache_mb(
                request=request,
                profile=profile,
                progress_fraction=(
                    progress_fraction
                ),
            )
        )

        if migration_policy in {
            "keep",
            "oracle_guarded",
        }:
            old_latency = (
                shortest_path_latency_ms(
                    infra,
                    old_anchor,
                    execution_node,
                )
            )

            new_latency = (
                shortest_path_latency_ms(
                    infra,
                    new_anchor,
                    execution_node,
                )
            )

            if not np.isfinite(
                old_latency
            ):
                old_latency = 0.0

            if not np.isfinite(
                new_latency
            ):
                new_latency = (
                    old_latency
                    + handover_setup_ms
                )

            interruption_ms = (
                handover_setup_ms
                + max(
                    0.0,
                    new_latency
                    - old_latency,
                )
            )

            migrated_mb = 0.0

            if (
                migration_policy
                == "oracle_guarded"
            ):
                transfer_ms = transfer_time_ms(
                    infra,
                    old_anchor,
                    new_anchor,
                    kv_cache_mb,
                )

                if not np.isfinite(
                    transfer_ms
                ):
                    raise RuntimeError(
                        "No migration path from "
                        f"{old_anchor} to "
                        f"{new_anchor}"
                    )

                if prediction_correct:
                    prefetch_lead_ms = max(
                        event_ms
                        - prediction_available_ms,
                        0.0,
                    )

                    remaining_transfer_ms = max(
                        transfer_ms
                        - prefetch_lead_ms,
                        0.0,
                    )

                    prefetch_interruption_ms = (
                        2.0
                        + remaining_transfer_ms
                    )

                    prefetch_migrated_mb = (
                        kv_cache_mb
                    )

                else:
                    prefetch_interruption_ms = (
                        handover_setup_ms
                        + transfer_ms
                    )

                    prefetch_migrated_mb = (
                        2.0
                        * kv_cache_mb
                    )

                if (
                    prefetch_interruption_ms
                    < interruption_ms
                ):
                    interruption_ms = (
                        prefetch_interruption_ms
                    )

                    migrated_mb = (
                        prefetch_migrated_mb
                    )

        else:
            transfer_ms = (
                transfer_time_ms(
                    infra,
                    old_anchor,
                    new_anchor,
                    kv_cache_mb,
                )
            )

            if not np.isfinite(
                transfer_ms
            ):
                raise RuntimeError(
                    "No migration path from "
                    f"{old_anchor} to "
                    f"{new_anchor}"
                )

            if (
                migration_policy
                == "prefetch"
                and prediction_correct
            ):
                prefetch_lead_ms = max(
                    event_ms
                    - prediction_available_ms,
                    0.0,
                )

                remaining_transfer_ms = max(
                    transfer_ms
                    - prefetch_lead_ms,
                    0.0,
                )

                # A completed prefetch only pays a small
                # resume cost. An incomplete prefetch must
                # finish transferring the residual state.
                interruption_ms = (
                    2.0
                    + remaining_transfer_ms
                )

                migrated_mb = kv_cache_mb

            elif (
                migration_policy
                == "prefetch"
                and not prediction_correct
            ):
                interruption_ms = (
                    handover_setup_ms
                    + transfer_ms
                )

                migrated_mb = (
                    2.0
                    * kv_cache_mb
                )

            else:
                interruption_ms = (
                    handover_setup_ms
                    + transfer_ms
                )

                migrated_mb = kv_cache_mb

        total_overhead_ms += float(
            interruption_ms
        )

        total_migration_mb += float(
            migrated_mb
        )

        handover_events += 1

        state = create_mobility_state(
            request_id=request.request_id,
            current_anchor=new_anchor,
            now_ms=event_ms,
            transition_index=(
                state.transition_index
                + 1
            ),
            episode_seed=episode_seed,
            edge_ids=edge_ids,
            mode=mode,
            prediction_error=(
                prediction_error
            ),
        )

        prediction_available_ms = event_ms

        if handover_events >= 256:
            raise RuntimeError(
                "More than 256 handovers "
                "occurred during one base "
                "inference window"
            )

    return (
        total_overhead_ms,
        total_migration_mb,
        handover_events,
        state,
    )

def choose_action(
    observation,
    method: str,
    cfg: dict,
    agent: DDQNAgent | None,
    env: SchedulingEnv,
    mobility_prior_beta: float,
) -> int:
    if method == "full":
        if agent is None:
            raise RuntimeError(
                "Full requires an agent"
            )

        with torch.no_grad():
            q_values = (
                agent.online(
                    observation,
                    agent.device,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64
                )
            )

        if (
            mobility_prior_beta > 0.0
            and env.batch is not None
        ):
            if env.infra is None:
                raise RuntimeError(
                    "Environment infrastructure "
                    "is unavailable."
                )

            fallback_requests = (
                list(
                    env.batch.requests
                )
                if env.batch is not None
                else list(
                    env.queue[
                        : env.batcher.window_size
                    ]
                )
            )

            mobility_costs_ms = (
                candidate_mobility_costs_ms(
                    infra=env.infra,
                    actions=list(
                        observation
                        .candidate_payloads
                    ),
                    fallback_requests=(
                        fallback_requests
                    ),
                )
            )

            decision_scores = (
                adjust_q_values_with_mobility_prior(
                    q_values=q_values,
                    mobility_costs_ms=(
                        mobility_costs_ms
                    ),
                    beta=(
                        mobility_prior_beta
                    ),
                )
            )
        else:
            decision_scores = q_values

        return int(
            np.argmax(
                decision_scores
            )
        )

    if method == "lecu_rba":
        return choose_external_action(
            observation,
            "rba",
            cfg,
        )

    if method == "dybap_adapted":
        return choose_external_action(
            observation,
            "dybap_adapted",
            cfg,
        )

    return choose_min_lower_bound(
        observation
    )


def run_one_mapping(
    env: SchedulingEnv,
    infra,
    queue,
    now_ms: float,
    max_steps: int,
    method: str,
    cfg: dict,
    agent: DDQNAgent | None,
    mobility_prior_beta: float,
):
    observation = env.reset(
        infra,
        queue,
        now_ms=now_ms,
    )

    result = None

    for _ in range(max_steps):
        action = choose_action(
            observation=observation,
            method=method,
            cfg=cfg,
            agent=agent,
            env=env,
            mobility_prior_beta=(
                mobility_prior_beta
            ),
        )

        (
            next_observation,
            _reward,
            done,
            result,
        ) = env.step(action)

        if done:
            if result is None:
                raise RuntimeError(
                    "Mapping ended without "
                    "a result"
                )

            return result

        if next_observation is None:
            raise RuntimeError(
                "Non-terminal mapping returned "
                "no observation"
            )

        observation = next_observation

    raise RuntimeError(
        f"Mapping exceeded {max_steps} steps"
    )


def get_decode_times(
    env: SchedulingEnv,
    expected_decode_ms: float,
    batch_size: int,
) -> list[float]:
    function = getattr(
        env,
        "_per_request_decode_ms",
        None,
    )

    if callable(function):
        values = [
            float(value)
            for value in function()
        ]

        if len(values) == batch_size:
            return values

    return [
        float(expected_decode_ms)
        for _ in range(batch_size)
    ]


def run_dynamic_episode(
    cfg: dict,
    profile: ProfileTable,
    infra,
    queue,
    method: str,
    mobility_mode: str,
    migration_policy: str,
    prediction_error: float,
    handover_setup_ms: float,
    episode_seed: int,
    max_steps: int,
    max_batches: int,
    agent: DDQNAgent | None,
    mobility_prior_beta: float,
) -> dict:
    (
        future,
        mobility_states,
        edge_ids,
    ) = initialize_mobility(
        queue=queue,
        infra=infra,
        episode_seed=episode_seed,
        mode=mobility_mode,
        prediction_error=(
            prediction_error
        ),
    )

    future = sorted(
        future,
        key=lambda request:
            request.arrival_ms,
    )

    total_requests = len(future)

    if total_requests == 0:
        raise ValueError(
            "Empty request queue"
        )

    if method == "dybap_adapted":
        batcher = DyBAPFusionBatcher(
            cfg,
            profile,
        )
    else:
        batcher = (
            NodeConditionedDPBatcher(
                cfg,
                profile,
            )
        )

    pending = []
    cursor = 0
    now_ms = float(
        future[0].arrival_ms
    )

    served_requests = 0
    dropped_requests = 0
    successful_requests = 0
    slo_violations = 0

    batch_count = 0
    mapping_steps = 0
    total_reward = 0.0

    total_handover_events = 0
    handover_affected_requests = 0
    handover_successful_requests = 0
    total_migration_mb = 0.0
    total_migration_ms = 0.0

    batch_sizes = []
    request_e2e_ms = []

    failure_reason = ""

    while batch_count < max_batches:
        while (
            cursor < len(future)
            and future[cursor].arrival_ms
            <= now_ms + 1e-9
        ):
            pending.append(
                future[cursor]
            )

            cursor += 1

        if not pending:
            if cursor >= len(future):
                break

            now_ms = max(
                now_ms,
                float(
                    future[cursor]
                    .arrival_ms
                ),
            )

            continue

        pending = (
            refresh_pending_requests(
                pending=pending,
                states=mobility_states,
                now_ms=now_ms,
                episode_seed=(
                    episode_seed
                ),
                edge_ids=edge_ids,
                mode=mobility_mode,
                prediction_error=(
                    prediction_error
                ),
            )
        )

        env = SchedulingEnv(
            cfg,
            profile,
            batcher,
        )

        batch_start_ms = now_ms

        try:
            result = run_one_mapping(
                env=env,
                infra=infra,
                queue=pending,
                now_ms=now_ms,
                max_steps=max_steps,
                method=method,
                cfg=cfg,
                agent=agent,
                mobility_prior_beta=(
                    mobility_prior_beta
                ),
            )

        except RuntimeError as error:
            drop_index = min(
                range(len(pending)),
                key=lambda index: (
                    pending[index]
                    .remaining_deadline_ms(
                        now_ms
                    ),
                    pending[index]
                    .arrival_ms,
                ),
            )

            dropped_request = (
                pending.pop(drop_index)
            )

            mobility_states.pop(
                dropped_request.request_id,
                None,
            )

            dropped_requests += 1
            slo_violations += 1

            failure_reason = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            continue

        if env.batch is None:
            raise RuntimeError(
                "Environment completed without "
                "a selected batch"
            )

        selected = list(
            env.batch.requests
        )

        selected_ids = {
            request.request_id
            for request in selected
        }

        pending = [
            request
            for request in pending
            if request.request_id
            not in selected_ids
        ]

        decode_times = get_decode_times(
            env=env,
            expected_decode_ms=(
                result.expected_decode_ms
            ),
            batch_size=result.batch_size,
        )

        execution_node = (
            result.mapping[-1].node_id
            if result.mapping
            else infra.cloud_node
        )

        per_request_overhead_ms = []

        for request, decode_ms in zip(
            selected,
            decode_times,
        ):
            base_finish_ms = (
                batch_start_ms
                + result.total_prefill_ms
                + decode_ms
            )

            state = mobility_states[
                request.request_id
            ]

            (
                migration_ms,
                migration_mb,
                handover_events,
                final_state,
            ) = simulate_handover_overhead(
                request=request,
                state=state,
                service_start_ms=(
                    batch_start_ms
                ),
                base_finish_ms=(
                    base_finish_ms
                ),
                execution_node=(
                    execution_node
                ),
                infra=infra,
                profile=profile,
                episode_seed=(
                    episode_seed
                ),
                edge_ids=edge_ids,
                mode=mobility_mode,
                prediction_error=(
                    prediction_error
                ),
                migration_policy=(
                    migration_policy
                ),
                handover_setup_ms=(
                    handover_setup_ms
                ),
            )

            mobility_states[
                request.request_id
            ] = final_state

            completion_ms = (
                base_finish_ms
                + migration_ms
            )

            e2e_ms = max(
                0.0,
                completion_ms
                - request.arrival_ms,
            )

            violation = int(
                e2e_ms
                > request.deadline_ms
            )

            request_e2e_ms.append(
                e2e_ms
            )

            slo_violations += violation

            if not violation:
                successful_requests += 1

            if handover_events > 0:
                handover_affected_requests += 1

                if not violation:
                    handover_successful_requests += 1

            total_handover_events += (
                handover_events
            )

            total_migration_mb += (
                migration_mb
            )

            total_migration_ms += (
                migration_ms
            )

            per_request_overhead_ms.append(
                migration_ms
            )

        service_decode_ms = max(
            [
                decode_ms
                + migration_ms
                for decode_ms, migration_ms
                in zip(
                    decode_times,
                    per_request_overhead_ms,
                )
            ],
            default=(
                result.expected_decode_ms
            ),
        )

        now_ms = (
            batch_start_ms
            + result.total_prefill_ms
            + service_decode_ms
        )

        batch_count += 1

        served_requests += (
            result.batch_size
        )

        batch_sizes.append(
            result.batch_size
        )

        mapping_steps += len(
            result.mapping
        )

        total_reward += float(
            result.total_reward
        )

    unfinished = (
        len(pending)
        + len(future)
        - cursor
    )

    if unfinished > 0:
        dropped_requests += unfinished
        slo_violations += unfinished

    makespan_seconds = (
        now_ms / 1000.0
    )

    handover_slo = (
        handover_successful_requests
        / handover_affected_requests
        if handover_affected_requests > 0
        else np.nan
    )

    return {
        "method": method,
        "mobility_prior_beta": float(
            mobility_prior_beta
        ),
        "mobility_mode": mobility_mode,
        "migration_policy": (
            migration_policy
        ),
        "prediction_error": (
            prediction_error
        ),
        "served_requests": (
            served_requests
        ),
        "dropped_requests": (
            dropped_requests
        ),
        "completion_ratio": (
            served_requests
            / max(total_requests, 1)
        ),
        "batches": batch_count,
        "mean_batch_size": (
            float(
                np.mean(batch_sizes)
            )
            if batch_sizes
            else 0.0
        ),
        "makespan_ms": now_ms,
        "avg_e2e_ms": (
            float(
                np.mean(
                    request_e2e_ms
                )
            )
            if request_e2e_ms
            else np.nan
        ),
        "p95_e2e_ms": (
            float(
                np.quantile(
                    request_e2e_ms,
                    0.95,
                )
            )
            if request_e2e_ms
            else np.nan
        ),
        "slo_satisfaction": (
            1.0
            - slo_violations
            / max(total_requests, 1)
        ),
        "throughput_rps": (
            served_requests
            / makespan_seconds
            if makespan_seconds > 0
            else 0.0
        ),
        "goodput_rps": (
            successful_requests
            / makespan_seconds
            if makespan_seconds > 0
            else 0.0
        ),
        "handover_events": (
            total_handover_events
        ),
        "handover_affected_requests": (
            handover_affected_requests
        ),
        "handover_slo_satisfaction": (
            handover_slo
        ),
        "migration_mb": (
            total_migration_mb
        ),
        "handover_overhead_ms": (
            total_migration_ms
        ),
        "migration_ms": (
            total_migration_ms
            if migration_policy != "keep"
            else 0.0
        ),
        "avg_handover_overhead_ms_per_event": (
            total_migration_ms
            / total_handover_events
            if total_handover_events > 0
            else 0.0
        ),
        "avg_handover_overhead_ms_per_affected_request": (
            total_migration_ms
            / handover_affected_requests
            if handover_affected_requests > 0
            else 0.0
        ),
        "avg_migration_ms_per_event": (
            total_migration_ms
            / total_handover_events
            if (
                total_handover_events > 0
                and migration_policy != "keep"
            )
            else 0.0
        ),
        "avg_migration_ms_per_affected_request": (
            total_migration_ms
            / handover_affected_requests
            if (
                handover_affected_requests > 0
                and migration_policy != "keep"
            )
            else 0.0
        ),
        "avg_migration_mb_per_event": (
            total_migration_mb
            / total_handover_events
            if total_handover_events > 0
            else 0.0
        ),
        "mapping_steps": mapping_steps,
        "reward": total_reward,
        "failure_reason": (
            failure_reason
        ),
    }


def load_full_agent(
    cfg: dict,
    profile: ProfileTable,
    request_pool: pd.DataFrame,
    checkpoint_path: str,
    device: str,
) -> DDQNAgent:
    probe_rng = np.random.default_rng(
        int(cfg["seed"]) + 987654
    )

    probe_infra = (
        make_synthetic_infrastructure(
            cfg,
            probe_rng,
        )
    )

    probe_queue = make_request_queue(
        cfg,
        probe_rng,
        request_pool=request_pool,
    )[:1]

    probe_queue = (
        assign_poisson_arrivals(
            probe_queue,
            probe_rng,
            0.1,
        )
    )

    probe_batcher = (
        NodeConditionedDPBatcher(
            cfg,
            profile,
        )
    )

    probe_env = SchedulingEnv(
        cfg,
        profile,
        probe_batcher,
    )

    probe_observation = (
        probe_env.reset(
            probe_infra,
            probe_queue,
            now_ms=0.0,
        )
    )

    agent = DDQNAgent(
        cfg,
        node_dim=int(
            probe_observation
            .node_features.shape[1]
        ),
        edge_dim=int(
            probe_observation
            .edge_features.shape[1]
        ),
        batch_dim=int(
            probe_observation
            .batch_features.shape[0]
        ),
        device=device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=agent.device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint must be a dict"
        )

    online_state = None

    for key in [
        "online",
        "online_state_dict",
        "model_state_dict",
        "model",
    ]:
        if key in checkpoint:
            online_state = checkpoint[key]
            break

    if online_state is None:
        online_state = checkpoint

    agent.online.load_state_dict(
        online_state
    )

    agent.target.load_state_dict(
        agent.online.state_dict()
    )

    agent.online.eval()
    agent.target.eval()

    return agent


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "configs"
            / "train_online_cpu.yaml"
        ),
    )

    parser.add_argument(
        "--request-trace",
        required=True,
    )

    parser.add_argument(
        "--profile-csv",
        default=None,
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "node_dp",
            "lecu_rba",
            "dybap_adapted",
            "full",
        ],
    )

    parser.add_argument(
        "--agent-checkpoint",
        default=None,
    )

    parser.add_argument(
        "--mobility-mode",
        required=True,
        choices=list(
            MOBILITY_PROFILES
        ),
    )

    parser.add_argument(
        "--migration-policy",
        default="reactive",
        choices=[
            "keep",
            "reactive",
            "prefetch",
            "oracle_guarded",
        ],
    )

    parser.add_argument(
        "--prediction-error",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--handover-setup-ms",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--mobility-prior-beta",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--arrival-rate-rps",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-requests",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.prediction_error
        <= 1.0
    ):
        raise ValueError(
            "--prediction-error must "
            "be in [0, 1]"
        )

    if args.mobility_prior_beta < 0.0:
        raise ValueError(
            "--mobility-prior-beta must "
            "be non-negative"
        )

    if (
        args.method == "full"
        and not args.agent_checkpoint
    ):
        raise ValueError(
            "--method full requires "
            "--agent-checkpoint"
        )

    cfg = load_config(
        args.config
    )

    request_pool = pd.read_csv(
        args.request_trace,
        usecols=[
            "input_tokens",
            "output_tokens",
            "request_type",
            "source",
        ],
    )

    model = build_model_profile(
        cfg
    )

    profile = (
        ProfileTable.from_csv(
            model,
            args.profile_csv,
        )
        if args.profile_csv
        else ProfileTable(model)
    )

    agent = None

    if args.method == "full":
        agent = load_full_agent(
            cfg=cfg,
            profile=profile,
            request_pool=request_pool,
            checkpoint_path=(
                args.agent_checkpoint
            ),
            device=args.device,
        )

    rows = []

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.perf_counter()

    print(
        "Starting dynamic mobility test: "
        f"method={args.method}, "
        f"mode={args.mobility_mode}, "
        f"policy={args.migration_policy}, "
        f"prediction_error="
        f"{args.prediction_error}, "
        f"prior_beta="
        f"{args.mobility_prior_beta}, "
        f"rate={args.arrival_rate_rps}, "
        f"episodes={args.episodes}",
        flush=True,
    )

    for episode in range(
        args.episodes
    ):
        episode_seed = (
            int(cfg["seed"])
            + 20000
            + episode
        )

        rng = np.random.default_rng(
            episode_seed
        )

        infra = (
            make_synthetic_infrastructure(
                cfg,
                rng,
            )
        )

        queue = make_request_queue(
            cfg,
            rng,
            request_pool=request_pool,
        )

        queue = queue[
            :args.max_requests
        ]

        queue = assign_poisson_arrivals(
            queue=queue,
            rng=rng,
            rate_rps=(
                args.arrival_rate_rps
            ),
        )

        metrics = run_dynamic_episode(
            cfg=cfg,
            profile=profile,
            infra=infra,
            queue=queue,
            method=args.method,
            mobility_mode=(
                args.mobility_mode
            ),
            migration_policy=(
                args.migration_policy
            ),
            prediction_error=(
                args.prediction_error
            ),
            handover_setup_ms=(
                args.handover_setup_ms
            ),
            episode_seed=(
                episode_seed
            ),
            max_steps=args.max_steps,
            max_batches=args.max_batches,
            agent=agent,
            mobility_prior_beta=(
                args.mobility_prior_beta
            ),
        )

        rows.append(
            {
                "episode": episode,
                "episode_seed": (
                    episode_seed
                ),
                "arrival_rate_rps": (
                    args.arrival_rate_rps
                ),
                **metrics,
            }
        )

        if (
            episode == 0
            or (
                episode + 1
            )
            % max(
                args.log_every,
                1,
            )
            == 0
            or episode + 1
            == args.episodes
        ):
            pd.DataFrame(
                rows
            ).to_csv(
                output,
                index=False,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"episode={episode + 1:4d}/"
                f"{args.episodes} "
                f"elapsed={elapsed:8.1f}s "
                f"slo="
                f"{np.mean([row['slo_satisfaction'] for row in rows[-args.log_every:]]):.4f} "
                f"handover="
                f"{np.mean([row['handover_events'] for row in rows[-args.log_every:]]):.2f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)

    frame.to_csv(
        output,
        index=False,
    )

    columns = [
        "avg_e2e_ms",
        "p95_e2e_ms",
        "slo_satisfaction",
        "throughput_rps",
        "goodput_rps",
        "handover_events",
        "handover_slo_satisfaction",
        "migration_mb",
        "migration_ms",
        "mean_batch_size",
    ]

    print(
        frame[columns]
        .agg(["mean", "std"])
        .round(4)
        .to_string(),
        flush=True,
    )

    print(
        f"Saved {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
