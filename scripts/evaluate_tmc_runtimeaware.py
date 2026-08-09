from __future__ import annotations

import atexit
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.home() / "msn_graph_scheduler"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.evaluate_dynamic_mobility as evaluator
import msn_scheduler.continuity as continuity

from msn_scheduler.profiles import transfer_time_ms


TRACE_PATH = Path(
    os.environ[
        "TMC_PREFETCH_TRACE_OUTPUT"
    ]
)

TRACE = []

ETA_ERROR_BOUND = float(
    os.environ.get(
        "TMC_ETA_ERROR_BOUND",
        "0.0",
    )
)

RUNTIME_POLICY = os.environ.get(
    "TMC_RUNTIME_POLICY",
    "prefetch",
)

if RUNTIME_POLICY not in {
    "reactive",
    "prefetch",
}:
    raise ValueError(
        f"Unknown TMC_RUNTIME_POLICY: "
        f"{RUNTIME_POLICY}"
    )

ENABLE_PREFETCH = (
    RUNTIME_POLICY == "prefetch"
)

STATE = {
    "episode_seed": None,
}

ORIGINAL_EPISODE = (
    evaluator.run_dynamic_episode
)


_UNCACHED_PLANNER = (
    evaluator
    .find_optimal_continuation_mapping
)

_ORIGINAL_FEASIBLE = (
    continuity.feasible_nodes_for_group
)

_ORIGINAL_TRANSFER = (
    continuity.transfer_time_ms
)


def cached_exact_planner(
    *args,
    **kwargs,
):
    transfer_cache = {}
    feasible_cache = {}

    def cached_transfer(
        infra,
        source,
        destination,
        data_mb,
    ):
        key = (
            id(infra),
            str(source),
            str(destination),
            float(data_mb).hex(),
        )

        if key not in transfer_cache:
            transfer_cache[key] = (
                _ORIGINAL_TRANSFER(
                    infra,
                    source,
                    destination,
                    data_mb,
                )
            )

        return transfer_cache[key]

    def cached_feasible(
        *,
        infra,
        profile,
        start_block,
        group_size,
        batch_size,
        max_input_tokens,
        expected_output_tokens,
    ):
        key = (
            id(infra),
            id(profile),
            str(infra.anchor_node),
            int(start_block),
            int(group_size),
            int(batch_size),
            int(max_input_tokens),
            int(expected_output_tokens),
        )

        if key not in feasible_cache:
            feasible_cache[key] = tuple(
                _ORIGINAL_FEASIBLE(
                    infra=infra,
                    profile=profile,
                    start_block=start_block,
                    group_size=group_size,
                    batch_size=batch_size,
                    max_input_tokens=max_input_tokens,
                    expected_output_tokens=expected_output_tokens,
                )
            )

        return feasible_cache[key]

    saved_transfer = (
        continuity.transfer_time_ms
    )

    saved_feasible = (
        continuity.feasible_nodes_for_group
    )

    continuity.transfer_time_ms = (
        cached_transfer
    )

    continuity.feasible_nodes_for_group = (
        cached_feasible
    )

    try:
        return _UNCACHED_PLANNER(
            *args,
            **kwargs,
        )

    finally:
        continuity.transfer_time_ms = (
            saved_transfer
        )

        continuity.feasible_nodes_for_group = (
            saved_feasible
        )


evaluator.find_optimal_continuation_mapping = (
    cached_exact_planner
)


def delta_migration(
    token_count,
    old_mapping,
    new_mapping,
    infra,
    profile,
):
    token_count = max(
        int(token_count),
        0,
    )

    if token_count <= 0:
        return 0.0, 0.0

    num_blocks = int(
        profile.model.num_blocks
    )

    old_nodes = (
        continuity._mapping_block_nodes(
            old_mapping,
            num_blocks,
        )
    )

    new_nodes = (
        continuity._mapping_block_nodes(
            new_mapping,
            num_blocks,
        )
    )

    bytes_per_block = (
        token_count
        * profile.model
        .kv_bytes_per_token_per_block
    )

    total_ms = 0.0
    total_mb = 0.0

    block = 0

    while block < num_blocks:
        source = old_nodes[block]
        destination = new_nodes[block]

        if source == destination:
            block += 1
            continue

        end = block + 1

        while (
            end < num_blocks
            and old_nodes[end] == source
            and new_nodes[end]
            == destination
        ):
            end += 1

        count = end - block

        mb = (
            bytes_per_block
            * count
            / (1024.0 * 1024.0)
        )

        ms = transfer_time_ms(
            infra,
            source,
            destination,
            mb,
        )

        if not np.isfinite(ms):
            return (
                float("inf"),
                float("inf"),
            )

        total_ms += float(ms)
        total_mb += float(mb)

        block = end

    return (
        float(total_ms),
        float(total_mb),
    )


def traced_episode(
    *args,
    **kwargs,
):
    STATE["episode_seed"] = int(
        kwargs["episode_seed"]
    )

    try:
        return ORIGINAL_EPISODE(
            *args,
            **kwargs,
        )
    finally:
        STATE["episode_seed"] = None


def simulate_prefetch(
    request,
    initial_state,
    service_start_ms,
    prefill_ms,
    mapping,
    batch_size,
    infra,
    profile,
    policy,
    handover_setup_ms,
    replan_callback,
    next_state_callback,
    minimum_replan_gain_ms=0.0,
    max_handovers=256,
):
    current_mapping = list(mapping)

    current_state = initial_state

    current_anchor = str(
        current_state.current_anchor
    )

    current_ms = float(
        service_start_ms
        + prefill_ms
    )

    remaining_tokens = float(
        max(
            request.expected_output_tokens,
            0,
        )
    )

    generated_tokens = 0.0

    total_blocking_migration_ms = 0.0
    total_migration_mb = 0.0

    handover_events = 0
    replan_events = 0
    replanned_blocks = 0

    while remaining_tokens > 1e-9:
        per_token_ms = (
            continuity
            .mapping_decode_per_token_ms(
                request=request,
                mapping=current_mapping,
                anchor_node=current_anchor,
                batch_size=batch_size,
                infra=infra,
                profile=profile,
            )
        )

        if (
            not np.isfinite(per_token_ms)
            or per_token_ms <= 0.0
        ):
            raise RuntimeError(
                "Invalid decode time: "
                f"{per_token_ms}"
            )

        finish_ms = (
            current_ms
            + remaining_tokens
            * per_token_ms
        )

        event_ms = float(
            current_state.next_handover_ms
        )

        if (
            not np.isfinite(event_ms)
            or event_ms > finish_ms
        ):
            current_ms = finish_ms
            remaining_tokens = 0.0
            break

        lead_ms = max(
            event_ms - current_ms,
            0.0,
        )

        eta_rng = evaluator.deterministic_rng(
            episode_seed=int(
                STATE["episode_seed"]
            ),
            request_id=request.request_id,
            transition_index=int(
                current_state.transition_index
            ),
            stream="eta_prediction_error",
        )

        if ETA_ERROR_BOUND > 0.0:
            eta_relative_error = float(
                eta_rng.uniform(
                    -ETA_ERROR_BOUND,
                    ETA_ERROR_BOUND,
                )
            )
        else:
            eta_relative_error = 0.0

        predicted_lead_ms = max(
            lead_ms
            * (
                1.0
                + eta_relative_error
            ),
            0.0,
        )

        predicted_event_ms = (
            current_ms
            + predicted_lead_ms
        )

        predicted_tokens_before_event = min(
            remaining_tokens,
            predicted_lead_ms
            / per_token_ms,
        )

        predicted_future_remaining = (
            remaining_tokens
            - predicted_tokens_before_event
        )

        predicted_future_generated = (
            generated_tokens
            + predicted_tokens_before_event
        )

        tokens_before_event = min(
            remaining_tokens,
            lead_ms / per_token_ms,
        )

        future_remaining = (
            remaining_tokens
            - tokens_before_event
        )

        future_generated = (
            generated_tokens
            + tokens_before_event
        )

        predicted_anchor = str(
            current_state
            .predicted_next_anchor
        )

        actual_anchor = str(
            current_state
            .actual_next_anchor
        )

        plan = None

        prefetch_planner_ms = 0.0
        reactive_planner_ms = 0.0

        # --------------------------------------------------
        # Prefetch decision before handover.
        #
        # Do not prefetch if this handover already occurred
        # while we were blocked by a previous migration.
        # --------------------------------------------------

        if (
            ENABLE_PREFETCH
            and replan_callback is not None
            and predicted_future_remaining > 1e-9
            and predicted_lead_ms > 1e-9
        ):
            planner_start_ns = (
                time.perf_counter_ns()
            )

            candidate = (
                replan_callback(
                    predicted_anchor,
                    predicted_event_ms
                    + handover_setup_ms,
                    max(
                        int(
                            np.ceil(
                                predicted_future_remaining
                            )
                        ),
                        1,
                    ),
                    max(
                        int(
                            np.floor(
                                predicted_future_generated
                            )
                        ),
                        0,
                    ),
                    current_mapping,
                )
            )

            prefetch_planner_ms = (
                time.perf_counter_ns()
                - planner_start_ns
            ) / 1e6

            # Planning is causal. If the real handover
            # already arrives before planning finishes,
            # this prediction cannot launch a prefetch.
            if (
                prefetch_planner_ms
                >= lead_ms
            ):
                candidate = None

            if candidate:
                candidate = list(
                    candidate
                )

                snapshot_generated = max(
                    int(
                        np.floor(
                            generated_tokens
                        )
                    ),
                    0,
                )

                event_generated = max(
                    int(
                        np.floor(
                            future_generated
                        )
                    ),
                    snapshot_generated,
                )

                (
                    base_ms,
                    base_mb,
                    changed_blocks,
                ) = (
                    continuity
                    .estimate_mapping_migration(
                        request=request,
                        generated_tokens=(
                            snapshot_generated
                        ),
                        old_mapping=(
                            current_mapping
                        ),
                        new_mapping=candidate,
                        infra=infra,
                        profile=profile,
                    )
                )

                delta_tokens = max(
                    event_generated
                    - snapshot_generated,
                    0,
                )

                (
                    delta_ms,
                    delta_mb,
                ) = delta_migration(
                    token_count=(
                        delta_tokens
                    ),
                    old_mapping=(
                        current_mapping
                    ),
                    new_mapping=candidate,
                    infra=infra,
                    profile=profile,
                )

                if (
                    np.isfinite(base_ms)
                    and np.isfinite(delta_ms)
                ):
                    prefetch_start_ms = max(
                        current_ms
                        + prefetch_planner_ms,
                        predicted_event_ms
                        - float(base_ms),
                    )

                    actual_overlap_ms = max(
                        event_ms
                        - prefetch_start_ms,
                        0.0,
                    )

                    hidden_ms = min(
                        float(base_ms),
                        float(actual_overlap_ms),
                    )

                    residual_base_ms = max(
                        float(base_ms)
                        - hidden_ms,
                        0.0,
                    )

                    if float(base_ms) > 1e-12:
                        transferred_fraction = min(
                            max(
                                hidden_ms
                                / float(base_ms),
                                0.0,
                            ),
                            1.0,
                        )
                    else:
                        transferred_fraction = 0.0

                    transferred_base_mb = (
                        float(base_mb)
                        * transferred_fraction
                    )

                    plan = {
                        "candidate": candidate,
                        "predicted_anchor": (
                            predicted_anchor
                        ),
                        "base_ms": float(
                            base_ms
                        ),
                        "base_mb": float(
                            base_mb
                        ),
                        "hidden_ms": float(
                            hidden_ms
                        ),
                        "transferred_base_mb": float(
                            transferred_base_mb
                        ),
                        "residual_base_ms": (
                            float(
                                residual_base_ms
                            )
                        ),
                        "delta_ms": float(
                            delta_ms
                        ),
                        "delta_mb": float(
                            delta_mb
                        ),
                        "delta_tokens": int(
                            delta_tokens
                        ),
                        "changed_blocks": int(
                            changed_blocks
                        ),
                    }

        # --------------------------------------------------
        # Decode normally while prefetch happens in parallel.
        # --------------------------------------------------

        completed_tokens = (
            tokens_before_event
        )

        remaining_tokens -= (
            completed_tokens
        )

        generated_tokens += (
            completed_tokens
        )

        current_ms = max(
            current_ms
            + completed_tokens
            * per_token_ms,
            event_ms,
        )

        current_ms += float(
            handover_setup_ms
        )

        accepted_prefetch = False

        # --------------------------------------------------
        # Correct prediction: only residual base KV +
        # newly-generated delta KV blocks decode.
        # --------------------------------------------------

        if (
            plan is not None
            and plan[
                "predicted_anchor"
            ] == actual_anchor
            and remaining_tokens > 1e-9
        ):
            candidate = plan[
                "candidate"
            ]

            keep_per_token_ms = (
                continuity
                .mapping_decode_per_token_ms(
                    request=request,
                    mapping=current_mapping,
                    anchor_node=(
                        actual_anchor
                    ),
                    batch_size=batch_size,
                    infra=infra,
                    profile=profile,
                )
            )

            candidate_per_token_ms = (
                continuity
                .mapping_decode_per_token_ms(
                    request=request,
                    mapping=candidate,
                    anchor_node=(
                        actual_anchor
                    ),
                    batch_size=batch_size,
                    infra=infra,
                    profile=profile,
                )
            )

            keep_remaining_ms = (
                remaining_tokens
                * keep_per_token_ms
            )

            blocking_ms = (
                plan[
                    "residual_base_ms"
                ]
                + plan[
                    "delta_ms"
                ]
            )

            candidate_remaining_ms = (
                blocking_ms
                + remaining_tokens
                * candidate_per_token_ms
            )

            if (
                np.isfinite(
                    candidate_remaining_ms
                )
                and (
                    candidate_remaining_ms
                    + minimum_replan_gain_ms
                    < keep_remaining_ms
                )
            ):
                current_mapping = list(
                    candidate
                )

                current_ms += float(
                    blocking_ms
                )

                total_blocking_migration_ms += (
                    float(
                        blocking_ms
                    )
                )

                total_migration_mb += (
                    float(
                        plan["base_mb"]
                        + plan["delta_mb"]
                    )
                )

                replan_events += 1

                replanned_blocks += int(
                    plan[
                        "changed_blocks"
                    ]
                )

                accepted_prefetch = True

        # --------------------------------------------------
        # No usable prefetch:
        # fall back to corrected reactive Oracle.
        # --------------------------------------------------

        if (
            not accepted_prefetch
            and remaining_tokens > 1e-9
        ):
            candidate = None

            if replan_callback is not None:
                planner_start_ns = (
                    time.perf_counter_ns()
                )

                candidate = (
                    replan_callback(
                        actual_anchor,
                        current_ms,
                        max(
                            int(
                                np.ceil(
                                    remaining_tokens
                                )
                            ),
                            1,
                        ),
                        max(
                            int(
                                np.floor(
                                    generated_tokens
                                )
                            ),
                            0,
                        ),
                        current_mapping,
                    )
                )

                reactive_planner_ms = (
                    time.perf_counter_ns()
                    - planner_start_ns
                ) / 1e6

                # Reactive replanning happens after
                # handover and therefore blocks
                # continuation conservatively.
                current_ms += float(
                    reactive_planner_ms
                )

            if candidate:
                candidate = list(
                    candidate
                )

                (
                    migration_ms,
                    migration_mb,
                    changed_blocks,
                ) = (
                    continuity
                    .estimate_mapping_migration(
                        request=request,
                        generated_tokens=max(
                            int(
                                np.floor(
                                    generated_tokens
                                )
                            ),
                            0,
                        ),
                        old_mapping=(
                            current_mapping
                        ),
                        new_mapping=(
                            candidate
                        ),
                        infra=infra,
                        profile=profile,
                    )
                )

                keep_per_token_ms = (
                    continuity
                    .mapping_decode_per_token_ms(
                        request=request,
                        mapping=current_mapping,
                        anchor_node=(
                            actual_anchor
                        ),
                        batch_size=batch_size,
                        infra=infra,
                        profile=profile,
                    )
                )

                candidate_per_token_ms = (
                    continuity
                    .mapping_decode_per_token_ms(
                        request=request,
                        mapping=candidate,
                        anchor_node=(
                            actual_anchor
                        ),
                        batch_size=batch_size,
                        infra=infra,
                        profile=profile,
                    )
                )

                keep_remaining_ms = (
                    remaining_tokens
                    * keep_per_token_ms
                )

                candidate_remaining_ms = (
                    migration_ms
                    + remaining_tokens
                    * candidate_per_token_ms
                )

                if (
                    np.isfinite(
                        candidate_remaining_ms
                    )
                    and (
                        candidate_remaining_ms
                        + minimum_replan_gain_ms
                        < keep_remaining_ms
                    )
                ):
                    current_mapping = list(
                        candidate
                    )

                    current_ms += float(
                        migration_ms
                    )

                    total_blocking_migration_ms += (
                        float(
                            migration_ms
                        )
                    )

                    total_migration_mb += (
                        float(
                            migration_mb
                        )
                    )

                    replan_events += 1

                    replanned_blocks += int(
                        changed_blocks
                    )

        TRACE.append(
            {
                "episode_seed": (
                    STATE[
                        "episode_seed"
                    ]
                ),
                "request_id": (
                    request.request_id
                ),
                "handover_index": (
                    handover_events
                ),
                "runtime_policy": (
                    RUNTIME_POLICY
                ),
                "lead_ms": float(
                    lead_ms
                ),
                "prefetch_planner_ms": float(
                    prefetch_planner_ms
                ),
                "reactive_planner_ms": float(
                    reactive_planner_ms
                ),
                "blocking_planner_ms": float(
                    reactive_planner_ms
                ),
                "predicted_lead_ms": float(
                    predicted_lead_ms
                ),
                "eta_relative_error": float(
                    eta_relative_error
                ),
                "eta_error_bound": float(
                    ETA_ERROR_BOUND
                ),
                "prediction_correct": int(
                    predicted_anchor
                    == actual_anchor
                ),
                "had_prefetch_plan": int(
                    plan is not None
                ),
                "accepted_prefetch": int(
                    accepted_prefetch
                ),
                "base_migration_ms": (
                    plan["base_ms"]
                    if plan is not None
                    else 0.0
                ),
                "hidden_migration_ms": (
                    plan["hidden_ms"]
                    if (
                        plan is not None
                        and accepted_prefetch
                    )
                    else 0.0
                ),
                "prefetched_mb": (
                    plan["transferred_base_mb"]
                    if plan is not None
                    else 0.0
                ),
                "useful_prefetch_mb": (
                    plan["transferred_base_mb"]
                    if (
                        plan is not None
                        and accepted_prefetch
                    )
                    else 0.0
                ),
                "wasted_prefetch_mb": (
                    plan["transferred_base_mb"]
                    if (
                        plan is not None
                        and not accepted_prefetch
                    )
                    else 0.0
                ),
                "wrong_anchor_waste_mb": (
                    plan["transferred_base_mb"]
                    if (
                        plan is not None
                        and predicted_anchor
                        != actual_anchor
                    )
                    else 0.0
                ),
                "residual_base_ms": (
                    plan[
                        "residual_base_ms"
                    ]
                    if (
                        plan is not None
                        and accepted_prefetch
                    )
                    else 0.0
                ),
                "delta_sync_ms": (
                    plan["delta_ms"]
                    if (
                        plan is not None
                        and accepted_prefetch
                    )
                    else 0.0
                ),
                "delta_tokens": (
                    plan["delta_tokens"]
                    if plan is not None
                    else 0
                ),
                "changed_blocks": (
                    plan[
                        "changed_blocks"
                    ]
                    if plan is not None
                    else 0
                ),
            }
        )

        handover_events += 1

        current_anchor = (
            actual_anchor
        )

        current_state = (
            next_state_callback(
                current_state,
                event_ms,
                actual_anchor,
            )
        )

        if (
            handover_events
            >= max_handovers
        ):
            raise RuntimeError(
                "Stateful handover "
                "limit exceeded."
            )

    return (
        continuity.StatefulDecodeResult(
            completion_ms=float(
                current_ms
            ),
            migration_ms=float(
                total_blocking_migration_ms
            ),
            migration_mb=float(
                total_migration_mb
            ),
            handover_events=int(
                handover_events
            ),
            replan_events=int(
                replan_events
            ),
            replanned_blocks=int(
                replanned_blocks
            ),
            final_state=(
                current_state
            ),
        )
    )


def save_trace():
    TRACE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        TRACE
    ).to_csv(
        TRACE_PATH,
        index=False,
    )

    print(
        f"Saved trace: {TRACE_PATH}",
        flush=True,
    )


atexit.register(
    save_trace
)

evaluator.run_dynamic_episode = (
    traced_episode
)

evaluator.simulate_stateful_decode = (
    simulate_prefetch
)

evaluator.main()
