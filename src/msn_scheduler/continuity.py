from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .candidates import (
    feasible_nodes_for_group,
)
from .datatypes import (
    Infrastructure,
    MappingStep,
    Request,
)
from .profiles import (
    ProfileTable,
    transfer_time_ms,
)


@dataclass
class StatefulDecodeResult:
    completion_ms: float
    migration_ms: float
    migration_mb: float
    handover_events: int
    replan_events: int
    replanned_blocks: int
    final_state: object


def mapping_decode_per_token_ms(
    request: Request,
    mapping: Sequence[MappingStep],
    anchor_node: str,
    batch_size: int,
    infra: Infrastructure,
    profile: ProfileTable,
) -> float:
    """Match SchedulingEnv's decode model for an arbitrary anchor."""

    if not mapping:
        return 0.0

    token_mb = profile.intermediate_mb(
        batch_size,
        1,
        decode=True,
    )

    # Keep the original midpoint-context convention so that
    # stationary stateful_keep reproduces the legacy baseline.
    context_tokens = int(
        request.input_tokens
        + request.expected_output_tokens / 2
    )

    previous = anchor_node
    total_per_token_ms = 0.0

    for step in mapping:
        node = infra.nodes[step.node_id]

        total_per_token_ms += (
            profile.decode_per_token_ms(
                node=node,
                batch_size=batch_size,
                context_tokens=context_tokens,
                group_size=step.group_size,
            )
        )

        transfer_ms = transfer_time_ms(
            infra,
            previous,
            step.node_id,
            token_mb,
        )

        if not np.isfinite(transfer_ms):
            return float("inf")

        total_per_token_ms += float(
            transfer_ms
        )

        previous = step.node_id

    return float(total_per_token_ms)


def _mapping_block_nodes(
    mapping: Sequence[MappingStep],
    num_blocks: int,
) -> list[str]:
    nodes: list[str | None] = [
        None
        for _ in range(num_blocks)
    ]

    for step in mapping:
        start = int(step.start_block)
        end = start + int(step.group_size)

        if start < 0 or end > num_blocks:
            raise ValueError(
                "Mapping step is outside model blocks: "
                f"start={start}, end={end}, "
                f"num_blocks={num_blocks}"
            )

        for block in range(start, end):
            if nodes[block] is not None:
                raise ValueError(
                    f"Block {block} is mapped twice."
                )

            nodes[block] = str(
                step.node_id
            )

    if any(node is None for node in nodes):
        missing = [
            index
            for index, node in enumerate(nodes)
            if node is None
        ]

        raise ValueError(
            f"Mapping does not cover blocks: {missing}"
        )

    return [
        str(node)
        for node in nodes
    ]


def estimate_mapping_migration(
    request: Request,
    generated_tokens: int,
    old_mapping: Sequence[MappingStep],
    new_mapping: Sequence[MappingStep],
    infra: Infrastructure,
    profile: ProfileTable,
) -> tuple[float, float, int]:
    """Migrate KV only for blocks whose execution node changes."""

    num_blocks = int(
        profile.model.num_blocks
    )

    old_nodes = _mapping_block_nodes(
        old_mapping,
        num_blocks,
    )

    new_nodes = _mapping_block_nodes(
        new_mapping,
        num_blocks,
    )

    context_tokens = max(
        int(request.input_tokens)
        + int(generated_tokens),
        1,
    )

    bytes_per_block = (
        context_tokens
        * profile.model
        .kv_bytes_per_token_per_block
    )

    total_ms = 0.0
    total_mb = 0.0
    changed_blocks = 0

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
            and new_nodes[end] == destination
        ):
            end += 1

        group_size = end - block

        migration_mb = (
            bytes_per_block
            * group_size
            / (1024.0 * 1024.0)
        )

        migration_ms = transfer_time_ms(
            infra,
            source,
            destination,
            migration_mb,
        )

        if not np.isfinite(migration_ms):
            return (
                float("inf"),
                float("inf"),
                changed_blocks + group_size,
            )

        total_ms += float(migration_ms)
        total_mb += float(migration_mb)
        changed_blocks += group_size

        block = end

    return (
        float(total_ms),
        float(total_mb),
        int(changed_blocks),
    )



def _migration_segment_ms(
    infra: Infrastructure,
    source: str | None,
    destination: str | None,
    block_count: int,
    kv_mb_per_block: float,
) -> float:
    if (
        source is None
        or destination is None
        or block_count <= 0
    ):
        return 0.0

    value = transfer_time_ms(
        infra,
        source,
        destination,
        kv_mb_per_block
        * block_count,
    )

    return float(value)


def find_optimal_continuation_mapping(
    request: Request,
    old_mapping: Sequence[MappingStep],
    new_anchor: str,
    remaining_tokens: int,
    generated_tokens: int,
    batch_size: int,
    infra: Infrastructure,
    profile: ProfileTable,
    cfg: dict,
) -> list[MappingStep] | None:
    """Global DP oracle for post-handover continuation.

    Objective:
        changed-block KV migration time
        + remaining decode time.

    Unlike the frozen DDQN replan, this searches all
    feasible node/group continuations.
    """

    num_blocks = int(
        profile.model.num_blocks
    )

    old_nodes = _mapping_block_nodes(
        old_mapping,
        num_blocks,
    )

    remaining_tokens = max(
        int(remaining_tokens),
        1,
    )

    generated_tokens = max(
        int(generated_tokens),
        0,
    )

    batch_size = max(
        int(batch_size),
        1,
    )

    # KV already contains input tokens plus tokens
    # generated before this handover.
    kv_context_tokens = max(
        int(request.input_tokens)
        + generated_tokens,
        1,
    )

    kv_mb_per_block = (
        kv_context_tokens
        * profile.model
        .kv_bytes_per_token_per_block
        / (1024.0 * 1024.0)
    )

    # Keep exactly the same decode-context convention
    # as SchedulingEnv / mapping_decode_per_token_ms.
    decode_context_tokens = max(
        int(
            request.input_tokens
            + request.expected_output_tokens
            / 2
        ),
        1,
    )

    token_mb = (
        profile.intermediate_mb(
            batch_size,
            1,
            decode=True,
        )
    )

    group_sizes = sorted(
        {
            int(value)
            for value
            in cfg["scheduler"][
                "group_sizes"
            ]
            if int(value) > 0
        }
    )

    # feasible_nodes_for_group uses infra.anchor_node
    # only for reachability. Use the real post-handover
    # anchor for the oracle search.
    oracle_infra = Infrastructure(
        nodes=infra.nodes,
        links=infra.links,
        cloud_node=infra.cloud_node,
        anchor_node=str(new_anchor),
    )

    # DP state:
    #
    # (
    #   previous_execution_node,
    #   open_migration_source,
    #   open_migration_destination,
    #   open_migration_block_count,
    # )
    #
    # Migration transfer is charged only when a
    # contiguous source->destination block segment
    # closes. This matches estimate_mapping_migration().
    frontiers: list[dict] = [
        {}
        for _ in range(
            num_blocks + 1
        )
    ]

    initial_state = (
        str(new_anchor),
        None,
        None,
        0,
    )

    frontiers[0][
        initial_state
    ] = (
        0.0,
        [],
    )

    for start_block in range(
        num_blocks
    ):
        if not frontiers[
            start_block
        ]:
            continue

        remaining_blocks = (
            num_blocks
            - start_block
        )

        for (
            state,
            (
                accumulated_cost,
                partial_mapping,
            ),
        ) in list(
            frontiers[
                start_block
            ].items()
        ):
            (
                previous_node,
                open_source,
                open_destination,
                open_count,
            ) = state

            for group_size in group_sizes:
                if (
                    group_size
                    > remaining_blocks
                ):
                    continue

                feasible_nodes = (
                    feasible_nodes_for_group(
                        infra=oracle_infra,
                        profile=profile,
                        start_block=(
                            start_block
                        ),
                        group_size=(
                            group_size
                        ),
                        batch_size=(
                            batch_size
                        ),
                        max_input_tokens=(
                            kv_context_tokens
                        ),
                        expected_output_tokens=(
                            remaining_tokens
                        ),
                    )
                )

                for node_id in feasible_nodes:
                    node = infra.nodes[
                        node_id
                    ]

                    decode_compute_ms = (
                        profile
                        .decode_per_token_ms(
                            node=node,
                            batch_size=(
                                batch_size
                            ),
                            context_tokens=(
                                decode_context_tokens
                            ),
                            group_size=(
                                group_size
                            ),
                        )
                    )

                    decode_transfer_ms = (
                        transfer_time_ms(
                            infra,
                            previous_node,
                            node_id,
                            token_mb,
                        )
                    )

                    if not np.isfinite(
                        decode_transfer_ms
                    ):
                        continue

                    incremental_cost = (
                        remaining_tokens
                        * (
                            float(
                                decode_compute_ms
                            )
                            + float(
                                decode_transfer_ms
                            )
                        )
                    )

                    next_source = (
                        open_source
                    )

                    next_destination = (
                        open_destination
                    )

                    next_count = int(
                        open_count
                    )

                    valid = True

                    for block in range(
                        start_block,
                        start_block
                        + group_size,
                    ):
                        old_node = str(
                            old_nodes[block]
                        )

                        if old_node == node_id:
                            segment_cost = (
                                _migration_segment_ms(
                                    infra=infra,
                                    source=(
                                        next_source
                                    ),
                                    destination=(
                                        next_destination
                                    ),
                                    block_count=(
                                        next_count
                                    ),
                                    kv_mb_per_block=(
                                        kv_mb_per_block
                                    ),
                                )
                            )

                            if not np.isfinite(
                                segment_cost
                            ):
                                valid = False
                                break

                            incremental_cost += (
                                segment_cost
                            )

                            next_source = None
                            next_destination = None
                            next_count = 0

                            continue

                        pair = (
                            old_node,
                            str(node_id),
                        )

                        current_pair = (
                            (
                                next_source,
                                next_destination,
                            )
                            if next_source
                            is not None
                            else None
                        )

                        if current_pair == pair:
                            next_count += 1
                            continue

                        segment_cost = (
                            _migration_segment_ms(
                                infra=infra,
                                source=(
                                    next_source
                                ),
                                destination=(
                                    next_destination
                                ),
                                block_count=(
                                    next_count
                                ),
                                kv_mb_per_block=(
                                    kv_mb_per_block
                                ),
                            )
                        )

                        if not np.isfinite(
                            segment_cost
                        ):
                            valid = False
                            break

                        incremental_cost += (
                            segment_cost
                        )

                        next_source = pair[0]
                        next_destination = pair[1]
                        next_count = 1

                    if not valid:
                        continue

                    next_block = (
                        start_block
                        + group_size
                    )

                    next_state = (
                        str(node_id),
                        next_source,
                        next_destination,
                        next_count,
                    )

                    next_cost = (
                        accumulated_cost
                        + incremental_cost
                    )

                    existing = (
                        frontiers[
                            next_block
                        ].get(
                            next_state
                        )
                    )

                    if (
                        existing is not None
                        and existing[0]
                        <= next_cost
                        + 1e-12
                    ):
                        continue

                    step = MappingStep(
                        node_id=str(
                            node_id
                        ),
                        start_block=int(
                            start_block
                        ),
                        group_size=int(
                            group_size
                        ),
                        prefill_ms=0.0,
                        transfer_ms=0.0,
                        handover_ms=0.0,
                    )

                    frontiers[
                        next_block
                    ][
                        next_state
                    ] = (
                        float(
                            next_cost
                        ),
                        partial_mapping
                        + [step],
                    )

    best_cost = float("inf")
    best_mapping = None

    for (
        state,
        (
            accumulated_cost,
            candidate_mapping,
        ),
    ) in frontiers[
        num_blocks
    ].items():
        (
            _previous_node,
            open_source,
            open_destination,
            open_count,
        ) = state

        final_segment_ms = (
            _migration_segment_ms(
                infra=infra,
                source=open_source,
                destination=(
                    open_destination
                ),
                block_count=(
                    open_count
                ),
                kv_mb_per_block=(
                    kv_mb_per_block
                ),
            )
        )

        if not np.isfinite(
            final_segment_ms
        ):
            continue

        total_cost = (
            accumulated_cost
            + final_segment_ms
        )

        if total_cost < (
            best_cost - 1e-12
        ):
            best_cost = float(
                total_cost
            )

            best_mapping = list(
                candidate_mapping
            )

    return best_mapping


def simulate_stateful_decode(
    request: Request,
    initial_state: object,
    service_start_ms: float,
    prefill_ms: float,
    mapping: Sequence[MappingStep],
    batch_size: int,
    infra: Infrastructure,
    profile: ProfileTable,
    policy: str,
    handover_setup_ms: float,
    replan_callback: (
        Callable[
            [str, float, int, int],
            Sequence[MappingStep] | None,
        ]
        | None
    ),
    next_state_callback: Callable[
        [object, float, str],
        object,
    ],
    minimum_replan_gain_ms: float = 0.0,
    max_handovers: int = 256,
) -> StatefulDecodeResult:
    """Advance decode through real handover events.

    stateful_keep preserves the original block mapping.
    oracle_replan compares remaining keep time against a
    new MSN mapping plus partial block-level KV migration.
    """

    if policy not in {
        "stateful_keep",
        "oracle_replan",
    }:
        raise ValueError(
            f"Unsupported stateful policy: {policy}"
        )

    current_mapping = list(mapping)
    current_state = initial_state
    current_anchor = str(
        current_state.current_anchor
    )

    current_ms = float(
        service_start_ms + prefill_ms
    )

    remaining_tokens = float(
        max(
            request.expected_output_tokens,
            0,
        )
    )

    generated_tokens = 0.0

    total_migration_ms = 0.0
    total_migration_mb = 0.0
    handover_events = 0
    replan_events = 0
    replanned_blocks = 0

    while remaining_tokens > 1e-9:
        per_token_ms = (
            mapping_decode_per_token_ms(
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
                "Invalid decode-per-token time: "
                f"{per_token_ms}"
            )

        finish_without_handover_ms = (
            current_ms
            + remaining_tokens
            * per_token_ms
        )

        event_ms = float(
            current_state.next_handover_ms
        )

        if (
            not np.isfinite(event_ms)
            or event_ms
            > finish_without_handover_ms
        ):
            current_ms = (
                finish_without_handover_ms
            )

            remaining_tokens = 0.0
            break

        available_decode_ms = max(
            event_ms - current_ms,
            0.0,
        )

        completed_tokens = min(
            remaining_tokens,
            available_decode_ms
            / per_token_ms,
        )

        remaining_tokens -= completed_tokens
        generated_tokens += completed_tokens

        current_ms = max(
            current_ms
            + completed_tokens
            * per_token_ms,
            event_ms,
        )

        new_anchor = str(
            current_state.actual_next_anchor
        )

        current_ms += float(
            handover_setup_ms
        )

        if (
            policy == "oracle_replan"
            and replan_callback is not None
            and remaining_tokens > 1e-9
        ):
            candidate_mapping = (
                replan_callback(
                    new_anchor,
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
                )
            )

            if candidate_mapping:
                (
                    migration_ms,
                    migration_mb,
                    changed_blocks,
                ) = estimate_mapping_migration(
                    request=request,
                    generated_tokens=max(
                        int(
                            np.floor(
                                generated_tokens
                            )
                        ),
                        0,
                    ),
                    old_mapping=current_mapping,
                    new_mapping=candidate_mapping,
                    infra=infra,
                    profile=profile,
                )

                keep_per_token_ms = (
                    mapping_decode_per_token_ms(
                        request=request,
                        mapping=current_mapping,
                        anchor_node=new_anchor,
                        batch_size=batch_size,
                        infra=infra,
                        profile=profile,
                    )
                )

                replan_per_token_ms = (
                    mapping_decode_per_token_ms(
                        request=request,
                        mapping=candidate_mapping,
                        anchor_node=new_anchor,
                        batch_size=batch_size,
                        infra=infra,
                        profile=profile,
                    )
                )

                keep_remaining_ms = (
                    remaining_tokens
                    * keep_per_token_ms
                )

                replan_remaining_ms = (
                    migration_ms
                    + remaining_tokens
                    * replan_per_token_ms
                )

                if (
                    np.isfinite(
                        replan_remaining_ms
                    )
                    and (
                        replan_remaining_ms
                        + minimum_replan_gain_ms
                        < keep_remaining_ms
                    )
                ):
                    current_mapping = list(
                        candidate_mapping
                    )

                    current_ms += float(
                        migration_ms
                    )

                    total_migration_ms += float(
                        migration_ms
                    )

                    total_migration_mb += float(
                        migration_mb
                    )

                    replan_events += 1
                    replanned_blocks += int(
                        changed_blocks
                    )

        handover_events += 1
        current_anchor = new_anchor

        current_state = (
            next_state_callback(
                current_state,
                event_ms,
                new_anchor,
            )
        )

        if handover_events >= max_handovers:
            raise RuntimeError(
                "Stateful handover limit exceeded."
            )

    return StatefulDecodeResult(
        completion_ms=float(current_ms),
        migration_ms=float(
            total_migration_ms
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
        final_state=current_state,
    )
