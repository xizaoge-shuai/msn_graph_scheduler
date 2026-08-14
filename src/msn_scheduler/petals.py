from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

from .datatypes import (
    ComputeNode,
    Infrastructure,
)
from .profiles import (
    ProfileTable,
    shortest_path_latency_ms,
)


@dataclass(frozen=True)
class PetalsSpan:
    start_block: int
    end_block: int
    node_id: str

    @property
    def group_size(self) -> int:
        return (
            self.end_block
            - self.start_block
        )


# ============================================================
# PETALS-Adapted
#
# Reproduced idea:
#   - servers already hold consecutive model blocks
#   - heterogeneous server compute throughput
#   - network-latency-aware routing
#   - find a low-latency chain covering all model blocks
#
# Deliberately NOT reproduced:
#   - server failures
#   - dual KV-cache recovery
#   - D* Lite incremental replanning
#   - PETALS block rebalancing
#   - online batching
#   - request deadline awareness
#   - NODEGRAPH DDQN
#
# With no failures/topology changes, PETALS' dynamic
# shortest-path routing reduces naturally to a static
# shortest-path problem.
# ============================================================


NOMINAL_BATCH = 1
NOMINAL_CONTEXT = 128
NOMINAL_BACKGROUND_LOAD = 0.35


def _profile_node(
    node: ComputeNode,
) -> ComputeNode:
    """
    PETALS servers advertise measured compute
    throughput. For a conservative offline
    adaptation, retain hardware heterogeneity
    but do not exploit instantaneous queue/load.
    """

    return ComputeNode(
        node_id=node.node_id,
        node_type=node.node_type,
        total_memory_gb=node.total_memory_gb,
        free_memory_gb=node.free_memory_gb,
        compute_scale=node.compute_scale,
        queue_delay_ms=0.0,
        background_load=(
            NOMINAL_BACKGROUND_LOAD
        ),
        kv_cache_gb=node.kv_cache_gb,
        deployed_blocks=node.deployed_blocks,
    )


def _compute_cost_ms(
    profile: ProfileTable,
    node: ComputeNode,
    group_size: int,
) -> float:
    """
    PETALS routing targets autoregressive
    inference. Thus use a fixed one-token
    decode profile rather than the current
    request's exact input/output lengths.
    """

    profiled = _profile_node(
        node
    )

    return float(
        profile.decode_per_token_ms(
            profiled,
            NOMINAL_BATCH,
            NOMINAL_CONTEXT,
            group_size,
        )
    )


def _network_cost_ms(
    infra: Infrastructure,
    node_id: str,
) -> float:
    """
    PETALS measures client-server latency by
    ping. MSN stores link latency as one-way
    latency, so 2x approximates RTT.

    Importantly, this uses latency only rather
    than current-request payload size, matching
    PETALS' autoregressive routing motivation:
    one token activation is only a few KB.
    """

    one_way = (
        shortest_path_latency_ms(
            infra,
            infra.anchor_node,
            node_id,
        )
    )

    if not math.isfinite(
        one_way
    ):
        return float("inf")

    return float(
        2.0 * one_way
    )


def _memory_feasible(
    cfg: dict,
    infra: Infrastructure,
    profile: ProfileTable,
    node_id: str,
    group_size: int,
) -> bool:
    """
    Use workload bounds rather than the current
    request so that the chosen chain remains
    valid for the whole episode.
    """

    node = infra.nodes[
        node_id
    ]

    max_input = int(
        cfg["requests"][
            "input_token_max"
        ]
    )

    max_output = int(
        cfg["requests"][
            "output_token_max"
        ]
    )

    required = (
        profile.memory_gb(
            1,
            max_input,
            max_output,
            group_size,
        )
    )

    return bool(
        required
        <= node.free_memory_gb
    )


def build_petals_chain(
    cfg: dict,
    infra: Infrastructure,
    profile: ProfileTable,
) -> list[PetalsSpan]:
    """
    Find a PETALS-inspired shortest server chain.

    State:
        current transformer block

    Transition:
        choose a server that already hosts the
        next contiguous group of blocks.

    Cost:
        server inference time
        +
        client-server RTT

    This operates on the SAME fixed block
    deployment as NODEGRAPH and all baselines.
    """

    num_blocks = int(
        profile.model.num_blocks
    )

    group_sizes = sorted(
        {
            int(g)
            for g in cfg[
                "scheduler"
            ]["group_sizes"]
            if int(g) > 0
        },
        reverse=True,
    )

    node_ids = list(
        infra.nodes.keys()
    )

    @lru_cache(maxsize=None)
    def solve(
        start_block: int,
    ):
        if (
            start_block
            >= num_blocks
        ):
            return (
                0.0,
                (),
            )

        remaining = (
            num_blocks
            - start_block
        )

        best_cost = float(
            "inf"
        )

        best_chain = None

        for g in group_sizes:

            if g > remaining:
                continue

            for node_id in node_ids:

                node = infra.nodes[
                    node_id
                ]

                # Shared replica constraint.
                if not node.supports(
                    start_block,
                    g,
                ):
                    continue

                if not _memory_feasible(
                    cfg,
                    infra,
                    profile,
                    node_id,
                    g,
                ):
                    continue

                compute = (
                    _compute_cost_ms(
                        profile,
                        node,
                        g,
                    )
                )

                network = (
                    _network_cost_ms(
                        infra,
                        node_id,
                    )
                )

                if not (
                    math.isfinite(compute)
                    and
                    math.isfinite(network)
                ):
                    continue

                tail_cost, tail = solve(
                    start_block + g
                )

                if not math.isfinite(
                    tail_cost
                ):
                    continue

                total = (
                    compute
                    + network
                    + tail_cost
                )

                candidate = (
                    PetalsSpan(
                        start_block=
                            start_block,
                        end_block=
                            start_block + g,
                        node_id=
                            node_id,
                    ),
                ) + tail

                better = (
                    total
                    < best_cost
                    - 1e-9
                )

                tie = (
                    abs(
                        total
                        - best_cost
                    )
                    <= 1e-9
                )

                # Stable deterministic tie
                # breaking:
                # 1) fewer RPC stages
                # 2) larger current segment
                # 3) lexicographic node id
                if (
                    better
                    or (
                        tie
                        and (
                            best_chain is None
                            or len(candidate)
                            < len(best_chain)
                            or (
                                len(candidate)
                                == len(best_chain)
                                and (
                                    candidate[
                                        0
                                    ].group_size
                                    >
                                    best_chain[
                                        0
                                    ].group_size
                                )
                            )
                        )
                    )
                ):
                    best_cost = float(
                        total
                    )

                    best_chain = (
                        candidate
                    )

        if best_chain is None:
            return (
                float("inf"),
                (),
            )

        return (
            best_cost,
            best_chain,
        )

    total_cost, chain = solve(
        0
    )

    if (
        not chain
        or not math.isfinite(
            total_cost
        )
    ):
        raise RuntimeError(
            "PETALS-Adapted could not "
            "find a complete server chain "
            "on the shared block layout"
        )

    covered = sum(
        span.group_size
        for span in chain
    )

    if covered != num_blocks:
        raise RuntimeError(
            f"PETALS chain covers "
            f"{covered}/{num_blocks} "
            f"blocks"
        )

    return list(chain)


def get_petals_chain(
    cfg: dict,
    infra: Infrastructure,
    profile: ProfileTable,
) -> list[PetalsSpan]:
    """
    PETALS clients reuse routing information
    across inference calls. We therefore cache
    one chain per infrastructure realization.
    """

    cached = getattr(
        infra,
        "_petals_adapted_chain",
        None,
    )

    if cached is not None:
        return cached

    chain = build_petals_chain(
        cfg,
        infra,
        profile,
    )

    infra._petals_adapted_chain = (
        chain
    )

    return chain


def choose_petals_action(
    env,
) -> int:
    """
    Follow the fixed PETALS server chain.

    No request-aware optimization is performed.
    """

    chain = get_petals_chain(
        env.cfg,
        env.infra,
        env.profile,
    )

    block = int(
        env.current_block
    )

    target = None

    for span in chain:

        if (
            span.start_block
            <= block
            < span.end_block
        ):
            target = span
            break

    if target is None:
        raise RuntimeError(
            f"No PETALS stage covers "
            f"block {block}"
        )

    remaining = (
        target.end_block
        - block
    )

    feasible = []

    for i, action in enumerate(
        env.actions
    ):

        if (
            action.node_id
            != target.node_id
        ):
            continue

        g = int(
            action.group_size
        )

        if g <= remaining:
            feasible.append(
                (
                    i,
                    g,
                )
            )

    if not feasible:

        available = [
            (
                a.node_id,
                int(a.group_size),
            )
            for a
            in env.actions
        ]

        raise RuntimeError(
            "PETALS planned action "
            "unavailable: "
            f"block={block}, "
            f"target={target.node_id}, "
            f"remaining={remaining}, "
            f"available={available}"
        )

    # Follow the planned server; use the
    # largest supported execution group inside
    # its planned contiguous segment.
    idx, _ = max(
        feasible,
        key=lambda item:
            item[1],
    )

    return int(idx)
