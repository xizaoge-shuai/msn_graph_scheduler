from __future__ import annotations

import math

from .datatypes import ComputeNode
from .profiles import transfer_time_ms


# ============================================================
# EdgeShard-Adapted
#
# Idea-level adaptation only:
#
#   heterogeneous-compute awareness
# + communication-aware device selection
# + contiguous block-group scheduling
#
# NOT a reproduction of the original EdgeShard DP.
#
# It deliberately excludes:
#   - request-specific input/output knowledge
#   - queue/deadline information
#   - future-search/oracle DP
#   - NODEGRAPH batching
#   - DDQN
#   - deployment optimization
# ============================================================


NOMINAL_BATCH = 1
NOMINAL_INPUT = 128
NOMINAL_OUTPUT = 128
NOMINAL_CONTEXT = 192
NOMINAL_BACKGROUND_LOAD = 0.35


def _profile_node(
    node: ComputeNode,
) -> ComputeNode:
    """
    Offline nominal device profile.

    Keep hardware heterogeneity but hide
    instantaneous queue and load states.
    """

    return ComputeNode(
        node_id=node.node_id,
        node_type=node.node_type,
        total_memory_gb=node.total_memory_gb,
        free_memory_gb=node.free_memory_gb,
        compute_scale=node.compute_scale,
        queue_delay_ms=0.0,
        background_load=NOMINAL_BACKGROUND_LOAD,
        kv_cache_gb=node.kv_cache_gb,
        deployed_blocks=node.deployed_blocks,
    )


def _score(
    env,
    action,
) -> float:
    """
    Fixed-profile EdgeShard-inspired score.

    No information from the current request
    is used in the ranking.
    """

    if env.infra is None:
        return float("inf")

    node = env.infra.nodes[
        action.node_id
    ]

    profiled = _profile_node(
        node
    )

    g = int(
        action.group_size
    )

    if g <= 0:
        return float("inf")

    # --------------------------------------------------
    # Offline nominal compute profile
    # --------------------------------------------------

    prefill_ms = (
        env.profile.prefill_ms(
            profiled,
            NOMINAL_BATCH,
            NOMINAL_INPUT,
            g,
        )
    )

    decode_per_token_ms = (
        env.profile.decode_per_token_ms(
            profiled,
            NOMINAL_BATCH,
            NOMINAL_CONTEXT,
            g,
        )
    )

    decode_ms = (
        float(decode_per_token_ms)
        * NOMINAL_OUTPUT
    )

    compute_ms = (
        float(prefill_ms)
        + decode_ms
    )

    # --------------------------------------------------
    # Communication-aware device-selection proxy
    #
    # Always measure from the request source/anchor.
    #
    # This avoids the artificial "once cloud,
    # always cloud" zero-transfer attraction that
    # appears when adapting EdgeShard to MSN's
    # fixed-replica routing problem.
    # --------------------------------------------------

    prefill_payload_mb = (
        env.profile.intermediate_mb(
            NOMINAL_BATCH,
            NOMINAL_INPUT,
            decode=False,
        )
    )

    decode_payload_mb = (
        env.profile.intermediate_mb(
            NOMINAL_BATCH,
            1,
            decode=True,
        )
    )

    anchor = (
        env.infra.anchor_node
    )

    prefill_network_ms = (
        transfer_time_ms(
            env.infra,
            anchor,
            action.node_id,
            prefill_payload_mb,
        )
    )

    decode_network_ms = (
        transfer_time_ms(
            env.infra,
            anchor,
            action.node_id,
            decode_payload_mb,
        )
        * NOMINAL_OUTPUT
    )

    network_ms = (
        float(prefill_network_ms)
        + float(decode_network_ms)
    )

    total = (
        compute_ms
        + network_ms
    )

    if not math.isfinite(total):
        return float("inf")

    # Compare useful progress rather than
    # trivially favouring g=1.
    return float(
        total / g
    )


def choose_edgeshard_adapted(
    env,
) -> int:
    """
    Rank currently feasible contiguous
    block-group placements using the
    EdgeShard-inspired offline score.

    Feasibility still comes from the common
    MSN candidate generator, so all methods
    share the same model-block deployment.
    """

    if not env.actions:
        raise RuntimeError(
            "EdgeShard-Adapted received "
            "no feasible actions"
        )

    scored = []

    for idx, action in enumerate(
        env.actions
    ):

        score = _score(
            env,
            action,
        )

        scored.append(
            (
                score,
                -int(action.group_size),
                str(action.node_id),
                int(idx),
            )
        )

    scored.sort()

    return int(
        scored[0][-1]
    )
