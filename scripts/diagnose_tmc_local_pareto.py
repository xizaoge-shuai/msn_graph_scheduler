from __future__ import annotations

import atexit
import copy
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.evaluate_dynamic_mobility as evaluator


BASE_WEIGHT = 0.003

OUTPUT = Path(
    os.environ["TMC_PARETO_OUTPUT"]
)

ORIGINAL_RUN_ONE_MAPPING = (
    evaluator.run_one_mapping
)

ORIGINAL_RUN_DYNAMIC_EPISODE = (
    evaluator.run_dynamic_episode
)

STATE = {
    "episode_seed": None,
    "mapping_id": -1,
}

ROWS: list[dict] = []


def candidate_ids(candidate) -> tuple[str, ...]:
    return tuple(
        sorted(
            request.request_id
            for request in candidate.requests
        )
    )


def traced_run_one_mapping(*args, **kwargs):
    env = kwargs["env"]
    infra = kwargs["infra"]
    full_queue = list(
        kwargs["queue"]
    )

    window_size = int(
        env.batcher.window_size
    )

    # Freeze the original candidate universe.
    # Excluding a baseline request must not allow a
    # request outside the original rolling window
    # to enter the alternative search.
    search_queue = full_queue[
        :window_size
    ]

    now_ms = float(kwargs["now_ms"])

    result = ORIGINAL_RUN_ONE_MAPPING(
        *args,
        **kwargs,
    )

    STATE["mapping_id"] += 1

    if env.batch is None:
        return result

    batcher = copy.copy(env.batcher)
    batcher._cache = {}

    node_id = str(env.batch.node_id)
    group_size = int(env.batch.group_size)
    batch_size = len(env.batch.requests)

    selected_ids = candidate_ids(
        env.batch
    )

    baseline = batcher._best_batch_once(
        queue=search_queue,
        now_ms=now_ms,
        infra=infra,
        node_id=node_id,
        group_size=group_size,
        mobility_weight=BASE_WEIGHT,
        fixed_batch_size=batch_size,
    )

    if baseline is None:
        return result

    baseline_ids = candidate_ids(
        baseline
    )

    baseline_risk = (
        batcher
        ._candidate_prefill_mobility_risk(
            baseline,
            infra,
            node_id,
        )
    )

    candidates = {
        baseline_ids: baseline
    }

    exclusion_ids = list(
        baseline_ids
    )

    exclusions = [
        frozenset([request_id])
        for request_id in exclusion_ids
    ]

    exclusions.extend(
        frozenset(values)
        for values in itertools.combinations(
            exclusion_ids,
            2,
        )
    )

    for excluded in exclusions:
        filtered_queue = [
            request
            for request in search_queue
            if request.request_id
            not in excluded
        ]

        if len(filtered_queue) < batch_size:
            continue

        alternative = (
            batcher._best_batch_once(
                queue=filtered_queue,
                now_ms=now_ms,
                infra=infra,
                node_id=node_id,
                group_size=group_size,
                mobility_weight=BASE_WEIGHT,
                fixed_batch_size=batch_size,
            )
        )

        if alternative is None:
            continue

        ids = candidate_ids(
            alternative
        )

        incumbent = candidates.get(ids)

        if (
            incumbent is None
            or alternative.utility
            > incumbent.utility
        ):
            candidates[ids] = alternative

    risk_reducing = []

    for ids, candidate in candidates.items():
        if ids == baseline_ids:
            continue

        candidate_risk = (
            batcher
            ._candidate_prefill_mobility_risk(
                candidate,
                infra,
                node_id,
            )
        )

        risk_reduction = (
            baseline_risk
            - candidate_risk
        )

        if risk_reduction <= 1e-12:
            continue

        absolute_regret = (
            baseline.utility
            - candidate.utility
        )

        if absolute_regret < -1e-8:
            raise RuntimeError(
                "Alternative utility exceeds baseline "
                "utility under the same frozen window: "
                f"episode_seed={STATE['episode_seed']}, "
                f"mapping_id={STATE['mapping_id']}, "
                f"baseline_utility={baseline.utility}, "
                f"candidate_utility={candidate.utility}, "
                f"regret={absolute_regret}"
            )

        # Remove only floating-point noise.
        absolute_regret = max(
            0.0,
            float(absolute_regret),
        )

        utility_scale = max(
            abs(float(baseline.utility)),
            abs(float(candidate.utility)),
            1e-6,
        )

        relative_regret = (
            absolute_regret
            / utility_scale
        )

        risk_reducing.append(
            {
                "candidate_ids": "|".join(ids),
                "risk_reduction": float(
                    risk_reduction
                ),
                "absolute_regret": float(
                    absolute_regret
                ),
                "relative_regret": float(
                    relative_regret
                ),
            }
        )

    row = {
        "episode_seed": int(
            STATE["episode_seed"]
        ),
        "mapping_id": int(
            STATE["mapping_id"]
        ),
        "node_id": node_id,
        "group_size": group_size,
        "batch_size": batch_size,
        "selected_matches_baseline": int(
            selected_ids == baseline_ids
        ),
        "candidate_count": len(
            candidates
        ),
        "risk_reducing_count": len(
            risk_reducing
        ),
        "best_relative_regret": np.nan,
        "best_absolute_regret": np.nan,
        "best_risk_reduction": np.nan,
        "best_candidate_ids": "",
        "baseline_ids": "|".join(
            baseline_ids
        ),
    }

    if risk_reducing:
        best = min(
            risk_reducing,
            key=lambda item: (
                item["relative_regret"],
                -item["risk_reduction"],
            ),
        )

        row.update(
            {
                "best_relative_regret": (
                    best[
                        "relative_regret"
                    ]
                ),
                "best_absolute_regret": (
                    best[
                        "absolute_regret"
                    ]
                ),
                "best_risk_reduction": (
                    best[
                        "risk_reduction"
                    ]
                ),
                "best_candidate_ids": (
                    best[
                        "candidate_ids"
                    ]
                ),
            }
        )

    for threshold in [
        0.01,
        0.02,
        0.05,
        0.10,
    ]:
        tag = str(threshold).replace(
            ".",
            "p",
        )

        accepted = [
            item
            for item in risk_reducing
            if item["relative_regret"]
            <= threshold + 1e-12
        ]

        row[
            f"accepted_rel_{tag}"
        ] = int(bool(accepted))

        row[
            f"max_risk_reduction_rel_{tag}"
        ] = (
            max(
                item["risk_reduction"]
                for item in accepted
            )
            if accepted
            else np.nan
        )

    ROWS.append(row)

    return result


def traced_run_dynamic_episode(
    *args,
    **kwargs,
):
    STATE["episode_seed"] = int(
        kwargs["episode_seed"]
    )

    STATE["mapping_id"] = -1

    try:
        return ORIGINAL_RUN_DYNAMIC_EPISODE(
            *args,
            **kwargs,
        )
    finally:
        STATE["episode_seed"] = None


def save_output() -> None:
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(ROWS).to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"Saved Pareto diagnostics: {OUTPUT}",
        flush=True,
    )


atexit.register(save_output)

evaluator.run_one_mapping = (
    traced_run_one_mapping
)

evaluator.run_dynamic_episode = (
    traced_run_dynamic_episode
)

evaluator.main()
