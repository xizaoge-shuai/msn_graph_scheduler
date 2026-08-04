from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from inspect import signature
from pathlib import Path
import sys

import numpy as np

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from msn_scheduler.batching import (
    NodeConditionedDPBatcher,
)
from msn_scheduler.config import load_config
from msn_scheduler.env import SchedulingEnv
from msn_scheduler.profiles import (
    ModelProfile,
    ProfileTable,
)
from msn_scheduler.synthetic import (
    make_request_queue,
    make_synthetic_infrastructure,
)


def build_profile(
    cfg: dict,
) -> ProfileTable:
    model_cfg = dict(
        cfg.get(
            "model",
            {},
        )
    )

    accepted = {
        name
        for name in signature(
            ModelProfile
        ).parameters
        if name != "self"
    }

    kwargs = {
        name: value
        for name, value in model_cfg.items()
        if name in accepted
    }

    missing = [
        name
        for name, parameter in signature(
            ModelProfile
        ).parameters.items()
        if (
            name != "self"
            and parameter.default
            is parameter.empty
            and name not in kwargs
        )
    ]

    print(
        "ModelProfile accepted parameters:",
        sorted(accepted),
    )

    print(
        "ModelProfile supplied parameters:",
        sorted(kwargs),
    )

    print(
        "Ignored model config fields:",
        sorted(
            set(model_cfg)
            - accepted
        ),
    )

    if missing:
        raise RuntimeError(
            "Missing required ModelProfile "
            f"parameters: {missing}"
        )

    model = ModelProfile(
        **kwargs
    )

    return ProfileTable(model)


def observation_items(
    observation,
):
    if is_dataclass(observation):
        for field in fields(observation):
            yield (
                field.name,
                getattr(
                    observation,
                    field.name,
                ),
            )
        return

    if hasattr(
        observation,
        "__dict__",
    ):
        yield from vars(
            observation
        ).items()
        return

    raise RuntimeError(
        "Cannot inspect observation object: "
        f"{type(observation)}"
    )


cfg = load_config(
    ROOT
    / "configs/train_online_cpu.yaml"
)

rng = np.random.default_rng(
    9201
)

infra = make_synthetic_infrastructure(
    cfg,
    rng,
)

queue = make_request_queue(
    cfg,
    rng,
    now_ms=0.0,
)

profile = build_profile(cfg)

full_cfg = deepcopy(cfg)

masked_cfg = deepcopy(cfg)

masked_cfg.setdefault(
    "ablation",
    {},
)

masked_cfg["ablation"][
    "mask_mobility_features"
] = True

masked_cfg["batching"][
    "dp_alpha_mobility"
] = 0.0

masked_cfg["batching"][
    "baseline_alpha_mobility"
] = 0.0

full_batcher = (
    NodeConditionedDPBatcher(
        full_cfg,
        profile,
    )
)

masked_batcher = (
    NodeConditionedDPBatcher(
        masked_cfg,
        profile,
    )
)

full_env = SchedulingEnv(
    full_cfg,
    profile,
    full_batcher,
)

masked_env = SchedulingEnv(
    masked_cfg,
    profile,
    masked_batcher,
)

full_observation = full_env.reset(
    infra,
    queue,
    0.0,
)

masked_observation = (
    masked_env.reset(
        infra,
        queue,
        0.0,
    )
)

full_items = dict(
    observation_items(
        full_observation
    )
)

masked_items = dict(
    observation_items(
        masked_observation
    )
)

print(
    "\nObservation type:",
    type(full_observation),
)

print(
    "Observation fields:",
    sorted(full_items),
)

print(
    "\n===== Numerical differences ====="
)

changed_fields = []

for name in sorted(
    set(full_items)
    & set(masked_items)
):
    try:
        full_value = np.asarray(
            full_items[name]
        )

        masked_value = np.asarray(
            masked_items[name]
        )

        if (
            full_value.shape
            != masked_value.shape
        ):
            print(
                f"{name}: shape mismatch "
                f"{full_value.shape} vs "
                f"{masked_value.shape}"
            )
            continue

        if not np.issubdtype(
            full_value.dtype,
            np.number,
        ):
            continue

        difference = np.abs(
            full_value.astype(
                np.float64
            )
            - masked_value.astype(
                np.float64
            )
        )

        maximum = (
            float(difference.max())
            if difference.size
            else 0.0
        )

        changed = int(
            np.count_nonzero(
                difference > 1e-12
            )
        )

        print(
            f"{name:<30} "
            f"shape={str(full_value.shape):<15} "
            f"changed={changed:<5} "
            f"max_diff={maximum:.12f}"
        )

        if changed:
            changed_fields.append(
                name
            )

            if name in {
                "batch_features",
                "mobility_features",
            }:
                print(
                    "  Full:"
                )
                print(full_value)

                print(
                    "  Masked:"
                )
                print(masked_value)

    except (
        TypeError,
        ValueError,
    ):
        continue

print(
    "\nChanged numerical fields:",
    changed_fields,
)

if changed_fields:
    print(
        "DECISION: mobility masking changes "
        "the environment observation."
    )
else:
    print(
        "DECISION: no observation difference "
        "was detected."
    )
