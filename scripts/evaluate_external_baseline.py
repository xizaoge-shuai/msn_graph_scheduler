from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from msn_scheduler.config import load_config
from msn_scheduler.external_baselines import (
    choose_external_action,
)


def find_option(
    arguments: list[str],
    option: str,
    default: str,
) -> str:
    for index, value in enumerate(
        arguments
    ):
        if value == option:
            if index + 1 >= len(arguments):
                raise ValueError(
                    f"{option} requires a value"
                )

            return arguments[
                index + 1
            ]

        prefix = option + "="

        if value.startswith(prefix):
            return value[
                len(prefix):
            ]

    return default


def main() -> None:
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    parser.add_argument(
        "--external-mapper",
        required=True,
        choices=[
            "rba",
            "dybap_core",
        ],
    )

    known, forwarded = (
        parser.parse_known_args()
    )

    config_path = find_option(
        forwarded,
        "--config",
        str(
            ROOT
            / "configs"
            / "default.yaml"
        ),
    )

    cfg = load_config(
        config_path
    )

    evaluator_path = (
        ROOT
        / "scripts"
        / "evaluate_batchers.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "_msn_evaluate_batchers",
            evaluator_path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Cannot load evaluate_batchers.py"
        )

    evaluator = (
        importlib.util
        .module_from_spec(spec)
    )

    spec.loader.exec_module(
        evaluator
    )

    original_mapper = (
        evaluator.choose_min_lower_bound
    )

    def adapted_mapper(obs):
        try:
            return choose_external_action(
                obs,
                known.external_mapper,
                cfg,
            )
        except Exception as error:
            print(
                "External mapper fallback to "
                "lower-bound greedy: "
                f"{type(error).__name__}: "
                f"{error}",
                file=sys.stderr,
                flush=True,
            )

            return original_mapper(
                obs
            )

    evaluator.choose_min_lower_bound = (
        adapted_mapper
    )

    print(
        "Using adapted external mapper: "
        f"{known.external_mapper}",
        flush=True,
    )

    sys.argv = [
        str(evaluator_path),
        *forwarded,
    ]

    evaluator.main()


if __name__ == "__main__":
    main()
