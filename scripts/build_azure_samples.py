from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def uniform_sample(
    paths: list[Path],
    sample_size: int,
    max_input: int,
    max_output: int,
    seed: int,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Uniformly sample rows using random priorities.

    Only a bounded reservoir and one CSV chunk are retained in memory.
    """
    rng = np.random.default_rng(seed)
    reservoir: pd.DataFrame | None = None
    valid_rows = 0

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

        print(f"Reading {path}", flush=True)

        for chunk in pd.read_csv(
            path,
            usecols=[
                "timestamp",
                "input_tokens",
                "output_tokens",
                "request_type",
                "source",
            ],
            chunksize=chunksize,
        ):
            chunk = chunk[
                (chunk["input_tokens"] > 0)
                & (chunk["input_tokens"] <= max_input)
                & (chunk["output_tokens"] > 0)
                & (chunk["output_tokens"] <= max_output)
            ].copy()

            if chunk.empty:
                continue

            valid_rows += len(chunk)
            chunk["_priority"] = rng.random(len(chunk))

            if reservoir is None:
                reservoir = chunk
            else:
                reservoir = pd.concat(
                    [reservoir, chunk],
                    ignore_index=True,
                )

            if len(reservoir) > sample_size * 2:
                reservoir = reservoir.nsmallest(
                    sample_size,
                    "_priority",
                ).reset_index(drop=True)

    if reservoir is None or reservoir.empty:
        raise RuntimeError("No valid Azure request was found")

    result = (
        reservoir.nsmallest(sample_size, "_priority")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    print(
        f"Valid rows before sampling: {valid_rows:,}",
        flush=True,
    )
    print(
        f"Rows retained: {len(result):,}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--max-input",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500_000,
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = args.processed_root

    train = uniform_sample(
        paths=[
            root / "azure_2023_code.csv",
            root / "azure_2023_conversation.csv",
        ],
        sample_size=args.train_size,
        max_input=args.max_input,
        max_output=args.max_output,
        seed=args.seed,
    )

    # 2023 本身不足十万条，实际会保留所有满足范围的请求。
    train_path = root / "azure_2023_train_sample.csv"
    train.to_csv(train_path, index=False)
    print(f"Saved {train_path}", flush=True)

    test = uniform_sample(
        paths=[
            root / "azure_2024_code.csv",
            root / "azure_2024_conversation.csv",
        ],
        sample_size=args.test_size,
        max_input=args.max_input,
        max_output=args.max_output,
        seed=args.seed + 1,
    )

    test_path = root / "azure_2024_test_sample.csv"
    test.to_csv(test_path, index=False)
    print(f"Saved {test_path}", flush=True)

    summary = (
        pd.concat(
            [
                train.assign(split="train"),
                test.assign(split="test"),
            ],
            ignore_index=True,
        )
        .groupby(["split", "request_type"])
        .agg(
            requests=("input_tokens", "size"),
            input_mean=("input_tokens", "mean"),
            input_p50=("input_tokens", "median"),
            input_p95=(
                "input_tokens",
                lambda x: x.quantile(0.95),
            ),
            output_mean=("output_tokens", "mean"),
            output_p50=("output_tokens", "median"),
            output_p95=(
                "output_tokens",
                lambda x: x.quantile(0.95),
            ),
        )
        .reset_index()
    )

    summary_path = root / "azure_workload_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
