from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_COLUMNS = {
    "TIMESTAMP",
    "ContextTokens",
    "GeneratedTokens",
}


def load_azure_trace(
    path: Path,
    source: str,
    request_type: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Azure trace does not exist: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Azure trace is empty: {path}"
        )

    print(f"Reading {path}", flush=True)

    raw = pd.read_csv(path)

    missing = sorted(
        REQUIRED_COLUMNS.difference(raw.columns)
    )
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                raw["TIMESTAMP"],
                errors="coerce",
                utc=True,
            ),
            "input_tokens": pd.to_numeric(
                raw["ContextTokens"],
                errors="coerce",
            ),
            "output_tokens": pd.to_numeric(
                raw["GeneratedTokens"],
                errors="coerce",
            ),
            "request_type": request_type,
            "source": source,
        }
    )

    before = len(normalized)

    normalized = normalized.dropna(
        subset=[
            "timestamp",
            "input_tokens",
            "output_tokens",
        ]
    )

    normalized = normalized[
        (normalized["input_tokens"] > 0)
        & (normalized["output_tokens"] > 0)
    ].copy()

    normalized["input_tokens"] = (
        normalized["input_tokens"].astype("int64")
    )
    normalized["output_tokens"] = (
        normalized["output_tokens"].astype("int64")
    )

    normalized = normalized.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    removed = before - len(normalized)

    print(
        f"  source={source} "
        f"type={request_type} "
        f"valid={len(normalized):,} "
        f"removed={removed:,}",
        flush=True,
    )

    return normalized


def save_summary(
    combined: pd.DataFrame,
    output_path: Path,
) -> None:
    summary = (
        combined.groupby(
            ["source", "request_type"],
            dropna=False,
        )
        .agg(
            requests=("timestamp", "size"),
            input_mean=("input_tokens", "mean"),
            input_p50=(
                "input_tokens",
                lambda x: x.quantile(0.50),
            ),
            input_p95=(
                "input_tokens",
                lambda x: x.quantile(0.95),
            ),
            input_p99=(
                "input_tokens",
                lambda x: x.quantile(0.99),
            ),
            output_mean=("output_tokens", "mean"),
            output_p50=(
                "output_tokens",
                lambda x: x.quantile(0.50),
            ),
            output_p95=(
                "output_tokens",
                lambda x: x.quantile(0.95),
            ),
            output_p99=(
                "output_tokens",
                lambda x: x.quantile(0.99),
            ),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
        )
        .reset_index()
    )

    summary.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Azure LLM inference traces "
            "to a common request schema."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/processed",
    )
    args = parser.parse_args()

    specs = [
        (
            args.raw_root
            / "azure_2023"
            / "code.csv",
            "azure_2023",
            "code",
        ),
        (
            args.raw_root
            / "azure_2023"
            / "conversation.csv",
            "azure_2023",
            "conversation",
        ),
        (
            args.raw_root
            / "azure_2024"
            / "code.csv",
            "azure_2024",
            "code",
        ),
        (
            args.raw_root
            / "azure_2024"
            / "conversation.csv",
            "azure_2024",
            "conversation",
        ),
    ]

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames: list[pd.DataFrame] = []

    for path, source, request_type in specs:
        frame = load_azure_trace(
            path=path,
            source=source,
            request_type=request_type,
        )

        output_file = (
            args.output_root
            / f"{source}_{request_type}.csv"
        )

        frame.to_csv(
            output_file,
            index=False,
        )

        print(
            f"  saved: {output_file}",
            flush=True,
        )

        frames.append(frame)

    if not frames:
        raise RuntimeError(
            "No Azure trace was processed."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    combined_path = (
        args.output_root
        / "azure_all_requests.csv"
    )
    combined.to_csv(
        combined_path,
        index=False,
    )

    summary_path = (
        args.output_root
        / "azure_summary.csv"
    )
    save_summary(
        combined,
        summary_path,
    )

    print("", flush=True)
    print(
        f"Combined requests: {len(combined):,}",
        flush=True,
    )
    print(
        f"Combined trace:    {combined_path}",
        flush=True,
    )
    print(
        f"Summary:           {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
