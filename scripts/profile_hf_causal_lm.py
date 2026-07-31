from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)


CORE_FIELDS = [
    "node_type",
    "phase",
    "batch_size",
    "token_length",
    "group_size",
    "latency_ms",
    "peak_memory_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--node-type",
        default="edge_fast",
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[
            1,
            2,
            4,
            8,
        ],
    )

    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[
            128,
            256,
            512,
            1024,
            2048,
        ],
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "auto",
            "float16",
            "bfloat16",
        ],
        default="auto",
    )

    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--metadata-output",
        required=True,
    )

    return parser.parse_args()


def select_dtype(
    name: str,
) -> torch.dtype:
    if name == "float16":
        return torch.float16

    if name == "bfloat16":
        return torch.bfloat16

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def sync() -> None:
    torch.cuda.synchronize()


def peak_gb() -> float:
    return (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def median_ms(
    values: list[float],
) -> float:
    return float(
        statistics.median(values)
    )


def timed_prefill(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            del output
            sync()

        latencies = []
        peaks = []

        for _ in range(repeats):
            clear_cuda()
            sync()

            start = time.perf_counter()

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            sync()

            latencies.append(
                (
                    time.perf_counter()
                    - start
                )
                * 1000.0
            )

            peaks.append(
                peak_gb()
            )

            del output

        return (
            median_ms(latencies),
            max(peaks),
        )


def timed_decode(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    batch_size = int(
        input_ids.shape[0]
    )

    device = input_ids.device

    with torch.inference_mode():
        for _ in range(warmup):
            prefix = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            next_ids = torch.randint(
                low=3,
                high=max(
                    vocab_size - 1,
                    4,
                ),
                size=(
                    batch_size,
                    1,
                ),
                device=device,
            )

            next_mask = torch.ones(
                (
                    batch_size,
                    input_ids.shape[1] + 1,
                ),
                dtype=attention_mask.dtype,
                device=device,
            )

            decoded = model(
                input_ids=next_ids,
                attention_mask=next_mask,
                past_key_values=(
                    prefix.past_key_values
                ),
                use_cache=True,
            )

            del decoded
            del prefix
            sync()

        latencies = []
        peaks = []

        for _ in range(repeats):
            prefix = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            sync()
            torch.cuda.reset_peak_memory_stats()

            next_ids = torch.randint(
                low=3,
                high=max(
                    vocab_size - 1,
                    4,
                ),
                size=(
                    batch_size,
                    1,
                ),
                device=device,
            )

            next_mask = torch.ones(
                (
                    batch_size,
                    input_ids.shape[1] + 1,
                ),
                dtype=attention_mask.dtype,
                device=device,
            )

            start = time.perf_counter()

            decoded = model(
                input_ids=next_ids,
                attention_mask=next_mask,
                past_key_values=(
                    prefix.past_key_values
                ),
                use_cache=True,
            )

            sync()

            latencies.append(
                (
                    time.perf_counter()
                    - start
                )
                * 1000.0
            )

            peaks.append(
                peak_gb()
            )

            del decoded
            del prefix

        return (
            median_ms(latencies),
            max(peaks),
        )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    dtype = select_dtype(
        args.dtype
    )

    config = AutoConfig.from_pretrained(
        args.model,
    )

    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    try:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                args.model,
                attn_implementation=(
                    args.attn_implementation
                ),
                **load_kwargs,
            )
        )
    except TypeError:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                args.model,
                **load_kwargs,
            )
        )

    model = model.eval().to(
        "cuda"
    )

    num_layers = int(
        getattr(
            config,
            "num_hidden_layers",
        )
    )

    vocab_size = int(
        getattr(
            config,
            "vocab_size",
        )
    )

    rows = []

    for batch_size in args.batch_sizes:
        for token_length in args.seq_lengths:
            print(
                "Profiling:",
                f"B={batch_size}",
                f"S={token_length}",
                flush=True,
            )

            try:
                input_ids = torch.randint(
                    low=3,
                    high=max(
                        vocab_size - 1,
                        4,
                    ),
                    size=(
                        batch_size,
                        token_length,
                    ),
                    device="cuda",
                )

                attention_mask = torch.ones(
                    (
                        batch_size,
                        token_length,
                    ),
                    dtype=torch.long,
                    device="cuda",
                )

                prefill_ms, prefill_peak = (
                    timed_prefill(
                        model,
                        input_ids,
                        attention_mask,
                        args.warmup,
                        args.repeats,
                    )
                )

                decode_ms, decode_peak = (
                    timed_decode(
                        model,
                        input_ids,
                        attention_mask,
                        vocab_size,
                        args.warmup,
                        args.repeats,
                    )
                )

                rows.extend(
                    [
                        {
                            "node_type":
                                args.node_type,
                            "phase":
                                "prefill",
                            "batch_size":
                                batch_size,
                            "token_length":
                                token_length,
                            "group_size":
                                num_layers,
                            "latency_ms":
                                prefill_ms,
                            "peak_memory_gb":
                                prefill_peak,
                        },
                        {
                            "node_type":
                                args.node_type,
                            "phase":
                                "decode",
                            "batch_size":
                                batch_size,
                            "token_length":
                                token_length,
                            "group_size":
                                num_layers,
                            "latency_ms":
                                decode_ms,
                            "peak_memory_gb":
                                decode_peak,
                        },
                    ]
                )

                print(
                    f"  prefill={prefill_ms:.3f} ms, "
                    f"decode={decode_ms:.3f} ms, "
                    f"peak={max(prefill_peak, decode_peak):.3f} GB",
                    flush=True,
                )

                del input_ids
                del attention_mask
                clear_cuda()

            except torch.cuda.OutOfMemoryError:
                print(
                    "  OOM: skipped",
                    flush=True,
                )

                clear_cuda()

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CORE_FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "model": args.model,
        "node_type": args.node_type,
        "gpu": (
            torch.cuda.get_device_name(0)
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(dtype),
        "num_hidden_layers": num_layers,
        "hidden_size": int(
            getattr(
                config,
                "hidden_size",
            )
        ),
        "vocab_size": vocab_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "attention_implementation": (
            args.attn_implementation
        ),
        "profile_points": len(rows),
    }

    metadata_output = Path(
        args.metadata_output
    )

    metadata_output.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved {output}"
    )

    print(
        f"Saved {metadata_output}"
    )


if __name__ == "__main__":
    main()
