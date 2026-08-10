from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)


def sync():
    torch.cuda.synchronize()


def timed_call(fn):
    sync()
    start = time.perf_counter()

    result = fn()

    sync()

    return (
        (time.perf_counter() - start)
        * 1000.0,
        result,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=(
            "Qwen/"
            "Qwen2.5-1.5B-Instruct"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--meta-output",
        required=True,
    )

    parser.add_argument(
        "--node-type",
        default="edge_fast",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "bfloat16",
            "float16",
        ],
        default="bfloat16",
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )

    parser.add_argument(
        "--tokens",
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
        "--repeat",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    cfg = AutoConfig.from_pretrained(
        args.model
    )

    print(
        "Loading model:",
        args.model,
        flush=True,
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        .to(args.device)
        .eval()
    )

    num_layers = int(
        cfg.num_hidden_layers
    )

    hidden_size = int(
        cfg.hidden_size
    )

    num_heads = int(
        cfg.num_attention_heads
    )

    num_kv_heads = int(
        cfg.num_key_value_heads
    )

    head_dim = int(
        getattr(
            cfg,
            "head_dim",
            hidden_size // num_heads,
        )
    )

    bytes_per_element = int(
        torch.tensor(
            [],
            dtype=dtype,
        ).element_size()
    )

    kv_bytes_per_token_per_block = (
        2
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )

    layers = model.model.layers

    layer_parameter_bytes = []

    for layer in layers:
        layer_parameter_bytes.append(
            sum(
                p.numel()
                * p.element_size()
                for p in layer.parameters()
            )
        )

    block_parameter_gb = (
        sum(layer_parameter_bytes)
        / len(layer_parameter_bytes)
        / (1024 ** 3)
    )

    total_parameter_bytes = sum(
        p.numel()
        * p.element_size()
        for p in model.parameters()
    )

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(
            torch.device(args.device)
        ),
        "num_blocks": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": (
            num_kv_heads
        ),
        "head_dim": head_dim,
        "bytes_per_element": (
            bytes_per_element
        ),
        "kv_bytes_per_token_per_block": (
            kv_bytes_per_token_per_block
        ),
        "block_parameter_gb": (
            block_parameter_gb
        ),
        "total_parameter_gb": (
            total_parameter_bytes
            / (1024 ** 3)
        ),
    }

    print()
    print("===== MODEL META =====")

    for key, value in meta.items():
        print(
            f"{key:32s} = {value}"
        )

    meta_path = Path(
        args.meta_output
    )

    meta_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []

    vocab_size = int(
        cfg.vocab_size
    )

    device = torch.device(
        args.device
    )

    for batch_size in args.batch_sizes:

        for tokens in args.tokens:

            print()
            print(
                "PROFILE",
                f"batch={batch_size}",
                f"tokens={tokens}",
                flush=True,
            )

            try:
                input_ids = torch.randint(
                    low=0,
                    high=vocab_size,
                    size=(
                        batch_size,
                        tokens,
                    ),
                    device=device,
                )

                # ----------------------------
                # PREFILL
                # ----------------------------

                with torch.inference_mode():

                    for _ in range(
                        args.warmup
                    ):
                        out = model(
                            input_ids=input_ids,
                            use_cache=True,
                        )
                        del out

                    sync()

                    prefill_samples = []

                    for _ in range(
                        args.repeat
                    ):
                        latency, out = (
                            timed_call(
                                lambda:
                                model(
                                    input_ids=(
                                        input_ids
                                    ),
                                    use_cache=True,
                                )
                            )
                        )

                        prefill_samples.append(
                            latency
                        )

                        del out

                prefill_ms = float(
                    torch.tensor(
                        prefill_samples
                    ).median()
                )

                # ----------------------------
                # DECODE
                # ----------------------------

                with torch.inference_mode():

                    initial = model(
                        input_ids=input_ids,
                        use_cache=True,
                    )

                    past = (
                        initial.past_key_values
                    )

                    del initial

                    next_token = torch.randint(
                        low=0,
                        high=vocab_size,
                        size=(
                            batch_size,
                            1,
                        ),
                        device=device,
                    )

                    for _ in range(
                        args.warmup
                    ):
                        out = model(
                            input_ids=next_token,
                            past_key_values=past,
                            use_cache=True,
                        )

                        past = (
                            out.past_key_values
                        )

                        del out

                    decode_samples = []

                    for _ in range(
                        args.repeat
                    ):
                        latency, out = (
                            timed_call(
                                lambda:
                                model(
                                    input_ids=(
                                        next_token
                                    ),
                                    past_key_values=(
                                        past
                                    ),
                                    use_cache=True,
                                )
                            )
                        )

                        decode_samples.append(
                            latency
                        )

                        past = (
                            out.past_key_values
                        )

                        del out

                decode_ms = float(
                    torch.tensor(
                        decode_samples
                    ).median()
                )

                rows.append(
                    {
                        "node_type":
                            args.node_type,
                        "phase":
                            "prefill",
                        "batch_size":
                            batch_size,
                        "tokens":
                            tokens,
                        "group_size":
                            num_layers,
                        "latency_ms":
                            prefill_ms,
                    }
                )

                rows.append(
                    {
                        "node_type":
                            args.node_type,
                        "phase":
                            "decode",
                        "batch_size":
                            batch_size,
                        "tokens":
                            tokens,
                        "group_size":
                            num_layers,
                        "latency_ms":
                            decode_ms,
                    }
                )

                print(
                    f"prefill={prefill_ms:.3f} ms "
                    f"decode={decode_ms:.3f} ms/token",
                    flush=True,
                )

                del input_ids
                del past
                del next_token

                torch.cuda.empty_cache()

            except torch.OutOfMemoryError:

                print(
                    "OOM: skip",
                    batch_size,
                    tokens,
                    flush=True,
                )

                torch.cuda.empty_cache()

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise RuntimeError(
            "No profiling rows generated"
        )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output,
        index=False,
    )

    print()
    print(
        "===== PROFILE TABLE ====="
    )

    print(
        result.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
