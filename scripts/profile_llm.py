from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile full-model prefill/decode latency for later calibration.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="profiles/qwen_full_model.csv")
    parser.add_argument("--node-type", default=None, help="Logical simulator node type, e.g. edge_fast")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--tokens", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=8)
    args = parser.parse_args()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("Install optional dependencies: pip install -e '.[profile]'") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    rows = []
    vocab = int(model.config.vocab_size)
    num_blocks = int(getattr(model.config, "num_hidden_layers", 1))
    for b in args.batches:
        for t in args.tokens:
            input_ids = torch.randint(0, vocab, (b, t), device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                for _ in range(args.warmup):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                synchronize()
                samples = []
                peak = 0.0
                for _ in range(args.repeat):
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                    synchronize()
                    samples.append((time.perf_counter() - start) * 1000.0)
                    if device == "cuda":
                        peak = max(peak, torch.cuda.max_memory_allocated() / 1024**3)
                rows.append(
                    {
                        "node_type": args.node_type or (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
                        "phase": "prefill",
                        "batch_size": b,
                        "tokens": t,
                        "group_size": num_blocks,
                        "latency_ms": sum(samples) / len(samples),
                        "peak_memory_gb": peak,
                    }
                )
                # One-token decode using returned KV cache.
                next_token = torch.randint(0, vocab, (b, 1), device=device)
                past = out.past_key_values
                decode_samples = []
                for _ in range(args.repeat):
                    start = time.perf_counter()
                    _ = model(input_ids=next_token, past_key_values=past, use_cache=True)
                    synchronize()
                    decode_samples.append((time.perf_counter() - start) * 1000.0)
                rows.append(
                    {
                        "node_type": args.node_type or (torch.cuda.get_device_name(0) if device == "cuda" else "cpu"),
                        "phase": "decode",
                        "batch_size": b,
                        "tokens": t,
                        "group_size": num_blocks,
                        "latency_ms": sum(decode_samples) / len(decode_samples),
                        "peak_memory_gb": peak,
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
