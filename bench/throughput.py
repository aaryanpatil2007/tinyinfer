"""Tokens/second harness. Every optimization phase ends with a number from this file.

    python -m bench.throughput                  # defaults: 128 new tokens
    python -m bench.throughput --new-tokens 256

Keep a log of results in bench/RESULTS.md — the whole project's story is that
table: naive -> KV cache -> quantized -> fused kernel, same machine, same prompt.
"""

import argparse
import time

import torch

from engine.loader import load_model

PROMPT = "In a shocking finding, scientists discovered a herd of unicorns living in"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--cache", action="store_true", help="use the KV cache (phase 3)")
    parser.add_argument("--model", default="gpt2", choices=("gpt2", "qwen2"))
    parser.add_argument("--bits", type=int, choices=(8, 4), help="weight-only quantization (phase 5)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    if args.bits:
        from engine.quantize import quantize_model
        model = quantize_model(model, args.bits)
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(args.device)

    for _ in range(args.warmup):
        model.generate(ids, max_new_tokens=8, use_cache=args.cache)

    rates = []
    for run in range(args.runs):
        start = time.perf_counter()
        out = model.generate(ids, max_new_tokens=args.new_tokens, use_cache=args.cache)
        if args.device in ("cuda", "mps"):
            torch.cuda.synchronize() if args.device == "cuda" else torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        rate = args.new_tokens / elapsed
        rates.append(rate)
        print(f"run {run + 1}: {rate:.2f} tok/s  ({elapsed:.2f}s for {args.new_tokens} tokens)")

    print(f"\nbest of {args.runs}: {max(rates):.2f} tok/s on {args.device} "
          f"({tokenizer.decode(out[0, -12:])!r}...)")


if __name__ == "__main__":
    main()
