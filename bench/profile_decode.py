"""Where does a decode step actually spend its time? (Phase 6a referee — done for you.)

    python -m bench.profile_decode --model qwen2
    python -m bench.profile_decode --model qwen2 --bits 8

Times each component across cached decode steps and prints time next to weight
bytes. The point: on a memory-bound workload the time column tracks the bytes
column — the profile tells you which kernel to write.
"""

import argparse
import time
from collections import defaultdict

import torch

from engine.loader import load_model

PROMPT = "The history of computing is usually told as a story about machines, but"


def classify(name: str) -> str | None:
    if name == "lm_head":
        return "lm_head"
    if name.endswith((".attn", ".self_attn")):
        return "attention"
    if name.endswith(".mlp"):
        return "mlp"
    if name in ("ln_f", "norm") or name.endswith(
            (".ln_1", ".ln_2", ".input_layernorm", ".post_attention_layernorm")):
        return "norm"
    if name in ("wte", "wpe", "embed_tokens"):
        return "embed"
    return None


def module_bytes(module) -> int:
    seen, total = set(), 0
    for t in list(module.parameters()) + list(module.buffers()):
        if t is not None and id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2", choices=("gpt2", "qwen2"))
    parser.add_argument("--bits", type=int, choices=(8, 4))
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    if args.bits:
        from engine.quantize import quantize_model
        model = quantize_model(model, args.bits)

    times = defaultdict(float)
    bytes_per = defaultdict(int)
    starts = {}

    for name, mod in model.named_modules():
        bucket = classify(name)
        if bucket is None:
            continue
        bytes_per[bucket] += module_bytes(mod)
        mod.register_forward_pre_hook(
            lambda m, i: starts.__setitem__(id(m), time.perf_counter()))
        mod.register_forward_hook(
            lambda m, i, o, b=bucket: times.__setitem__(
                b, times[b] + time.perf_counter() - starts[id(m)]))

    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        logits, past = model(ids)          # prefill, not measured
        times.clear()

        t0 = time.perf_counter()
        for _ in range(args.steps):
            tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            logits, past = model(tok, past)
        total = time.perf_counter() - t0

    label = f"int{args.bits}" if args.bits else "fp32"
    step_ms = total / args.steps * 1e3
    print(f"{args.model} @ {label} — {args.steps} cached decode steps, "
          f"{step_ms:.2f} ms/step ({args.steps / total:.1f} tok/s)\n")
    print(f"{'component':>10} {'ms/step':>9} {'% time':>7} {'weights MB':>11}")
    accounted = 0.0
    for bucket, t in sorted(times.items(), key=lambda kv: -kv[1]):
        accounted += t
        print(f"{bucket:>10} {t / args.steps * 1e3:>9.3f} {t / total * 100:>6.1f}% "
              f"{bytes_per[bucket] / 1e6:>11.1f}")
    print(f"{'other':>10} {(total - accounted) / args.steps * 1e3:>9.3f} "
          f"{(total - accounted) / total * 100:>6.1f}% {'—':>11}")
    print("\n(lm_head weights are tied to the embedding — same tensor counted in both rows)")


if __name__ == "__main__":
    main()
