"""Phase 5 referee: quantization trades precision for bytes — this measures both sides.

    python -m scripts.check_quality --model qwen2             # fp32 baseline
    python -m scripts.check_quality --model qwen2 --bits 8
    python -m scripts.check_quality --model qwen2 --bits 4

Reports weight memory, perplexity on bench/data/eval.txt, and (for quantized
runs) max logit error vs the fp32 model. Judge a scheme by the TRADE, not one
number: int8 should be ~4x smaller with near-zero perplexity drift; int4
~7x smaller with a small but visible drift.
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.loader import get_tokenizer, load_model

EVAL_TEXT = Path(__file__).resolve().parent.parent / "bench" / "data" / "eval.txt"
DRIFT_PROMPT = "The capital of France is"
WINDOW = 512


def weight_bytes(model) -> int:
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t is not None and id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()
    return total


@torch.no_grad()
def perplexity(model, tokenizer) -> float:
    ids = tokenizer(EVAL_TEXT.read_text(), return_tensors="pt").input_ids
    total_nll, total_tokens = 0.0, 0
    for start in range(0, ids.size(1) - 1, WINDOW):
        chunk = ids[:, start:start + WINDOW + 1]
        if chunk.size(1) < 2:
            break
        logits, _ = model(chunk[:, :-1])
        targets = chunk[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                              targets.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tokens += targets.numel()
    return math.exp(total_nll / total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=("gpt2", "qwen2"))
    parser.add_argument("--bits", type=int, choices=(8, 4))
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    model, _ = load_model(args.model)

    if args.bits:
        from engine.quantize import quantize_model

        fp32_model, _ = load_model(args.model)
        with torch.no_grad():
            fp32_logits, _ = fp32_model(
                tokenizer(DRIFT_PROMPT, return_tensors="pt").input_ids)
        del fp32_model

        model = quantize_model(model, args.bits)

    mb = weight_bytes(model) / 1e6
    ppl = perplexity(model, tokenizer)
    label = f"int{args.bits}" if args.bits else "fp32"
    print(f"{args.model} @ {label}:")
    print(f"  weights:    {mb:8.1f} MB")
    print(f"  perplexity: {ppl:8.3f}   (eval: {EVAL_TEXT.name}, window {WINDOW})")

    if args.bits:
        with torch.no_grad():
            q_logits, _ = model(tokenizer(DRIFT_PROMPT, return_tensors="pt").input_ids)
        drift = (q_logits - fp32_logits).abs().max().item()
        print(f"  max|Δlogits| vs fp32: {drift:.4f}  ({DRIFT_PROMPT!r})")
        print("\nRecord all three numbers next to the fp32 baseline in bench/RESULTS.md.")


if __name__ == "__main__":
    main()
