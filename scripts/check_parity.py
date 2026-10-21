"""Parity referee: does your forward pass produce the same logits as HuggingFace?

    python -m scripts.check_parity                  # gpt2
    python -m scripts.check_parity --model qwen2

Comparison runs in float64: an algorithmically identical forward pass agrees to
~1e-13, while float32 kernel rounding noise alone reaches ~1e-4. float64
separates "correct" from "noise" by several orders of magnitude.
"""

import argparse

import torch

from engine.loader import get_tokenizer, load_model, load_reference

TOLERANCE = 1e-8

PROMPTS = [
    "The capital of France is",
    "import torch\n\ndef attention(q, k, v):",
    "1, 1, 2, 3, 5, 8, 13,",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=("gpt2", "qwen2"))
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    mine, _ = load_model(args.model)
    mine = mine.double()
    theirs = load_reference(args.model).double()

    worst = 0.0
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            my_logits, _ = mine(ids)
            hf_logits = theirs(ids).logits
        diff = (my_logits - hf_logits).abs().max().item()
        worst = max(worst, diff)
        status = "OK " if diff < TOLERANCE else "FAIL"
        print(f"[{status}] max|Δlogits| = {diff:.2e}  {prompt!r}")

    if worst < TOLERANCE:
        print(f"\nParity achieved (worst {worst:.2e} < {TOLERANCE}). Go measure tokens/sec.")
    else:
        print(f"\nNot yet (worst {worst:.2e}). Debug tip: compare intermediate activations "
              "layer by layer against HF's (hooks on the reference model's layers).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
