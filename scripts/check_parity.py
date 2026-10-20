"""Phase 1 referee: does your forward pass produce the same logits as HuggingFace?

    python -m scripts.check_parity

Green means your GPT-2 implementation is numerically correct and you move on to
generation. Red prints the max absolute difference so you can bisect layer by layer.
"""

import torch

from engine.loader import get_tokenizer, load_gpt2

# Comparison runs in float64: an algorithmically identical forward pass agrees
# to ~1e-13, while float32 kernel rounding noise alone reaches ~2e-4 (HF's
# Conv1D and our transposed nn.Linear take different BLAS paths). float64
# separates "correct" from "noise" by eight orders of magnitude.
TOLERANCE = 1e-8

PROMPTS = [
    "The capital of France is",
    "import torch\n\ndef attention(q, k, v):",
    "1, 1, 2, 3, 5, 8, 13,",
]


def main() -> None:
    from transformers import GPT2LMHeadModel

    tokenizer = get_tokenizer()
    mine = load_gpt2().double()
    theirs = GPT2LMHeadModel.from_pretrained("gpt2").eval().double()

    worst = 0.0
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            my_logits = mine(ids)
            hf_logits = theirs(ids).logits
        diff = (my_logits - hf_logits).abs().max().item()
        worst = max(worst, diff)
        status = "OK " if diff < TOLERANCE else "FAIL"
        print(f"[{status}] max|Δlogits| = {diff:.2e}  {prompt!r}")

    if worst < TOLERANCE:
        print(f"\nParity achieved (worst {worst:.2e} < {TOLERANCE}). Phase 1 done — go measure tokens/sec.")
    else:
        print(f"\nNot yet (worst {worst:.2e}). Debug tip: compare intermediate activations "
              "block by block against HF's (model.transformer.h[i] outputs via hooks).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
