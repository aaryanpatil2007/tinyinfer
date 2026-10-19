# Phase 1–3: from empty forward pass to first real speedup

This is the "week 1" plan. Each day has a done-criterion you can run. The split:
**you write everything in `engine/models/gpt2.py`**; the loader, parity checker,
and benchmark harness are scaffolding and already work.

## Setup (once)

```bash
source .venv/bin/activate
python -c "from engine.loader import load_gpt2"   # should import cleanly
```

## Day 1–2 — MLP, Block wiring, embeddings

Fill in `MLP.forward`, `Block.forward`, and `GPT2.forward` (treat attention as a
black box for now — it will just raise). Things that will bite you if skipped:

- GELU must be the **tanh approximation** (`F.gelu(x, approximate="tanh")`).
- Positions are `0..T-1`, looked up in `wpe`, added to token embeddings.
- Pre-norm: the residual stream is never normalized in place; `ln_1`/`ln_2`
  apply only to the branch input.

**Done when:** `GPT2()(torch.randint(0, 50257, (2, 16)))` raises
NotImplementedError *inside CausalSelfAttention* and nowhere else.

## Day 3–4 — causal self-attention

The six steps are spelled out in the docstring. Write it with explicit
reshape/transpose/matmul ops — no `F.scaled_dot_product_attention`. You will
rewrite this exact code as a fused kernel in a later phase; the point of this
week is knowing every intermediate tensor and its shape.

**Done when:** `python -m scripts.check_parity` prints `Parity achieved`.
Max |Δlogits| vs HuggingFace must be < 1e-4 on all three prompts.

Debugging when it isn't: hook HF's `model.transformer.h[i]` outputs and compare
block-by-block. The first block that diverges contains the bug. Usual suspects:
QKV split order, missing scale factor, mask applied after softmax, wrong GELU.

## Day 5 — naive generation + baseline number

Fill in `GPT2.generate`: greedy argmax, recompute the full sequence every step.
This is deliberately O(T²·T) wasteful — it's the baseline.

**Done when:** `python -m bench.throughput` prints a tokens/sec number and the
unicorn continuation reads as coherent English. Record the number in
`bench/RESULTS.md` — it is row 1 of the table the whole project builds.

## Day 6–7 — KV cache (Phase 3, first real optimization)

Each decode step only needs the new token's Q against all cached K, V. Add a
cache (per-layer K and V tensors, appended each step) and a path through
`forward` that takes one token + the cache instead of the full sequence.

**Done when:**
1. Cached and uncached generation produce **identical token sequences**.
2. `python -m bench.throughput` shows the speedup (expect several ×, growing
   with sequence length). Row 2 of the table.

Then stop and write up: naive vs cached tok/s at 128/256/512 new tokens, and a
one-paragraph explanation of *why* the gap grows with length. That write-up is
the template for every phase after this.

## What comes next (Phase 4+)

Port to Qwen2.5-0.5B — RoPE, RMSNorm, GQA, SwiGLU. Your `rms_normalization.py`
exercise in `foundations/` is the warm-up for exactly this. Same drill: match
HF logits, then optimize.
