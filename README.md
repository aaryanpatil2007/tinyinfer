# tinyinfer — a from-scratch LLM inference engine

**Aaryan Patil** · Started July 2026 · Reframed October 2026

A hand-written inference engine that runs real pretrained models (GPT-2, then
Qwen2.5), rebuilt layer by layer for speed: KV caching, quantization, custom
GPU kernels, speculative decoding — each step verified against HuggingFace
logits and measured in tokens/second.

This repo began as "build a GPT from scratch." The training half is done and
archived in [`foundations/`](foundations/); the project is now the half that
actually interested me — **inference**: how a forward pass becomes fast.

## The rules

1. **Real weights, hand-written forward pass.** The model code in
   [`engine/`](engine/) is mine; the weights are the actual pretrained
   checkpoints. Logits must match HuggingFace to 1e-4 before any optimization
   is allowed to touch the code.
2. **Every phase ends in a number.** Tokens/second on the same machine, same
   prompt, logged in `bench/RESULTS.md`. No "it works" without "it's this fast."
3. **No black boxes on the hot path.** PyTorch provides tensors; attention,
   caching, quantization, and (later) the kernels themselves are hand-written.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.check_parity    # does my GPT-2 match HuggingFace?
python -m bench.throughput        # how fast is it?
```

## Roadmap

**Build the engine**
- [ ] GPT-2 (124M) forward pass from scratch, real HF weights, logits match to 1e-4
- [ ] Naive autoregressive generation — baseline tokens/sec
- [ ] KV cache — first real optimization, measured against the baseline
- [ ] Port to Qwen2.5-0.5B: RoPE, RMSNorm, GQA, SwiGLU (the modern stack), logits matched again

**Make it fast**
- [ ] INT8 / INT4 weight-only quantization
- [ ] Fused attention kernel — Metal (MLX) on Apple Silicon, CUDA/Triton on NVIDIA
- [ ] Batched decode
- [ ] Speculative decoding with a small draft model, acceptance-rate math included

**Prove it**
- [ ] Benchmarks vs. HuggingFace transformers and llama.cpp/MLX on the same hardware,
      with roofline analysis of where each gap comes from

**Reaching (pick one, after the above)**
- [ ] Beat MLX/llama.cpp on one specific kernel (int4 batch-1 decode matmul)
- [ ] Continuous batching + paged KV cache with a latency/throughput Pareto curve
- [ ] Tokens-per-joule across quantization levels, measured properly

## Layout

| Path | What it is |
|---|---|
| `engine/models/` | Hand-written model forward passes (the point of the repo) |
| `engine/loader.py` | Maps HF checkpoints onto the hand-written modules |
| `scripts/check_parity.py` | Correctness referee: my logits vs. HuggingFace's |
| `bench/` | Tokens/sec harness and the results log |
| `docs/PHASE1.md` | Current working plan with per-day done-criteria |
| `foundations/` | Archived: backprop, training loops, norm variants written by hand |

**External dependencies:** PyTorch for tensor ops; `transformers` only for
checkpoint download and tokenization — never for the forward pass.
