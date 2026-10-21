# Phase 4: Qwen2.5-0.5B — the modern stack

Same drill as GPT-2: hand-write the forward pass in `engine/models/qwen2.py`,
real weights load via `engine/loader.py` (already working), referee is
`python -m scripts.check_parity --model qwen2`.

Build order, easiest to hardest — each step's contract is in the skeleton docstrings:

1. **RMSNorm** — 3 lines. Watch the float32 upcast noted in the docstring.
2. **SwiGLU MLP** — 1 line once you see it.
3. **DecoderLayer** — identical wiring to your GPT-2 Block.
4. **apply_rope** — the conceptually new piece. Half-split convention, cos/sin
   in float32. Get this wrong and parity fails at ~1e0, not 1e-13.
5. **Attention with GQA** — your GPT-2 attention + RoPE on Q,K + expanding
   2 KV heads to serve 14 Q heads. Your rectangular mask carries over as-is.
6. **Qwen2.forward** — no position embedding add; build `positions` from the
   cache length and pass it down.

**Done when:**
- `check_parity --model qwen2` is green (float64, 1e-8).
- `check_kv --model qwen2` is green — note your KV cache works unchanged
  *because* you cache rotated keys; be able to say why that's lossless.
- `bench/RESULTS.md` gets Qwen rows (naive + cached). Note the cache memory:
  2 KV heads instead of 14 — compute bytes/token and compare against a
  hypothetical full-MHA cache. That ratio is why GQA exists.

**Debugging parity:** hook the reference model's `model.layers[i]` outputs and
bisect. RoPE bugs look like "fine at position 0, garbage after" — test a prompt
of one token first.
