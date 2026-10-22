# Phase 5: weight-only quantization (INT8, then INT4)

Everything you write lives in `engine/quantize.py`. Referee:
`python -m scripts.check_quality --model qwen2 --bits 8` (and `--bits 4`).

The referee changed character. Phases 1–4 had an exact answer (parity).
Quantization deliberately loses precision, so "correct" is now a trade you
measure: **bytes saved vs quality lost**, reported as weight MB, perplexity on
a fixed text, and logit drift. Get the fp32 baselines first (no --bits) so the
drift has something to be measured against.

Build order:

1. **Int8Linear** — per-output-channel symmetric. scale = max|row|/127,
   q = round(W/scale). Forward dequantizes and calls F.linear.
2. **quantize_model** — swap every nn.Linear except lm_head (weight-tied to
   the embedding). After this, `check_quality --bits 8` runs end to end.
3. **Int4Linear** — the fiddly one: group-wise scales (64 weights per group)
   and two-nibbles-per-byte packing. Write a tiny round-trip test first:
   pack then unpack a known 2x8 matrix and assert exact recovery of q.

**Done when** `bench/RESULTS.md` has a table like:

| model | precision | weights MB | perplexity | max logit drift | tok/s (cached) |
|---|---|---|---|---|---|
| qwen2.5 | fp32 | ~1977 | baseline | — | |
| qwen2.5 | int8 | ~5xx | +~0–1% | small | |
| qwen2.5 | int4 | ~3xx | +few % | larger | |

(embedding stays fp32, which is why int8 isn't a clean 4x on total MB —
compute the expected number yourself and check it matches.)

**Expectations, so you aren't surprised:**
- tok/s will likely get WORSE — naive dequant-then-matmul adds work and saves
  no memory traffic (the fp32 weights get rebuilt every forward). Measure it
  anyway and write down why. A quantization scheme only pays off when a kernel
  computes directly on the packed weights — that is exactly Phase 6.
- If int8 perplexity jumps more than ~1–2%, the bug is usually scale shape
  (broadcasting a (out,) scale instead of (out, 1)) or quantizing lm_head.
- If int4 is garbage (perplexity 10x), the packing round-trip is broken —
  test it in isolation.

**Worth 20 minutes of thought:** why does per-ROW int8 work fine but per-row
int4 need groups? (Hint: one outlier weight sets the scale for everything
sharing it; 4 bits means only 15 usable levels.)
