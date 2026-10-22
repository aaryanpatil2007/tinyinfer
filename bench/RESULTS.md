# Results log

Same machine, same prompt (`bench/throughput.py` defaults), best of 3 runs.
One row per optimization — this table is the project.

| Date | Change | Model | Device | New tokens | tok/s | vs. previous |
|------|--------|-------|--------|-----------:|------:|-------------:|
| 2026-10-19 | naive generation (baseline) | gpt2 (124M) | cpu | 128 | 43.92 | — |
| 2026-10-20 | KV cache | gpt2 (124M) | cpu | 128 | 162.06 | 3.7x |
| 2026-10-21 | naive generation (baseline) | qwen2.5 (0.5B) | cpu | 128 | 12.21 | — |
| 2026-10-21 | KV cache | qwen2.5 (0.5B) | cpu | 128 | 56.49 | 4.6x |

## Phase 5: weight-only quantization (naive dequant-in-forward)

Perplexity on `bench/data/eval.txt` (window 512); tok/s = cached decode, 128 new tokens, cpu.

| model | precision | weights MB | perplexity | max logit drift | tok/s (cached) |
|---|---|---:|---:|---:|---:|
| gpt2 (124M) | fp32 | 497.8 | 52.961 | — | 162.06 |
| gpt2 (124M) | int8 | 243.3 | 53.036 (+0.1%) | 1.20 | 88.12 |
| gpt2 (124M) | int4 g64 | 205.8 | 59.237 (+11.9%) | 10.54 | 34.33 |
| qwen2.5 (0.5B) | fp32 | 1976.1 | 34.452 | — | 56.49 |
| qwen2.5 (0.5B) | int8 | 903.9 | 34.470 (+0.05%) | 3.78 | 24.90 |
| qwen2.5 (0.5B) | int4 g64 | 746.1 | 42.872 (+24.4%) | 17.14 | 9.19 |
