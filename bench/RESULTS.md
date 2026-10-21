# Results log

Same machine, same prompt (`bench/throughput.py` defaults), best of 3 runs.
One row per optimization — this table is the project.

| Date | Change | Model | Device | New tokens | tok/s | vs. previous |
|------|--------|-------|--------|-----------:|------:|-------------:|
| 2026-10-19 | naive generation (baseline) | gpt2 (124M) | cpu | 128 | 43.92 | — |
| 2026-10-20 | KV cache | gpt2 (124M) | cpu | 128 | 162.06 | 3.7x |
| 2026-10-21 | naive generation (baseline) | qwen2.5 (0.5B) | cpu | 128 | 12.21 | — |
| 2026-10-21 | KV cache | qwen2.5 (0.5B) | cpu | 128 | 56.49 | 4.6x |
