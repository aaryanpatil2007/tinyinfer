"""Phase 3 referee: the KV cache must change the SPEED, never the TOKENS.

    python -m scripts.check_kv

Checks that generate(use_cache=True) emits token-for-token identical output to
the naive path (greedy decoding makes this an exact equality test), then times
both at increasing lengths so you can watch the gap grow — that growth curve is
your write-up.
"""

import time

import torch

from engine.loader import get_tokenizer, load_gpt2

PROMPT = "The history of computing begins with"
LENGTHS = (32, 128, 256)


def timed_generate(model, ids, n, use_cache):
    start = time.perf_counter()
    out = model.generate(ids, max_new_tokens=n, use_cache=use_cache)
    return out, n / (time.perf_counter() - start)


def main() -> None:
    tokenizer = get_tokenizer()
    model = load_gpt2()
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    print(f"{'tokens':>7} {'naive tok/s':>12} {'cached tok/s':>13} {'speedup':>8}  identical?")
    all_ok = True
    for n in LENGTHS:
        naive, naive_rate = timed_generate(model, ids, n, use_cache=False)
        cached, cached_rate = timed_generate(model, ids, n, use_cache=True)
        same = torch.equal(naive, cached)
        all_ok &= same
        print(f"{n:>7} {naive_rate:>12.2f} {cached_rate:>13.2f} {cached_rate / naive_rate:>7.2f}x  "
              f"{'yes' if same else 'NO — first mismatch at token ' + str((naive[0] != cached[0]).nonzero()[0].item())}")

    if all_ok:
        print("\nKV cache is lossless. Log the numbers in bench/RESULTS.md — phase 3 done.")
    else:
        print("\nCached output diverges from naive output. The cache changed the math somewhere —"
              "\nusual suspects: position embedding offset, stale mask, Q/K/V for the wrong token.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
