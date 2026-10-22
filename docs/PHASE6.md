# Phase 6: your first GPU kernel — fused int8 matvec in Metal

Phase 5 ended with a trap deliberately armed: int8 weights are half the bytes
but decode got SLOWER, because the fp32 matrix gets rebuilt in memory every
forward. This phase springs the trap: a Metal kernel that dequantizes each
weight in a register, so the fp32 matrix never exists. On a memory-bound
matvec that converts the byte saving directly into speed.

You write ONE thing: `KERNEL_BODY` in `engine/kernels/int8_matvec.py` —
actual Metal Shading Language, run on the M4 Max GPU via MLX's
`mx.fast.metal_kernel`. Everything else (wrapper, dispatch, referee) is done.

## 6a — profile first (done tooling, your observation)

    python -m bench.profile_decode --model qwen2
    python -m bench.profile_decode --model qwen2 --bits 8

Confirm where the time goes and how it tracks weight bytes (mlp dominates).
Note where the CPU numbers DON'T track bytes — attention runs over its byte
share (cache traffic + many small ops), and that gap is worth a sentence in
your write-up.

## 6b — learn just enough Metal

The mental model, smallest to largest:
- a **thread** runs your KERNEL_BODY once; `thread_position_in_grid.x` says which one you are
- threads come in **simdgroups of 32** that execute in lockstep and can share
  values in one instruction (`simd_sum(v)` — no memory involved)
- simdgroups come in **threadgroups** (our wrapper launches one per row group)

Read: MLX custom Metal kernels docs (mlx docs → "Custom Metal Kernels") and
skim the Metal Shading Language spec sections on vector types (char4/float4).
Everything in scope inside KERNEL_BODY is listed in the kernel file's docstring.

## 6c — the climb (referee: `python -m scripts.check_kernel`)

The referee prints effective GB/s against a measured ~309 GB/s streaming
ceiling for this machine. A fused matvec does ~2 ops per byte — pure memory
traffic — so **% of ceiling is the score**, that's the roofline argument in
one number.

- **v1 — correct.** `THREADS_PER_ROW = 1`: each thread owns a row, scalar loop
  over the columns, multiply-accumulate in fp32, `y[row] = scales[row] * acc`.
  Expect single-digit % of ceiling. Correctness gate: referee says yes on all
  four shapes (they include non-square and the big 1024x8192).
- **v2 — parallel within the row.** `THREADS_PER_ROW = 32`: one simdgroup per
  row. Lane k accumulates columns k, k+32, k+64, …; one `simd_sum` merges the
  32 partials; lane 0 writes. The W reads become coalesced (adjacent lanes
  read adjacent bytes) — this is the big jump.
- **v3 — vectorize.** Reinterpret the row as `const device char4*` and process
  4 weights per load. Optionally several rows per threadgroup. Chase the
  ceiling; past ~70% you're winning.

Log each variant as a row in bench/RESULTS.md: variant, GB/s, % ceiling.

## Done when

- all shapes correct, best variant well past the naive unfused baseline and
  at a defensible fraction of the ceiling, with a short write-up: bytes moved,
  GB/s achieved, why the remaining gap exists (scales/x reads, dispatch
  overhead, small-shape underutilization).
- you can answer: why does THIS op benefit from int8 while the Phase 5
  CPU path got slower? (one sentence, bytes in vs bytes out)

## Stretch (pick later, not now)

- int4 variant: unpack two nibbles per byte in-register — another 2x fewer bytes
- a full MLX decode path for Qwen using your kernel for every Linear
  (torch↔MLX per-op copying would eat the win; port the loop, not the op)
- fused attention kernel for single-token decode (reads the whole KV cache:
  the other big byte-mover, and the Phase 7 speculative-decode substrate)
