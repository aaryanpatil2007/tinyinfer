"""Phase 6 — your first GPU kernel. You write KERNEL_BODY (Metal Shading Language).

The op: fused int8 matrix–vector product for batch-1 decode.
    y[r] = scales[r] * Σ_c  W[r, c] * x[c]
W is (ROWS, COLS) int8, scales (ROWS,) fp32, x (COLS,) fp32, y (ROWS,) fp32.
FUSED means the fp32 weight matrix never exists in memory: each weight is read
as 1 byte, dequantized in a register, used, and discarded. That is the entire
mechanism by which quantization becomes speed.

The wrapper below handles everything except the kernel itself. Inside
KERNEL_BODY you can use:
    W, scales, x        device pointers (const device char*, const device float*, ...)
    y                   output pointer (device float*)
    W_shape[0/1]        ROWS / COLS (because the body references W_shape)
    TPR                 literal int, threads cooperating per row (set below)
    thread_position_in_grid.x       global thread index
and Metal builtins: simd_sum(v) sums a value across the 32 threads of a
simdgroup; char4/float4 vector loads; metal_stdlib is already included.

Climb in three steps, measuring with `python -m scripts.check_kernel` each time:
  v1  THREADS_PER_ROW = 1: one thread owns one row, scalar loop over COLS.
      Correct first. It will be slow — one thread can't saturate bandwidth.
  v2  THREADS_PER_ROW = 32: a simdgroup owns one row. Lane k handles columns
      k, k+32, k+64, ... then one simd_sum combines the 32 partials and lane 0
      writes y[row]. This is the big jump.
  v3  Vectorize: load 4 weights per instruction via char4 (reinterpret the row
      pointer), multiply against float4 of x. Stride pattern same as v2 but in
      units of 4. Squeezes out the last factor.
"""

import mlx.core as mx

THREADS_PER_ROW = 1

KERNEL_BODY: str | None = None


_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_cache: dict = {}


def _build(tpr: int):
    if KERNEL_BODY is None:
        raise NotImplementedError(
            "write KERNEL_BODY in engine/kernels/int8_matvec.py (see docs/PHASE6.md)")
    source = KERNEL_BODY.replace("TPR", str(tpr))
    key = source
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name="int8_matvec",
            input_names=["W", "scales", "x"],
            output_names=["y"],
            header=_HEADER,
            source=source,
        )
    return _cache[key]


def int8_matvec(W: mx.array, scales: mx.array, x: mx.array) -> mx.array:
    assert W.dtype == mx.int8 and W.ndim == 2
    rows, _ = W.shape
    tpr = THREADS_PER_ROW
    kernel = _build(tpr)
    (y,) = kernel(
        inputs=[W, scales, x],
        grid=(rows * tpr, 1, 1),
        threadgroup=(min(max(tpr, 32), 1024), 1, 1),
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
    )
    return y
