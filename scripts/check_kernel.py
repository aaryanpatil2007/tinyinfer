"""Phase 6 referee: is the kernel correct, and how close to the memory ceiling is it?

    python -m scripts.check_kernel --selftest   # proves MLX+Metal works on this machine
    python -m scripts.check_kernel              # correctness + bandwidth of YOUR kernel

Correctness: your fused int8 matvec vs the same math in plain MLX ops, on the
engine's real layer shapes. Speed: effective GB/s (bytes of W moved / kernel
time), reported against (a) the naive unfused dequant-matmul, and (b) a
measured streaming-bandwidth probe — the practical ceiling for this machine.
A fused matvec is pure memory traffic, so GB/s as a fraction of the probe IS
the roofline verdict.
"""

import argparse
import time

import mlx.core as mx

SHAPES = [(896, 896), (4864, 896), (896, 4864), (1024, 1024 * 8)]
ITERS = 100


def bench(fn, iters=ITERS) -> float:
    # MLX is lazy: every iteration must be forced with mx.eval or it never runs.
    for _ in range(3):
        mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def bandwidth_probe() -> float:
    a = mx.random.normal((256 * 1024 * 1024 // 4,))  # 256 MB fp32
    mx.eval(a)
    t = bench(lambda: mx.sum(a), iters=20)
    return a.nbytes / t / 1e9


def selftest() -> None:
    src = """
        uint i = thread_position_in_grid.x;
        out[i] = inp[i] * 2.0f;
    """
    k = mx.fast.metal_kernel(name="selftest_double", input_names=["inp"],
                             output_names=["out"], source=src)
    a = mx.arange(1000).astype(mx.float32)
    (res,) = k(inputs=[a], grid=(a.size, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[a.shape], output_dtypes=[mx.float32])
    assert mx.allclose(res, a * 2).item(), "selftest kernel produced wrong values"
    print(f"selftest OK — custom Metal kernels run on this machine "
          f"(streaming bandwidth ≈ {bandwidth_probe():.0f} GB/s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    from engine.kernels.int8_matvec import THREADS_PER_ROW, int8_matvec

    peak = bandwidth_probe()
    print(f"streaming-bandwidth probe: {peak:.0f} GB/s (practical ceiling)")
    print(f"kernel config: THREADS_PER_ROW = {THREADS_PER_ROW}\n")
    print(f"{'shape':>14} {'correct':>8} {'kernel':>10} {'naive':>10} "
          f"{'GB/s':>7} {'% ceiling':>10}")

    for rows, cols in SHAPES:
        W = (mx.random.normal((rows, cols)) * 40).astype(mx.int8)
        scales = mx.abs(mx.random.normal((rows,))) * 0.01
        x = mx.random.normal((cols,))
        mx.eval(W, scales, x)

        ref = (W.astype(mx.float32) * scales[:, None]) @ x
        y = int8_matvec(W, scales, x)
        mx.eval(ref, y)
        rel = (mx.abs(y - ref).max() / (mx.abs(ref).max() + 1e-9)).item()
        ok = rel < 1e-3

        t_kernel = bench(lambda: int8_matvec(W, scales, x))
        t_naive = bench(lambda: (W.astype(mx.float32) * scales[:, None]) @ x)

        gbs = W.nbytes / t_kernel / 1e9
        print(f"{f'{rows}x{cols}':>14} {'yes' if ok else 'NO':>8} "
              f"{t_kernel * 1e6:>8.1f}us {t_naive * 1e6:>8.1f}us "
              f"{gbs:>7.1f} {gbs / peak * 100:>9.1f}%")

    print("\nTargets: correct everywhere, beat naive, then climb toward the ceiling "
          "(v2/v3 in the kernel file). Log the best row per variant in bench/RESULTS.md.")


if __name__ == "__main__":
    main()
