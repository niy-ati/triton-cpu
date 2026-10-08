"""Does running torch ops and triton-cpu kernels back to back cost more than each alone?"""
import sys
import time

import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active(triton.backends.backends["cpu"].driver())


@triton.jit
def scale_kernel(X, Y, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(Y + off, tl.load(X + off, mask=m) * 2.0 + 1.0, mask=m)


N = 1 << 20
a = torch.randn(N)
b = torch.empty(N)
c = torch.empty(N)
grid = (triton.cdiv(N, 4096), )


def torch_op():
    torch.exp(a, out=c)


def triton_op():
    scale_kernel[grid](a, b, N, BLOCK=4096)


def pair():
    torch_op()
    triton_op()


def med(fn, reps=400):
    for _ in range(30):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    ts.sort()
    return ts[len(ts) // 2] / 1e3, ts[int(0.99 * (len(ts) - 1))] / 1e3


tag = sys.argv[1] if len(sys.argv) > 1 else "default"
t_torch, t_torch99 = med(torch_op)
t_triton, t_triton99 = med(triton_op)
t_pair, t_pair99 = med(pair)
print(f"{tag:22s} torch alone {t_torch:7.0f} | triton alone {t_triton:7.0f} | "
      f"sum {t_torch + t_triton:7.0f} | interleaved pair {t_pair:7.0f} us "
      f"({t_pair / (t_torch + t_triton):4.2f}x)  p99 pair {t_pair99:7.0f}")
assert torch.allclose(b, a * 2 + 1)
