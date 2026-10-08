"""Row-wise softmax on triton-cpu vs torch.softmax (all threads), with a correctness check."""
import sys
import time

import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active(triton.backends.backends["cpu"].driver())
torch.manual_seed(0)


@triton.jit
def softmax_kernel(X, Y, N_COLS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < N_COLS
    x = tl.load(X + row * N_COLS + cols, mask=m, other=-float("inf"))
    z = x - tl.max(x, axis=0)
    num = tl.exp(z)
    tl.store(Y + row * N_COLS + cols, num / tl.sum(num, axis=0), mask=m)


def med(fn, reps=50):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    ts.sort()
    return ts[len(ts) // 2] / 1e3


tag = sys.argv[1] if len(sys.argv) > 1 else ""
for rows, cols in ((4096, 1024), (1024, 4096)):
    x = torch.randn(rows, cols)
    y = torch.empty_like(x)
    blk = triton.next_power_of_2(cols)
    run = lambda: softmax_kernel[(rows, )](x, y, cols, BLOCK=blk)
    run()
    assert torch.allclose(y, torch.softmax(x, dim=1), atol=1e-6, rtol=1e-5), (y - torch.softmax(x, 1)).abs().max()
    tt = med(run)
    tr = med(lambda: torch.softmax(x, dim=1))
    print(f"[{tag}] softmax {rows}x{cols}: triton {tt:8.0f} us | torch {tr:8.0f} us | {tt / tr:5.2f}x")
