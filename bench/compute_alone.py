"""triton-cpu alone: a compute-bound kernel, median launch-to-finish time."""
import sys
import time

import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active(triton.backends.backends["cpu"].driver())


@triton.jit
def work_kernel(X, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.full([BLOCK], 0.0, tl.float32) + pid
    for _ in range(ITERS):
        acc = acc * 0.5 + 1.0
    tl.store(X + pid * BLOCK + tl.arange(0, BLOCK), acc)


x = torch.empty(96 * 256)
for _ in range(20):
    work_kernel[(96, )](x, 2000, 256)
ts = []
for _ in range(200):
    t0 = time.perf_counter_ns()
    work_kernel[(96, )](x, 2000, 256)
    ts.append(time.perf_counter_ns() - t0)
ts.sort()
print(f"{sys.argv[1]:22s} compute-bound triton alone (96 blocks): median {ts[100] / 1e3:7.0f} us")
