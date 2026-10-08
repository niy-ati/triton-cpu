"""A transformer MLP block on CPU: torch matmuls with Triton elementwise kernels between them.

Compares the step time against the same block written in torch only, and against the
sum of its parts timed separately, to show whether interleaving costs extra.
"""
import sys
import time

import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active(triton.backends.backends["cpu"].driver())
torch.manual_seed(0)


@triton.jit
def gelu_kernel(X, Y, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m)
    # Sigmoid approximation of GELU (tl.erf does not compile on CPU yet).
    y = x / (1.0 + tl.exp(-1.702 * x))
    tl.store(Y + off, y, mask=m)


@triton.jit
def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(Z + off, tl.load(X + off, mask=m) + tl.load(Y + off, mask=m), mask=m)


T, D, H = 64, 1024, 4096  # tokens, model dim, hidden dim
x = torch.randn(T, D)
w1 = torch.randn(D, H) / D**0.5
w2 = torch.randn(H, D) / H**0.5
h = torch.empty(T, H)
g = torch.empty(T, H)
o = torch.empty(T, D)
y = torch.empty(T, D)
BLOCK = 4096


def triton_gelu():
    gelu_kernel[(triton.cdiv(h.numel(), BLOCK), )](h, g, h.numel(), BLOCK=BLOCK)


def triton_add():
    add_kernel[(triton.cdiv(o.numel(), BLOCK), )](o, x, y, o.numel(), BLOCK=BLOCK)


def mm1():
    torch.matmul(x, w1, out=h)


def mm2():
    torch.matmul(g, w2, out=o)


def step_triton():
    mm1()
    triton_gelu()
    mm2()
    triton_add()


def step_torch():
    torch.matmul(x, w1, out=h)
    torch.mul(h, torch.sigmoid(1.702 * h), out=g)
    torch.matmul(g, w2, out=o)
    torch.add(o, x, out=y)


def med(fn, reps=200):
    for _ in range(20):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    ts.sort()
    return ts[len(ts) // 2] / 1e3


tag = sys.argv[1] if len(sys.argv) > 1 else "default"
parts = med(mm1) + med(triton_gelu) + med(mm2) + med(triton_add)
s_tri = med(step_triton)
s_torch = med(step_torch)
step_triton()
hr = x @ w1
ref = (hr * torch.sigmoid(1.702 * hr)) @ w2 + x
assert torch.allclose(y, ref, atol=1e-3, rtol=1e-3), (y - ref).abs().max()
print(f"{tag:24s} MLP step: torch-only {s_torch:8.0f} us | with triton kernels {s_tri:8.0f} us "
      f"| sum of parts {parts:8.0f} us | interleave cost {s_tri / parts:5.2f}x")
