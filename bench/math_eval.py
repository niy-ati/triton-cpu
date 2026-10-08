"""Accuracy (max ULP vs float64) and speed (vs torch) of tl math functions on triton-cpu."""
import re
import sys
import time

import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active(triton.backends.backends["cpu"].driver())
torch.manual_seed(0)

# name: (triton fn, float64 reference, input sampler)
OPS = {
    "exp": (tl.exp, torch.exp, lambda n: torch.empty(n).uniform_(-87, 88)),
    "exp2": (tl.exp2, torch.exp2, lambda n: torch.empty(n).uniform_(-126, 127)),
    "log": (tl.log, torch.log, lambda n: torch.exp(torch.empty(n).uniform_(-80, 80))),
    "log2": (tl.log2, torch.log2, lambda n: torch.exp(torch.empty(n).uniform_(-80, 80))),
    "sin": (tl.sin, torch.sin, lambda n: torch.empty(n).uniform_(-100, 100)),
    "cos": (tl.cos, torch.cos, lambda n: torch.empty(n).uniform_(-100, 100)),
    "erf": (tl.erf, torch.erf, lambda n: torch.empty(n).uniform_(-5, 5)),
    "sigmoid": (tl.sigmoid, torch.sigmoid, lambda n: torch.empty(n).uniform_(-30, 30)),
}


def make():
    @triton.jit
    def kernel(X, Y, BLOCK: tl.constexpr, FN: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Y + off, FN(tl.load(X + off)))

    return kernel


def ulp_err(y, ref64):
    ref32 = ref64.float()
    ulp = (torch.nextafter(ref32.abs(), torch.tensor(float("inf"))) - ref32.abs()).double()
    return ((y.double() - ref64).abs() / ulp).max().item()


def med(fn, reps=100):
    for _ in range(10):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    ts.sort()
    return ts[len(ts) // 2] / 1e3


n, B = 1 << 16, 1024
tag = sys.argv[1] if len(sys.argv) > 1 else ""
print(f"[{tag}] threads={torch.get_num_threads()}")
for name, (tfn, ref, sample) in OPS.items():
    x = sample(n)
    y = torch.empty(n)
    k = make()
    try:
        c = k[(n // B, )](x, y, BLOCK=B, FN=tfn)
    except Exception as e:
        print(f"  {name:8s} COMPILE FAIL: {str(e).strip().splitlines()[-1][:90]}")
        continue
    calls = len(re.findall(r"call\w*\s+\S*@PLT", c.asm["asm"]))
    ref64 = ref(x.double())
    err = ulp_err(y, ref64)
    err_torch = ulp_err(ref(x), ref64)
    tt = med(lambda: k[(n // B, )](x, y, BLOCK=B, FN=tfn))
    tr = med(lambda: ref(x, out=y))
    print(f"  {name:8s} max ulp {err:9.1f} (torch f32 {err_torch:5.1f}) | libm calls/block {calls:5d} | "
          f"triton {tt:7.1f} us | torch {tr:6.1f} us | {tt / tr:5.2f}x")
