"""Print the OpenMP runtimes loaded once torch and a triton-cpu kernel have both run."""
import ctypes
import os
import sys

import torch
import triton
import triton.language as tl

triton.runtime.driver.set_active(triton.backends.backends["cpu"].driver())


@triton.jit
def k(X):
    pass


k[(4, )](torch.empty(4))
torch.exp(torch.randn(1 << 20))

if sys.platform == "darwin":
    dyld = ctypes.CDLL(None)
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    images = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
else:
    images = [line.split()[-1] for line in open("/proc/self/maps") if "/" in line]
print("OpenMP runtimes loaded:", sorted({p for p in images if "omp" in os.path.basename(p)}))
print("torch threads:", torch.get_num_threads(), "| cpu count:", os.cpu_count())
