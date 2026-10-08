import re

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def unary_kernel(X, Y, BLOCK: tl.constexpr, FN: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + off, FN(tl.load(X + off)))


CASES = [
    (tl.exp, torch.exp, "exp", lambda n: torch.empty(n).uniform_(-87, 88)),
    (tl.log, torch.log, "log", lambda n: torch.exp(torch.empty(n).uniform_(-80, 80))),
    (tl.log2, torch.log2, "log2", lambda n: torch.exp(torch.empty(n).uniform_(-80, 80))),
    (tl.erf, torch.erf, "erf", lambda n: torch.empty(n).uniform_(-5, 5)),
]


@pytest.mark.parametrize("tile_and_fuse", [False, True])
@pytest.mark.parametrize("fn, ref, libm_name, sample", CASES, ids=[c[2] for c in CASES])
def test_math_approximation(fn, ref, libm_name, sample, tile_and_fuse):
    driver = triton.backends.backends['cpu'].driver()
    triton.runtime.driver.set_active(driver)

    torch.manual_seed(0)
    n, block = 4096, 1024
    x = sample(n)
    y = torch.empty_like(x)
    compiled = unary_kernel[(n // block, )](x, y, BLOCK=block, FN=fn, tile_and_fuse=tile_and_fuse)
    torch.testing.assert_close(y, ref(x), rtol=1e-6, atol=1e-6)

    # In fused regions the functions are expanded into polynomials that
    # vectorize, instead of becoming one libm call per element.
    calls = re.findall(rf"call\w*\s+{libm_name}f\b", compiled.asm["asm"])
    if tile_and_fuse:
        assert not calls
