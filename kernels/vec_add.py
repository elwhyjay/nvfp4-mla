import pathlib

import torch
from torch.utils.cpp_extension import load

mod = load("vec_add", sources=[str(pathlib.Path(__file__).with_suffix(".cu"))])

n = 1 << 24
a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
c = mod.vec_add_launch(a, b)
torch.cuda.synchronize()
assert torch.allclose(c, a + b)
print("ok", n)
