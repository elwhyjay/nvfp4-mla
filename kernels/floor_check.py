"""Where does trtllm-gen MLA's small-batch ~85 us floor come from: GPU kernels or host launch overhead?

  GPU=B200 uv run modal run bench.py::sh --cmd "python kernels/floor_check.py"
Per provider: CUDA-event time (what bench_local reports), CUDA-graph time (no host overhead), host time per call,
and the kernels one call launches with their GPU time (torch.profiler / CUPTI).
"""
import pathlib
import statistics
import sys
import time

import torch
from flashinfer.testing.utils import bench_gpu_time
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import baselines  # noqa: E402

print(torch.cuda.get_device_name())
for B, S in ((1, 1024), (1, 65536), (64, 65536)):
    inp = baselines.Inputs(B, S, 64, 16)
    for p in ("dense:fp8", "dense:bf16", "wrapper:fa2"):
        fn = baselines.make(p, inp)
        fn()
        torch.cuda.synchronize()
        med = lambda **kw: statistics.median(bench_gpu_time(fn=fn, dry_run_iters=5, repeat_iters=30, **kw))
        ev = med(cold_l2_cache=True)
        try:
            graph = f"{med(cold_l2_cache=False, use_cuda_graph=True) * 1e3:7.1f}"
        except Exception as e:
            graph = f"n/a ({type(e).__name__})"
        t0 = time.perf_counter()
        for _ in range(200):
            fn()
        host = (time.perf_counter() - t0) / 200 * 1e6
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
        kern = {}
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                k = kern.setdefault(e.name, [0, 0.0])
                k[0] += 1
                k[1] += e.device_time
        print(f"\nH16 B{B} S{S} {p}: event {ev * 1e3:7.1f} us | graph {graph} us | host/call {host:6.1f} us")
        for name, (n, t) in sorted(kern.items(), key=lambda x: -x[1][1]):
            print(f"    {n / 10:4.1f}x/call {t / n:8.1f} us  {name[:110]}")
        del fn
    del inp
    torch.cuda.empty_cache()
