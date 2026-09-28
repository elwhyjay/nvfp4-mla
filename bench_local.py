"""Our NVFP4 kernels vs FlashInfer baselines in one session (clocks can't be locked on Modal), q_len=1.

  uv run modal run bench.py::sh --cmd "python bench_local.py"                  (GPU=B200 for SM100)
  ncu: uv run modal run bench.py::main --script "bench_local.py --batch 64 --s-kv 16384 --heads 16 --providers v2" \
       --kernel regex:mla_decode_v2
GB/s uses each format's own bytes (NVFP4 352, FP8 576, BF16 1152 B/token); %BW is vs the Step 0 read BW.
Times are GPU kernel time per call with cold L2, no host launch overhead (baselines.gpu_ms).
"""
import argparse
import csv
import pathlib
import time

import torch

import baselines
import ref
from test_decode import ext

STEP0_BW = {(12, 0): 1559, (10, 0): 7300}  # GB/s, kernels/step0.py 4 GB read, 2026-09-27
BYTES = {"fp4": 352, "fp8": 576, "bf16": 1152}


def ours(version, c, max_len):
    f = getattr(ext[False], f"decode_{version}")
    extra = (max_len, 0) if version == "v4" else ()  # v4: heuristic split count
    return lambda: f(c.q_nope, c.q_pe, c.ckv, c.sf, c.kpe, c.page_table, c.kv_lens, ref.SM_SCALE, 1.0, 1.0, *extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="1,4,16,64")
    ap.add_argument("--s-kv", default="1024,4096,16384,65536")
    ap.add_argument("--heads", default="16,128")
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--providers", default="v4,triton:fp8,deq+fa2,wrapper:fa2,dense:fp8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--csv-dir", default="/root/.cache/results")
    a = ap.parse_args()
    ints = lambda s: [int(x) for x in s.split(",")]
    providers = a.providers.split(",")
    gpu = torch.cuda.get_device_name().replace(" ", "_")
    peak = STEP0_BW[torch.cuda.get_device_capability()]
    print(f"{gpu}  step0 BW {peak} GB/s  page_size={a.page_size}", flush=True)

    rows, reported = [], set()
    hdr = f"{'H':>4} {'B':>3} {'s_kv':>6} |" + "".join(f" {p + ' ms/GBs/%':>22}" for p in providers)
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    for H in ints(a.heads):
        for B in ints(a.batch):
            for S in ints(a.s_kv):
                c = ref.Cache([S] * B, a.page_size, H)
                inp = baselines.Inputs(B, S, a.page_size, H) if any(":" in p for p in providers) else None
                line = f"{H:>4} {B:>3} {S:>6} |"
                for p in providers:
                    fmt = "fp8" if p.endswith("fp8") else "bf16" if ":" in p else "fp4"
                    try:
                        fn = ours(p, c, S) if p.startswith("v") else baselines.make(p, inp) if ":" in p else \
                            ref.fa2_dequant(c, lambda: ext[False].dequant(c.ckv, c.sf, c.kpe, 1.0, 1.0, True))
                        ms = baselines.gpu_ms(fn, iters=a.iters)
                    except Exception as e:
                        ms = float("nan")
                        if (p, H) not in reported:
                            reported.add((p, H))
                            print(f"  [{p} H={H}] {baselines.first_line(e)}", flush=True)
                    gbs = B * S * BYTES[fmt] / (ms * 1e-3) / 1e9
                    rows.append(dict(gpu=gpu, provider=p, heads=H, batch=B, s_kv=S, page_size=a.page_size,
                                     ms=ms, gbps=gbs, pct_bw=100 * gbs / peak))
                    line += f" {ms:9.3f}/{gbs:5.0f}/{100 * gbs / peak:3.0f}%" if ms == ms else f" {'--':>22}"
                print(line, flush=True)
                del c, inp
                torch.cuda.empty_cache()
    d = pathlib.Path(a.csv_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"bench_{gpu}_{time.strftime('%Y%m%d-%H%M')}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
