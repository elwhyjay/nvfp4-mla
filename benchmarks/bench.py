"""nvfp4_mla vs MLA decode baselines on the local GPU, q_len = 1, DeepSeek geometry (512 + 64).

  python benchmarks/bench.py                                   # default sweep
  python benchmarks/bench.py --batch 64 --s-kv 65536 --heads 16 --providers nvfp4,trtllm:fp8

Times are GPU kernel time per call with a cold L2 (CUPTI; host launch overhead excluded, see baselines.gpu_ms).
GB/s counts each format's own bytes per token: NVFP4 352, FP8 576, BF16 1152. Providers:
  nvfp4[/N[/S]]   this kernel (N KV splits, 0 = auto; S pipeline stages)
  nvfp4+fa2       nvfp4_mla.dequantize to BF16, then FlashInfer fa2 ("dequantize, then a BF16 kernel")
  fa2, cutlass    FlashInfer BatchMLAPagedAttentionWrapper, BF16 KV
  trtllm:bf16|fp8 FlashInfer trtllm_batch_decode_with_kv_cache_mla (trtllm-gen on SM100, XQA on SM120)
  triton:bf16|fp8 SGLang's Triton MLA decode kernel (vendored in triton_mla.py), the vLLM/SGLang default on SM120
  cake            FlashInfer Cake NVFP4 MLA decode on the same cache (SM100, CUDA 13.4, FlashInfer >= 0.7.4rc1);
                  its query is NVFP4, quantized once outside the timed call
"""
import argparse
import csv
import pathlib
import time

import torch

import baselines
import nvfp4_mla
from nvfp4_mla.testing import SM_SCALE, RandomCache, fa2_on_dequantized

BYTES = {"nvfp4": 352, "fp8": 576, "bf16": 1152}
# Achievable DRAM read BW (GB/s) from roofline.py, for the %BW column; override with --bw.
KNOWN_BW = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": 1559, "NVIDIA B200": 7300}


def ours(name, c, max_len):
    _, splits, stages = (name.split("/") + ["0", "3"])[:3]
    f = nvfp4_mla._load().decode
    return lambda: f(*c.args(), max_len, int(splits), int(stages))


def cake(c, max_len):
    from flashinfer.experimental.cake_mla_nvfp4_paged_decode import cake_backend as cb
    from flashinfer.mla import prepare_cake_mla_nvfp4_paged_decode, quantize_mla_nvfp4_query

    q = torch.cat([c.q_nope, c.q_pe], -1).unsqueeze(1)  # [B, q_len=1, H, 576]
    lead, rows, dev = tuple(q.shape[:-1]), q[..., 0].numel(), q.device
    qbuf = torch.empty(cb.query_workspace_bytes(rows), dtype=torch.uint8, device=dev)
    qq = quantize_mla_nvfp4_query(q, c.ckv_scale, c.kpe_scale, out=cb.mla_nvfp4_query_buffers(qbuf, lead))
    out = torch.empty(*lead, 512, dtype=torch.bfloat16, device=dev)
    ws = torch.empty(cb.max_workspace_bytes(rows), dtype=torch.uint8, device=dev)
    run = prepare_cake_mla_nvfp4_paged_decode(*qq, c.ckv_cache, c.ckv_sf_cache, c.kpe_cache, c.page_table, c.kv_lens,
                                              out, ws, sm_scale=SM_SCALE, ckv_scale=c.ckv_scale, max_seq_len=max_len)
    return lambda: run.launch().squeeze(1)


def make(name, c, inp, max_len):
    if name.startswith("nvfp4/") or name == "nvfp4":
        return ours(name, c, max_len)
    if name == "cake":
        return cake(c, max_len)
    if name == "nvfp4+fa2":
        return fa2_on_dequantized(c, lambda: nvfp4_mla.dequantize(c.ckv_cache, c.ckv_sf_cache, c.kpe_cache))
    return baselines.make(name, inp)


def kv_format(name):
    return "nvfp4" if name.startswith(("nvfp4", "cake")) else "fp8" if name.endswith("fp8") else "bf16"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="1,4,16,64")
    ap.add_argument("--s-kv", default="1024,4096,16384,65536")
    ap.add_argument("--heads", default="16,128")
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--providers", default="nvfp4,triton:fp8,fa2,trtllm:fp8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--bw", type=float, help="achievable read BW in GB/s for the %%BW column (see roofline.py)")
    ap.add_argument("--csv-dir", help="also write bench_<gpu>_<time>.csv here")
    a = ap.parse_args()
    ints = lambda s: [int(x) for x in s.split(",")]
    providers = a.providers.split(",")
    name = torch.cuda.get_device_name()
    peak = a.bw or KNOWN_BW.get(name)
    print(f"{name}  read BW {peak or '?'} GB/s  page_size={a.page_size}", flush=True)

    rows, reported = [], set()
    hdr = f"{'H':>4} {'B':>3} {'s_kv':>6} |" + "".join(f" {p + ' ms/GBs/%':>24}" for p in providers)
    print(hdr + "\n" + "-" * len(hdr), flush=True)
    for H in ints(a.heads):
        for B in ints(a.batch):
            for S in ints(a.s_kv):
                c = RandomCache([S] * B, a.page_size, H)
                need_bf16 = any(kv_format(p) != "nvfp4" for p in providers)
                inp = baselines.Inputs(B, S, a.page_size, H) if need_bf16 else None
                line = f"{H:>4} {B:>3} {S:>6} |"
                for p in providers:
                    try:
                        ms = baselines.gpu_ms(make(p, c, inp, S), iters=a.iters)
                    except Exception as e:
                        ms = float("nan")
                        if (p, H) not in reported:
                            reported.add((p, H))
                            print(f"  [{p} H={H}] {baselines.first_line(e)}", flush=True)
                    gbs = B * S * BYTES[kv_format(p)] / (ms * 1e-3) / 1e9
                    pct = 100 * gbs / peak if peak else float("nan")
                    rows.append(dict(gpu=name, provider=p, heads=H, batch=B, s_kv=S, page_size=a.page_size,
                                     ms=ms, gbps=gbs, pct_bw=pct))
                    line += f" {ms:9.3f}/{gbs:6.0f}/{pct:4.0f}%" if ms == ms else f" {'--':>24}"
                print(line, flush=True)
                del c, inp
                torch.cuda.empty_cache()
    if a.csv_dir:
        d = pathlib.Path(a.csv_dir)
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"bench_{name.replace(' ', '_')}_{time.strftime('%Y%m%d-%H%M')}.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
