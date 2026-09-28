"""MLA decode baselines via FlashInfer 0.7.0 (bf16 + fp8), DeepSeek geometry (ckv 512 + kpe 64), q_len=1.

  uv run modal run bench.py::sh --cmd "python baselines.py"             # RTX PRO 6000
  GPU=B200 uv run modal run bench.py::sh --cmd "python baselines.py"    # B200
Raw log -> out/logs/, CSV -> out/results/ (pulled automatically by bench.py::sh)
"""
import argparse
import csv
import math
import pathlib
import statistics

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
from torch.profiler import ProfilerActivity, profile, record_function

import triton_mla

CKV, KPE, QK_NOPE = 512, 64, 128
SM_SCALE = (QK_NOPE + KPE) ** -0.5
BYTES_PER_TOKEN = {"bf16": (CKV + KPE) * 2, "fp8": CKV + KPE}
DEV = torch.device("cuda")
WS = torch.zeros(128 << 20, dtype=torch.int8, device=DEV)


class Inputs:
    def __init__(self, batch, s_kv, page_size, heads):
        self.batch, self.s_kv, self.page_size, self.heads = batch, s_kv, page_size, heads
        pages = math.ceil(s_kv / page_size)
        total = batch * pages
        self.q_nope = torch.randn(batch, heads, CKV, dtype=torch.bfloat16, device=DEV)
        self.q_pe = torch.randn(batch, heads, KPE, dtype=torch.bfloat16, device=DEV)
        self.ckv = torch.randn(total, page_size, CKV, dtype=torch.bfloat16, device=DEV)
        self.kpe = torch.randn(total, page_size, KPE, dtype=torch.bfloat16, device=DEV)
        self.kv_lens = torch.full((batch,), s_kv, dtype=torch.int32, device=DEV)
        self.page_table = torch.arange(total, dtype=torch.int32, device=DEV).reshape(batch, pages)


def triton(inp, variant):
    """SGLang Triton MLA with SGLang's default split policy. fp8 also casts q to fp8 for the nope QK dot."""
    B, H, ps = inp.batch, inp.heads, inp.page_size
    q = torch.cat([inp.q_nope, inp.q_pe], -1).contiguous()
    kv = torch.cat([inp.ckv, inp.kpe], -1).to({"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[variant])
    kv = kv.reshape(-1, 1, CKV + KPE)
    # token slots of our paged layout; addressed with PAGE_SIZE=1, the SGLang Triton backend default
    slots = (inp.page_table[:, :, None] * ps + torch.arange(ps, device=DEV)).reshape(B, -1)[:, :inp.s_kv]
    kv_indices = slots.reshape(-1).int().contiguous()
    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device=DEV) * inp.s_kv
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    max_splits = triton_mla.mla_max_kv_splits(sms, max_context_len=163840)  # DeepSeek-V3 context length
    splits = triton_mla.mla_num_kv_splits(inp.kv_lens, H, max_splits, sms)
    logits = torch.empty(B, H, max_splits, CKV, dtype=torch.float32, device=DEV)
    lse = torch.empty(B, H, max_splits, dtype=torch.float32, device=DEV)
    o = torch.empty(B, H, CKV, dtype=torch.bfloat16, device=DEV)
    pdl = torch.cuda.get_device_capability()[0] >= 9  # is_arch_support_pdl
    fwd = lambda: triton_mla.mla_decode_fwd(q, kv, o, kv_indptr, kv_indices, logits, lse, splits, max_splits,
                                            SM_SCALE, 1.0, 1.0, use_pdl=pdl)
    return lambda: (fwd(), o)[1]


def make(provider, inp):
    kind, variant = provider.split(":")
    if kind == "triton":
        return triton(inp, variant)
    if kind == "wrapper":  # fa2: any arch (JIT) / cutlass: SM100, H=128 / cutile: needs cuda.tile
        packed = variant == "cutlass"  # cutlass requires packed [B,H,576] / [P,ps,576]
        layout = "packed" if packed else "split"
        w = BatchMLAPagedAttentionWrapper(WS, backend=variant)
        md = MLAPlanMetadata.dense(
            cum_seq_lens_q=torch.arange(inp.batch + 1, dtype=torch.int32, device=DEV),
            block_tables=inp.page_table, seq_lens=inp.kv_lens)
        w.plan(metadata=md, num_heads=inp.heads, head_dim_ckv=CKV, head_dim_kpe=KPE,
               page_size=inp.page_size, causal=False, sm_scale=SM_SCALE,
               q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
               query_layout=layout, kv_cache_layout=layout)
        if packed:
            q = torch.cat([inp.q_nope, inp.q_pe], -1).contiguous()
            kv = torch.cat([inp.ckv, inp.kpe], -1).contiguous()
            return lambda: w.run(query=q, kv_cache=kv)
        return lambda: w.run(query=(inp.q_nope, inp.q_pe), kv_cache=(inp.ckv, inp.kpe))
    # dense:* -> backend="auto": trtllm-gen on SM100, XQA on SM120 (XQA: H=128, q_len=1 only)
    dt = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[variant]
    q = torch.cat([inp.q_nope, inp.q_pe], -1).unsqueeze(1).to(dt).contiguous()   # [B,1,H,576]
    kv = torch.cat([inp.ckv, inp.kpe], -1).unsqueeze(1).to(dt).contiguous()       # [P,1,ps,576]
    return lambda: trtllm_batch_decode_with_kv_cache_mla(
        query=q, kv_cache=kv, workspace_buffer=WS, qk_nope_head_dim=QK_NOPE,
        kv_lora_rank=CKV, qk_rope_head_dim=KPE, block_tables=inp.page_table,
        seq_lens=inp.kv_lens, max_seq_len=inp.s_kv, bmm1_scale=SM_SCALE, bmm2_scale=1.0)


def gpu_ms(fn, iters=20):
    """Median per-call sum of GPU kernel durations (CUPTI via torch.profiler), L2 flushed before each call.
    Eager event timing would count host launch time instead: FlashInfer's trtllm-gen wrapper spends ~135 us/call."""
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.int8, device="cuda")
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            flush.zero_()
            with record_function("fn"):
                fn()
        torch.cuda.synchronize()
    gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    per_call = [sum(k.device_time for k in gpu if k.name != "fn" and a.time_range.start <= k.time_range.start
                    and k.time_range.end <= a.time_range.end)  # the GPU-side "fn" range; gaps not counted
                for a in gpu if a.name == "fn"]
    return statistics.median(per_call) / 1e3


def bench_ms(provider, inp, iters):
    return gpu_ms(make(provider, inp), iters)


def first_line(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:110] if str(e) else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="1,4,16,64")
    ap.add_argument("--s-kv", default="1024,4096,16384,65536")
    ap.add_argument("--heads", default="16,128")
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--providers", default="wrapper:fa2,wrapper:cutlass,dense:bf16,dense:fp8")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--csv-dir", default="/root/.cache/results")
    a = ap.parse_args()
    ints = lambda s: [int(x) for x in s.split(",")]
    providers = a.providers.split(",")
    gpu = torch.cuda.get_device_name().replace(" ", "_")
    cc = torch.cuda.get_device_capability()
    print(f"{gpu} CC{cc[0]}.{cc[1]} page_size={a.page_size} q_len=1 iters={a.iters}", flush=True)

    # sanity: cosine similarity vs fa2 at one config (also triggers JIT once)
    inp = Inputs(4, 2048, a.page_size, 128)
    ref = make("wrapper:fa2", inp)().reshape(4, 128, CKV).float().flatten()
    for p in providers:
        try:
            out = make(p, inp)().reshape(4, 128, CKV).float().flatten()
            print(f"  cos({p} vs fa2) = {torch.nn.functional.cosine_similarity(out, ref, dim=0):.5f}")
        except Exception as e:
            print(f"  {p}: {first_line(e)}")
    del inp, ref

    rows, reported = [], set()
    hdr = f"{'H':>4} {'batch':>5} {'s_kv':>6} |" + "".join(f" {p + ' ms/GBs':>16}" for p in providers)
    print(hdr, flush=True)
    print("-" * len(hdr))
    for H in ints(a.heads):
        for B in ints(a.batch):
            for S in ints(a.s_kv):
                inp = Inputs(B, S, a.page_size, H)
                line = f"{H:>4} {B:>5} {S:>6} |"
                for p in providers:
                    try:
                        ms = bench_ms(p, inp, a.iters)
                    except Exception as e:
                        ms = float("nan")
                        if (p, H) not in reported:
                            reported.add((p, H))
                            print(f"  [{p} H={H}] {first_line(e)}", flush=True)
                    fmt = "fp8" if p.endswith("fp8") else "bf16"
                    gbs = B * S * BYTES_PER_TOKEN[fmt] / (ms * 1e-3) / 1e9 if ms == ms else float("nan")
                    rows.append(dict(gpu=gpu, provider=p, heads=H, batch=B, s_kv=S,
                                     page_size=a.page_size, ms=ms, gbps=gbs))
                    line += f" {ms:9.3f}/{gbs:6.0f}" if ms == ms else f" {'--':>16}"
                print(line, flush=True)
                del inp
                torch.cuda.empty_cache()
    d = pathlib.Path(a.csv_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"baselines_{gpu}.csv"
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
