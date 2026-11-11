"""MLA decode baselines (BF16 / FP8 KV) on the same random inputs, and the GPU timer used by bench.py.

  python benchmarks/baselines.py      # cosine similarity of each baseline against FlashInfer fa2
"""
import functools
import math
import statistics

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
from torch.profiler import ProfilerActivity, profile, record_function

import triton_mla

CKV, KPE, QK_NOPE = 512, 64, 128
SM_SCALE = (QK_NOPE + KPE) ** -0.5
PROVIDERS = ["fa2", "cutlass", "cutile", "cute-dsl-monolithic", "cute-dsl-modular", "trtllm:bf16", "trtllm:fp8",
             "triton:bf16", "triton:fp8", "b12x:bf16", "b12x:fp8"]
PACKED = ("cutlass", "cute-dsl-monolithic", "cute-dsl-modular")  # these take [.., 576] query and KV


@functools.lru_cache(maxsize=1)
def workspace():
    return torch.zeros(128 << 20, dtype=torch.int8, device="cuda")


class Inputs:
    """Random BF16 q and paged KV, every request of length s_kv."""

    def __init__(self, batch, s_kv, page_size, heads, device="cuda"):
        self.batch, self.s_kv, self.page_size, self.heads = batch, s_kv, page_size, heads
        pages = math.ceil(s_kv / page_size)
        total = batch * pages
        self.q_nope = torch.randn(batch, heads, CKV, dtype=torch.bfloat16, device=device)
        self.q_pe = torch.randn(batch, heads, KPE, dtype=torch.bfloat16, device=device)
        self.ckv = torch.randn(total, page_size, CKV, dtype=torch.bfloat16, device=device)
        self.kpe = torch.randn(total, page_size, KPE, dtype=torch.bfloat16, device=device)
        self.kv_lens = torch.full((batch,), s_kv, dtype=torch.int32, device=device)
        self.page_table = torch.arange(total, dtype=torch.int32, device=device).reshape(batch, pages)


def triton(inp, variant):
    """SGLang Triton MLA with SGLang's default split policy. fp8 also casts q to fp8 for the nope QK dot."""
    B, H, ps, dev = inp.batch, inp.heads, inp.page_size, inp.q_nope.device
    q = torch.cat([inp.q_nope, inp.q_pe], -1).contiguous()
    kv = torch.cat([inp.ckv, inp.kpe], -1).to({"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[variant])
    kv = kv.reshape(-1, 1, CKV + KPE)
    # token slots of the paged layout, addressed with PAGE_SIZE=1 (the SGLang Triton backend default)
    slots = (inp.page_table[:, :, None] * ps + torch.arange(ps, device=dev)).reshape(B, -1)[:, :inp.s_kv]
    kv_indices = slots.reshape(-1).int().contiguous()
    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device=dev) * inp.s_kv
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    max_splits = triton_mla.mla_max_kv_splits(sms, max_context_len=163840)  # DeepSeek-V3 context length
    splits = triton_mla.mla_num_kv_splits(inp.kv_lens, H, max_splits, sms)
    logits = torch.empty(B, H, max_splits, CKV, dtype=torch.float32, device=dev)
    lse = torch.empty(B, H, max_splits, dtype=torch.float32, device=dev)
    o = torch.empty(B, H, CKV, dtype=torch.bfloat16, device=dev)
    pdl = torch.cuda.get_device_capability(dev)[0] >= 9  # SGLang's is_arch_support_pdl
    fwd = lambda: triton_mla.mla_decode_fwd(q, kv, o, kv_indptr, kv_indices, logits, lse, splits, max_splits,
                                            SM_SCALE, 1.0, 1.0, use_pdl=pdl)
    return lambda: (fwd(), o)[1]


def wrapper(inp, backend):
    """FlashInfer BatchMLAPagedAttentionWrapper, BF16 KV. fa2 runs anywhere; cutlass needs SM100 and 128 heads; cutile
    (cuTile) runs on SM100 and SM12x; the cute-dsl backends (CuTe DSL) target SM100."""
    packed = backend in PACKED
    layout = "packed" if packed else "split"
    w = BatchMLAPagedAttentionWrapper(workspace(), backend=backend)
    md = MLAPlanMetadata.dense(cum_seq_lens_q=torch.arange(inp.batch + 1, dtype=torch.int32, device="cuda"),
                               block_tables=inp.page_table, seq_lens=inp.kv_lens)
    w.plan(metadata=md, num_heads=inp.heads, head_dim_ckv=CKV, head_dim_kpe=KPE, page_size=inp.page_size,
           causal=False, sm_scale=SM_SCALE, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
           query_layout=layout, kv_cache_layout=layout)
    if packed:
        q = torch.cat([inp.q_nope, inp.q_pe], -1).contiguous()
        kv = torch.cat([inp.ckv, inp.kpe], -1).contiguous()
        out = torch.empty(inp.batch, inp.heads, CKV, dtype=torch.bfloat16, device="cuda")  # cute-dsl needs it
        return lambda: w.run(query=q, kv_cache=kv, out=out)
    return lambda: w.run(query=(inp.q_nope, inp.q_pe), kv_cache=(inp.ckv, inp.kpe))


def trtllm(inp, variant):
    """trtllm-gen on SM100, XQA on SM120 (XQA needs 128 heads). Named explicitly: since FlashInfer 0.7.3
    backend="auto" may pick fa2 on SM12x."""
    dt = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[variant]
    backend = "xqa" if torch.cuda.get_device_capability()[0] == 12 else "trtllm-gen"
    q = torch.cat([inp.q_nope, inp.q_pe], -1).unsqueeze(1).to(dt).contiguous()   # [B, 1, H, 576]
    kv = torch.cat([inp.ckv, inp.kpe], -1).unsqueeze(1).to(dt).contiguous()       # [P, 1, ps, 576]
    return lambda: trtllm_batch_decode_with_kv_cache_mla(
        query=q, kv_cache=kv, workspace_buffer=workspace(), qk_nope_head_dim=QK_NOPE, kv_lora_rank=CKV,
        qk_rope_head_dim=KPE, block_tables=inp.page_table, seq_lens=inp.kv_lens, max_seq_len=inp.s_kv,
        bmm1_scale=SM_SCALE, bmm2_scale=1.0, backend=backend)


def b12x(inp, variant):
    """FlashInfer's experimental b12x dense MLA (SM12x only), BF16 or FP8 (E4M3) KV in the combined 576-wide record.
    With an FP8 cache the BF16 query goes through b12x's static FP8 quantizer (inside the timed call) with a per-call
    scale, amax / 448, computed beforehand. Its only tuning knob is the KV split count; b12x's own autotuner needs
    compiler worker processes that do not see the GPU on Modal, so the same candidates (powers of two up to one split
    per 64 tokens, and its default) are timed here and the fastest is used."""
    cap = max(1, (inp.s_kv + 63) // 64)
    best = None
    for splits in [None] + [1 << k for k in range(8) if 1 << k <= cap]:
        f, close = _b12x_call(inp, variant, splits)
        t = gpu_ms(f, iters=5)
        if best is None or t < best[0]:
            best, close = (t, f, close), best and best[2]
        if close:  # a candidate that lost: close its session as b12x's tests do, not left to the garbage collector
            close()
    return best[1]


def _b12x_call(inp, variant, max_splits):
    from b12x.attention import dense_mla
    from b12x.preparation import PreparationSession, PreparedCall

    dt = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[variant]
    B, H, dev = inp.batch, inp.heads, inp.q_nope.device
    q = torch.cat([inp.q_nope, inp.q_pe], -1).contiguous()
    kv = torch.cat([inp.ckv, inp.kpe], -1).to(dt).contiguous()
    kw = dict(q=q, kv_cache=kv, output=torch.empty(B, H, CKV, dtype=torch.bfloat16, device=dev),
              page_table=inp.page_table, cache_seqlens=inp.kv_lens,
              cu_seqlens_q=torch.arange(B + 1, dtype=torch.int32, device=dev),
              kv_scale=torch.ones((), device=dev) if variant == "fp8" else None,
              q_scale=(q.abs().amax().float() / 448).reshape(()) if variant == "fp8" else None)
    plan = dense_mla.plan(dense_mla.Caps(
        device=dev, mode="decode", kv_dtype=dt, q_dtype=torch.bfloat16, num_q_heads=H, page_size=inp.page_size,
        max_total_q=B, max_batch=B, max_cache_tokens=inp.s_kv, max_page_table_width=inp.page_table.shape[1],
        num_cache_pages=kv.shape[0]), override=None if max_splits is None else dense_mla.DenseMlaConfig(max_splits=max_splits))

    def call(state):  # b12x's prepared-call lifecycle: scratch, bind, prime
        (spec,) = state.scratch_specs()
        scr = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(scratch=scr, sm_scale=SM_SCALE, **kw)
        state.prime(binding)
        return PreparedCall(run=lambda: state.run(binding), output=kw["output"], owners=(scr, binding))

    session = PreparationSession(device=dev, autotune=False)
    result = session.prepare((plan.request(name="bench", prepare_call=call),))
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = dense_mla.bind(plan, scratch=scratch, sm_scale=SM_SCALE, **kw)
    keep = (session, result, scratch)

    def close():
        torch.cuda.synchronize()
        result.close()
        session.close()

    return (lambda: (keep, dense_mla.run(binding=binding)[0])[1]), close


def make(provider, inp):
    kind, _, variant = provider.partition(":")
    if kind in ("fa2", "cutile") or kind in PACKED:
        return wrapper(inp, kind)
    if kind == "trtllm":
        return trtllm(inp, variant)
    if kind == "triton":
        return triton(inp, variant)
    if kind == "b12x":
        return b12x(inp, variant)
    raise ValueError(f"unknown provider {provider!r}; one of {PROVIDERS}")


def gpu_ms(fn, iters=20):
    """Median per-call sum of GPU kernel durations (CUPTI via torch.profiler), L2 flushed before each call.

    Eager CUDA-event timing would count host launch time instead: FlashInfer's trtllm-gen wrapper spends ~135 us of
    host time per call, more than its kernel at small batch.
    """
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


def first_line(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:110] if str(e) else ''}"


if __name__ == "__main__":
    print(torch.cuda.get_device_name())
    inp = Inputs(4, 2048, 64, 128)
    ref = make("fa2", inp)().float().flatten()
    for p in PROVIDERS[1:]:
        try:
            out = make(p, inp)().reshape(4, 128, CKV).float().flatten()
            print(f"  cos({p} vs fa2) = {torch.nn.functional.cosine_similarity(out, ref, dim=0):.5f}")
        except Exception as e:
            print(f"  {p}: {first_line(e)}")
