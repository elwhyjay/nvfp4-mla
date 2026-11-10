"""v0: NVFP4 MLA paged cache (flashinfer#4676 layout) -> PyTorch dequant -> fp32 reference attention.

  uv run modal run bench.py::sh --cmd "python ref.py"
Prints rel-L2 of the NVFP4 path vs the unquantized BF16 cache, and of fa2-on-dequantized-cache vs the fp32 reference.
"""
import math

import flashinfer
import torch

CKV, KPE = 512, 64
SM_SCALE = (128 + KPE) ** -0.5
DEV = torch.device("cuda")
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=DEV)


class Cache:
    """Random q + paged KV in BF16 and NVFP4. Physical pages are shuffled and unused slots hold random bytes
    (NaN scales included), so a kernel that reads past kv_len or ignores the page table produces NaN/garbage."""

    def __init__(self, kv_lens, page_size, heads, ckv_scale=1.0, kpe_scale=1.0, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        B, ps = len(kv_lens), page_size
        n_pages = [math.ceil(L / ps) for L in kv_lens]
        P, nnz = sum(n_pages), sum(kv_lens)
        self.kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=DEV)
        self.page_size, self.ckv_scale, self.kpe_scale = ps, ckv_scale, kpe_scale

        kv_indices = torch.randperm(P, generator=g, device=DEV).int()
        kv_indptr = torch.tensor([0] + n_pages, device=DEV).cumsum(0).int()
        self.page_table = torch.zeros(B, max(n_pages), dtype=torch.int32, device=DEV)
        for b in range(B):
            self.page_table[b, :n_pages[b]] = kv_indices[kv_indptr[b]:kv_indptr[b + 1]]
        last_len = torch.tensor([L - (n - 1) * ps for L, n in zip(kv_lens, n_pages)], dtype=torch.int32, device=DEV)
        append_indptr = torch.tensor([0] + list(kv_lens), device=DEV).cumsum(0).int()
        bi, pos = flashinfer.get_batch_indices_positions(append_indptr, self.kv_lens, nnz)

        rn = lambda *s: torch.randn(*s, generator=g, device=DEV, dtype=torch.bfloat16)
        self.q_nope, self.q_pe = rn(B, heads, CKV), rn(B, heads, KPE)
        ckv, kpe = rn(nnz, CKV), rn(nnz, KPE)
        self.ckv_bf16 = torch.zeros(P, ps, CKV, dtype=torch.bfloat16, device=DEV)
        self.kpe_bf16 = torch.zeros(P, ps, KPE, dtype=torch.bfloat16, device=DEV)
        flashinfer.append_paged_mla_kv_cache(ckv, kpe, bi, pos, self.ckv_bf16, self.kpe_bf16,
                                             kv_indices, kv_indptr, last_len)
        junk = lambda n: torch.randint(0, 256, (P, ps, n), generator=g, device=DEV, dtype=torch.uint8)
        self.ckv, self.sf, self.kpe = junk(CKV // 2), junk(CKV // 16).view(torch.float8_e4m3fn), junk(KPE).view(torch.float8_e4m3fn)
        flashinfer.nvfp4_quantize_append_paged_mla_kv_cache(ckv, kpe, bi, pos, self.ckv, self.sf, self.kpe,
                                                            kv_indices, kv_indptr, last_len, ckv_scale, kpe_scale)

    def dequant(self):
        """fp32 [P, ps, 512], [P, ps, 64]: e2m1 * block_scale * ckv_scale (low nibble = even channel)."""
        codes = torch.stack([self.ckv & 15, self.ckv >> 4], -1).flatten(-2).long()
        ckv = E2M1[codes] * self.sf.float().repeat_interleave(16, -1) * self.ckv_scale
        return ckv, self.kpe.float() * self.kpe_scale


def attention(q_nope, q_pe, ckv, kpe, page_table, kv_lens, sm_scale=SM_SCALE):
    """fp32 absorbed-MLA decode (q_len=1): out[b,h] = softmax(q·[ckv|kpe]^T * scale) @ ckv."""
    ps, out = ckv.shape[1], []
    for b, L in enumerate(kv_lens.tolist()):
        pages = page_table[b, :math.ceil(L / ps)]
        kc, kp = ckv[pages].reshape(-1, CKV)[:L].float(), kpe[pages].reshape(-1, KPE)[:L].float()
        s = (q_nope[b].float() @ kc.T + q_pe[b].float() @ kp.T) * sm_scale
        out.append(s.softmax(-1) @ kc)
    return torch.stack(out)


def ref_fp4(c: Cache):
    return attention(c.q_nope, c.q_pe, *c.dequant(), c.page_table, c.kv_lens)


def ref_bf16(c: Cache):
    return attention(c.q_nope, c.q_pe, c.ckv_bf16, c.kpe_bf16, c.page_table, c.kv_lens)


def fa2_dequant(c: Cache, dequant=None):
    """Dequant-then-kernel: returns fn() = NVFP4 -> BF16 cache -> FlashInfer fa2 (plan done up front).
    dequant() -> (ckv, kpe) in BF16; defaults to the PyTorch dequant (NaN padding slots zeroed).
    Lossless in BF16 for power-of-two ckv_scale (E2M1 x E4M3 fits in 6 significant bits)."""
    dequant = dequant or (lambda: [x.nan_to_num().bfloat16() for x in c.dequant()])
    B, H = c.q_nope.shape[:2]
    ws = torch.zeros(128 << 20, dtype=torch.int8, device=DEV)
    w = flashinfer.mla.BatchMLAPagedAttentionWrapper(ws, backend="fa2")
    md = flashinfer.mla.MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.arange(B + 1, dtype=torch.int32, device=DEV),
        block_tables=c.page_table, seq_lens=c.kv_lens)
    w.plan(metadata=md, num_heads=H, head_dim_ckv=CKV, head_dim_kpe=KPE, page_size=c.page_size,
           causal=False, sm_scale=SM_SCALE, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
           query_layout="split", kv_cache_layout="split")
    def fn():
        ckv, kpe = dequant()
        return w.run(query=(c.q_nope, c.q_pe), kv_cache=(ckv, kpe)).reshape(B, H, CKV)
    return fn


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


if __name__ == "__main__":
    for H in (16, 128):
        c = Cache([1, 100, 4096, 5000], page_size=64, heads=H)
        r = ref_fp4(c)
        print(f"H={H:3d}  rel-L2 nvfp4 vs bf16 = {rel_l2(r, ref_bf16(c)):.4f}   "
              f"fa2(dequant) vs fp32 ref = {rel_l2(fa2_dequant(c)(), r):.2e}")
