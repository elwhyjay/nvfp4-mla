"""Test and benchmark helpers: random NVFP4 MLA caches written by FlashInfer, and an fp32 PyTorch reference.

Needs ``flashinfer-python >= 0.7.0``. ``python -m nvfp4_mla.testing`` prints the NVFP4 error against a BF16 cache.
"""
import math

import flashinfer
import torch

CKV, KPE = 512, 64
SM_SCALE = (128 + KPE) ** -0.5  # DeepSeek: qk_nope_head_dim 128 + qk_rope_head_dim 64
_E2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]


class RandomCache:
    """Random q and the same random KV in a BF16 and an NVFP4 paged cache.

    Physical pages are shuffled and unused slots hold random bytes (NaN scales included), so a kernel that reads
    past ``kv_lens`` or ignores the page table produces NaN or garbage.
    """

    def __init__(self, kv_lens, page_size, heads, ckv_scale=1.0, kpe_scale=1.0, seed=0, device="cuda"):
        g = torch.Generator(device=device).manual_seed(seed)
        B, ps = len(kv_lens), page_size
        n_pages = [math.ceil(L / ps) for L in kv_lens]
        P, nnz = sum(n_pages), sum(kv_lens)
        self.kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
        self.page_size, self.ckv_scale, self.kpe_scale = ps, ckv_scale, kpe_scale

        kv_indices = torch.randperm(P, generator=g, device=device).int()
        kv_indptr = torch.tensor([0] + n_pages, device=device).cumsum(0).int()
        self.page_table = torch.zeros(B, max(n_pages), dtype=torch.int32, device=device)
        for b in range(B):
            self.page_table[b, :n_pages[b]] = kv_indices[kv_indptr[b]:kv_indptr[b + 1]]
        last_len = torch.tensor([L - (n - 1) * ps for L, n in zip(kv_lens, n_pages)], dtype=torch.int32,
                                device=device)
        append_indptr = torch.tensor([0] + list(kv_lens), device=device).cumsum(0).int()
        bi, pos = flashinfer.get_batch_indices_positions(append_indptr, self.kv_lens, nnz)

        rn = lambda *s: torch.randn(*s, generator=g, device=device, dtype=torch.bfloat16)
        self.q_nope, self.q_pe = rn(B, heads, CKV), rn(B, heads, KPE)
        ckv, kpe = rn(nnz, CKV), rn(nnz, KPE)
        self.ckv_bf16 = torch.zeros(P, ps, CKV, dtype=torch.bfloat16, device=device)
        self.kpe_bf16 = torch.zeros(P, ps, KPE, dtype=torch.bfloat16, device=device)
        flashinfer.append_paged_mla_kv_cache(ckv, kpe, bi, pos, self.ckv_bf16, self.kpe_bf16,
                                             kv_indices, kv_indptr, last_len)
        junk = lambda n: torch.randint(0, 256, (P, ps, n), generator=g, device=device, dtype=torch.uint8)
        self.ckv_cache = junk(CKV // 2)
        self.ckv_sf_cache = junk(CKV // 16).view(torch.float8_e4m3fn)
        self.kpe_cache = junk(KPE).view(torch.float8_e4m3fn)
        flashinfer.nvfp4_quantize_append_paged_mla_kv_cache(
            ckv, kpe, bi, pos, self.ckv_cache, self.ckv_sf_cache, self.kpe_cache, kv_indices, kv_indptr, last_len,
            ckv_scale, kpe_scale)

    def args(self, sm_scale=SM_SCALE):
        """Positional arguments of ``nvfp4_mla.decode``."""
        return (self.q_nope, self.q_pe, self.ckv_cache, self.ckv_sf_cache, self.kpe_cache, self.page_table,
                self.kv_lens, sm_scale, self.ckv_scale, self.kpe_scale)

    def dequantize(self):
        return dequantize(self.ckv_cache, self.ckv_sf_cache, self.kpe_cache, self.ckv_scale, self.kpe_scale)

    def reference(self, sm_scale=SM_SCALE):
        """fp32 attention over the dequantized NVFP4 cache: what the kernel should produce."""
        return attention(self.q_nope, self.q_pe, *self.dequantize(), self.page_table, self.kv_lens, sm_scale)

    def reference_bf16(self, sm_scale=SM_SCALE):
        """fp32 attention over the unquantized BF16 cache: the quantization error baseline."""
        return attention(self.q_nope, self.q_pe, self.ckv_bf16, self.kpe_bf16, self.page_table, self.kv_lens,
                         sm_scale)


def dequantize(ckv_cache, ckv_sf_cache, kpe_cache, ckv_scale=1.0, kpe_scale=1.0):
    """fp32 ``[pages, page_size, 512]``, ``[pages, page_size, 64]``: e2m1 * block_scale * ckv_scale."""
    e2m1 = torch.tensor(_E2M1, device=ckv_cache.device)
    codes = torch.stack([ckv_cache & 15, ckv_cache >> 4], -1).flatten(-2).long()  # low nibble = even channel
    ckv = e2m1[codes] * ckv_sf_cache.float().repeat_interleave(16, -1) * ckv_scale
    return ckv, kpe_cache.float() * kpe_scale


def attention(q_nope, q_pe, ckv, kpe, page_table, kv_lens, sm_scale=SM_SCALE):
    """fp32 absorbed-MLA decode (q_len = 1): out[b, h] = softmax(q . [ckv | kpe]^T * sm_scale) @ ckv."""
    ps, out = ckv.shape[1], []
    for b, L in enumerate(kv_lens.tolist()):
        pages = page_table[b, :math.ceil(L / ps)]
        kc, kp = ckv[pages].reshape(-1, CKV)[:L].float(), kpe[pages].reshape(-1, KPE)[:L].float()
        s = (q_nope[b].float() @ kc.T + q_pe[b].float() @ kp.T) * sm_scale
        out.append(s.softmax(-1) @ kc)
    return torch.stack(out)


def fa2_on_dequantized(c: RandomCache, dequant=None, sm_scale=SM_SCALE):
    """Dequant-then-kernel: returns fn() = NVFP4 -> BF16 cache -> FlashInfer fa2 (planned up front).

    ``dequant() -> (ckv, kpe)`` in BF16; defaults to the PyTorch dequant with NaN padding slots zeroed. BF16 is
    lossless for power-of-two ``ckv_scale`` (E2M1 x E4M3 fits in 6 significant bits).
    """
    dequant = dequant or (lambda: [x.nan_to_num().bfloat16() for x in c.dequantize()])
    B, H = c.q_nope.shape[:2]
    dev = c.q_nope.device
    w = flashinfer.mla.BatchMLAPagedAttentionWrapper(torch.zeros(128 << 20, dtype=torch.int8, device=dev),
                                                     backend="fa2")
    md = flashinfer.mla.MLAPlanMetadata.dense(cum_seq_lens_q=torch.arange(B + 1, dtype=torch.int32, device=dev),
                                              block_tables=c.page_table, seq_lens=c.kv_lens)
    w.plan(metadata=md, num_heads=H, head_dim_ckv=CKV, head_dim_kpe=KPE, page_size=c.page_size, causal=False,
           sm_scale=sm_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, query_layout="split",
           kv_cache_layout="split")

    def fn():
        ckv, kpe = dequant()
        return w.run(query=(c.q_nope, c.q_pe), kv_cache=(ckv, kpe)).reshape(B, H, CKV)

    return fn


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


if __name__ == "__main__":
    for H in (16, 128):
        c = RandomCache([1, 100, 4096, 5000], page_size=64, heads=H)
        r = c.reference()
        print(f"H={H:3d}  rel-L2 NVFP4 vs BF16 cache = {rel_l2(r, c.reference_bf16()):.4f}   "
              f"fa2 on the dequantized cache vs fp32 reference = {rel_l2(fa2_on_dequantized(c)(), r):.2e}")
