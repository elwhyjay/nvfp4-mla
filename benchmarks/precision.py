"""Error of MLA decode precision options on an NVFP4 cache, simulated on the CPU with random data.

  python benchmarks/precision.py

Every option reads the same NVFP4 cache (flashinfer#4676 rounding: E4M3 scale = amax / 6 per 16 channels, E2M1 codes
round to nearest). The error is relative L2 against fp32 attention over the unquantized BF16 cache, so it includes
the format's own error; the second column is against attention over the dequantized cache (the kernel's own error).
"""
import torch

F8 = torch.float8_e4m3fn
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
SM_SCALE = (128 + 64) ** -0.5


def e2m1(x):
    i = (x.abs().clamp(max=6).unsqueeze(-1) - E2M1).abs().argmin(-1)
    return E2M1[i] * x.sign()


def nvfp4(x):
    b = x.reshape(*x.shape[:-1], -1, 16)
    sf = (b.abs().amax(-1, keepdim=True) / 6).to(F8).float().clamp(min=1e-30)
    return (e2m1(b / sf) * sf).reshape(x.shape)


def q_nvfp4(q):
    """FlashInfer Cake's query: a per-row FP32 scale amax / (6 * 448), then per-16 NVFP4."""
    g = q.abs().amax(-1, keepdim=True) / (6 * 448)
    return nvfp4(q / g) * g


def fp8_rows(x):
    s = x.abs().amax(-1, keepdim=True) / 448
    return (x / s).to(F8).float() * s


def terms(q, quant, k):
    """q as k residual terms, each quantized with quant: the terms accumulate into one S on the tensor cores."""
    out = torch.zeros_like(q)
    for _ in range(k):
        out = out + quant(q - out)
    return out


def attend(q_nope, q_pe, ckv, kpe, p_fp8=False):
    p = torch.softmax((q_nope @ ckv.T + q_pe @ kpe.T) * SM_SCALE, -1)
    p = p / p.amax(-1, keepdim=True)  # P relative to the running max, as in the kernel
    p = (p * 448).to(F8).float() / 448 if p_fp8 else p.bfloat16().float()
    return (p @ ckv) / p.sum(-1, keepdim=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    for H, S in [(16, 4096), (128, 4096), (16, 16384)]:
        qn, qp = torch.randn(H, 512), torch.randn(H, 64)
        ckv, kpe = torch.randn(S, 512).bfloat16().float(), torch.randn(S, 64).bfloat16().float()
        ref = torch.softmax((qn @ ckv.T + qp @ kpe.T) * SM_SCALE, -1) @ ckv
        ck, kp = nvfp4(ckv), kpe.to(F8).float()
        deq = torch.softmax((qn @ ck.T + qp @ kp.T) * SM_SCALE, -1) @ ck
        options = {
            "BF16 Q, BF16 P (v5)": attend(qn.bfloat16().float(), qp.bfloat16().float(), ck, kp),
            "FP8 Q, BF16 P": attend(fp8_rows(qn), fp8_rows(qp), ck, kp),
            "FP8 Q, FP8 K/V, FP8 P": attend(fp8_rows(qn), fp8_rows(qp), ck.to(F8).float(), kp, p_fp8=True),
            "FP4 Q, BF16 P": attend(q_nvfp4(qn), fp8_rows(qp), ck, kp),
            "FP4 Q, FP8 P": attend(q_nvfp4(qn), fp8_rows(qp), ck, kp, p_fp8=True),
            "2 FP4 Q terms, 2 FP8 RoPE terms": attend(terms(qn, q_nvfp4, 2), terms(qp, fp8_rows, 2), ck, kp),
            "3 FP4 Q terms, 2 FP8 RoPE terms": attend(terms(qn, q_nvfp4, 3), terms(qp, fp8_rows, 2), ck, kp),
            "FP8 KV cache, FP8 Q and P (XQA)": attend(fp8_rows(qn), fp8_rows(qp), ckv.to(F8).float(), kp, p_fp8=True),
        }
        print(f"H={H} S={S}")
        for name, out in options.items():
            print(f"  {name:32s} {rel(out, ref):.4f}  {rel(out, deq):.4f}")
