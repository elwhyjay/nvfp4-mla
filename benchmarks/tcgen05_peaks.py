"""tcgen05 MMA throughput on SM100 for the instruction shapes the SM100 kernel issues, and large-N shapes as the peak.

  python benchmarks/tcgen05_peaks.py          # B200, needs nvidia-cutlass-dsl

One CTA per SM; one thread issues MMAs back to back into one accumulator, operands resident in shared memory (or A in
tensor memory), no loads. Two run lengths, so launch and setup cancel. M = 128, cta_group::1.
"""
import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as bsl
from cutlass import Float4E2M1FN, Float8E4M3FN, Float32, Int32
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05

M, KSTEPS = 128, 4


class Probe:
    def __init__(self, kind: str, n: int, a_tmem: bool = False):
        self.kind, self.n, self.a_tmem = kind, n, a_tmem
        self.fp4 = kind == "mxf4nvf4"
        self.scaled = kind in ("mxf4nvf4", "mxf8f6f4")
        self.k = 64 if self.fp4 else 32  # K of one instruction
        self.dt = Float4E2M1FN if self.fp4 else Float8E4M3FN
        self.sf_t = Float8E4M3FN if self.fp4 else cutlass.Float8E8M0FNU
        self.vec = 16 if self.fp4 else 32

    def flop(self):
        return 2 * M * self.n * self.k

    @cute.jit
    def __call__(self, dev: cute.Tensor, iters: Int32, stream):  # dev: any CUDA tensor, for the launch stream
        n, kt = self.n, KSTEPS * self.k
        if cutlass.const_expr(self.kind == "mxf4nvf4"):
            mma = cute.make_tiled_mma(tcgen05.MmaMXF4NVF4Op(Float8E4M3FN, (M, n, 64), tcgen05.CtaGroup.ONE,
                                                            tcgen05.OperandSource.SMEM))
        elif cutlass.const_expr(self.kind == "mxf8f6f4"):
            mma = cute.make_tiled_mma(tcgen05.MmaMXF8F6F4Op(Float8E4M3FN, Float8E4M3FN, (M, n, 32), tcgen05.CtaGroup.ONE,
                                                            tcgen05.OperandSource.SMEM, OperandMajorMode.K,
                                                            OperandMajorMode.K))
        else:
            mma = sm100_utils.make_trivial_tiled_mma(
                Float8E4M3FN, Float8E4M3FN, OperandMajorMode.K, OperandMajorMode.K, Float32, tcgen05.CtaGroup.ONE,
                (M, n), tcgen05.OperandSource.TMEM if self.a_tmem else tcgen05.OperandSource.SMEM)
        atom = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.K_SW128, self.dt)
        la = cute.tile_to_shape(atom, (M, kt, 1), order=(0, 1, 2))
        lb = cute.tile_to_shape(atom, (n, kt, 1), order=(0, 1, 2))
        if cutlass.const_expr(self.scaled):
            lsfa = bsl.make_smem_layout_sfa(mma, (M, n, kt), self.vec, 1)
            lsfb = bsl.make_smem_layout_sfb(mma, (M, n, kt), self.vec, 1)
        else:  # unused
            lsfa, lsfb = cute.make_layout(1), cute.make_layout(1)
        self.kernel(iters, mma, la, lb, lsfa, lsfb).launch(grid=[148, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, iters: Int32, mma: cute.TiledMma, la: cute.ComposedLayout, lb: cute.ComposedLayout,
               lsfa: cute.Layout, lsfb: cute.Layout):
        n, kt = self.n, KSTEPS * self.k
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)

        @cute.struct
        class Storage:
            bar: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_buf: Int32

        smem = utils.SmemAllocator()
        stg = smem.allocate(Storage)
        sA = smem.allocate_tensor(self.dt, la.outer, 1024, la.inner)
        sB = smem.allocate_tensor(self.dt, lb.outer, 1024, lb.inner)
        smem.allocate_tensor(cutlass.Uint8, cute.make_layout(128 * 1024), 1024)  # one CTA per SM: each takes all TMEM
        bar = stg.bar.data_ptr()
        if tidx == 0:
            cute.arch.mbarrier_init(bar, 1)
        cute.arch.mbarrier_init_fence()
        tmem = utils.TmemAllocator(stg.tmem_buf.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128))
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        if cutlass.const_expr(self.scaled):
            sSFA = smem.allocate_tensor(self.sf_t, lsfa, 128)
            sSFB = smem.allocate_tensor(self.sf_t, lsfb, 128)
        cute.arch.sync_threads()
        if warp == 0:
            thr = mma.get_slice(0)
            tCrB = mma.make_fragment_B(thr.partition_B(sB))
            tCtD = cute.make_tensor(tmem_ptr, mma.make_fragment_C(mma.partition_shape_C((M, n))).layout)
            col = tcgen05.find_tmem_tensor_col_offset(tCtD)
            if cutlass.const_expr(self.a_tmem):
                tCrA = cute.make_tensor(cute.recast_ptr(tmem_ptr + col, dtype=self.dt),
                                        thr.make_fragment_A(mma.partition_shape_A((M, kt))).layout)
            else:
                tCrA = mma.make_fragment_A(thr.partition_A(sA))
            if cutlass.const_expr(self.scaled):
                tCtSFA = cute.make_tensor(cute.recast_ptr(tmem_ptr + col, dtype=self.sf_t),
                                          bsl.make_tmem_layout_sfa(mma, (M, n, kt), self.vec,
                                                                   cute.slice_(lsfa, (None, None, None, 0))))
                c2 = col + tcgen05.find_tmem_tensor_col_offset(tCtSFA)
                tCtSFB = cute.make_tensor(cute.recast_ptr(tmem_ptr + c2, dtype=self.sf_t),
                                          bsl.make_tmem_layout_sfb(mma, (M, n, kt), self.vec,
                                                                   cute.slice_(lsfb, (None, None, None, 0))))
                for tt, ss in ((tCtSFA, sSFA), (tCtSFB, sSFB)):
                    cp = tcgen05.make_s2t_copy(cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
                                                                   self.sf_t), cute.filter_zeros(tt))
                    th = cp.get_slice(0)
                    cute.copy(cp, tcgen05.get_s2t_smem_desc_tensor(cp, th.partition_S(cute.filter_zeros(ss)))
                              [None, None, None, None, 0], th.partition_D(cute.filter_zeros(tt)))
            mma.set(tcgen05.Field.ACCUMULATE, True)
            for _ in cutlass.range(iters):
                for k in cutlass.range_constexpr(KSTEPS):
                    if cutlass.const_expr(self.scaled):
                        mma.set(tcgen05.Field.SFA, tCtSFA[None, None, k].iterator)
                        mma.set(tcgen05.Field.SFB, tCtSFB[None, None, k].iterator)
                    if cutlass.const_expr(self.a_tmem):
                        cute.gemm(mma, tCtD, tCrA[None, None, k], tCrB[None, None, k, 0], tCtD)
                    else:
                        cute.gemm(mma, tCtD, tCrA[None, None, k, 0], tCrB[None, None, k, 0], tCtD)
            with cute.arch.elect_one():
                tcgen05.commit(bar)
            cute.arch.mbarrier_wait(bar, 0)
        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


def rate(kind, n, a_tmem=False):
    p = Probe(kind, n, a_tmem)
    dev = torch.empty(1, device="cuda")
    f = cute.compile(p, cute.runtime.make_fake_compact_tensor(Float32, (1,)), Int32(1),
                     cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def ms(iters):
        f(dev, Int32(iters))
        torch.cuda.synchronize()
        s.record()
        f(dev, Int32(iters))
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)

    i1, i2 = 2000, 20000
    t = ms(i2) - ms(i1)
    return p.flop() * KSTEPS * (i2 - i1) * 148 / (t * 1e-3) / 1e12  # TFLOPS


if __name__ == "__main__":
    clk = 1.74e9  # under load in this kernel's profiles; the per-SM-cycle column uses it
    print(torch.cuda.get_device_name())
    for kind, n, a_tmem, what in [("mxf4nvf4", 256, False, "FP4 peak shape"),
                                   ("mxf4nvf4", 96, False, "QK, 3 query terms x 32 heads"),
                                   ("mxf4nvf4", 64, False, "QK, 2 terms x 32 heads"),
                                   ("mxf4nvf4", 32, False, "QK, 1 term x 32 heads"),
                                   ("mxf8f6f4", 32, False, "RoPE, 32 heads (block-scaled FP8)"),
                                   ("f8f6f4", 256, False, "FP8 peak shape"),
                                   ("f8f6f4", 32, True, "PV, 32 heads, A (V) in tensor memory")]:
        tf = rate(kind, n, a_tmem)
        print(f"{kind:9s} N={n:<3} {'A in TMEM' if a_tmem else 'A in smem':9s} {tf:7.0f} TFLOPS "
              f"{tf * 1e12 / 148 / clk:7.0f} FLOP/cycle/SM  ({what})", flush=True)
