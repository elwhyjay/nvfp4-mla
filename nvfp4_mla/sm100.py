"""SM100 (B200) NVFP4 MLA decode with tcgen05 MMAs, written in CuTe DSL.

Swap-AB: tokens are the MMA M (128 per tile), heads the N (16 or 32 per CTA), so 16 heads need no padding.
v0: one warpgroup runs every step in order (convert -> QK -> softmax -> PV), no pipelining. The FP4 cache is
converted to FP16 (e2m1 x e4m3 is exact in FP16) once per tile into shared memory; the same buffer is the K operand
(K-major) of QK and the V^T operand (MN-major) of PV.
v1a: the same structure, but QK reads the cache as stored: the latent part is block-scaled FP4 x FP4
(kind::mxf4nvf4, the cache's block scales as SFA) against the query written as 4 NVFP4 terms with one shared scale per
head (query_terms in the extension), stacked along N so K is read once; the RoPE part FP16 (kpe converted exactly)
into a second accumulator.
v1b (default): as v1a, but V is converted from the raw tile straight into TMEM as the A operand of PV.
"""
import functools
import math

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float16, Float32, Int32, Int64, Uint8, Uint32
import cutlass.utils.blockscaled_layout as bsl
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

CKV, KPE, D = 512, 64, 576
TILE = 128  # tokens per tile = MMA M
LOG2E = 1.4426950408889634


def _asm(ret, args, code, cons, loc, ip):
    return llvm.inline_asm(ret, [a.ir_value(loc=loc, ip=ip) for a in args], code, cons, has_side_effects=False,
                           is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def _unpack4(r, loc, ip):
    return tuple(Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def addr_of(t: cute.Tensor, *, loc=None, ip=None) -> Int64:
    return Int64(llvm.ptrtoint(T.i64(), t.iterator.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def ld_v4(addr: Int64, *, loc=None, ip=None):
    r = _asm(llvm.StructType.get_literal([T.i32()] * 4), [Int64(addr)], "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
             "=r,=r,=r,=r,l", loc, ip)
    return _unpack4(r, loc, ip)


@dsl_user_op
def e2m1x8_f16(w: Uint32, s2: Uint32, *, loc=None, ip=None):
    """8 e2m1 (low nibble first) times an f16x2 scale -> 4 f16x2."""
    r = _asm(llvm.StructType.get_literal([T.i32()] * 4), [Uint32(w), Uint32(s2)],
             "{\n.reg .b8 b0, b1, b2, b3;\n.reg .b32 h0, h1, h2, h3;\nmov.b32 {b0, b1, b2, b3}, $4;\n"
             "cvt.rn.f16x2.e2m1x2 h0, b0;\ncvt.rn.f16x2.e2m1x2 h1, b1;\n"
             "cvt.rn.f16x2.e2m1x2 h2, b2;\ncvt.rn.f16x2.e2m1x2 h3, b3;\n"
             "mul.rn.f16x2 $0, h0, $5;\nmul.rn.f16x2 $1, h1, $5;\nmul.rn.f16x2 $2, h2, $5;\nmul.rn.f16x2 $3, h3, $5;\n}",
             "=r,=r,=r,=r,r,r", loc, ip)
    return _unpack4(r, loc, ip)


@dsl_user_op
def e4m3_splat_f16x2(byte: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [Uint32(byte * Uint32(0x0101))],
                       "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $1;\ncvt.rn.f16x2.e4m3x2 $0, lo;\n}", "=r,r", loc, ip))


@dsl_user_op
def e4m3x4_f16(w: Uint32, *, loc=None, ip=None):
    r = _asm(llvm.StructType.get_literal([T.i32()] * 2), [Uint32(w)],
             "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $2;\ncvt.rn.f16x2.e4m3x2 $0, lo;\ncvt.rn.f16x2.e4m3x2 $1, hi;\n}",
             "=r,=r,r", loc, ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def bf16x2_f16x2(w: Uint32, scale: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [Uint32(w), Float32(scale)],
                       "{\n.reg .b16 lo, hi;\n.reg .f32 a, b;\nmov.b32 {lo, hi}, $1;\ncvt.f32.bf16 a, lo;\n"
                       "cvt.f32.bf16 b, hi;\nmul.f32 a, a, $2;\nmul.f32 b, b, $2;\ncvt.rn.f16x2.f32 $0, b, a;\n}",
                       "=r,r,f", loc, ip))


@dsl_user_op
def ld_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [Int64(addr)], "ld.global.nc.u32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def ldsm_t16(addr: Uint32, *, loc=None, ip=None):
    """ldmatrix.m16n16.x1.trans.b8: lane (g, t) gets byte columns g and g + 8 of rows 4t .. 4t + 3."""
    r = _asm(llvm.StructType.get_literal([T.i32()] * 2), [Uint32(addr)],
             "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];", "=r,=r,r", loc, ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def v_words(w: Uint32, sf4: Uint32, *, loc=None, ip=None):
    """4 tokens x 1 byte (e2m1 channels 2c, 2c + 1) and the 4 tokens' block scales -> f16x2 words of token pairs:
    (2c: tokens 0, 1), (2c: 2, 3), (2c + 1: 0, 1), (2c + 1: 2, 3). The product is exact in f16."""
    r = _asm(llvm.StructType.get_literal([T.i32()] * 4), [Uint32(w), Uint32(sf4)],
             "{\n.reg .b8 b0, b1, b2, b3;\n.reg .b16 s0, s1;\n.reg .b32 h0, h1, h2, h3, p01, p23, e, f;\n"
             "mov.b32 {b0, b1, b2, b3}, $4;\nmov.b32 {s0, s1}, $5;\n"
             "cvt.rn.f16x2.e4m3x2 p01, s0;\ncvt.rn.f16x2.e4m3x2 p23, s1;\n"
             "cvt.rn.f16x2.e2m1x2 h0, b0;\ncvt.rn.f16x2.e2m1x2 h1, b1;\n"
             "cvt.rn.f16x2.e2m1x2 h2, b2;\ncvt.rn.f16x2.e2m1x2 h3, b3;\n"
             "prmt.b32 e, h0, h1, 0x5410;\nmul.rn.f16x2 $0, e, p01;\n"
             "prmt.b32 f, h2, h3, 0x5410;\nmul.rn.f16x2 $1, f, p23;\n"
             "prmt.b32 e, h0, h1, 0x7632;\nmul.rn.f16x2 $2, e, p01;\n"
             "prmt.b32 f, h2, h3, 0x7632;\nmul.rn.f16x2 $3, f, p23;\n}",
             "=r,=r,=r,=r,r,r", loc, ip)
    return _unpack4(r, loc, ip)


@dsl_user_op
def tmem_st16x256(taddr: Uint32, w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None):
    """tcgen05.st.16x256b.x1: thread (g, t) writes lane g columns 2t, 2t + 1 (w0, w1) and lane g + 8 (w2, w3)."""
    llvm.inline_asm(None, [Uint32(x).ir_value(loc=loc, ip=ip) for x in (taddr, w0, w1, w2, w3)],
                    "tcgen05.st.sync.aligned.16x256b.x1.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@cute.jit
def store8(dst: cute.Tensor, u0: Uint32, u1: Uint32, u2: Uint32, u3: Uint32):
    """Store 16 bytes held as 4 words to a shared-memory slice of 16 bytes (8 f16 or 32 e2m1)."""
    r = cute.make_rmem_tensor(cute.make_layout((1, 4)), Uint32)
    r[0], r[1], r[2], r[3] = u0, u1, u2, u3
    cute.autovec_copy(cute.recast_tensor(r, dst.element_type), dst)


class DecodeV0:
    def __init__(self, n: int, page_size: int):
        self.n, self.ps = n, page_size
        self.tmem_cols = 1 << math.ceil(math.log2(5 * n))  # S: n columns, O^T: 4 tiles of n

    @cute.jit
    def __call__(self, q_nope: cute.Tensor, q_pe: cute.Tensor, ckv: cute.Tensor, sf: cute.Tensor, kpe: cute.Tensor,
                 page_table: cute.Tensor, kv_lens: cute.Tensor, out: cute.Tensor, o_part: cute.Tensor,
                 lse_part: cute.Tensor, chunk: Int32, splits: Int32, scale_log2: Float32, kpe_ratio: Float32,
                 out_scale: Float32, stream):
        n = self.n
        mma_kq = sm100_utils.make_trivial_tiled_mma(Float16, OperandMajorMode.K, OperandMajorMode.K, Float32,
                                                    tcgen05.CtaGroup.ONE, (TILE, n))
        mma_vp = sm100_utils.make_trivial_tiled_mma(Float16, OperandMajorMode.MN, OperandMajorMode.MN, Float32,
                                                    tcgen05.CtaGroup.ONE, (TILE, n))
        atom_k = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.K_SW128, Float16)
        atom_mn = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.MN_SW128, Float16)
        # Token groups of 8 innermost, then 64-channel blocks: the K-major (tok, d) tile and the MN-major (d, tok)
        # view below address the same bytes.
        kv_layout = cute.tile_to_shape(atom_k, (TILE, D), order=(0, 1))
        vt_layout = cute.tile_to_shape(atom_mn, (CKV, TILE), order=(1, 0))
        q_layout = cute.tile_to_shape(atom_k, (n, D), order=(0, 1))
        p_layout = sm100_utils.make_smem_layout_b(mma_vp, (TILE, n, TILE), Float16, 1)
        self.kernel(q_nope, q_pe, ckv, sf, kpe, page_table, kv_lens, out, o_part, lse_part, chunk, scale_log2,
                    kpe_ratio, out_scale, mma_kq, mma_vp, kv_layout, vt_layout, q_layout, p_layout).launch(
            grid=[q_nope.shape[1] // n, q_nope.shape[0], splits], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQn: cute.Tensor, mQp: cute.Tensor, mCkv: cute.Tensor, mSf: cute.Tensor, mKpe: cute.Tensor,
               mPt: cute.Tensor, mLens: cute.Tensor, mOut: cute.Tensor, mOpart: cute.Tensor, mLse: cute.Tensor,
               chunk: Int32, scale_log2: Float32, kpe_ratio: Float32, out_scale: Float32, mma_kq: cute.TiledMma,
               mma_vp: cute.TiledMma, kv_layout: cute.ComposedLayout, vt_layout: cute.ComposedLayout,
               q_layout: cute.ComposedLayout, p_layout: cute.ComposedLayout):
        n, ps = self.n, self.ps
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = cute.arch.lane_idx()
        ht, b, z = cute.arch.block_idx()
        _, _, splits = cute.arch.grid_dim()
        h0 = ht * n

        @cute.struct
        class Storage:
            s_bar: cute.struct.MemRange[cutlass.Int64, 2]
            o_bar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_buf: Int32

        smem = utils.SmemAllocator()
        st = smem.allocate(Storage)
        sKV = smem.allocate_tensor(Float16, kv_layout.outer, 1024, kv_layout.inner)
        sQ = smem.allocate_tensor(Float16, q_layout.outer, 1024, q_layout.inner)
        sP = smem.allocate_tensor(Float16, p_layout.outer, 1024, p_layout.inner)
        sM = smem.allocate_tensor(Float32, cute.make_layout((TILE, n), stride=(0, 1)), 16)  # running column max
        sL = smem.allocate_tensor(Float32, cute.make_layout((4, n), stride=(n, 1)), 16)
        sVt = cute.make_tensor(cute.recast_ptr(sKV.iterator, vt_layout.inner, dtype=Float16), vt_layout.outer)

        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        all_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1, producer_group=one, consumer_group=all_threads, barrier_storage=st.s_bar.data_ptr(),
            defer_sync=True).make_participants()
        o_prod, o_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1, producer_group=one, consumer_group=all_threads, barrier_storage=st.o_bar.data_ptr(),
            defer_sync=True).make_participants()
        tmem = utils.TmemAllocator(st.tmem_buf.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128))
        tmem.allocate(self.tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        # MMA operands
        thr_kq = mma_kq.get_slice(0)
        thr_vp = mma_vp.get_slice(0)
        tCrK = mma_kq.make_fragment_A(thr_kq.partition_A(sKV))  # (MMA, 1, 36)
        tCrQ = mma_kq.make_fragment_B(thr_kq.partition_B(sQ))  # (MMA, 1, 36)
        tCrV = mma_vp.make_fragment_A(thr_vp.partition_A(cute.flat_divide(sVt, (TILE, TILE))))  # (MMA, 1, 8, 4, 1)
        tCrP = mma_vp.make_fragment_B(sP)  # (MMA, 1, 8, 1)

        s_shape = mma_kq.partition_shape_C((TILE, n, 1))  # (MMA, 1, 1, 1)
        o_shape = mma_vp.partition_shape_C((TILE, n, 4))  # (MMA, 1, 1, 4): O^T in 4 tiles of 128 channels
        tCtS = cute.make_tensor(tmem_ptr, mma_kq.make_fragment_C(s_shape).layout)
        tCtO = cute.make_tensor(tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtS),
                                mma_vp.make_fragment_C(o_shape).layout)

        # P^T (heads contiguous) seen as the (tok, head) C tile of QK, so softmax threads store their own rows
        tCsP = cute.composition(sP, cute.make_ordered_layout(s_shape, order=((2, 0), 3, 1, 4)))

        rep = tcgen05.Repetition(n)
        ld_s = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.Ld32x32bOp(rep), Float32), tCtS[None, None, None, 0])
        st_o = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(rep), Float32), tCtO[None, None, None, 0])
        thr_ld = ld_s.get_slice(tidx)
        thr_st = st_o.get_slice(tidx)
        tStS = thr_ld.partition_S(tCtS)
        tSsP = thr_ld.partition_D(tCsP)
        tStO = thr_ld.partition_S(tCtO)
        tOtO = thr_st.partition_D(tCtO)
        tSsM = thr_ld.partition_D(thr_kq.partition_C(sM))
        tScS = thr_ld.partition_D(thr_kq.partition_C(cute.make_identity_tensor((TILE, n))))
        frag = tSsP.shape[:-1]

        if warp == 0 and lane < n:
            tSsM[lane] = -Float32.inf
        cute.arch.sync_threads()

        # Q -> FP16 (the RoPE part scaled by kpe_scale / ckv_scale, so one accumulator serves both parts)
        for i in cutlass.range(tidx, n * (D // 8), 128):
            r, c = i // (D // 8), i % (D // 8)
            qa = addr_of(mQn) + (Int64(b * mQn.shape[1] + h0 + r) * CKV + c * 8) * 2
            f = Float32(1.0)
            if c >= CKV // 8:
                qa = addr_of(mQp) + (Int64(b * mQp.shape[1] + h0 + r) * KPE + (c - CKV // 8) * 8) * 2
                f = kpe_ratio
            w = ld_v4(qa)
            store8(cute.local_tile(sQ, (1, 8), (r, c)), bf16x2_f16x2(w[0], f), bf16x2_f16x2(w[1], f),
                   bf16x2_f16x2(w[2], f), bf16x2_f16x2(w[3], f))

        t_beg = z * chunk
        t_end = cutlass.min(mLens[b], t_beg + chunk)
        n_tiles = cutlass.max(t_end - t_beg + TILE - 1, 0) // TILE
        m_prev = cute.make_rmem_tensor(frag, Float32)
        m_prev.fill(-Float32.inf)
        l_acc = cute.make_rmem_tensor(frag, Float32)
        l_acc.fill(0.0)
        row = tScS[0][0]

        for it in cutlass.range(n_tiles):
            # ---- convert token t0 + tidx into row tidx of the FP16 tile (zeros past the end)
            tok = t_beg + it * TILE + tidx
            if tok < t_end:
                page = mPt[b, tok // ps]
                slot = Int64(page) * ps + tok % ps
                pc = addr_of(mCkv) + slot * (CKV // 2)
                sw = ld_v4(addr_of(mSf) + slot * (CKV // 16))
                sw2 = ld_v4(addr_of(mSf) + slot * (CKV // 16) + 16)
                sfw = (sw[0], sw[1], sw[2], sw[3], sw2[0], sw2[1], sw2[2], sw2[3])
                for j in cutlass.range_constexpr(16):
                    w = ld_v4(pc + 16 * j)
                    for k in cutlass.range_constexpr(4):
                        q = 2 * j + k // 2
                        s2 = e4m3_splat_f16x2((sfw[q // 4] >> (8 * (q % 4))) & Uint32(0xFF))
                        h = e2m1x8_f16(w[k], s2)
                        store8(cute.local_tile(sKV, (1, 8), (tidx, 4 * j + k)), h[0], h[1], h[2], h[3])
                pk = addr_of(mKpe) + slot * KPE
                for j in cutlass.range_constexpr(4):
                    w = ld_v4(pk + 16 * j)
                    a0, a1 = e4m3x4_f16(w[0])
                    a2, a3 = e4m3x4_f16(w[1])
                    store8(cute.local_tile(sKV, (1, 8), (tidx, CKV // 8 + 2 * j)), a0, a1, a2, a3)
                    a0, a1 = e4m3x4_f16(w[2])
                    a2, a3 = e4m3x4_f16(w[3])
                    store8(cute.local_tile(sKV, (1, 8), (tidx, CKV // 8 + 2 * j + 1)), a0, a1, a2, a3)
            else:
                for c in cutlass.range_constexpr(D // 8):
                    store8(cute.local_tile(sKV, (1, 8), (tidx, c)), Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            # ---- S^T = K Q^T
            if warp == 0:
                hs = s_prod.acquire_and_advance()
                mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(D // 16):
                    cute.gemm(mma_kq, tCtS[None, None, None, 0], tCrK[None, None, k], tCrQ[None, None, k],
                              tCtS[None, None, None, 0])
                    mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                hs.commit()

            # ---- online softmax over tokens (columns are heads)
            hsc = s_cons.wait_and_advance()
            sS = cute.make_rmem_tensor(frag, Float32)
            cute.copy(ld_s, tStS[None, None, None, None, 0], sS)
            cute.arch.fence_view_async_tmem_load()
            hsc.release()
            if t_beg + it * TILE + row >= t_end:
                sS.fill(-Float32.inf)

            m_lane = Float32(0.0)
            for i in cutlass.range_constexpr(n):
                mi = cute.arch.warp_redux_sync(sS[i], kind="fmax", nan=True)
                if i == lane:
                    m_lane = mi
            if lane < n:
                cute.arch.atomic_fmax(tSsM.iterator + tSsM.layout(lane), m_lane, sem="relaxed", scope="cta")
            cute.arch.sync_threads()
            m_cur = cute.make_rmem_tensor(frag, Float32)
            cute.autovec_copy(tSsM, m_cur)

            pf = cute.make_rmem_tensor(frag, Float32)
            corr = cute.make_rmem_tensor(frag, Float32)
            for i in cutlass.range_constexpr(n):
                pf[i] = cute.math.exp2((sS[i] - m_cur[i]) * scale_log2, fastmath=True)
                corr[i] = cute.math.exp2((m_prev[i] - m_cur[i]) * scale_log2, fastmath=True)
                l_acc[i] = l_acc[i] * corr[i] + pf[i]
                m_prev[i] = m_cur[i]
            p16 = cute.make_rmem_tensor(frag, Float16)
            p16.store(pf.load().to(Float16))
            cute.autovec_copy(p16, tSsP[None, None, None, None, 0])
            cute.arch.fence_view_async_shared()

            if it > 0:  # rescale O^T rows (this thread's channel) by each head's correction
                for dm in cutlass.range_constexpr(4):
                    o = cute.make_rmem_tensor(frag, Float32)
                    cute.copy(ld_s, tStO[None, None, None, None, dm], o)
                    for i in cutlass.range_constexpr(n):
                        o[i] = o[i] * corr[i]
                    cute.copy(st_o, o, tOtO[None, None, None, None, dm])
                cute.arch.fence_view_async_tmem_store()
            cute.arch.sync_threads()

            # ---- O^T += V^T P^T
            if warp == 0:
                ho = o_prod.acquire_and_advance()
                for dm in cutlass.range_constexpr(4):
                    mma_vp.set(tcgen05.Field.ACCUMULATE, it > 0)
                    for k in cutlass.range_constexpr(TILE // 16):
                        cute.gemm(mma_vp, tCtO[None, None, None, dm], tCrV[None, None, k, dm, 0],
                                  tCrP[None, None, k, 0], tCtO[None, None, None, dm])
                        mma_vp.set(tcgen05.Field.ACCUMULATE, True)
                ho.commit()
            hoc = o_cons.wait_and_advance()  # PV done: sKV and sP may be overwritten
            hoc.release()

        # ---- epilogue: column sums across the 4 warps, then O^T / l
        l_lane = Float32(0.0)
        for i in cutlass.range_constexpr(n):
            li = cute.arch.warp_reduction_sum(l_acc[i])
            if i == lane:
                l_lane = li
        if lane < n:
            sL[warp, lane] = l_lane
        cute.arch.sync_threads()

        for dm in cutlass.range_constexpr(4):
            o = cute.make_rmem_tensor(frag, Float32)
            if n_tiles > 0:
                cute.copy(ld_s, tStO[None, None, None, None, dm], o)
            for i in cutlass.range_constexpr(n):
                head = tScS[i][1]
                l = sL[0, head] + sL[1, head] + sL[2, head] + sL[3, head]
                v = Float32(0.0)
                if l > 0.0:
                    v = o[i] * (out_scale / l)
                ch = dm * TILE + tScS[i][0]
                if splits == 1:
                    mOut[b, h0 + head, ch] = v.to(BFloat16)
                else:
                    mOpart[z, b, h0 + head, ch] = v
        if splits > 1 and warp == 0 and lane < n:
            l = sL[0, lane] + sL[1, lane] + sL[2, lane] + sL[3, lane]
            lse = -Float32.inf
            if l > 0.0:
                lse = m_prev[lane] * scale_log2 + cute.math.log2(l, fastmath=False)
            mLse[z, b, h0 + lane] = lse

        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


SF_VEC = 16  # channels per E4M3 block scale


class DecodeV1:
    """v1a' (V through shared memory as FP16) or, with v_tmem, v1b (V converted straight into TMEM)."""

    def __init__(self, n: int, page_size: int, terms: int, dbg: bool = False, v_tmem: bool = False):
        self.n, self.ps, self.terms, self.dbg, self.v_tmem = n, page_size, terms, dbg, v_tmem

    @cute.jit
    def __call__(self, q4: cute.Tensor, qsf: cute.Tensor, qg: cute.Tensor, q_pe: cute.Tensor, ckv: cute.Tensor,
                 sf: cute.Tensor, kpe: cute.Tensor, page_table: cute.Tensor, kv_lens: cute.Tensor, out: cute.Tensor,
                 o_part: cute.Tensor, lse_part: cute.Tensor, chunk: Int32, splits: Int32, scale_log2: Float32,
                 kpe_ratio: Float32, out_scale: Float32, stream):
        n, terms = self.n, self.terms
        nq = terms * n  # B rows: term t of head h at row t * n + h, so one MMA per K block reads K once
        mma_qk = cute.make_tiled_mma(tcgen05.MmaMXF4NVF4Op(Float8E4M3FN, (TILE, nq, 64), tcgen05.CtaGroup.ONE,
                                                           tcgen05.OperandSource.SMEM))
        mma_pe = sm100_utils.make_trivial_tiled_mma(Float16, OperandMajorMode.K, OperandMajorMode.K, Float32,
                                                    tcgen05.CtaGroup.ONE, (TILE, n))
        if cutlass.const_expr(self.v_tmem):  # V^T from TMEM (A), P^T from shared memory (B)
            mma_vp = sm100_utils.make_trivial_tiled_mma(Float16, Float16, OperandMajorMode.K, OperandMajorMode.MN,
                                                        Float32, tcgen05.CtaGroup.ONE, (TILE, n),
                                                        tcgen05.OperandSource.TMEM)
        else:
            mma_vp = sm100_utils.make_trivial_tiled_mma(Float16, OperandMajorMode.MN, OperandMajorMode.MN, Float32,
                                                        tcgen05.CtaGroup.ONE, (TILE, n))
        # the latent accumulator's TMEM layout (M = 128, N = terms x n, fp32), taken from an FP16 MMA of that shape
        mma_sn = sm100_utils.make_trivial_tiled_mma(Float16, OperandMajorMode.K, OperandMajorMode.K, Float32,
                                                    tcgen05.CtaGroup.ONE, (TILE, nq))
        atom4 = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.K_SW128, Float4E2M1FN)
        atom_k = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.K_SW128, Float16)
        atom_mn = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.MN_SW128, Float16)
        tiler = (TILE, nq, CKV)
        layouts = (
            cute.tile_to_shape(atom4, (TILE, CKV), order=(0, 1)),
            cute.tile_to_shape(atom4, (nq, CKV), order=(0, 1)),
            cute.tile_to_shape(atom_k, (TILE, KPE), order=(0, 1)),
            cute.tile_to_shape(atom_k, (n, KPE), order=(0, 1)),
            # V as v0 (latent channels only): K-major rows for the converting threads, MN-major for PV
            cute.tile_to_shape(atom_k, (TILE, CKV), order=(0, 1)),
            cute.tile_to_shape(atom_mn, (CKV, TILE), order=(1, 0)),
            sm100_utils.make_smem_layout_b(mma_vp, (TILE, n, TILE), Float16, 1),
            bsl.make_smem_layout_sfa(mma_qk, tiler, SF_VEC, 1),
            bsl.make_smem_layout_sfb(mma_qk, tiler, SF_VEC, 1))
        self.kernel(q4, qsf, qg, q_pe, ckv, sf, kpe, page_table, kv_lens, out, o_part, lse_part, chunk, scale_log2,
                    kpe_ratio, out_scale, mma_qk, mma_pe, mma_vp, mma_sn, *layouts).launch(
            grid=[q_pe.shape[1] // n, q_pe.shape[0], splits], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mQ4: cute.Tensor, mQsf: cute.Tensor, mQg: cute.Tensor, mQp: cute.Tensor, mCkv: cute.Tensor,
               mSf: cute.Tensor, mKpe: cute.Tensor, mPt: cute.Tensor, mLens: cute.Tensor, mOut: cute.Tensor,
               mOpart: cute.Tensor, mLse: cute.Tensor, chunk: Int32, scale_log2: Float32, kpe_ratio: Float32,
               out_scale: Float32, mma_qk: cute.TiledMma, mma_pe: cute.TiledMma, mma_vp: cute.TiledMma,
               mma_sn: cute.TiledMma, lk4: cute.ComposedLayout, lq4: cute.ComposedLayout, lkpe: cute.ComposedLayout,
               lqpe: cute.ComposedLayout, lv: cute.ComposedLayout, lvt: cute.ComposedLayout, lp: cute.ComposedLayout,
               lsfa: cute.Layout, lsfb: cute.Layout):
        n, ps, terms = self.n, self.ps, self.terms
        nq = terms * n
        L = dict(k4=lk4, q4=lq4, kpe=lkpe, qpe=lqpe, v=lv, vt=lvt, p=lp, sfa=lsfa, sfb=lsfb)
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = cute.arch.lane_idx()
        ht, b, z = cute.arch.block_idx()
        _, _, splits = cute.arch.grid_dim()
        h0 = ht * n

        @cute.struct
        class Storage:
            s_bar: cute.struct.MemRange[cutlass.Int64, 2]
            o_bar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_buf: Int32

        smem = utils.SmemAllocator()
        st = smem.allocate(Storage)
        sK4 = smem.allocate_tensor(Float4E2M1FN, L["k4"].outer, 1024, L["k4"].inner)
        sQ4 = smem.allocate_tensor(Float4E2M1FN, L["q4"].outer, 1024, L["q4"].inner)
        sKpe = smem.allocate_tensor(Float16, L["kpe"].outer, 1024, L["kpe"].inner)
        sQpe = smem.allocate_tensor(Float16, L["qpe"].outer, 1024, L["qpe"].inner)
        sP = smem.allocate_tensor(Float16, L["p"].outer, 1024, L["p"].inner)
        sSFA = smem.allocate_tensor(Float8E4M3FN, L["sfa"], 128)
        sM = smem.allocate_tensor(Float32, cute.make_layout((TILE, n), stride=(0, 1)), 16)
        sL = smem.allocate_tensor(Float32, cute.make_layout((4, n), stride=(n, 1)), 16)
        sG = smem.allocate_tensor(Float32, cute.make_layout(n), 16)
        if cutlass.const_expr(self.v_tmem):
            sSFB = smem.allocate_tensor(Float8E4M3FN, L["sfb"], 128)
            # block scales again, block-major: one 32-bit load gives 4 tokens' scales of one block
            sSfT = smem.allocate_tensor(Uint32, cute.make_layout((CKV // 16, TILE // 4), stride=(TILE // 4, 1)), 16)
            sSfT8 = cute.make_tensor(cute.recast_ptr(sSfT.iterator, dtype=Uint8),
                                     cute.make_layout((CKV // 16, TILE), stride=(TILE, 1)))
            sfb_base = sSFB.iterator
        else:
            sV = smem.allocate_tensor(Float16, L["v"].outer, 1024, L["v"].inner)
            sVt = cute.make_tensor(cute.recast_ptr(sV.iterator, L["vt"].inner, dtype=Float16), L["vt"].outer)
            # The query's scale factors live in the V buffer until they are in TMEM (prologue only).
            sSFB = cute.make_tensor(cute.recast_ptr(sV.iterator, dtype=Float8E4M3FN), L["sfb"])
            sfb_base = sV.iterator
        # Byte (row m, scale j) of an SF tile sits at 512 (j / 4) + 16 (m % 32) + 4 (m / 32) + j % 4 (the SF atom
        # tiled along K); as words: [m % 32, m / 32, j / 4] (+ 1024 words per query term).
        sfa_w = cute.make_tensor(cute.recast_ptr(sSFA.iterator, dtype=Uint32),
                                 cute.make_layout((32, 4, CKV // 64), stride=(4, 1, 128)))
        sfb_w = cute.make_tensor(cute.recast_ptr(sfb_base, dtype=Uint32),
                                 cute.make_layout((32, 4, CKV // 64), stride=(4, 1, 128)))

        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        all_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        s_prod, s_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1, producer_group=one, consumer_group=all_threads, barrier_storage=st.s_bar.data_ptr(),
            defer_sync=True).make_participants()
        o_prod, o_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1, producer_group=one, consumer_group=all_threads, barrier_storage=st.o_bar.data_ptr(),
            defer_sync=True).make_participants()
        tmem = utils.TmemAllocator(st.tmem_buf.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128))
        tmem.allocate(512 if self.v_tmem else 256)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        # MMA operands
        thr_qk = mma_qk.get_slice(0)
        thr_pe = mma_pe.get_slice(0)
        thr_vp = mma_vp.get_slice(0)
        tCrK4 = mma_qk.make_fragment_A(thr_qk.partition_A(sK4))  # (MMA, 1, 8)
        tCrQ4 = mma_qk.make_fragment_B(thr_qk.partition_B(sQ4))  # (MMA, 1, 8)
        tCrKpe = mma_pe.make_fragment_A(thr_pe.partition_A(sKpe))  # (MMA, 1, 4)
        tCrQpe = mma_pe.make_fragment_B(thr_pe.partition_B(sQpe))  # (MMA, 1, 4)
        if cutlass.const_expr(self.v_tmem):
            # V^T in TMEM columns 256-511: 4 channel tiles x 64 columns (2 tokens per column)
            tAtV = cute.make_tensor(cute.recast_ptr(tmem_ptr + 256, dtype=Float16),
                                    thr_vp.make_fragment_A(mma_vp.partition_shape_A((TILE, TILE, 4))).layout)
            tmem_v = Uint32(tmem_ptr.toint()) + Uint32(256)
            k4_base = Uint32(sK4.iterator.toint())
        else:
            tCrV = mma_vp.make_fragment_A(thr_vp.partition_A(cute.flat_divide(sVt, (TILE, TILE))))  # (MMA, 1, 8, 4, 1)
        tCrP = mma_vp.make_fragment_B(sP)  # (MMA, 1, 8, 1)

        # TMEM: S latent (terms x n columns), S RoPE (n), O^T (4 tiles of n), then the scale factors
        s_shape = mma_pe.partition_shape_C((TILE, n, 1))  # (MMA, 1, 1, 1)
        o_shape = mma_vp.partition_shape_C((TILE, n, 4))
        s_layout = mma_pe.make_fragment_C(s_shape).layout
        tCtSn = cute.make_tensor(tmem_ptr, mma_sn.make_fragment_C(mma_sn.partition_shape_C((TILE, nq, 1))).layout)
        col_n = tcgen05.find_tmem_tensor_col_offset(tCtSn)
        tCtSp = cute.make_tensor(tmem_ptr + col_n, s_layout)
        col = tcgen05.find_tmem_tensor_col_offset(tCtSp)
        tCtO = cute.make_tensor(tmem_ptr + col_n + col, mma_vp.make_fragment_C(o_shape).layout)
        col_sf = col_n + col + tcgen05.find_tmem_tensor_col_offset(tCtO)
        tCtSFA = cute.make_tensor(cute.recast_ptr(tmem_ptr + col_sf, dtype=Float8E4M3FN),
                                  bsl.make_tmem_layout_sfa(mma_qk, (TILE, n, CKV), SF_VEC,
                                                           cute.slice_(L["sfa"], (None, None, None, 0))))
        col_sfa = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
        sfb_tmem_layout = bsl.make_tmem_layout_sfb(mma_qk, (TILE, n, CKV), SF_VEC,
                                                   cute.slice_(L["sfb"], (None, None, None, 0)))
        tCtSFB = cute.make_tensor(cute.recast_ptr(tmem_ptr + col_sf + col_sfa, dtype=Float8E4M3FN), sfb_tmem_layout)

        s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), Float8E4M3FN)
        cp_sfa = tcgen05.make_s2t_copy(s2t, cute.filter_zeros(tCtSFA))
        thr_cpa = cp_sfa.get_slice(0)
        sfa_src = tcgen05.get_s2t_smem_desc_tensor(cp_sfa, thr_cpa.partition_S(cute.filter_zeros(sSFA)))
        sfa_dst = thr_cpa.partition_D(cute.filter_zeros(tCtSFA))
        cp_sfb = tcgen05.make_s2t_copy(s2t, cute.filter_zeros(tCtSFB))
        thr_cpb = cp_sfb.get_slice(0)
        sfb_src = tcgen05.get_s2t_smem_desc_tensor(cp_sfb, thr_cpb.partition_S(cute.filter_zeros(sSFB)))
        sfb_dst = thr_cpb.partition_D(cute.filter_zeros(tCtSFB))

        tCsP = cute.composition(sP, cute.make_ordered_layout(s_shape, order=((2, 0), 3, 1, 4)))
        rep = tcgen05.Repetition(n)
        ld_s = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.Ld32x32bOp(rep), Float32), tCtSp[None, None, None, 0])

        st_o = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(rep), Float32), tCtO[None, None, None, 0])
        thr_ld = ld_s.get_slice(tidx)
        thr_st = st_o.get_slice(tidx)
        # the latent accumulator read as `terms` (TILE, n) blocks: 32x32b loads with 64 columns crash the DSL compiler
        tStSn = [thr_ld.partition_S(cute.make_tensor(tmem_ptr + t * col, s_layout)) for t in range(terms)]
        tStSp = thr_ld.partition_S(tCtSp)
        tSsP = thr_ld.partition_D(tCsP)
        tStO = thr_ld.partition_S(tCtO)
        tOtO = thr_st.partition_D(tCtO)
        tSsM = thr_ld.partition_D(thr_pe.partition_C(sM))
        tSsG = thr_ld.partition_D(thr_pe.partition_C(cute.make_tensor(sG.iterator, cute.make_layout((TILE, n),
                                                                                                        stride=(0, 1)))))
        tScS = thr_ld.partition_D(thr_pe.partition_C(cute.make_identity_tensor((TILE, n))))
        frag = tSsP.shape[:-1]

        # ---- prologue: the query's terms, scale factors and per-head scale; RoPE part to FP16 (x kpe / ckv scale)
        for i in cutlass.range(tidx, 1024, 128):  # zero the SFB rows past terms x n (the copy moves 128 rows)
            sfb_w[i % 32, (i // 32) % 4, i // 128] = Uint32(0)
        cute.arch.sync_threads()
        for i in cutlass.range(tidx, n * terms * (CKV // 32), 128):
            r, t, c = i // (terms * (CKV // 32)), (i // (CKV // 32)) % terms, i % (CKV // 32)
            w = ld_v4(addr_of(mQ4) + ((Int64(b * mQ4.shape[1] + h0 + r) * terms + t) * (CKV // 2) + 16 * c))
            store8(cute.local_tile(sQ4, (1, 32), (t * n + r, c)), w[0], w[1], w[2], w[3])
        for i in cutlass.range(tidx, n * terms * (CKV // 64), 128):
            r, t, a = i // (terms * (CKV // 64)), (i // (CKV // 64)) % terms, i % (CKV // 64)
            sfb_w[(t * n + r) % 32, (t * n + r) // 32, a] = ld_u32(addr_of(mQsf) + ((Int64(b * mQsf.shape[1] + h0 + r) * terms + t)
                                                                   * (CKV // 16) + 4 * a))
        for i in cutlass.range(tidx, n * (KPE // 8), 128):
            r, c = i // (KPE // 8), i % (KPE // 8)
            w = ld_v4(addr_of(mQp) + (Int64(b * mQp.shape[1] + h0 + r) * KPE + c * 8) * 2)
            store8(cute.local_tile(sQpe, (1, 8), (r, c)), bf16x2_f16x2(w[0], kpe_ratio), bf16x2_f16x2(w[1], kpe_ratio),
                   bf16x2_f16x2(w[2], kpe_ratio), bf16x2_f16x2(w[3], kpe_ratio))
        if warp == 0 and lane < n:
            tSsM[lane] = -Float32.inf
            sG[lane] = mQg[b, h0 + lane]
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        if warp == 0:  # SFB into TMEM once; wait for it, since its smem is the V buffer
            hs = s_prod.acquire_and_advance()
            cute.copy(cp_sfb, sfb_src[None, None, None, None, 0], sfb_dst)
            hs.commit()
        hsc = s_cons.wait_and_advance()
        hsc.release()
        g_head = cute.make_rmem_tensor(frag, Float32)
        cute.autovec_copy(tSsG, g_head)

        t_beg = z * chunk
        t_end = cutlass.min(mLens[b], t_beg + chunk)
        n_tiles = cutlass.max(t_end - t_beg + TILE - 1, 0) // TILE
        m_prev = cute.make_rmem_tensor(frag, Float32)
        m_prev.fill(-Float32.inf)
        l_acc = cute.make_rmem_tensor(frag, Float32)
        l_acc.fill(0.0)
        row = tScS[0][0]

        for it in cutlass.range(n_tiles):
            # ---- token t0 + tidx: raw bytes into the FP4 tile, its scales into SFA, RoPE and V converted to FP16
            tok = t_beg + it * TILE + tidx
            # Scales for every row, outside the branch: past the end, a valid row's words masked to zero.
            tokc = cutlass.min(tok, t_end - 1)
            slotc = Int64(mPt[b, tokc // ps]) * ps + tokc % ps
            keep = Uint32(~((t_end - 1 - tok) >> 31))  # all ones for a valid row, zero past the end
            sw = ld_v4(addr_of(mSf) + slotc * (CKV // 16))
            sw2 = ld_v4(addr_of(mSf) + slotc * (CKV // 16) + 16)
            sfw = tuple(x & keep for x in (sw[0], sw[1], sw[2], sw[3], sw2[0], sw2[1], sw2[2], sw2[3]))
            for a in cutlass.range_constexpr(CKV // 64):
                sfa_w[tidx % 32, tidx // 32, a] = sfw[a]
            if cutlass.const_expr(self.v_tmem):
                for jj in cutlass.range_constexpr(CKV // 16):
                    sSfT8[jj, tidx] = ((sfw[jj // 4] >> (8 * (jj % 4))) & Uint32(0xFF)).to(Uint8)
            # Raw bytes, RoPE (and in v1a' V) the same way: rows past the end are a valid row masked to zero.
            pc = addr_of(mCkv) + slotc * (CKV // 2)
            for j in cutlass.range_constexpr(16):
                w = ld_v4(pc + 16 * j)
                w = (w[0] & keep, w[1] & keep, w[2] & keep, w[3] & keep)
                store8(cute.local_tile(sK4, (1, 32), (tidx, j)), w[0], w[1], w[2], w[3])
                if cutlass.const_expr(not self.v_tmem):
                    for k in cutlass.range_constexpr(4):
                        q = 2 * j + k // 2
                        s2 = e4m3_splat_f16x2((sfw[q // 4] >> (8 * (q % 4))) & Uint32(0xFF))
                        h = e2m1x8_f16(w[k], s2)
                        store8(cute.local_tile(sV, (1, 8), (tidx, 4 * j + k)), h[0], h[1], h[2], h[3])
            pk = addr_of(mKpe) + slotc * KPE
            for j in cutlass.range_constexpr(4):
                w = ld_v4(pk + 16 * j)
                a0, a1 = e4m3x4_f16(w[0] & keep)
                a2, a3 = e4m3x4_f16(w[1] & keep)
                store8(cute.local_tile(sKpe, (1, 8), (tidx, 2 * j)), a0, a1, a2, a3)
                a0, a1 = e4m3x4_f16(w[2] & keep)
                a2, a3 = e4m3x4_f16(w[3] & keep)
                store8(cute.local_tile(sKpe, (1, 8), (tidx, 2 * j + 1)), a0, a1, a2, a3)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()

            # ---- S^T: latent = sum over terms of block-scaled FP4 MMAs; RoPE = FP16 MMAs, its own accumulator
            if warp == 0:
                hs = s_prod.acquire_and_advance()
                cute.copy(cp_sfa, sfa_src[None, None, None, None, 0], sfa_dst)
                mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(CKV // 64):
                    mma_qk.set(tcgen05.Field.SFA, tCtSFA[None, None, k].iterator)
                    mma_qk.set(tcgen05.Field.SFB, tCtSFB[None, None, k].iterator)
                    cute.gemm(mma_qk, tCtSn[None, None, None, 0], tCrK4[None, None, k], tCrQ4[None, None, k],
                              tCtSn[None, None, None, 0])
                    mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                mma_pe.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(KPE // 16):
                    cute.gemm(mma_pe, tCtSp[None, None, None, 0], tCrKpe[None, None, k], tCrQpe[None, None, k],
                              tCtSp[None, None, None, 0])
                    mma_pe.set(tcgen05.Field.ACCUMULATE, True)
                hs.commit()

            if cutlass.const_expr(self.v_tmem):
                # ---- V^T into TMEM while the QK MMAs run. Warp w, channel tile dm: raw bytes 64 dm + 16 w .. + 15
                # (16-byte chunk 4 dm + w), i.e. channels 128 dm + 32 w .. + 31, one 16-token block per ldmatrix.
                # Lane group G (lanes + 16 G) takes byte columns 8 G .. 8 G + 7: lane g -> channel 2g, g + 8 -> 2g + 1.
                t4 = lane % 4
                for dm in cutlass.range_constexpr(4):
                    for blk in cutlass.range_constexpr(TILE // 16):
                        vrow = 16 * blk + lane % 16  # not `row`: that one masks this thread's scores below
                        c = 4 * dm + warp
                        off = (c // 8) * 16384 + (vrow // 8) * 1024 + (vrow % 8) * 128 + ((c % 8) ^ (vrow % 8)) * 16
                        r = ldsm_t16(k4_base + Uint32(off))
                        for G in cutlass.range_constexpr(2):
                            w = v_words(r[G], sSfT[8 * dm + 2 * warp + G, 4 * blk + t4])
                            taddr = tmem_v + Uint32(((32 * warp + 16 * G) << 16) + dm * 64 + blk * 8)
                            tmem_st16x256(taddr, w[0], w[1], w[2], w[3])
                cute.arch.fence_view_async_tmem_store()

            # ---- online softmax over tokens (columns are heads); S = g_head * latent + RoPE
            hsc = s_cons.wait_and_advance()
            sS = cute.make_rmem_tensor(frag, Float32)
            sR = cute.make_rmem_tensor(frag, Float32)
            cute.copy(ld_s, tStSn[0][None, None, None, None, 0], sS)
            for t in cutlass.range_constexpr(1, terms):
                cute.copy(ld_s, tStSn[t][None, None, None, None, 0], sR)
                for i in cutlass.range_constexpr(n):
                    sS[i] = sS[i] + sR[i]
            cute.copy(ld_s, tStSp[None, None, None, None, 0], sR)
            cute.arch.fence_view_async_tmem_load()
            hsc.release()
            if cutlass.const_expr(self.dbg):  # first tile's scores: o_part[0] = latent * g, o_part[1] = RoPE
                if it == 0:
                    for i in cutlass.range_constexpr(n):
                        mOpart[0, b, h0 + tScS[i][1], row] = sS[i] * g_head[i]
                        mOpart[1, b, h0 + tScS[i][1], row] = sR[i]
            for i in cutlass.range_constexpr(n):
                sS[i] = sS[i] * g_head[i] + sR[i]
            if t_beg + it * TILE + row >= t_end:
                sS.fill(-Float32.inf)

            m_lane = Float32(0.0)
            for i in cutlass.range_constexpr(n):
                mi = cute.arch.warp_redux_sync(sS[i], kind="fmax", nan=True)
                if i == lane:
                    m_lane = mi
            if lane < n:
                cute.arch.atomic_fmax(tSsM.iterator + tSsM.layout(lane), m_lane, sem="relaxed", scope="cta")
            cute.arch.sync_threads()
            m_cur = cute.make_rmem_tensor(frag, Float32)
            cute.autovec_copy(tSsM, m_cur)

            pf = cute.make_rmem_tensor(frag, Float32)
            corr = cute.make_rmem_tensor(frag, Float32)
            for i in cutlass.range_constexpr(n):
                pf[i] = cute.math.exp2((sS[i] - m_cur[i]) * scale_log2, fastmath=True)
                corr[i] = cute.math.exp2((m_prev[i] - m_cur[i]) * scale_log2, fastmath=True)
                l_acc[i] = l_acc[i] * corr[i] + pf[i]
                m_prev[i] = m_cur[i]
            p16 = cute.make_rmem_tensor(frag, Float16)
            p16.store(pf.load().to(Float16))
            cute.autovec_copy(p16, tSsP[None, None, None, None, 0])
            cute.arch.fence_view_async_shared()

            if it > 0:
                for dm in cutlass.range_constexpr(4):
                    o = cute.make_rmem_tensor(frag, Float32)
                    cute.copy(ld_s, tStO[None, None, None, None, dm], o)
                    for i in cutlass.range_constexpr(n):
                        o[i] = o[i] * corr[i]
                    cute.copy(st_o, o, tOtO[None, None, None, None, dm])
                cute.arch.fence_view_async_tmem_store()
            cute.arch.sync_threads()

            # ---- O^T += V^T P^T
            if warp == 0:
                ho = o_prod.acquire_and_advance()
                for dm in cutlass.range_constexpr(4):
                    mma_vp.set(tcgen05.Field.ACCUMULATE, it > 0)
                    for k in cutlass.range_constexpr(TILE // 16):
                        if cutlass.const_expr(self.v_tmem):
                            cute.gemm(mma_vp, tCtO[None, None, None, dm], tAtV[None, None, k, dm],
                                      tCrP[None, None, k, 0], tCtO[None, None, None, dm])
                        else:
                            cute.gemm(mma_vp, tCtO[None, None, None, dm], tCrV[None, None, k, dm, 0],
                                      tCrP[None, None, k, 0], tCtO[None, None, None, dm])
                        mma_vp.set(tcgen05.Field.ACCUMULATE, True)
                ho.commit()
            hoc = o_cons.wait_and_advance()
            hoc.release()

        # ---- epilogue (as v0)
        l_lane = Float32(0.0)
        for i in cutlass.range_constexpr(n):
            li = cute.arch.warp_reduction_sum(l_acc[i])
            if i == lane:
                l_lane = li
        if lane < n:
            sL[warp, lane] = l_lane
        cute.arch.sync_threads()

        for dm in cutlass.range_constexpr(4):
            o = cute.make_rmem_tensor(frag, Float32)
            if n_tiles > 0:
                cute.copy(ld_s, tStO[None, None, None, None, dm], o)
            for i in cutlass.range_constexpr(n):
                head = tScS[i][1]
                l = sL[0, head] + sL[1, head] + sL[2, head] + sL[3, head]
                v = Float32(0.0)
                if l > 0.0:
                    v = o[i] * (out_scale / l)
                lrow = tScS[i][0]
                if cutlass.const_expr(self.v_tmem):  # lanes of a 16-lane group: even channels, then odd ones
                    lrow = (lrow & 0x70) | ((lrow & 7) << 1) | ((lrow >> 3) & 1)
                ch = dm * TILE + lrow
                if splits == 1:
                    mOut[b, h0 + head, ch] = v.to(BFloat16)
                else:
                    mOpart[z, b, h0 + head, ch] = v
        if splits > 1 and warp == 0 and lane < n:
            l = sL[0, lane] + sL[1, lane] + sL[2, lane] + sL[3, lane]
            lse = -Float32.inf
            if l > 0.0:
                lse = m_prev[lane] * scale_log2 + cute.math.log2(l, fastmath=False)
            mLse[z, b, h0 + lane] = lse

        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


@functools.lru_cache(maxsize=None)
def _compiled(n: int, page_size: int):
    sym = cute.sym_int
    b, h, p, mp, s = sym(), sym(), sym(), sym(), sym()

    def t(dtype, shape, align=16):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=tuple(range(len(shape)))[::-1],
                                                     assumed_align=align)

    return cute.compile(
        DecodeV0(n, page_size), t(BFloat16, (b, h, CKV)), t(BFloat16, (b, h, KPE)), t(Uint8, (p, page_size, CKV // 2)),
        t(Uint8, (p, page_size, CKV // 16)), t(Uint8, (p, page_size, KPE)), t(Int32, (b, mp), 4), t(Int32, (b,), 4),
        t(BFloat16, (b, h, CKV)), t(Float32, (s, b, h, CKV)), t(Float32, (s, b, h), 4), Int32(TILE), Int32(1),
        Float32(1.0), Float32(1.0), Float32(1.0), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi")


@functools.lru_cache(maxsize=None)
def _compiled1(n: int, page_size: int, terms: int, dbg: bool = False, v_tmem: bool = False):
    sym = cute.sym_int
    b, h, p, mp, s = sym(), sym(), sym(), sym(), sym()

    def t(dtype, shape, align=16):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=tuple(range(len(shape)))[::-1],
                                                     assumed_align=align)

    return cute.compile(
        DecodeV1(n, page_size, terms, dbg, v_tmem), t(Uint8, (b, h, terms, CKV // 2)), t(Uint8, (b, h, terms, CKV // 16), 4),
        t(Float32, (b, h), 4), t(BFloat16, (b, h, KPE)), t(Uint8, (p, page_size, CKV // 2)),
        t(Uint8, (p, page_size, CKV // 16)), t(Uint8, (p, page_size, KPE)), t(Int32, (b, mp), 4), t(Int32, (b,), 4),
        t(BFloat16, (b, h, CKV)), t(Float32, (s, b, h, CKV)), t(Float32, (s, b, h), 4), Int32(TILE), Int32(1),
        Float32(1.0), Float32(1.0), Float32(1.0), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi")


def decode(q_nope, q_pe, ckv_cache, ckv_sf_cache, kpe_cache, page_table, kv_lens, sm_scale, ckv_scale=1.0,
           kpe_scale=1.0, max_kv_len=None, num_splits=0, version=2, heads_per_cta=None, terms=4):
    """Same contract as nvfp4_mla.decode (BF16-or-better accuracy); SM100 only. version 0, 1 (v1a') or 2 (v1b)."""
    from . import _load

    B, H = q_nope.shape[:2]
    if H % 16:
        raise ValueError(f"heads must be a multiple of 16, got {H}")
    # v1a keeps v0's FP16 V tile in shared memory and v1b keeps V in TMEM: both leave room for 16 heads per CTA only
    n = heads_per_cta or (32 if version == 0 and H % 32 == 0 else 16)
    max_len = max(int(kv_lens.max()) if max_kv_len is None else int(max_kv_len), 1)
    tiles = -(-max_len // TILE)
    if num_splits <= 0:  # ponytail: one wave of one CTA per SM; v0 has no tuned split policy
        sms = torch.cuda.get_device_properties(q_nope.device).multi_processor_count
        num_splits = max(1, min(tiles, round(sms / (B * H // n))))
    chunk = -(-tiles // min(num_splits, tiles)) * TILE
    splits = -(-max_len // chunk)

    out = torch.empty(B, H, CKV, dtype=torch.bfloat16, device=q_nope.device)
    o_part = torch.empty(splits, B, H, CKV, dtype=torch.float32, device=q_nope.device)
    lse_part = torch.empty(splits, B, H, dtype=torch.float32, device=q_nope.device)
    args = (ckv_cache, ckv_sf_cache.view(torch.uint8), kpe_cache.view(torch.uint8), page_table, kv_lens, out, o_part,
            lse_part, chunk, splits, sm_scale * ckv_scale * LOG2E, kpe_scale / ckv_scale, ckv_scale)
    if version == 0:
        _compiled(n, ckv_cache.shape[1])(q_nope, q_pe, *args)
    else:
        q4, qsf, g = _load().query_terms(q_nope, terms)
        _compiled1(n, ckv_cache.shape[1], terms, False, version == 2)(q4, qsf, g, q_pe, *args)
    if splits > 1:
        _load().combine(o_part, lse_part, out)
    return out
