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
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
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
def v_words8(w: Uint32, sf4: Uint32, *, loc=None, ip=None):
    """As v_words, rounded to E4M3: (channel 2c: tokens 0-3), (channel 2c + 1: tokens 0-3), token 0 in the low byte.
    The values are (e2m1 x scale) / 8, at most 336: e2m1 x scale reaches 2688, past E4M3's 448, when the cache's
    global scale is calibrated (block scales near 448)."""
    r = _asm(llvm.StructType.get_literal([T.i32()] * 2), [Uint32(w), Uint32(sf4)],
             "{\n.reg .b8 b0, b1, b2, b3;\n.reg .b16 s0, s1, q0, q1, q2, q3;\n"
             ".reg .b32 h0, h1, h2, h3, p01, p23, e, f, g, k, c;\n"
             "mov.b32 {b0, b1, b2, b3}, $2;\nmov.b32 {s0, s1}, $3;\nmov.b32 c, 0x30003000;\n"
             "cvt.rn.f16x2.e4m3x2 p01, s0;\ncvt.rn.f16x2.e4m3x2 p23, s1;\n"
             "mul.rn.f16x2 p01, p01, c;\nmul.rn.f16x2 p23, p23, c;\n"
             "cvt.rn.f16x2.e2m1x2 h0, b0;\ncvt.rn.f16x2.e2m1x2 h1, b1;\n"
             "cvt.rn.f16x2.e2m1x2 h2, b2;\ncvt.rn.f16x2.e2m1x2 h3, b3;\n"
             "prmt.b32 e, h0, h1, 0x5410;\nmul.rn.f16x2 e, e, p01;\nprmt.b32 f, h2, h3, 0x5410;\nmul.rn.f16x2 f, f, p23;\n"
             "prmt.b32 g, h0, h1, 0x7632;\nmul.rn.f16x2 g, g, p01;\nprmt.b32 k, h2, h3, 0x7632;\nmul.rn.f16x2 k, k, p23;\n"
             "cvt.rn.satfinite.e4m3x2.f16x2 q0, e;\ncvt.rn.satfinite.e4m3x2.f16x2 q1, f;\n"
             "cvt.rn.satfinite.e4m3x2.f16x2 q2, g;\ncvt.rn.satfinite.e4m3x2.f16x2 q3, k;\n"
             "mov.b32 $0, {q0, q1};\nmov.b32 $1, {q2, q3};\n}",
             "=r,=r,r,r", loc, ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def tmem_st16x128(taddr: Uint32, w0: Uint32, w1: Uint32, *, loc=None, ip=None):
    """tcgen05.st.16x128b.x1: thread (g, t) writes lane g column t (w0) and lane g + 8 column t (w1)."""
    llvm.inline_asm(None, [Uint32(x).ir_value(loc=loc, ip=ip) for x in (taddr, w0, w1)],
                    "tcgen05.st.sync.aligned.16x128b.x1.b32 [$0], {$1, $2};", "r,r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def tmem_st16x256(taddr: Uint32, w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None):
    """tcgen05.st.16x256b.x1: thread (g, t) writes lane g columns 2t, 2t + 1 (w0, w1) and lane g + 8 (w2, w3)."""
    llvm.inline_asm(None, [Uint32(x).ir_value(loc=loc, ip=ip) for x in (taddr, w0, w1, w2, w3)],
                    "tcgen05.st.sync.aligned.16x256b.x1.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r",
                    has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def cp_async16(dst: Uint32, src: Int64, nbytes: Uint32, *, loc=None, ip=None):
    """16 bytes global -> shared, bypassing L1; nbytes = 0 zero-fills the destination without reading."""
    llvm.inline_asm(None, [Uint32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip),
                           Uint32(nbytes).ir_value(loc=loc, ip=ip)],
                    "cp.async.cg.shared.global [$0], [$1], 16, $2;", "r,l,r", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.commit_group;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def cp_async_wait(pending: int, *, loc=None, ip=None):
    llvm.inline_asm(None, [], f"cp.async.wait_group {pending};", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


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


class DecodeV2:
    """v2: v1b's data path with each step in its own warps, so consecutive tiles overlap.

    Warps 0-3 load (thread = token: cp.async of the raw bytes into a ring of tiles, then the scale layouts from shared
    memory), 4-7 convert V into TMEM, 8-11 run the softmax, warp 12 issues the MMAs; warps 13-15 only complete its
    warpgroup. Within a role, warp w % 4 owns TMEM lanes 32 (w % 4) .. + 31, its threads' tokens / channels.
    Registers are assigned per warpgroup, so a 13th warp would cost a whole one: 512 threads start at 128 registers
    and setmaxnreg moves them from load and MMA to convert and softmax (ptxas drops setmaxnreg unless the launch
    fixes the registers at entry: min_blocks_per_mp=1).

    feat (v2.2, each on its own for the ablation):
      lazy   a head's maximum moves only when it grows by more than 8 (log2), so most tiles skip the O rescale and
             the wait for PV(i - 1); two P buffers.
      rope8  RoPE as E4M3 x E4M3 on the raw kpe bytes (query as 3 E4M3 terms along K); the ring holds raw bytes only
             (4 stages), the scale layouts made from them have 2 buffers.
      qksm   QK(i + 1) issued by the first softmax warp right after S(i) is read; the MMA warp keeps only PV.
      vhalf  V converted and consumed in two 64-token halves (PV half by half).
    feat (v2.4, page size >= 16, needs rope8):
      tma    the raw tile comes by TMA (one box per page, issued by warp 13) instead of cp.async; warps 0-3 only
             write the scale layouts.
      cvt8   (needs tma) warps 0-3 convert too, two 128-channel chunks per warp; warps 14-15 write the scale
             layouts.
    feat (v2.5):
      v8     V and P in E4M3 (PV is kind::f8f6f4; V rounded as value / 8, P <= 256), so V takes 128 TMEM columns.
      st3    3 raw stages instead of 4.
      h32    32 heads per CTA (needs v8 for TMEM, st3 for shared memory, cvt8): the cache is read once per 32
             heads. The softmax gets 232 registers and reads the per-head scales from shared memory.
    """

    REGS = {0: 72, 1: 152, 2: 176, 3: 112}  # per role; sum x 128 threads = 65536
    REGS8 = {0: 128, 1: 128, 2: 176, 3: 80}  # cvt8
    REGS32 = {0: 104, 1: 104, 2: 232, 3: 72}  # h32: the softmax holds one value per head in 6 arrays
    FEATS = ("lazy", "rope8", "qksm", "vhalf", "tma", "cvt8", "v8", "st3", "h32")

    def __init__(self, n: int, page_size: int, terms: int, feat: str = "", stub: str = ""):
        # stub (measurement only, wrong results): roles in it skip their work but keep their pipeline signals
        self.n, self.ps, self.terms, self.stub = n, page_size, terms, stub
        (self.lazy, self.rope8, self.qksm, self.vhalf, self.tma, self.cvt8, self.v8, self.st3,
         self.h32) = (f in feat.split(",") for f in self.FEATS)
        if self.h32 != (n == 32) or self.h32 and not (self.v8 and self.st3 and self.cvt8):
            raise ValueError("32 heads per CTA go with h32, which needs v8, st3 and cvt8")
        self.regs = self.REGS32 if self.h32 else self.REGS8 if self.cvt8 else self.REGS
        if self.tma and not (self.rope8 and page_size >= 16) or self.cvt8 and not self.tma:
            raise ValueError("tma needs rope8 and page size >= 16; cvt8 needs tma")
        self.box = min(page_size, TILE)  # tokens per TMA box (one page, or a 128-token part of one)
        self.stages = 3 if self.st3 else 4 if self.rope8 else 2  # raw ring
        self.pv_t = Float8E4M3FN if self.v8 else Float16  # V and P: the PV MMA operands
        self.vbase = 512 - TILE * self.pv_t.width // 8  # TMEM column of V^T: 4 chunks of TILE x TILE values at the end
        self.vshift = 8.0 if self.v8 else 1.0  # v8 rounds V as value / 8
        self.nd = 2  # buffers of the scale layouts derived from the raw scales (tile index % nd)
        self.np = 2 if self.lazy else 1  # P buffers
        self.nv = 2 if self.vhalf else 1  # V parts

    @cute.jit
    def __call__(self, q4: cute.Tensor, qsf: cute.Tensor, qg: cute.Tensor, q_pe: cute.Tensor, qpe8: cute.Tensor,
                 qps: cute.Tensor, ckv: cute.Tensor, sf: cute.Tensor, kpe: cute.Tensor, page_table: cute.Tensor,
                 kv_lens: cute.Tensor, out: cute.Tensor, o_part: cute.Tensor, lse_part: cute.Tensor, chunk: Int32,
                 splits: Int32, scale_log2: Float32, kpe_ratio: Float32, out_scale: Float32, stream):
        n, terms, st = self.n, self.terms, self.stages
        nq = terms * n
        mma_qk = cute.make_tiled_mma(tcgen05.MmaMXF4NVF4Op(Float8E4M3FN, (TILE, nq, 64), tcgen05.CtaGroup.ONE,
                                                           tcgen05.OperandSource.SMEM))
        pe_t = Float8E4M3FN if self.rope8 else Float16
        mma_pe = sm100_utils.make_trivial_tiled_mma(pe_t, pe_t, OperandMajorMode.K, OperandMajorMode.K, Float32,
                                                    tcgen05.CtaGroup.ONE, (TILE, n))
        mma_vp = sm100_utils.make_trivial_tiled_mma(self.pv_t, self.pv_t, OperandMajorMode.K, OperandMajorMode.MN,
                                                    Float32, tcgen05.CtaGroup.ONE, (TILE, n), tcgen05.OperandSource.TMEM)
        mma_sn = sm100_utils.make_trivial_tiled_mma(Float16, OperandMajorMode.K, OperandMajorMode.K, Float32,
                                                    tcgen05.CtaGroup.ONE, (TILE, nq))
        atom4 = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.K_SW128, Float4E2M1FN)
        kind_pe = tcgen05.mma.SmemLayoutAtomKind.K_SW64 if self.rope8 else tcgen05.mma.SmemLayoutAtomKind.K_SW128
        atom_pe = tcgen05.make_smem_layout_atom(kind_pe, pe_t)
        tiler = (TILE, nq, CKV)
        npe = 3 if self.rope8 else 1  # query RoPE terms, along K
        tma = (None,) * 6
        if cutlass.const_expr(self.tma):  # the cache as [pages x page size, bytes per token]; one box = one page
            rows, bx, op = ckv.shape[0] * self.ps, self.box, cpasync.CopyBulkTensorTileG2SOp()
            flat = lambda t, w: cute.make_tensor(t.iterator, cute.make_layout((rows, w), stride=(w, 1)))
            rowl = lambda w: cute.make_layout((bx, w), stride=(w, 1))
            tma_k, mTK = cpasync.make_tiled_tma_atom(
                op, flat(ckv, CKV // 2), cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, rowl(128)), (bx, 128))
            tma_sf, mTSf = cpasync.make_tiled_tma_atom(op, flat(sf, CKV // 16), rowl(CKV // 16), (bx, CKV // 16))
            tma_pe, mTPe = cpasync.make_tiled_tma_atom(
                op, flat(kpe, KPE), cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0, rowl(KPE)), (bx, KPE))
            tma = (tma_k, mTK, tma_sf, mTSf, tma_pe, mTPe)
        self.kernel(q4, qsf, qg, q_pe, qpe8, qps, ckv, sf, kpe, page_table, kv_lens, out, o_part, lse_part, chunk,
                    scale_log2, kpe_ratio, out_scale, *tma, mma_qk, mma_pe, mma_vp, mma_sn,
                    cute.tile_to_shape(atom4, (TILE, CKV, st), order=(0, 1, 2)),
                    cute.tile_to_shape(atom4, (nq, CKV), order=(0, 1)),
                    cute.tile_to_shape(atom_pe, (TILE, KPE, st if self.rope8 else self.nd), order=(0, 1, 2)),
                    cute.tile_to_shape(atom_pe, (n, KPE, npe), order=(0, 1, 2)),
                    sm100_utils.make_smem_layout_b(mma_vp, (TILE, n, TILE), self.pv_t, self.np),
                    sm100_utils.make_smem_layout_b(mma_vp, (TILE, n, TILE), self.pv_t, 1),
                    bsl.make_smem_layout_sfa(mma_qk, tiler, SF_VEC, self.nd),
                    bsl.make_smem_layout_sfb(mma_qk, tiler, SF_VEC, 1)).launch(
            grid=[q_pe.shape[1] // n, q_pe.shape[0], splits], block=[512, 1, 1], min_blocks_per_mp=1, stream=stream)

    def views(self, mma_qk, mma_pe, mma_vp, mma_sn, lsfa, lsfb, lp1, tmem_ptr, sK4, sQ4, sKpe, sQpe, sP, sSFA, sSFB,
              sG, sC, wt):
        """Operand fragments, TMEM tensors and copy partitions. Called inside each role after its setmaxnreg, so none
        of it is live across the register reassignment (unused parts are dead code in each role)."""
        n, terms = self.n, self.terms
        nq = terms * n
        thr_qk = mma_qk.get_slice(0)
        thr_pe = mma_pe.get_slice(0)
        thr_vp = mma_vp.get_slice(0)
        tCrK4 = mma_qk.make_fragment_A(thr_qk.partition_A(sK4))  # (MMA, 1, 8, stages)
        tCrQ4 = mma_qk.make_fragment_B(thr_qk.partition_B(sQ4))  # (MMA, 1, 8)
        tCrKpe = mma_pe.make_fragment_A(thr_pe.partition_A(sKpe))  # (MMA, 1, kpe k-steps, buffers)
        tCrQpe = mma_pe.make_fragment_B(thr_pe.partition_B(sQpe))  # (MMA, 1, k-steps, query terms)
        tCrP = mma_vp.make_fragment_B(sP)  # (MMA, 1, 8, P buffers)

        # TMEM: S latent (terms x n), S RoPE (n), O^T (4 x n), SFA, SFB; V^T in columns 256-511
        s_shape = mma_vp.partition_shape_C((TILE, n, 1))
        s_layout = mma_vp.make_fragment_C(s_shape).layout
        tCtSn = cute.make_tensor(tmem_ptr, mma_sn.make_fragment_C(mma_sn.partition_shape_C((TILE, nq, 1))).layout)
        col_n = tcgen05.find_tmem_tensor_col_offset(tCtSn)
        tCtSp = cute.make_tensor(tmem_ptr + col_n, s_layout)
        col = tcgen05.find_tmem_tensor_col_offset(tCtSp)
        tCtO = cute.make_tensor(tmem_ptr + col_n + col,
                                mma_vp.make_fragment_C(mma_vp.partition_shape_C((TILE, n, 4))).layout)
        col_sf = col_n + col + tcgen05.find_tmem_tensor_col_offset(tCtO)
        tCtSFA = cute.make_tensor(cute.recast_ptr(tmem_ptr + col_sf, dtype=Float8E4M3FN),
                                  bsl.make_tmem_layout_sfa(mma_qk, (TILE, n, CKV), SF_VEC,
                                                           cute.slice_(lsfa, (None, None, None, 0))))
        col_sfa = tcgen05.find_tmem_tensor_col_offset(tCtSFA)
        tCtSFB = cute.make_tensor(cute.recast_ptr(tmem_ptr + col_sf + col_sfa, dtype=Float8E4M3FN),
                                  bsl.make_tmem_layout_sfb(mma_qk, (TILE, n, CKV), SF_VEC,
                                                           cute.slice_(lsfb, (None, None, None, 0))))
        tAtV = cute.make_tensor(cute.recast_ptr(tmem_ptr + self.vbase, dtype=self.pv_t),
                                thr_vp.make_fragment_A(mma_vp.partition_shape_A((TILE, TILE, 4))).layout)

        s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), Float8E4M3FN)
        cp_sfa = tcgen05.make_s2t_copy(s2t, cute.filter_zeros(tCtSFA))
        thr_cpa = cp_sfa.get_slice(0)
        sfa_src = tcgen05.get_s2t_smem_desc_tensor(cp_sfa, thr_cpa.partition_S(cute.filter_zeros(sSFA)))
        sfa_dst = thr_cpa.partition_D(cute.filter_zeros(tCtSFA))
        cp_sfb = tcgen05.make_s2t_copy(s2t, cute.filter_zeros(tCtSFB))
        thr_cpb = cp_sfb.get_slice(0)
        sfb_src = tcgen05.get_s2t_smem_desc_tensor(cp_sfb, thr_cpb.partition_S(cute.filter_zeros(sSFB)))
        sfb_dst = thr_cpb.partition_D(cute.filter_zeros(tCtSFB))

        # P^T buffer b as the (token, head) C tile of QK, so softmax threads store their own rows
        tCsP = [cute.composition(cute.make_tensor(cute.recast_ptr(sP.iterator + b * cute.cosize(lp1), lp1.inner,
                                                                  dtype=self.pv_t), lp1.outer),
                                 cute.make_ordered_layout(s_shape, order=((2, 0), 3, 1, 4))) for b in range(self.np)]
        rep = tcgen05.Repetition(n)
        ld_s = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.Ld32x32bOp(rep), Float32), tCtSp[None, None, None, 0])
        st_o = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(rep), Float32), tCtO[None, None, None, 0])
        thr_ld = ld_s.get_slice(wt)
        thr_st = st_o.get_slice(wt)
        tStSn = [thr_ld.partition_S(cute.make_tensor(tmem_ptr + t * col, s_layout)) for t in range(terms)]
        tStSp = thr_ld.partition_S(tCtSp)
        tSsP = [thr_ld.partition_D(x) for x in tCsP]
        tStO = thr_ld.partition_S(tCtO)
        tOtO = thr_st.partition_D(tCtO)
        bcast = cute.make_layout((TILE, n), stride=(0, 1))
        tSsG = thr_ld.partition_D(thr_vp.partition_C(cute.make_tensor(sG.iterator, bcast)))
        tSsC = thr_ld.partition_D(thr_vp.partition_C(cute.make_tensor(sC.iterator, bcast)))
        tScS = thr_ld.partition_D(thr_vp.partition_C(cute.make_identity_tensor((TILE, n))))
        frag = tSsP[0].shape[:-1]
        return locals()

    @cute.kernel
    def kernel(self, mQ4: cute.Tensor, mQsf: cute.Tensor, mQg: cute.Tensor, mQp: cute.Tensor, mQp8: cute.Tensor,
               mQps: cute.Tensor, mCkv: cute.Tensor, mSf: cute.Tensor, mKpe: cute.Tensor, mPt: cute.Tensor,
               mLens: cute.Tensor, mOut: cute.Tensor, mOpart: cute.Tensor, mLse: cute.Tensor, chunk: Int32,
               scale_log2: Float32, kpe_ratio: Float32, out_scale: Float32, tma_k, mTK, tma_sf, mTSf, tma_pe, mTPe,
               mma_qk: cute.TiledMma,
               mma_pe: cute.TiledMma, mma_vp: cute.TiledMma, mma_sn: cute.TiledMma, lk4: cute.ComposedLayout,
               lq4: cute.ComposedLayout, lkpe: cute.ComposedLayout, lqpe: cute.ComposedLayout,
               lp: cute.ComposedLayout, lp1: cute.ComposedLayout, lsfa: cute.Layout, lsfb: cute.Layout):
        n, ps, terms, nst, nd, npb, nv = self.n, self.ps, self.terms, self.stages, self.nd, self.np, self.nv
        nq = terms * n
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = cute.arch.lane_idx()
        role = warp // 4  # 0 load, 1 convert, 2 softmax, 3 MMA
        lw, wt = warp % 4, tidx % 128  # TMEM quadrant, thread within the role
        ht, b, z = cute.arch.block_idx()
        _, _, splits = cute.arch.grid_dim()
        h0 = ht * n

        @cute.struct
        class Storage:
            kvm: cute.struct.MemRange[cutlass.Int64, 2 * nst]
            kvc: cute.struct.MemRange[cutlass.Int64, 2 * nst]
            s_bar: cute.struct.MemRange[cutlass.Int64, 2]
            v_bar: cute.struct.MemRange[cutlass.Int64, 2 * nv]
            p_bar: cute.struct.MemRange[cutlass.Int64, 2 * npb]
            o_bar: cute.struct.MemRange[cutlass.Int64, 1]  # one arrival per finished PV (lazy)
            tma_bar: cute.struct.MemRange[cutlass.Int64, nst]  # tma: the raw bytes of a stage have landed
            tmem_buf: Int32

        smem = utils.SmemAllocator()
        stg = smem.allocate(Storage)
        sK4 = smem.allocate_tensor(Float4E2M1FN, lk4.outer, 1024, lk4.inner)
        sQ4 = smem.allocate_tensor(Float4E2M1FN, lq4.outer, 1024, lq4.inner)
        pe_t = Float8E4M3FN if self.rope8 else Float16
        sKpe = smem.allocate_tensor(pe_t, lkpe.outer, 1024, lkpe.inner)  # rope8: the raw ring; else FP16, nd buffers
        sQpe = smem.allocate_tensor(pe_t, lqpe.outer, 1024, lqpe.inner)
        sP = smem.allocate_tensor(self.pv_t, lp.outer, 1024, lp.inner)
        sSFA = smem.allocate_tensor(Float8E4M3FN, lsfa, 128)
        sSFB = smem.allocate_tensor(Float8E4M3FN, lsfb, 128)
        # block scales again, block-major: one 32-bit load gives 4 tokens' scales of one block
        sSfT = smem.allocate_tensor(Uint32, cute.make_layout((CKV // 16, TILE // 4, nd), stride=(TILE // 4, 1, 1024)), 16)
        sSfT8 = cute.make_tensor(cute.recast_ptr(sSfT.iterator, dtype=Uint8),
                                 cute.make_layout((CKV // 16, TILE, nd), stride=(TILE, 1, 4096)))
        # raw scales (and without rope8 the raw RoPE bytes) as copied, per stage: [token][32 B], [token][64 B]
        sSfR = smem.allocate_tensor(Uint32, cute.make_layout((nst, TILE, 8), stride=(TILE * 8, 8, 1)), 16)
        kr = (1, 1) if self.rope8 else (nst, TILE)  # rope8 copies RoPE straight into sKpe
        sKpR = smem.allocate_tensor(Uint32, cute.make_layout((kr[0], kr[1], 16), stride=(kr[1] * 16, 16, 1)), 16)
        sMx = smem.allocate_tensor(Float32, cute.make_layout((2, 4, n), stride=(4 * n, n, 1)), 16)  # tile maxima
        sL = smem.allocate_tensor(Float32, cute.make_layout((4, n), stride=(n, 1)), 16)
        sG = smem.allocate_tensor(Float32, cute.make_layout(n), 16)
        sC = smem.allocate_tensor(Float32, cute.make_layout(n), 16)  # rope8: RoPE scale per head
        # SF tiles as words: [m % 32, m / 32, j / 4] (byte 512 (j / 4) + 16 (m % 32) + 4 (m / 32) + j % 4)
        sfa_w = cute.make_tensor(cute.recast_ptr(sSFA.iterator, dtype=Uint32),
                                 cute.make_layout((32, 4, CKV // 64, nd), stride=(4, 1, 128, 1024)))
        sfb_w = cute.make_tensor(cute.recast_ptr(sSFB.iterator, dtype=Uint32),
                                 cute.make_layout((32, 4, CKV // 64), stride=(4, 1, 128)))
        o_bar, tma_bar = stg.o_bar.data_ptr(), stg.tma_bar.data_ptr()

        thr128 = lambda: pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        thr = lambda k: pipeline.CooperativeGroup(pipeline.Agent.Thread, k)
        one = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        n_lay = 64 if self.cvt8 else 128  # threads writing the scale layouts (they publish a stage)
        n_cvt = 256 if self.cvt8 else 128
        kvm = pipeline.PipelineAsyncUmma.create(num_stages=nst, producer_group=thr(n_lay), consumer_group=one,
                                                barrier_storage=stg.kvm.data_ptr(), defer_sync=True)
        kvc = pipeline.PipelineAsync.create(num_stages=nst, producer_group=thr(n_lay), consumer_group=thr(n_cvt),
                                            barrier_storage=stg.kvc.data_ptr(), defer_sync=True)
        _, kvm_c = kvm.make_participants()
        _, kvc_c = kvc.make_participants()
        s_p, s_c = pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=one, consumer_group=thr128(),
                                                     barrier_storage=stg.s_bar.data_ptr(), defer_sync=True
                                                     ).make_participants()
        v_p, v_c = pipeline.PipelineAsyncUmma.create(num_stages=nv, producer_group=thr(n_cvt), consumer_group=one,
                                                     barrier_storage=stg.v_bar.data_ptr(), defer_sync=True
                                                     ).make_participants()
        p_p, p_c = pipeline.PipelineAsyncUmma.create(num_stages=npb, producer_group=thr128(), consumer_group=one,
                                                     barrier_storage=stg.p_bar.data_ptr(), defer_sync=True
                                                     ).make_participants()
        softmax_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        if tidx == 0:
            cute.arch.mbarrier_init(o_bar, 1)
            if cutlass.const_expr(self.tma):
                for s in cutlass.range_constexpr(nst):
                    cute.arch.mbarrier_init(tma_bar + s, 1)
        cute.arch.mbarrier_init_fence()
        tmem = utils.TmemAllocator(stg.tmem_buf.ptr,
                                   barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=512))
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)

        # ---- prologue (all 512 threads): query terms and scales; RoPE part (E4M3 terms, or FP16 x kpe / ckv scale)
        for i in cutlass.range(tidx, 1024, 512):
            sfb_w[i % 32, (i // 32) % 4, i // 128] = Uint32(0)
        cute.arch.sync_threads()
        for i in cutlass.range(tidx, n * terms * (CKV // 32), 512):
            r, t, c = i // (terms * (CKV // 32)), (i // (CKV // 32)) % terms, i % (CKV // 32)
            w = ld_v4(addr_of(mQ4) + ((Int64(b * mQ4.shape[1] + h0 + r) * terms + t) * (CKV // 2) + 16 * c))
            store8(cute.local_tile(sQ4, (1, 32), (t * n + r, c)), w[0], w[1], w[2], w[3])
        for i in cutlass.range(tidx, n * terms * (CKV // 64), 512):
            r, t, a = i // (terms * (CKV // 64)), (i // (CKV // 64)) % terms, i % (CKV // 64)
            sfb_w[(t * n + r) % 32, (t * n + r) // 32, a] = ld_u32(
                addr_of(mQsf) + ((Int64(b * mQsf.shape[1] + h0 + r) * terms + t) * (CKV // 16) + 4 * a))
        if cutlass.const_expr(self.rope8):
            for i in cutlass.range(tidx, n * 3 * (KPE // 16), 512):
                r, t, c = i // (3 * (KPE // 16)), (i // (KPE // 16)) % 3, i % (KPE // 16)
                w = ld_v4(addr_of(mQp8) + ((Int64(b * mQp8.shape[1] + h0 + r) * 3 + t) * KPE + 16 * c))
                store8(cute.local_tile(sQpe[None, None, t], (1, 16), (r, c)), w[0], w[1], w[2], w[3])
            if tidx < n:
                sC[tidx] = mQps[b, h0 + tidx] * kpe_ratio
        else:
            for i in cutlass.range(tidx, n * (KPE // 8), 512):
                r, c = i // (KPE // 8), i % (KPE // 8)
                w = ld_v4(addr_of(mQp) + (Int64(b * mQp.shape[1] + h0 + r) * KPE + c * 8) * 2)
                store8(cute.local_tile(sQpe[None, None, 0], (1, 8), (r, c)), bf16x2_f16x2(w[0], kpe_ratio),
                       bf16x2_f16x2(w[1], kpe_ratio), bf16x2_f16x2(w[2], kpe_ratio), bf16x2_f16x2(w[3], kpe_ratio))
            if tidx < n:
                sC[tidx] = Float32(1.0)
        if tidx < n:
            sG[tidx] = mQg[b, h0 + tidx]
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        t_beg = z * chunk
        t_end = cutlass.min(mLens[b], t_beg + chunk)
        n_tiles = cutlass.max(t_end - t_beg + TILE - 1, 0) // TILE
        VA = (mma_qk, mma_pe, mma_vp, mma_sn, lsfa, lsfb, lp1, tmem_ptr, sK4, sQ4, sKpe, sQpe, sP, sSFA, sSFB, sG, sC, wt)

        if role == (-1 if self.tma else 0):
            cute.arch.setmaxregister_decrease(self.REGS[0])
            # ---- load (cp.async): iteration it starts the copies of tile it into its stage and finishes tile
            # it - (stages - 1) (scale layouts; without rope8 also RoPE to FP16) once its copies have landed. Thread
            # wt = token wt of the tile; rows past the end are zero-filled by the copies.
            st_issue = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, nst)
            st_done = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, nst)
            k4b, sfrb = Uint32(sK4.iterator.toint()), Uint32(sSfR.iterator.toint())
            kpb = Uint32(sKpe.iterator.toint()) if self.rope8 else Uint32(sKpR.iterator.toint())
            nl = 0 if "load" in self.stub else 1  # stub: no global access at all (page table, copies)
            for it in cutlass.range(n_tiles + nst - 1):
                # Finish before acquiring: tile j is then published once tile j - 2 is released. Acquiring first
                # waited for tile j - 1, which put this loop's own latency into every consumer's tile period.
                if it >= nst - 1:
                    cp_async_wait(nst - 2)
                    sx = st_done.index
                    dx = (it - (nst - 1)) % nd
                    self.scale_layouts(sSfR, sfa_w, sSfT8, sx, dx, wt, None, None)
                    if cutlass.const_expr(not self.rope8):
                        for j in cutlass.range_constexpr(4):
                            a0, a1 = e4m3x4_f16(sKpR[sx, wt, 4 * j])
                            a2, a3 = e4m3x4_f16(sKpR[sx, wt, 4 * j + 1])
                            store8(cute.local_tile(sKpe[None, None, dx], (1, 8), (wt, 2 * j)), a0, a1, a2, a3)
                            a0, a1 = e4m3x4_f16(sKpR[sx, wt, 4 * j + 2])
                            a2, a3 = e4m3x4_f16(sKpR[sx, wt, 4 * j + 3])
                            store8(cute.local_tile(sKpe[None, None, dx], (1, 8), (wt, 2 * j + 1)), a0, a1, a2, a3)
                    cute.arch.fence_view_async_shared()
                    kvm.producer_commit(st_done)
                    kvc.producer_commit(st_done)
                    st_done.advance()
                # Acquired even past the last tile: the previous iteration's acquire is what proves tile
                # it - stages - 1 released, so the derived buffer (nd = 2) the step above overwrites is no longer read.
                kvm.producer_acquire(st_issue)
                kvc.producer_acquire(st_issue)
                if it < n_tiles:
                    sx = st_issue.index
                    tok = t_beg + it * TILE + wt
                    tokc = cutlass.min(tok, t_end - 1)
                    slot = Int64(0)
                    if cutlass.const_expr(nl == 1):
                        slot = Int64(mPt[b, tokc // ps]) * ps + tokc % ps
                    nb = Uint32(16) & Uint32(~((t_end - 1 - tok) >> 31))  # 16 for a valid row, 0 past the end
                    for j in cutlass.range_constexpr(nl * 16):
                        off = (j // 8) * 16384 + (wt // 8) * 1024 + (wt % 8) * 128 + ((j % 8) ^ (wt % 8)) * 16
                        cp_async16(k4b + Uint32(sx * (TILE * CKV // 2) + off), addr_of(mCkv) + slot * (CKV // 2) + 16 * j, nb)
                    for j in cutlass.range_constexpr(nl * 2):
                        cp_async16(sfrb + Uint32((sx * TILE + wt) * 32 + 16 * j), addr_of(mSf) + slot * (CKV // 16) + 16 * j, nb)
                    for j in cutlass.range_constexpr(nl * 4):
                        if cutlass.const_expr(self.rope8):  # straight into the E4M3 MMA layout (64-byte rows, SW64)
                            off = (wt // 8) * 512 + (wt % 8) * 64 + ((j ^ ((wt % 8) >> 1)) & 3) * 16
                            cp_async16(kpb + Uint32(sx * (TILE * KPE) + off), addr_of(mKpe) + slot * KPE + 16 * j, nb)
                        else:
                            cp_async16(kpb + Uint32((sx * TILE + wt) * 64 + 16 * j), addr_of(mKpe) + slot * KPE + 16 * j, nb)
                st_issue.advance()
                cp_async_commit()

        if role <= 1 and role >= (0 if self.cvt8 else 1):
            if cutlass.const_expr(not self.cvt8):
                cute.arch.setmaxregister_increase(self.REGS[1])
            elif cutlass.const_expr(self.regs[1] < 128):
                cute.arch.setmaxregister_decrease(self.regs[1])
            # ---- convert: V^T of the tile into TMEM (quadrant lw, chunk 4 dm + lw), in nv parts of 128 / nv tokens.
            # cvt8: warpgroup r converts dm = 2 r, 2 r + 1.
            ndm = 2 if self.cvt8 else 4
            dm0 = 2 * role if self.cvt8 else 0
            tmem_v = Uint32(tmem_ptr.toint()) + Uint32(self.vbase)
            k4_base = Uint32(sK4.iterator.toint())
            t4 = lane % 4
            for it in cutlass.range(n_tiles):
                hk = kvc_c.wait_and_advance()
                sx, dx = hk.index, it % nd
                for part in cutlass.range_constexpr(nv):
                    hv = v_p.acquire_and_advance()
                    for dmi in cutlass.range_constexpr(0 if "cvt" in self.stub else ndm):
                        dm = dm0 + dmi
                        for bb in cutlass.range_constexpr(TILE // 16 // nv):
                            blk = part * (TILE // 16 // nv) + bb
                            vrow = 16 * blk + lane % 16
                            c = 4 * dm + lw
                            off = (c // 8) * 16384 + (vrow // 8) * 1024 + (vrow % 8) * 128 + ((c % 8) ^ (vrow % 8)) * 16
                            r = ldsm_t16(k4_base + Uint32(sx * (TILE * CKV // 2) + off))
                            for G in cutlass.range_constexpr(2):
                                lanes = (32 * lw + 16 * G) << 16
                                if cutlass.const_expr(self.v8):  # 4 tokens per column
                                    w = v_words8(r[G], sSfT[8 * dm + 2 * lw + G, 4 * blk + t4, dx])
                                    tmem_st16x128(tmem_v + Uint32(lanes + dm * 32 + blk * 4), w[0], w[1])
                                else:
                                    w = v_words(r[G], sSfT[8 * dm + 2 * lw + G, 4 * blk + t4, dx])
                                    tmem_st16x256(tmem_v + Uint32(lanes + dm * 64 + blk * 8), w[0], w[1], w[2], w[3])
                    cute.arch.fence_view_async_tmem_store()
                    hv.commit()
                hk.release()

        if role == 3:
            cute.arch.setmaxregister_decrease(self.regs[3])
        if cutlass.const_expr(self.tma):
            if warp == 13:
                # ---- TMA warp: tile it into its stage as soon as the stage is free
                tv = self.tma_views(tma_k, mTK, tma_sf, mTSf, tma_pe, mTPe, sK4, sSfR, sKpe)
                st_issue = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, nst)
                for it in cutlass.range(n_tiles):
                    kvm.producer_acquire(st_issue)
                    kvc.producer_acquire(st_issue)
                    self.issue_tma(tv, (tma_k, tma_sf, tma_pe), tma_bar + st_issue.index, st_issue.index, mPt, b,
                                   t_beg + it * TILE, t_end)
                    st_issue.advance()
            # ---- scale layouts once a tile has landed: warps 0-3 (a token each), or with cvt8 warps 14-15 (two
            # each). They do not acquire the stage, so before writing the layouts of tile j they wait for tile j - nd
            # (the last user of that derived buffer) to be released: both empty barriers of its stage.
            lay0, nlay = (14, 2) if self.cvt8 else (0, 4)
            if warp >= lay0 and warp < lay0 + nlay:
                if cutlass.const_expr(not self.cvt8):
                    cute.arch.setmaxregister_decrease(self.REGS[0])
                tl = tidx - 32 * lay0
                st_done = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, nst)
                for j in cutlass.range(n_tiles):
                    sx = st_done.index
                    cute.arch.mbarrier_wait(tma_bar + sx, (j // nst) % 2)
                    if j >= nd:
                        sp, ph = (j - nd) % nst, ((j - nd) // nst) % 2
                        cute.arch.mbarrier_wait(stg.kvm.data_ptr() + nst + sp, ph)
                        cute.arch.mbarrier_wait(stg.kvc.data_ptr() + nst + sp, ph)
                    for q in cutlass.range_constexpr(4 // nlay):
                        tk = tl + 32 * nlay * q
                        self.scale_layouts(sSfR, sfa_w, sSfT8, sx, j % nd, tk, t_beg + j * TILE + tk, t_end)
                    cute.arch.fence_view_async_shared()
                    kvm.producer_commit(st_done)
                    kvc.producer_commit(st_done)
                    st_done.advance()
        if warp == 12:
            # ---- MMA warp. Without qksm: SFB once; iteration it issues QK(it), then PV(it - 1). With qksm: PV only.
            v = self.views(*VA)
            tCrK4, tCrQ4, tCrKpe, tCrQpe, tCrP = v["tCrK4"], v["tCrQ4"], v["tCrKpe"], v["tCrQpe"], v["tCrP"]
            tCtSn, tCtSp, tCtO, tCtSFA, tCtSFB, tAtV = (v["tCtSn"], v["tCtSp"], v["tCtO"], v["tCtSFA"], v["tCtSFB"],
                                                        v["tAtV"])
            cp_sfa, sfa_src, sfa_dst, cp_sfb, sfb_src, sfb_dst = (v["cp_sfa"], v["sfa_src"], v["sfa_dst"], v["cp_sfb"],
                                                                  v["sfb_src"], v["sfb_dst"])
            if cutlass.const_expr(not self.qksm):
                if n_tiles > 0:
                    cute.copy(cp_sfb, sfb_src[None, None, None, None, 0], sfb_dst)
            for it in cutlass.range(n_tiles + 1):
                if cutlass.const_expr(not self.qksm):
                    if it < n_tiles:
                        hk = kvm_c.wait_and_advance()
                        hs = s_p.acquire_and_advance()
                        cute.copy(cp_sfa, sfa_src[None, None, None, None, it % nd], sfa_dst)
                        mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                        for k in cutlass.range_constexpr(0 if "qk" in self.stub else CKV // 64):
                            mma_qk.set(tcgen05.Field.SFA, tCtSFA[None, None, k].iterator)
                            mma_qk.set(tcgen05.Field.SFB, tCtSFB[None, None, k].iterator)
                            cute.gemm(mma_qk, tCtSn[None, None, None, 0], tCrK4[None, None, k, hk.index],
                                      tCrQ4[None, None, k], tCtSn[None, None, None, 0])
                            mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                        mma_pe.set(tcgen05.Field.ACCUMULATE, False)
                        for t in cutlass.range_constexpr(3 if self.rope8 else 1):  # rope8: the 3 query terms along K
                            for k in cutlass.range_constexpr(2 if self.rope8 else KPE // 16):
                                cute.gemm(mma_pe, tCtSp[None, None, None, 0], tCrKpe[None, None, k, hk.index if self.rope8 else it % nd],
                                          tCrQpe[None, None, k, t], tCtSp[None, None, None, 0])
                                mma_pe.set(tcgen05.Field.ACCUMULATE, True)
                        hs.commit()
                        hk.release()
                if it > 0:
                    hp = p_c.wait_and_advance()
                    for part in cutlass.range_constexpr(nv):
                        hv = v_c.wait_and_advance()
                        for dm in cutlass.range_constexpr(0 if "pv" in self.stub else 4):
                            nk = TILE // (32 if self.v8 else 16) // nv  # MMA k-steps per V part
                            for kk in cutlass.range_constexpr(nk):
                                k = part * nk + kk
                                mma_vp.set(tcgen05.Field.ACCUMULATE, (it > 1) | (k > 0))
                                cute.gemm(mma_vp, tCtO[None, None, None, dm], tAtV[None, None, k, dm],
                                          tCrP[None, None, k, hp.index], tCtO[None, None, None, dm])
                        hv.release()
                    hp.release()
                    if cutlass.const_expr(self.lazy):
                        with cute.arch.elect_one():
                            tcgen05.commit(o_bar)

        if role == 2:
            cute.arch.setmaxregister_increase(self.regs[2])
            # ---- softmax: thread wt = token row wt of S^T (TMEM lane wt)
            v = self.views(*VA)
            ld_s, st_o, tStSn, tStSp, tSsP, tStO, tOtO = (v["ld_s"], v["st_o"], v["tStSn"], v["tStSp"], v["tSsP"],
                                                          v["tStO"], v["tOtO"])
            tSsG, tSsC, tScS, frag = v["tSsG"], v["tSsC"], v["tScS"], v["frag"]
            g_head = cute.make_rmem_tensor(frag, Float32)
            c_head = cute.make_rmem_tensor(frag, Float32)
            if cutlass.const_expr(not self.h32):  # h32 reads them from shared memory each tile
                cute.autovec_copy(tSsG, g_head)
                cute.autovec_copy(tSsC, c_head)
            m_prev = cute.make_rmem_tensor(frag, Float32)
            m_prev.fill(-Float32.inf)
            l_acc = cute.make_rmem_tensor(frag, Float32)
            l_acc.fill(0.0)
            row = tScS[0][0]
            if cutlass.const_expr(self.qksm):  # QK(0) from the first softmax warp
                if lw == 0:
                    if n_tiles > 0:
                        cute.copy(v["cp_sfb"], v["sfb_src"][None, None, None, None, 0], v["sfb_dst"])
                        hk = kvm_c.wait_and_advance()
                        hs = s_p.acquire_and_advance()
                        cute.copy(v["cp_sfa"], v["sfa_src"][None, None, None, None, 0], v["sfa_dst"])
                        mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                        for k in cutlass.range_constexpr(0 if "qk" in self.stub else CKV // 64):
                            mma_qk.set(tcgen05.Field.SFA, v["tCtSFA"][None, None, k].iterator)
                            mma_qk.set(tcgen05.Field.SFB, v["tCtSFB"][None, None, k].iterator)
                            cute.gemm(mma_qk, v["tCtSn"][None, None, None, 0], v["tCrK4"][None, None, k, hk.index],
                                      v["tCrQ4"][None, None, k], v["tCtSn"][None, None, None, 0])
                            mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                        mma_pe.set(tcgen05.Field.ACCUMULATE, False)
                        for t in cutlass.range_constexpr(3 if self.rope8 else 1):  # rope8: the 3 query terms along K
                            for k in cutlass.range_constexpr(2 if self.rope8 else KPE // 16):
                                cute.gemm(mma_pe, v["tCtSp"][None, None, None, 0], v["tCrKpe"][None, None, k, hk.index if self.rope8 else 0],
                                          v["tCrQpe"][None, None, k, t], v["tCtSp"][None, None, None, 0])
                                mma_pe.set(tcgen05.Field.ACCUMULATE, True)
                        hs.commit()
                        hk.release()
            for it in cutlass.range(n_tiles):
                hs = s_c.wait_and_advance()
                sS = cute.make_rmem_tensor(frag, Float32)
                sR = cute.make_rmem_tensor(frag, Float32)
                cute.copy(ld_s, tStSn[0][None, None, None, None, 0], sS)
                for t in cutlass.range_constexpr(1, terms):
                    cute.copy(ld_s, tStSn[t][None, None, None, None, 0], sR)
                    for i in cutlass.range_constexpr(n):
                        sS[i] = sS[i] + sR[i]
                cute.copy(ld_s, tStSp[None, None, None, None, 0], sR)
                cute.arch.fence_view_async_tmem_load()
                hs.release()
                for i in cutlass.range_constexpr(n):
                    if cutlass.const_expr(self.h32):
                        sS[i] = sS[i] * sG[tScS[i][1]] + sR[i] * sC[tScS[i][1]]
                    else:
                        sS[i] = sS[i] * g_head[i] + sR[i] * c_head[i]
                if t_beg + it * TILE + row >= t_end:
                    sS.fill(-Float32.inf)
                # tile maximum per head: warp reduction, one slot per warp (parity double-buffered), named barrier
                m_lane = Float32(0.0)
                for i in cutlass.range_constexpr(n):
                    mi = cute.arch.warp_redux_sync(sS[i], kind="fmax", nan=True)
                    if i == lane:
                        m_lane = mi
                if lane < n:
                    sMx[it % 2, lw, lane] = m_lane
                softmax_bar.arrive_and_wait()  # also: all four warps have read S(it)
                if cutlass.const_expr(self.qksm):
                    if lw == 0:
                        if it + 1 < n_tiles:
                            hk = kvm_c.wait_and_advance()
                            hs2 = s_p.acquire_and_advance()
                            cute.copy(v["cp_sfa"], v["sfa_src"][None, None, None, None, (it + 1) % nd], v["sfa_dst"])
                            mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                            for k in cutlass.range_constexpr(0 if "qk" in self.stub else CKV // 64):
                                mma_qk.set(tcgen05.Field.SFA, v["tCtSFA"][None, None, k].iterator)
                                mma_qk.set(tcgen05.Field.SFB, v["tCtSFB"][None, None, k].iterator)
                                cute.gemm(mma_qk, v["tCtSn"][None, None, None, 0], v["tCrK4"][None, None, k, hk.index],
                                          v["tCrQ4"][None, None, k], v["tCtSn"][None, None, None, 0])
                                mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                            mma_pe.set(tcgen05.Field.ACCUMULATE, False)
                            for t in cutlass.range_constexpr(3 if self.rope8 else 1):  # rope8: the 3 query terms along K
                                for k in cutlass.range_constexpr(2 if self.rope8 else KPE // 16):
                                    cute.gemm(mma_pe, v["tCtSp"][None, None, None, 0], v["tCrKpe"][None, None, k, hk.index if self.rope8 else (it + 1) % nd],
                                              v["tCrQpe"][None, None, k, t], v["tCtSp"][None, None, None, 0])
                                    mma_pe.set(tcgen05.Field.ACCUMULATE, True)
                            hs2.commit()
                            hk.release()
                pf = cute.make_rmem_tensor(frag, Float32)
                corr = cute.make_rmem_tensor(frag, Float32)
                need = Int32(0)
                for i in cutlass.range_constexpr(n):
                    hd = tScS[i][1]
                    mt = cutlass.max(cutlass.max(sMx[it % 2, 0, hd], sMx[it % 2, 1, hd]),
                                     cutlass.max(sMx[it % 2, 2, hd], sMx[it % 2, 3, hd]))
                    m_new = cutlass.max(m_prev[i], mt)
                    if cutlass.const_expr(self.lazy):  # keep the old maximum unless it grows by more than 2^8
                        m_new = m_prev[i]
                        # the load stub leaves stale shared memory as K: keep its rescale count at the real ~0
                        if (mt - m_prev[i]) * scale_log2 > (Float32.inf if "load" in self.stub else 8.0):
                            m_new = mt
                            if it > 0:
                                need = Int32(1)
                    pf[i] = cute.math.exp2((sS[i] - m_new) * scale_log2, fastmath=True)
                    corr[i] = cute.math.exp2((m_prev[i] - m_new) * scale_log2, fastmath=True)
                    l_acc[i] = l_acc[i] * corr[i] + pf[i]
                    m_prev[i] = m_new
                hp = p_p.acquire_and_advance()  # P buffer free (PV(it - np) done)
                if cutlass.const_expr(self.lazy):
                    if need > 0:  # same in every softmax thread: wait for PV(it - 1), then rescale O
                        cute.arch.mbarrier_wait(o_bar, (it - 1) % 2)
                        self.rescale(ld_s, st_o, tStO, tOtO, corr, frag)
                else:
                    if it > 0:
                        self.rescale(ld_s, st_o, tStO, tOtO, corr, frag)
                p16 = cute.make_rmem_tensor(frag, self.pv_t)
                p16.store(pf.load().to(self.pv_t))
                if cutlass.const_expr(npb == 1):
                    cute.autovec_copy(p16, tSsP[0][None, None, None, None, 0])
                else:
                    if hp.index == 0:
                        cute.autovec_copy(p16, tSsP[0][None, None, None, None, 0])
                    else:
                        cute.autovec_copy(p16, tSsP[1][None, None, None, None, 0])
                cute.arch.fence_view_async_shared()
                hp.commit()

            # ---- epilogue: wait for the last PV, then O^T / l
            if cutlass.const_expr(self.lazy):
                if n_tiles > 0:
                    cute.arch.mbarrier_wait(o_bar, (n_tiles - 1) % 2)
            else:
                hp = p_p.acquire_and_advance()
            l_lane = Float32(0.0)
            for i in cutlass.range_constexpr(n):
                li = cute.arch.warp_reduction_sum(l_acc[i])
                if i == lane:
                    l_lane = li
            if lane < n:
                sL[lw, lane] = l_lane
            softmax_bar.arrive_and_wait()
            for dm in cutlass.range_constexpr(4):
                o = cute.make_rmem_tensor(frag, Float32)
                if n_tiles > 0:
                    cute.copy(ld_s, tStO[None, None, None, None, dm], o)
                for i in cutlass.range_constexpr(n):
                    head = tScS[i][1]
                    l = sL[0, head] + sL[1, head] + sL[2, head] + sL[3, head]
                    val = Float32(0.0)
                    if l > 0.0:
                        val = o[i] * (out_scale * self.vshift / l)
                    lrow = tScS[i][0]
                    ch = dm * TILE + ((lrow & 0x70) | ((lrow & 7) << 1) | ((lrow >> 3) & 1))
                    if splits == 1:
                        mOut[b, h0 + head, ch] = val.to(BFloat16)
                    else:
                        mOpart[z, b, h0 + head, ch] = val
            if splits > 1 and lw == 0 and lane < n:
                l = sL[0, lane] + sL[1, lane] + sL[2, lane] + sL[3, lane]
                lse = -Float32.inf
                if l > 0.0:
                    lse = m_prev[lane] * scale_log2 + cute.math.log2(l, fastmath=False)
                mLse[z, b, h0 + lane] = lse

        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    def scale_layouts(self, sSfR, sfa_w, sSfT8, sx, dx, tk, tok, t_end):
        """Scales of token tk of the tile in stage sx -> buffer dx: SFA words (QK) and block-major bytes (V). With
        TMA, rows past the end hold whatever the page had: their scales become 0, so V is exactly 0 there."""
        msk = None if tok is None else Uint32(~((t_end - 1 - tok) >> 31))
        for a in range(CKV // 64):
            w = Uint32(0x38383838) if "load" in self.stub else sSfR[sx, tk, a]
            if msk is not None:
                w = w & msk
            sfa_w[tk % 32, tk // 32, a, dx] = w
            for i in range(4):
                sSfT8[4 * a + i, tk, dx] = ((w >> (8 * i)) & Uint32(0xFF)).to(Uint8)

    def tma_views(self, tma_k, mTK, tma_sf, mTSf, tma_pe, mTPe, sK4, sSfR, sKpe):
        """TMA partitions: shared memory as (box, box index in tile, [half,] stage) in the layouts the consumers use
        (FP4 tile SWIZZLE_128B in two 128-byte halves, scales plain, RoPE SWIZZLE_64B); the cache as boxes of rows."""
        bx, nb, nst = self.box, TILE // self.box, self.stages
        sk = cute.make_tensor(cute.recast_ptr(sK4.iterator, cute.make_swizzle(3, 4, 3), dtype=Uint8),
                              cute.make_layout(((bx, 128), nb, 2, nst), stride=((128, 1), bx * 128, 16384, 32768)))
        ss = cute.make_tensor(cute.recast_ptr(sSfR.iterator, dtype=Uint8),
                              cute.make_layout(((bx, 32), nb, nst), stride=((32, 1), bx * 32, TILE * 32)))
        sp = cute.make_tensor(cute.recast_ptr(sKpe.iterator, cute.make_swizzle(2, 4, 3), dtype=Uint8),
                              cute.make_layout(((bx, 64), nb, nst), stride=((64, 1), bx * 64, TILE * 64)))
        one = cute.make_layout(1)
        part = lambda atom, s, g, w: cpasync.tma_partition(atom, 0, one, s,
                                                           cute.group_modes(cute.local_tile(g, (bx, w), (None, None)), 0, 2))
        return part(tma_k, sk, mTK, 128), part(tma_sf, ss, mTSf, 32), part(tma_pe, sp, mTPe, 64)

    def issue_tma(self, tv, atoms, bar, sx, mPt, b, tile0, t_end):
        """One warp (the DSL elects the lane of each TMA copy itself): the raw bytes of the tile starting at token tile0
        into stage sx. Boxes past the end load the last page again (all rows past the end are ignored), so the byte
        count is fixed."""
        (tks, tkg), (tss, tsg), (tps, tpg) = tv
        tma_k, tma_sf, tma_pe = atoms
        bx, ps = self.box, self.ps
        if "load" in self.stub:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(bar)
            return
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bar, TILE * (CKV // 2 + CKV // 16 + KPE))
        for pg in range(TILE // bx):
            tok = cutlass.min(tile0 + pg * bx, t_end - 1)
            rt = Int32(mPt[b, tok // ps]) * (ps // bx) + (tok % ps) // bx
            for hf in range(2):
                cute.copy(tma_k, tkg[None, rt, hf], tks[None, pg, hf, sx], tma_bar_ptr=bar)
            cute.copy(tma_sf, tsg[None, rt, 0], tss[None, pg, sx], tma_bar_ptr=bar)
            cute.copy(tma_pe, tpg[None, rt, 0], tps[None, pg, sx], tma_bar_ptr=bar)

    def rescale(self, ld_s, st_o, tStO, tOtO, corr, frag):
        for dm in range(0 if "rescale" in self.stub else 4):
            o = cute.make_rmem_tensor(frag, Float32)
            cute.copy(ld_s, tStO[None, None, None, None, dm], o)
            for i in range(self.n):
                o[i] = o[i] * corr[i]
            cute.copy(st_o, o, tOtO[None, None, None, None, dm])
        cute.arch.fence_view_async_tmem_store()


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


@functools.lru_cache(maxsize=None)
def _compiled2(n: int, page_size: int, terms: int, feat: str = "", stub: str = ""):
    sym = cute.sym_int
    b, h, p, mp, s = sym(), sym(), sym(), sym(), sym()

    def t(dtype, shape, align=16):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=tuple(range(len(shape)))[::-1],
                                                     assumed_align=align)

    return cute.compile(
        DecodeV2(n, page_size, terms, feat, stub), t(Uint8, (b, h, terms, CKV // 2)),
        t(Uint8, (b, h, terms, CKV // 16), 4), t(Float32, (b, h), 4), t(BFloat16, (b, h, KPE)),
        t(Uint8, (b, h, 3, KPE)), t(Float32, (b, h), 4), t(Uint8, (p, page_size, CKV // 2)),
        t(Uint8, (p, page_size, CKV // 16)), t(Uint8, (p, page_size, KPE)), t(Int32, (b, mp), 4), t(Int32, (b,), 4),
        t(BFloat16, (b, h, CKV)), t(Float32, (s, b, h, CKV)), t(Float32, (s, b, h), 4), Int32(TILE), Int32(1),
        Float32(1.0), Float32(1.0), Float32(1.0), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi")


def default_feat(page_size: int) -> str:
    """DecodeV2 options used by decode(); TMA boxes are whole pages, so pages under 16 tokens keep the cp.async load."""
    return "lazy,rope8,tma,cvt8,vhalf" if page_size >= 16 else "lazy,rope8"


def decode(q_nope, q_pe, ckv_cache, ckv_sf_cache, kpe_cache, page_table, kv_lens, sm_scale, ckv_scale=1.0,
           kpe_scale=1.0, max_kv_len=None, num_splits=0, version=3, heads_per_cta=None, terms=4,
           feat=None):
    """Same contract as nvfp4_mla.decode (BF16-or-better accuracy); SM100 only.

    version: 0 (v0), 1 (v1a'), 2 (v1b), 3 (v2, warp-specialized)."""
    from . import _load

    B, H = q_nope.shape[:2]
    if H % 16:
        raise ValueError(f"heads must be a multiple of 16, got {H}")
    if version == 3 and feat is None:
        feat = default_feat(ckv_cache.shape[1])
    h32 = version == 3 and "h32" in feat.split(",")
    if h32 and H % 32:
        raise ValueError(f"h32 needs a multiple of 32 heads, got {H}")
    # v1a keeps v0's FP16 V tile in shared memory and v1b keeps V in TMEM: both leave room for 16 heads per CTA only;
    # v2 with E4M3 V (v8) has room for 32 (h32)
    n = heads_per_cta or (32 if (version == 0 and H % 32 == 0) or h32 else 16)
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
        if version == 3:
            if "rope8" in feat:
                qp8, qps = _load().rope_terms(q_pe, 3)
            else:  # unused by the kernel
                qp8, qps = q_pe.new_empty(B, H, 3, KPE, dtype=torch.uint8), g
            _compiled2(n, ckv_cache.shape[1], terms, feat)(q4, qsf, g, q_pe, qp8, qps, *args)
        else:
            _compiled1(n, ckv_cache.shape[1], terms, False, version == 2)(q4, qsf, g, q_pe, *args)
    if splits > 1:
        _load().combine(o_part, lse_part, out)
    return out
