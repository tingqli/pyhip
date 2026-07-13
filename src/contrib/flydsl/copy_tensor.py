import flydsl.expr as fx
from . import utils as fxu

# fxu.enable_dump_ir(True)

"""
Try to implement a general vectorized copy according to 
3.4.2 Application: Vectorization Example in <CUTE LAYOUT REPRESENTATION AND ALGEBRA> by Cris Cecka

LDS src/dst usually has swizzled layout which prevents coalescing
"""
def copy_tensor(src, dst, num_threads=256, atom_neles = None, verbose=False):
    tid = fx.thread_idx.x
    assert src.dtype == dst.dtype
    src_size = fx.size(src.layout.shape).to_py_value()
    dst_size = fx.size(dst.layout.shape).to_py_value()
    assert src_size == dst_size, f"src_size={src_size} != dst_size={dst_size}"
    dtype = src.dtype
    count = fx.size(src.layout).to_py_value()

    # only 1 of src/dst can be composed layout
    is_src_composed = isinstance(src.layout, fx.ComposedLayout)
    is_dst_composed = isinstance(dst.layout, fx.ComposedLayout)
    assert not (is_src_composed and is_dst_composed)
    non_composed_layout = dst.layout if is_src_composed else src.layout

    # composed layout imposes additional limit on how many elements can be copied in one atom copy,
    # because the composed layout may not be contiguous in physical memory
    composed_limit_neles = 0x7fffffff
    if is_src_composed:
        composed_limit_neles = min(composed_limit_neles, (1 << src.layout.inner.base) if isinstance(src.layout.inner, fx.Swizzle) else 1)
    if is_dst_composed:
        composed_limit_neles = min(composed_limit_neles, (1 << dst.layout.inner.base) if isinstance(dst.layout.inner, fx.Swizzle) else 1)

    # sort modes in stride ascending order, so that we can traverse physical contiguous elements first
    strides = [fx.size(non_composed_layout.stride[i]).to_py_value() for i in fx.range_constexpr(non_composed_layout.stride.rank)]
    sorted_id = sorted(range(len(strides)), key=lambda i: strides[i])
    src = fx.select(src, sorted_id)
    dst = fx.select(dst, sorted_id)
    src_layout = src.layout.outer if is_src_composed else src.layout
    dst_layout = dst.layout.outer if is_dst_composed else dst.layout

    if atom_neles is None:
        # 3.4.2 Application: Vectorization Example in <CUTE LAYOUT REPRESENTATION AND ALGEBRA> by Cris Cecka
        # find vectorized pattern for copy

        src_inv = fx.right_inverse(src_layout)
        dst_inv = fx.right_inverse(dst_layout)
        src_dstinv = fx.composition(src_layout, dst_inv)

        max_count = num_threads * (128 // dtype.width)

        for K in range(max_count):
            # physical dst offset -> logical dst coord -> logical src coord -> physical src offset
            # 0 logical coord always maps to 0 physical offset and common sub-vector always maps 
            # contiguous physical offsets, so we can just check the first element of each sub-vector
            if fx.get_scalar(fx.get_1d_coord(K, src_dstinv)) != K:
                K -= 1
                break
        K += 1
        K = min(K, composed_limit_neles)

        for atom_bit_size in [128, 64, 32, 16, 8]:
            if atom_bit_size // dtype.width <= K:
                break
        
        atom_neles = atom_bit_size // dtype.width
    else:
        K = 1

    if verbose: print(f"Up to {K}/{count} elements can be copied in atom size {atom_neles}x'elements' per copy")

    cp_atom = fx.make_copy_atom(fx.UniversalCopy(atom_neles * dtype.width), dtype)

    if K > 1:
        # this divide is not tiler mode, so it generate 2 modes
        #  1.  traverse physical contiguous K elements
        #  2.  the rest mode
        # truncated-at-size-K
        Ik = fx.make_layout(K, 1)
        src_inv_trunc_K = fx.composition(src_inv, Ik) # 64:64  0,64,128,...
        dst_inv_trunc_K = fx.composition(dst_inv, Ik)

        if verbose: print(" src/dst truncated-at-size-K1: ", src, dst)
        src = fx.logical_divide(src, src_inv_trunc_K) 
        dst = fx.logical_divide(dst, dst_inv_trunc_K)
        if verbose: print(" src/dst truncated-at-size-K2: ", src, dst)

        # tv-layout, here m only means first mode, n only means second mode
        num_threads_m = min(K // atom_neles, num_threads)
        num_threads_n = num_threads // num_threads_m
        
        tile_mn = (num_threads_m * atom_neles, num_threads_n)
        tv_layout = fx.make_layout(((num_threads_m, num_threads_n           ), atom_neles),
                                ((atom_neles   , num_threads_m*atom_neles), 1))
    else:

        # 
        shape = list(src.layout.shape.to_py_value())
        assert shape[0] % atom_neles == 0, f"shape[0]={shape[0]} not divisible by atom_neles={atom_neles}"
        shape[0] //= atom_neles

        thread_shape = []
        thread_stride = []
        tile_mn = []
        stride = atom_neles
        left_threads = num_threads
        for s in shape:
            sz = min(s, left_threads)
            thread_shape.append(sz)
            thread_stride.append(stride)
            assert left_threads % sz == 0, f"left_threads={left_threads} not divisible by sz={sz}"
            left_threads //= sz
            stride *= sz
            if left_threads == 1:
                break
        assert left_threads == 1, f"left_threads={left_threads} not fully consumed by shape={shape}"
        tile_mn = [t for t in thread_shape]
        tile_mn[0] *= atom_neles
        tv_layout = fx.make_layout((thread_shape,atom_neles),(thread_stride,1))
        if verbose: print(" src/dst truncated-at-size-K2: ", src, dst)

    thr_copy = fx.make_tiled_copy(cp_atom, tv_layout, tile_mn).get_slice(tid)
    tvsrc = thr_copy.partition_S(src)
    tvdst = thr_copy.partition_D(dst)
    frag = fx.make_fragment_like(tvdst)

    if verbose: fxu.printv(src, dst, tv_layout, tile_mn, cp_atom, tvsrc, tvdst, frag)

    fx.copy(cp_atom, tvsrc, frag)
    fx.copy(cp_atom, frag, tvdst)
    fx.gpu.barrier()

def test(verbose=False):
    import torch

    import pyhip
    _,stream = pyhip.set_device()
    # pytest --pyargs pyhip.contrib.flydsl.utils
    @fxu.fly
    def general_copy(A0, B0):
        copy_tensor(A0, B0, verbose=verbose)

    A0 = torch.randn((64, 256), dtype=torch.float32)
    B0 = torch.zeros((64, 256), dtype=torch.float32)

    B0[...] = 0
    general_copy([1,],[256], A0[:,:64], B0[:,:64])
    assert pyhip.allclose(A0[:,:64], B0[:,:64])

    B0[...] = 0
    general_copy([1,],[256], A0[:,:64].T, B0[:,:64].T)
    assert pyhip.allclose(A0[:,:64], B0[:,:64])

    B0[...] = 0
    general_copy([1,],[256], A0[:,:128], B0[:,:128])
    assert pyhip.allclose(A0[:,:128], B0[:,:128])

    B0[...] = 0
    general_copy([1,],[256], A0[:,:128].T, B0[:,:128].T)
    assert pyhip.allclose(A0[:,:128], B0[:,:128])

    B0[...] = 0
    general_copy([1,],[256], A0[:,:64].T, B0[:,:64])
    assert pyhip.allclose(A0[:,:64].T, B0[:,:64])

    A0 = torch.randn((64, 256, 32, 8), dtype=torch.float32)
    B0 = torch.zeros((64, 256, 32, 8), dtype=torch.float32)
    src = A0[:8,:16,:,:1]
    dst = B0[:8,:16,:,:1]

    general_copy([1,],[256], src, dst)
    assert pyhip.allclose(src, dst)

    BLOCK_M, BLOCK_K = 64, 128

    @fxu.fly
    def general_copy(A0, B0):
        copy_tensor(A0, B0)

        @fx.struct
        class SharedStorage:
            A: fx.Array[A0.dtype, BLOCK_M * BLOCK_K]

        # swizzle happens in unit of 128b, 
        swz_c = fx.SwizzleType.get(3, 3, 3)

        print(swz_c, swz_c.base, swz_c.mask, swz_c.shift)

        # mask=3, base=3, shift=3
        #   base=3:  in unit of 8xbf16 elements (DW4), 
        #   mask=3:  
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        swz_layout = fx.make_composed_layout(fx.static(swz_c), fx.make_layout((BLOCK_M, BLOCK_K), (BLOCK_K, 1)))
        ldsA0 = lds.A.view(swz_layout)
        
        copy_tensor(A0, ldsA0, verbose=verbose)
        copy_tensor(ldsA0, B0, verbose=verbose)

    A0 = torch.randn((64, 256), dtype=torch.bfloat16)
    B0 = torch.zeros((64, 256), dtype=torch.bfloat16)
    src = A0[:,:128]
    dst = B0[:,:128]
    general_copy([1,],[256], src, dst)
    assert pyhip.allclose(src, dst)

if __name__ == "__main__":
    test(True)