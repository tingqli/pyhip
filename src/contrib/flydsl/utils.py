import os
import types
import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.utils.env import DebugEnvManager
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, scf, arith
import torch
import functools

def div_e(a, b):
    assert a % b == 0, f"div_e expect {a} evenly divisible by {b}"
    q = a // b
    assert q > 0, f"div_e expect {a} // {b} > 0"
    return q

"""
compare if two htuple(int) or htuple(layout) are the same, recursively
"""
def is_same(a, b):
    if type(a) != type(b):
        return False

    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(is_same(a[i], b[i]) for i in range(len(a)))
    
    if isinstance(a, fx.IntTuple):
        return a.to_py_value() == b.to_py_value()
    
    if isinstance(a, fx.Layout):
        return is_same(a.stride, b.stride) and is_same(a.shape, b.shape)
    
    if isinstance(getattr(a, "type", None), fx.CoordTensorType):
        base_a = fx.get_iter(a).to_py_value()
        base_b = fx.get_iter(b).to_py_value()
        return is_same(a.layout, b.layout) and (base_a == base_b)

    if isinstance(a, fx.Tensor):
        return is_same(a.layout, b.layout) and a.dtype == b.dtype

    assert 0, f"is_same: unsupported type {type(a)} & {type(b)}"

@flyc.jit
def _test_is_same():
    layout1 = fx.make_layout((4, (16, 16)), (32, (128, 1)))
    layout2 = fx.make_layout((4, (16, 16)), (32, (128, 1)))
    layout3 = fx.make_layout((4, (16, 16)), (32, (128, 2)))
    layout4 = fx.make_layout((4, 16, 16), (32, 128, 2))
    assert is_same(layout1, layout2)
    assert not is_same(layout1, layout3)
    assert not is_same(layout1, layout4)

def test_is_same(): _test_is_same()

def to_py_value(a):
    if isinstance(a, (tuple, list)):
        return a
    if getattr(a, "to_py_value", None):
        return a.to_py_value()
    if isinstance(a, fx.Tile):
        # Tile<[a|b|c]>
        txt = str(a)
        assert txt.startswith("Tile<") and txt.endswith(">"), "tile_mn must be a Tile"
        ret = []
        for v in txt[6:-2].split("|"):
            if ":" in v: assert int(v.split(":")[1]) == 1, v
            ret.append(int(v.split(":")[0]))
        return ret
    assert 0, f"to_py_value: unsupported type {type(a)}"

"""
passing single layout as tile do not work for fx.make_tile()
"""
def make_tile(*args):
    def _resolve(m):
        if isinstance(m, int) or m is None:
            return m
        if isinstance(m, tuple):
            return tuple(_resolve(e) for e in m)
        if isinstance(m, fx.Layout):
            return m.type
        raise ValueError(f"make_tile: expected int, None, tuple, or Layout, got {isinstance(m, fx.Layout)}")
    resolved = [_resolve(m) for m in args]            
    return fly.static(fly.TileType.get(resolved))

"""
a useful helper to build layout maps N-dimension to M-dimensions
for example, to build a TV-layout maps (T256, V8)->(M16, N128)

    tv_layout = make_layout_nd(((16, 16), 8),               # (T256, V8)
                                (16, 128),                  # (M16, N128)
                                (([0, 8], [1, 0]), [0, 1])) # list is stride in 2D 

the list-style vector stride will be transformed to 1d-offset by crd2idx(crd, co_shape)
"""
def make_layout_nd(shape, co_shape, strides):
    def convert_stride(coord):
        if isinstance(coord, list):
            # return crd2idx(coord, co_shape)
            s = 1
            sum = 0
            for c, sz in zip(coord, co_shape):
                sum += c * s
                s *= sz
            return sum
        else:
            assert isinstance(coord, tuple)
            return tuple(convert_stride(c) for c in coord)

    return fx.make_layout(shape, convert_stride(strides))

"""
partition a tensor into TV-tiles, each tile with specific TV-layout.

       tensor: (128,128):(1,128)
    tv_layout: Layout<((8,32),4):((128,1),32)>
    tv_tilemn: (32, 32)
         part: (((8,32),4),(4,4)):(((512,1),128),(32,4096))

 meaning of modes:
     1. (threads, values) within a tv-tile
     2. layout of tv-tile elements that covers entire input tensor
"""
def tv_partition(input, tv_layout, tv_tilemn):
    input_tiled = fx.zipped_divide(input, fx.make_tile(*tv_tilemn))
    #tile_tv = fly.static(fly.TileType.get([tv_layout.type]))
    tile_tv = fx.make_tile(tv_layout, None)
    tv_part = fx.composition(input_tiled, tile_tv)
    return tv_part

"""
put all threads as contiguous as possible until tileN elements are reached
    num_threads: number of threads in a block
    num_values: number of values per thread
    tileN: optional, if unspecified, put all threads along the line
"""
def make_mem_coalescing_2d_tv_layout(num_threads, num_values, tileN = None):
    threads_n = num_threads if tileN is None else min(tileN//num_values, num_threads)
    threads_m = num_threads//threads_n
    tv_tilemn = (threads_m, threads_n * num_values)
    tv_layout = make_layout_nd(((threads_n, threads_m), num_values),
                                    tv_tilemn,
                                    (([0, num_values], [1, 0]), [0, 1]))
    return tv_layout, tv_tilemn

def concat_modes(*modes, base=None):
    for i in fx.range_constexpr(0, len(modes)):
        m = modes[i]
        if isinstance(m, tuple) or isinstance(m, list):
            base = concat_modes(*m, base=base)
            base = fx.group(base, base.rank - len(m), -1)
        else:
            if base is None:
                base = m
            else:
                base = fx.append(base, m)
    return base

# inside fx.copy: recursively expand until inner-most copy_atom
# this works
def recurisve_apply(atom_op, *tensors, idx=None):
    if tensors[0].layout.rank == 1:
        if tensors[0].layout.shape.is_leaf:
            if idx is not None:
                atom_op(*tensors, idx=idx)
                idx += 1
            else:
                atom_op(*tensors)
        else:
            idx = recurisve_apply(atom_op, *[fx.get_(t,0) for t in tensors], idx=idx)
    else:
        tensors = [fx.group(t, 1, -1) for t in tensors]
        size = fx.size(tensors[0].layout[1])
        assert size.is_static, f"Expected static size, got {size}"
        for i in fx.range_constexpr(size.get_static_leaf_int):
            idx = recurisve_apply(atom_op, *[t[None, i] for t in tensors], idx=idx)
    return idx

def view_as(tensor, new_layout, dtype=None):
    iter = fx.get_iter(tensor)
    if dtype is not None:
        iter = fx.recast_iter(dtype, iter)
    return fx.Tensor(fx.make_view(iter, new_layout))

def layout2str(layout, image):
    # image: "m3 n6"
    shape = layout.shape.to_py_value()
    stride = layout.stride.to_py_value()
    names = [s[0] for s in image.split()]
    sizes = [int(s[1:]) for s in image.split()]
    steps,_ = shape2strides(sizes)

    domain = []
    for i in range(layout.shape.rank):
        sz = fx.get_scalar(fx.size(layout.shape[i]))
        n = chr(ord('a')+i) + str(sz)
        domain.append(n)
    domain = ",".join(domain)

    def stride2names(s):
        if isinstance(s, int):
            if s == 0: return "0"
            valid_steps = list((val, i) for i, val in enumerate(steps) if val <= s)
            step, i = max(valid_steps, key=lambda item: item[0])
            assert s % step == 0, f"stride {s} is not divisible by step {step}"
            return f"{s//step}{names[i]}"
        return "(" + ",".join([stride2names(e) for e in s]) + ")"
    return f"({domain})=>({image}) {shape}:{stride2names(stride)}"

def gemm(mma_atom, D, A, B, C):
    """
    simulate behavior of fx.gemm
    input layouts
        C/D : (c-tile, loop_m, loop_n)    always rank-3
         A  : (a-tile, loop_m, [loop_k])  [.] means optional, if not present, means 1
         B  : (b-tile, loop_n, [loop_k])  [.] means optional, if not present, means 1
    """
    loop_m = fx.get_scalar(fx.size(C.layout.shape[1]))
    loop_n = fx.get_scalar(fx.size(C.layout.shape[2]))
    loop_k = fx.get_scalar(fx.size(A.layout.shape[2])) if A.layout.rank > 2 else 1
    assert loop_m == fx.get_scalar(fx.size(A.layout.shape[1]))
    assert loop_n == fx.get_scalar(fx.size(B.layout.shape[1]))
    assert loop_k == fx.get_scalar(fx.size(B.layout.shape[2])) if B.layout.rank > 2 else 1

    cSrc = {}
    for m in range(loop_m):
        for n in range(loop_n):
            cSrc[(m,n)] = C

    for m in range(loop_m):
        for n in range(loop_n):
            for k in range(loop_k):
                fx.mma_atom_call(mma_atom, D[None, m, n], A[None, m, k], B[None, n, k], cSrc[(m,n)][None, m, n])
                cSrc[(m,n)] = D # visited, next iter use D as source

def is_layout_htuple(args):
    if isinstance(args, (tuple, list)):
        return all(is_layout_htuple(a) for a in args)
    return isinstance(args, fx.Layout)

# convert hierarchical-tuple of int_tuple/layouts to int_tuple/layout
# using fx.group & fx.append to combine them
# for example, ((m0,m1),m2)
def ht2fly(hierarchical_tuple):
    if not isinstance(hierarchical_tuple, (tuple, list)):
        return hierarchical_tuple # the single element itself
    layout = None
    for element in hierarchical_tuple:
        if isinstance(element, (tuple,list)):
            mode = ht2fly(element)
        else:
            mode = element
        if layout is None:
            layout = fx.group(mode, 0, -1)
        else:
            layout = fx.append(layout, mode)
    return layout

def shape2strides(shape, base_stride=1):
    if isinstance(shape, int): return base_stride, base_stride*shape
    assert isinstance(shape, (tuple, list))
    all_strides = []
    for s in shape:
        strides, next_base = shape2strides(s, base_stride)
        all_strides.append(strides)
        base_stride = next_base
    return all_strides, base_stride

# enhanced to support cutlass's orginal feature: can used to concat layouts
def make_layout(*args):
    if is_layout_htuple(args):
        return ht2fly(args)
    # shape/strides form
    if len(args) == 1:
        shape = args[0]
        strides, _ = shape2strides(shape)
    else:
        shape, strides = args
    return fx.make_layout(shape, strides)

# divide by mode
def _op_by(layout, tile, op):
    # tile is nested tupple
    if not isinstance(tile, (tuple,list)):
        # no further tile
        if isinstance(tile, int):
            divisor = fx.make_layout(tile, 1)
        else:
            divisor = tile
        return op(layout, divisor)

    result_modes = []
    for i in range(layout.rank):
        mode = layout[i]
        if i < len(tile) and tile[i] is not None:
            div = _op_by(mode, tile[i], op)
            result_modes.append(div)
        else:
            result_modes.append(mode)
    return result_modes

basic_composition = functools.partial(_op_by, op=fx.composition)
basic_divide = functools.partial(_op_by, op=fx.logical_divide)


class Tensor2Layout:
    def __init__(self, tensor_or_layout):
        if isinstance(tensor_or_layout, fx.Tensor):
            self.iterator = fx.get_iter(tensor_or_layout)
            self.layout = tensor_or_layout.layout
        else:
            self.iterator = None
            self.layout = tensor_or_layout
    
    def __call__(self, layout):
        return fx.Tensor(fx.make_view(self.iterator, layout)) if self.iterator is not None else layout

def logical_divide(tensor_or_layout, tile):
    t = Tensor2Layout(tensor_or_layout)
    result_modes = basic_divide(t.layout, tile)
    return t(ht2fly(result_modes))

def composition(tensor_or_layout, tile):
    t = Tensor2Layout(tensor_or_layout)
    result_modes = basic_composition(t.layout, tile)
    return t(ht2fly(result_modes))

def zip2_by(modes, guide):
    if not isinstance(guide, (tuple,list)):
        if isinstance(modes, (tuple,list)):
            assert len(modes) == 2
        else:
            assert modes.rank == 2
        return modes[0], modes[1]
    # recursively zip by guide
    ht0 = []
    ht1 = []
    for i in range(len(modes)):
        if i < len(guide):
            a, b = zip2_by(modes[i], guide[i])
            ht0.append(a)
            ht1.append(b)
        else:
            ht1.append(modes[i])
    return ht0, ht1

"""
zip(   ((a0,a1,...), (b0,b1,...), (c0,c1,...))    )
  =    ((a0,b0,c0,...), (a1,b1,...), ...     )
"""
def zip_(layout):
    for i in range(layout.rank):
        assert layout[i].rank == layout[0].rank, f"zip_: all modes must have same rank, got {layout[i].rank} != {layout[0].rank}"
    ret = []
    for j in range(layout[0].rank):
        ret.append([layout[i][j] for i in range(layout.rank)])
    return ht2fly(ret)

@flyc.jit
def _test_zip():
    layout = fx.make_layout(((4, 2), (16, 16)), ((32, 9), (128, 1)))
    assert is_same(zip_(layout), fx.make_layout(((4,16),(2,16)),((32,128),(9,1))))

    layout = fx.make_layout(((1, 2, 3), (4, 5, 6), (7, 8, 9)),
                            ((10, 20, 30), (40,50,60), (70,80,90)))
    assert is_same(zip_(layout), fx.make_layout(((1,4,7),(2,5,8),(3,6,9)),((10,40,70),(20,50,80),(30,60,90))))

def test_zip(): _test_zip()

def zipped_divide(tensor_or_layout, tile):
    t = Tensor2Layout(tensor_or_layout)
    result_modes = basic_divide(t.layout, tile)
    ht0, ht1 = zip2_by(result_modes, tile)
    return t(ht2fly([ht0, ht1]))

def to_str(a):
    if isinstance(a, (tuple, list)):
        return "(" + ", ".join([to_str(e) for e in a]) + ")"
    else:
        return str(a)

@flyc.jit
def _test_div():
    layout = fx.make_layout((4, (16, 16)), (32, (128, 1)))
    
    ret = logical_divide(layout, (2, (4, 8)))
    ref = fx.make_layout(((2,2),((4,4),(8,2))), ((32,64),((128,512),(1,8))))
    assert is_same(ret, ref), f"{ret} != ref:{ref}"
    
    ret = zipped_divide(layout, (2, (4, 8)))
    ref = fx.make_layout(((2,(4,8)),(2,(4,2))), ((32,(128,1)),(64,(512,8))))
    assert is_same(ret, ref), f"{ret} != ref:{ref}"
def test_div(): _test_div()

@flyc.jit
def _test_slice():
    layout = fx.make_layout((4, (16, 16)), (32, (128, 1)))
    ret1 = layout(None,None)
    assert is_same(ret1, layout), f"should keep htuple() structure according to None, {ret1}"

    ret2 = layout(None,(None, None))
    ref2 = fx.make_layout((4, 16, 16), (32, 128, 1))
    assert is_same(ret2, ref2), f"should expand the hierarchy according to (None, None), {ret2}"
def test_slice(): _test_slice()


def inspect(x):
    color0 = f"\033[0;{30+(2 % 8)}m"
    color1 = f"\033[0m"

    if isinstance(x, fx.TiledMma):
        print(f"{color0}", end="")
        print("TiledMma:")
        print("  mma_atom:", x.mma_atom)
        print("  atom_layout:", x.atom_layout)
        print("  thr_layout_vmnk:", x.thr_layout_vmnk)
        print("  permutation_mnk:", x.permutation_mnk)
        print("  tile_size_mnk:", x.tile_size_mnk)
        tile_size_mnk = x.tile_size_mnk.to_py_value()
        print("  tv_layout_A_tiled:", layout2str(x.tv_layout_A_tiled, f"m{tile_size_mnk[0]} k{tile_size_mnk[2]}"))
        print("  tv_layout_B_tiled:", layout2str(x.tv_layout_B_tiled, f"n{tile_size_mnk[1]} k{tile_size_mnk[2]}"))
        print("  tv_layout_C_tiled:", layout2str(x.tv_layout_C_tiled, f"m{tile_size_mnk[0]} n{tile_size_mnk[1]}"))
        print(f"{color1}", end="")
        return

    print(f"inspect: unsupported type {type(x)}")


import types

class Fragment:
    """
    Fragment 是一个寄存器Tensor的 per-thread view，背后隐含 tv-layout 的 tiling
    下面的构造参数中都显式指定了 (s0, s1) 作为 2D block 的shape, 代表所有线程
    的 fragment view 拼凑起来的 2D block 的shape, 也就是一个 block 的 fragment view 的总shape.

      fragment 视图的基本布局为： (FrgV, repeat_s0, repeat_s1, ...)

       - FrgV : 单个 tv-layout 中 value 维度大小
       - repeat_s0 : 在 s0 方向上重复 tv-layout 的次数，等于 s0//tile_size[0]
       - repeat_s1 : 在 s1 方向上重复 tv-layout 的次数，等于 s1//tile_size[1]
       - 其余维度仅仅当 partition_S/partition_D 时才会出现，直接继承自 src/dst
         tensor 的头2维之后的维度，代表 fragment view 整体重复次数

    常见的 tv-layout 包括：
     - 按照 TiledMMA 对A/B/C的要求摆放的 layout
     - 按照 mem-coelasing 的要求摆放的 layout

    虽然layout是固定的，但是一个 Fragment 可能同时作为多种不同的算子的操作数，例如:
      - 使用 128b 的方式 fx.copy 读写
      - 使用 32b 的方式 fx.copy 读写
      - 使用 fx.gemm 计算
    在参与这些不同算子时，每种算子需要 fragment 参数遵守某种特定布局，类似于 call convention
    此时一块物理同源的 fragment 需要局部多种不同的布局 view 才能参与这些操作。这些布局
    的区别通常不涉及 tv-layout (否则就需要利用dpp/ds-permute/LDS进行真正的搬运转换)，
    但是会涉及 per-thread value view 的不同变换形式。 这些形式基本遵循：
       (FrgV, repeat_M, repeat_K, ...)
    的形式，但是在 FrgV 这个维度上的要求则各不相同，例如
       - fx.gemm 要求 FrgV 维度是 mma-atom 的 per-thread value count
       - fx.copy 要求 FrgV 维度是 copy-atom 的 per-thread value count

    因此如果我们可以事先知道 fragment 会被用于什么算子时，就可以事先retile出
    不同的 view 来满足不同算子的要求。

    from_tvlayout/from_mmagemm 会根据 tv-layout(单独指定或者来自TileMMA) 构造出
    fragment的基本view, 构造时同时传入的多个 copy-atom 会用来构造可以被这些 copy-atom
    使用的不同的 tiled-copy 对象，并且保存其 retile 后的 fragment view, 以供后继
    copy_from/copy_to 使用。

    这些方法被动态附加到原始 fragment Tensor 对象上，因此如果将这些对象进一步加工，例如
    对其进行 layout algebra 操作后，附加的方法就会丢失。但是好在目前没有发现需要对 fragment
    进行进一步加工的需求，因此暂时不考虑这个问题。
    """

    @staticmethod
    def _attach_methods(frag):
        frag.copy_from  = types.MethodType(Fragment.copy_from, frag)
        frag.copy_to    = types.MethodType(Fragment.copy_to, frag)
        frag._check_is_unpartioned = types.MethodType(Fragment._check_is_unpartioned, frag)
        frag._check_is_partioned   = types.MethodType(Fragment._check_is_partioned, frag)
        frag.partition_S = types.MethodType(Fragment.partition_S, frag)
        frag.partition_D = types.MethodType(Fragment.partition_D, frag)
        frag.selfclone = types.MethodType(Fragment.selfclone, frag)

    @staticmethod
    def selfclone(self):
        frag = fx.make_fragment_like(self)
        copy_ops = {}
        for copy_atom, v in self._copy_ops.items():
            thr_copy, _ = v
            copy_ops[copy_atom] = (thr_copy, thr_copy.retile(frag))
        frag._copy_ops = copy_ops
        frag._block_shape = self._block_shape
        frag._tile_size = self._tile_size
        if hasattr(self, "tiled_mma"):
            frag.tiled_mma = self.tiled_mma
        Fragment._attach_methods(frag)
        return frag

    @staticmethod
    def from_tvlayout(dtype, s0:int, s1:int,
                      tv_layout, tile_size, copy_atoms: list[fx.CopyAtom]):

        if not isinstance(copy_atoms, (tuple, list)):
            assert isinstance(copy_atoms, fx.CopyAtom)
            copy_atoms = [copy_atoms]

        assert tv_layout.rank == 2
        assert tile_size[0] > 0 and tile_size[1] > 0
        
        tcopy = fx.make_tiled_copy(copy_atoms[0], tv_layout, tile_size)
        thr_copy = tcopy.get_slice(fx.thread_idx.x)

        dtype = fx.PointerType.get(dtype.ir_type, 1, 16)
        ptr = fx.inttoptr(dtype, fx.Int32(0))
        fake_block = fx.make_view(ptr, fx.make_layout((s0, s1), (s1, 1)))

        partS = thr_copy.partition_S(fake_block)
        partD = thr_copy.partition_D(fake_block)
        fragS = fx.make_fragment_like(partS)
        fragD = fx.make_fragment_like(partD)

        assert is_same(fragS.layout, fragD.layout), f"fragS.layout {fragS.layout} != fragD.layout {fragD.layout}"

        # build dict for each supported copy_atom
        copy_ops = {copy_atoms[0]: (thr_copy, fragS)}
        for copy_atom in copy_atoms[1:]:
            tcopy = fx.make_tiled_copy(copy_atom, tv_layout, tile_size)
            thr_copy = tcopy.get_slice(fx.thread_idx.x)
            copy_ops[copy_atom] = (thr_copy, thr_copy.retile(fragS))
        fragS._copy_ops = copy_ops
        fragS._block_shape = (s0, s1)
        fragS._tile_size = tile_size
        Fragment._attach_methods(fragS)
        return fragS

    @staticmethod
    def from_tiledmma(tiled_mma: fx.TiledMma, s0:int, s1:int, abc: str, copy_atoms: list[fx.CopyAtom], dtype=None):
        assert abc in ["A", "B", "C"]

        if not isinstance(copy_atoms, (tuple, list)):
            assert isinstance(copy_atoms, fx.CopyAtom)
            copy_atoms = [copy_atoms]

        tile_size_mnk = tiled_mma.tile_size_mnk.to_py_value()
        thr_mma = tiled_mma.thr_slice(fx.thread_idx.x)

        ptr_type = fx.PointerType.get(fx.Int8.ir_type, 1, 16)
        ptr = fx.inttoptr(ptr_type, fx.Int32(0))
        fake_block = fx.make_view(ptr, fx.make_layout((s0, s1), (s1, 1)))
    
        if abc == "A":
            tile_size = (tile_size_mnk[0], tile_size_mnk[2])
            assert s0 % tile_size[0] == 0
            assert s1 % tile_size[1] == 0
            frag = thr_mma.make_fragment_A(fake_block)
        elif abc == "B":
            tile_size = (tile_size_mnk[1], tile_size_mnk[2])
            assert s0 % tile_size[0] == 0
            assert s1 % tile_size[1] == 0
            frag = thr_mma.make_fragment_B(fake_block)
        elif abc == "C":
            tile_size = (tile_size_mnk[0], tile_size_mnk[1])
            assert s0 % tile_size[0] == 0
            assert s1 % tile_size[1] == 0
            frag = thr_mma.make_fragment_C(fake_block)

        # override mma's default dtype with user-specified dtype
        if (dtype is not None) and (dtype != frag.dtype):
            frag = fx.make_fragment_like(frag, dtype=dtype)

        # build dict for each supported copy_atom
        copy_ops = {}
        for copy_atom in copy_atoms:
            if abc == "A":
                tcopy = fx.make_tiled_copy_A(copy_atom, tiled_mma)
            elif abc == "B":
                tcopy = fx.make_tiled_copy_B(copy_atom, tiled_mma)
            else:
                tcopy = fx.make_tiled_copy_C(copy_atom, tiled_mma)

            thr_copy = tcopy.get_slice(fx.thread_idx.x)
            copy_ops[copy_atom] = (thr_copy, thr_copy.retile(frag))
        frag._copy_ops = copy_ops
        frag._block_shape = (s0, s1)
        frag._tile_size = tile_size
        frag.tiled_mma = tiled_mma

        Fragment._attach_methods(frag)
        return frag

    """
    avoid using load/store as name, these are Tensor's methods
    copy_atom: optional copy atom to use for the copy operation
                for example： fx.rocdl.BufferCopy128b vs fx.UniversalCopy128b()
                if copy_atom is not none, it must be compatible with the copy_atom
                used to create the Fragment
    """
    @staticmethod    
    def copy_from(self, src: fx.Tensor, copy_atom = None):
        if copy_atom is None:
            copy_atom = next(iter(self._copy_ops))
        else:
            assert copy_atom in self._copy_ops
        thr_copy, copy_frag = self._copy_ops[copy_atom]

        if fx.const_expr(self._check_is_unpartioned(src)):
            copy_src = thr_copy.partition_S(src)
        elif fx.const_expr(self._check_is_partioned(src)):
            copy_src = src
        else:
            raise RuntimeError(f"src tensor {src} is not partitioned or unpartitioned")
        fx.copy(copy_atom, copy_src, copy_frag, pred=None)

    @staticmethod
    def copy_to(self, dst: fx.Tensor, copy_atom = None):
        if copy_atom is None:
            copy_atom = next(iter(self._copy_ops))
        else:
            assert copy_atom in self._copy_ops
        thr_copy, copy_frag = self._copy_ops[copy_atom]
        if fx.const_expr(self._check_is_unpartioned(dst)):
            copy_dst = thr_copy.partition_D(dst)
        elif fx.const_expr(self._check_is_partioned(dst)):
            copy_dst = dst
        else:
            raise RuntimeError(f"dst tensor {dst} is not partitioned or unpartitioned")
        fx.copy(copy_atom, copy_frag, copy_dst, pred=None)

    @staticmethod
    def _check_is_unpartioned(self, t: fx.Tensor):
        # compiled time check if a tensor has not been partitioned into tiles
        if t.layout.rank != 2: return False
        s0 = fx.size(t.shape[0]).to_py_value()
        s1 = fx.size(t.shape[1]).to_py_value()
        return s0 == self._block_shape[0] and s1 == self._block_shape[1]

    @staticmethod
    def _check_is_partioned(self, t: fx.Tensor):
        # compiled time check if a tensor has been partitioned into tiles
        if t.layout.rank != 3: return False
        num_mma_tiles_s0 = fx.size(t.shape[1]).to_py_value()
        num_mma_tiles_s1 = fx.size(t.shape[2]).to_py_value()
        return num_mma_tiles_s0 == self._block_shape[0] //self._tile_size[0] and \
                num_mma_tiles_s1 == self._block_shape[1] //self._tile_size[1]

    @staticmethod
    def partition_S(self, src: fx.Tensor, copy_atom = None):
        """ 
        input src:  (BLOCK_M, BLOCK_K, num_blocks_k, ...)
        
            let  num_mma_tiles_BM =  BLOCK_M//tile_size_m
            let  num_mma_tiles_BN =  BLOCK_N//tile_size_n
            let  num_mma_tiles_BK =  BLOCK_K//tile_size_k
        
        copy_src_A:  ((trg_val, rest_val), num_mma_tiles_BM, num_mma_tiles_BK, num_blocks_k, ...)
        copy_src_B:  ((trg_val, rest_val), num_mma_tiles_BN, num_mma_tiles_BK, num_blocks_k, ...)
        copy_src_C:  ((trg_val, rest_val), num_mma_tiles_BM, num_mma_tiles_BN, ...)

            - trg_val : copy-atom value size
            - rest_val: number of copy-atoms(TV tiles) to fill VECT_WIDTH
            - RestM : M/tv_tilemn[0]
            - RestN : N/tv_tilemn[1]
            - ...   : the other dimensions are the same as input tensor
        """
        if copy_atom is None:
            copy_atom = next(iter(self._copy_ops))
        else:
            assert copy_atom in self._copy_ops
        thr_copy, copy_frag = self._copy_ops[copy_atom]        
        s0 = fx.size(src.shape[0]).to_py_value()
        s1 = fx.size(src.shape[1]).to_py_value()
        assert s0 == self._block_shape[0] and s1 == self._block_shape[1]
        return thr_copy.partition_S(src)

    @staticmethod
    def partition_D(self, dst: fx.Tensor, copy_atom = None):
        if copy_atom is None:
            copy_atom = next(iter(self._copy_ops))
        else:
            assert copy_atom in self._copy_ops
        thr_copy, copy_frag = self._copy_ops[copy_atom]          
        s0 = fx.size(dst.shape[0]).to_py_value()
        s1 = fx.size(dst.shape[1]).to_py_value()
        assert s0 == self._block_shape[0] and s1 == self._block_shape[1]
        return thr_copy.partition_D(dst)


def enable_dump_ir(enable_debug_info = True):
    import os
    import flydsl
    from flydsl.utils.env import DebugEnvManager
    from flydsl._mlir import ir
    DebugEnvManager.enable_debug_info = enable_debug_info
    DebugEnvManager.dump_asm = True
    DebugEnvManager.dump_ir = True
    DebugEnvManager.dump_dir = "my_ir_dumps"
    ir._globals.register_traceback_file_inclusion(__file__)
    ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
    ir._globals.set_loc_tracebacks_frame_limit(40)
    ir._globals.set_loc_tracebacks_enabled(True)
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

