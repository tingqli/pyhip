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

        for t in tensors:
            size = fx.size(t.layout[1])
            if size.is_static:
                break
        assert size.is_static, f"Cannot find a static size, got {size}"
        for i in fx.range_constexpr(size.get_static_leaf_int):
            idx = recurisve_apply(atom_op, *[t[None, i] for t in tensors], idx=idx)
    return idx

# generator form of recurisve_apply, more pythonic
def all_elements(*tensors):
    if tensors[0].layout.rank == 1:
        if tensors[0].layout.shape.is_leaf:
            yield tensors
        else:
            yield from all_elements(*[fx.get_(t,0) for t in tensors])
    else:
        tensors = [fx.group(t, 1, -1) for t in tensors]
        for t in tensors:
            size = fx.size(t.layout[1])
            if size.is_static:
                break
        assert size.is_static, f"Cannot find a static size, got {size}"
        for i in fx.range_constexpr(size.get_static_leaf_int):
            yield from all_elements(*[t[None, i] for t in tensors])

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


def printv(*args, **kwargs):
    import inspect
    import re
    """
    Print variable names and their values.
    Usage: printv(var1, var2, ...)
    """
    # Get the caller's frame
    frame = inspect.currentframe().f_back
    
    # Get the source line
    import linecache
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    line = linecache.getline(filename, lineno).strip()
    
    # Extract argument names from the call
    # This handles: printv(var1, var2) 
    # and also: printv(var1, var2, sep=' ') etc.
    match = re.match(r'.*printv\s*\((.*)\)', line)
    if not match:
        # Fallback to generic names
        arg_names = [f'arg{i}' for i in range(len(args))]
    else:
        # Parse the arguments, handling nested parentheses and strings
        args_text = match.group(1)
        arg_names = []
        depth = 0
        current = ''
        in_string = False
        string_char = None
        
        for char in args_text:
            if char in ('"', "'") and (not current or current[-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                current += char
            elif char == '(' and not in_string:
                depth += 1
                current += char
            elif char == ')' and not in_string:
                depth -= 1
                current += char
            elif char == ',' and depth == 0 and not in_string:
                # Found a top-level comma
                clean_name = current.strip()
                if clean_name and not clean_name.startswith('**') and not clean_name.startswith('*'):
                    arg_names.append(clean_name)
                current = ''
            else:
                current += char
        
        # Add the last argument
        if current.strip():
            clean_name = current.strip()
            if clean_name and not clean_name.startswith('**') and not clean_name.startswith('*'):
                arg_names.append(clean_name)
    
    # Print each variable with its name
    print(f"{filename}:{lineno}")
    for i, (name, value) in enumerate(zip(arg_names, args)):
        if isinstance(value, torch.Tensor):
            value = f"{value.shape}_{value.dtype}"
        print(f"{name:>20} = {value}")

import types

class Fragment(fx.Tensor):
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

    load_gather_rows/load_scatter_rows 允许copy的源/目的矩阵的行由 gather/scatter indicies 里面的值指定
    调用 copy_from/copy_to 时传入额外的rows + cols参数辅助定位每个 copy-atom 的实际位置。
    """
    def __init__(self, tensor, copy_ops, block_shape, tile_size, tiled_mma=None):
        super().__init__(tensor)
        self._copy_ops = copy_ops
        self._block_shape = block_shape
        self._tile_size = tile_size
        self.tiled_mma = tiled_mma

    @classmethod
    def from_tvlayout(cls, dtype, s0:int, s1:int,
                      tv_layout, tile_size, copy_atoms: list[fx.CopyAtom]):
        """
        just 1 copy_atom:
            used for both loading & storing
        2 or more copy_atoms:
            first is default for loading copy_from/partition_S
            last is default for storing copy_to/partition_D
        """
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
        return cls(fragS, copy_ops, (s0, s1), tile_size)

    @classmethod
    def from_tiledmma(cls, tiled_mma: fx.TiledMma, s0:int, s1:int, abc: str, copy_atoms: list[fx.CopyAtom] = (), dtype=None):
        """
        just 1 copy_atom:
            used for both loading & storing
        2 or more copy_atoms:
            first is default for loading copy_from/partition_S
            last is default for storing copy_to/partition_D
        """        
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
        return cls(frag, copy_ops, (s0, s1), tile_size, tiled_mma=tiled_mma)

    def selfclone(self):
        frag = fx.make_fragment_like(self)
        copy_ops = {}
        for copy_atom, v in self._copy_ops.items():
            thr_copy, _ = v
            copy_ops[copy_atom] = (thr_copy, thr_copy.retile(frag))
        return Fragment(frag, copy_ops, self._block_shape, self._tile_size, tiled_mma=self.tiled_mma)

    def _get_copy_assets(self, idx, copy_atom = None):
        if copy_atom is None:
            copy_atom = list(self._copy_ops)[idx]
        else:
            assert copy_atom in self._copy_ops
        thr_copy, copy_frag = self._copy_ops[copy_atom]
        return copy_atom, thr_copy, copy_frag

    def copy_from(self, src: fx.Tensor, copy_atom = None, rows = None, cols = None):
        copy_atom, thr_copy, copy_frag = self._get_copy_assets(0, copy_atom)
        if rows is not None:
            def gather_atom(dst, row, col):
                index = src.layout(row[0], col[0])
                iter = fx.add_offset(fx.get_iter(src), index)
                atom_A = fx.make_view(iter, copy_atom.layout_src_tv[1])
                fx.copy(copy_atom, atom_A, dst)
            recurisve_apply(gather_atom, copy_frag, rows, cols)
            return

        if fx.const_expr(self._check_is_unpartioned(src)):
            copy_src = thr_copy.partition_S(src)
        elif fx.const_expr(self._check_is_partioned(src)):
            copy_src = src
        else:
            raise RuntimeError(f"src tensor {src} is not partitioned or unpartitioned")
        fx.copy(copy_atom, copy_src, copy_frag, pred=None)

    def copy_to(self, dst: fx.Tensor, copy_atom = None, rows = None, cols = None):
        copy_atom, thr_copy, copy_frag = self._get_copy_assets(-1, copy_atom)
        if rows is not None:
            def scatter_atom(src, row, col):
                index = dst.layout(row[0], col[0])
                iter = fx.add_offset(fx.get_iter(dst), index)
                atom_D = fx.make_view(iter, copy_atom.layout_dst_tv[1])
                fx.copy(copy_atom, src, atom_D)
            recurisve_apply(scatter_atom, copy_frag, rows, cols)
            return

        if fx.const_expr(self._check_is_unpartioned(dst)):
            copy_dst = thr_copy.partition_D(dst)
        elif fx.const_expr(self._check_is_partioned(dst)):
            copy_dst = dst
        else:
            raise RuntimeError(f"dst tensor {dst} is not partitioned or unpartitioned")
        fx.copy(copy_atom, copy_frag, copy_dst, pred=None)

    def _check_is_unpartioned(self, t: fx.Tensor):
        # compiled time check if a tensor has not been partitioned into tiles
        if t.layout.rank != 2: return False
        s0 = fx.size(t.shape[0]).to_py_value()
        s1 = fx.size(t.shape[1]).to_py_value()
        return s0 == self._block_shape[0] and s1 == self._block_shape[1]

    def _check_is_partioned(self, t: fx.Tensor):
        # compiled time check if a tensor has been partitioned into tiles
        if t.layout.rank != 3: return False
        num_mma_tiles_s0 = fx.size(t.shape[1]).to_py_value()
        num_mma_tiles_s1 = fx.size(t.shape[2]).to_py_value()
        return num_mma_tiles_s0 == self._block_shape[0] //self._tile_size[0] and \
                num_mma_tiles_s1 == self._block_shape[1] //self._tile_size[1]

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
        copy_atom, thr_copy, copy_frag = self._get_copy_assets(0, copy_atom)
        s0 = fx.size(src.shape[0]).to_py_value()
        s1 = fx.size(src.shape[1]).to_py_value()
        assert s0 == self._block_shape[0] and s1 == self._block_shape[1]
        return thr_copy.partition_S(src)

    def partition_D(self, dst: fx.Tensor, copy_atom = None):
        copy_atom, thr_copy, copy_frag = self._get_copy_assets(-1, copy_atom)         
        s0 = fx.size(dst.shape[0]).to_py_value()
        s1 = fx.size(dst.shape[1]).to_py_value()
        assert s0 == self._block_shape[0] and s1 == self._block_shape[1]
        return thr_copy.partition_D(dst)

    def load_gather_rows(self, row_indicies_1d: fx.Tensor):
        return self._load_row_indicies(row_indicies_1d, for_D=False)

    def load_scatter_rows(self, row_indicies_1d: fx.Tensor):
        return self._load_row_indicies(row_indicies_1d, for_D=True)

    def _load_row_indicies(self, row_indicies_1d: fx.Tensor, for_D: bool = False):
        BLOCK_M, BLOCK_K = self._block_shape
        # 
        if row_indicies_1d.layout.rank == 1:
            row_indicies_2d = fx.make_view(fx.get_iter(row_indicies_1d), fx.make_layout((BLOCK_M, BLOCK_K), (1, 0)))
        else:
            assert row_indicies_1d.layout.rank == 2, f"row_indicies_1d must be rank-1 or rank-2, got {row_indicies_1d.layout.rank}"
            row_indicies_2d = row_indicies_1d

        row_indicies_tview = self.partition_S(row_indicies_2d) if not for_D else self.partition_D(row_indicies_2d)

        tview_shape = row_indicies_tview.shape.to_py_value()
        tview_stride = row_indicies_tview.stride.to_py_value()

        # make_fragment_layout_like() reserves space for dimension with 0 stride ???
        # frg_layout = fx.make_fragment_layout_like(row_indicies_tview)
        # print(" frg_layout: ", frg_layout, row_indicies_tview.layout)

        # only pick mode with non-zero stride
        nz_shape = []
        nz_stride = []
        fstride = 1
        def collect_nz_modes(shape, stride):
            nonlocal nz_shape, nz_stride, fstride
            frag_stride = [] # fragment stride is compact
            for s, d in zip(shape,stride):
                if isinstance(d, int):
                    if d != 0:
                        nz_shape.append(s)
                        nz_stride.append(d)
                        frag_stride.append(fstride)
                        fstride *= s
                    else:
                        frag_stride.append(0)
                else:
                    frag_stride.append(collect_nz_modes(s, d))
            return frag_stride
        frag_stride = collect_nz_modes(tview_shape, tview_stride)
        nz_cnt = fstride
        # print(" row_indicies_tview shape: ", nz_shape, " stride: ", nz_stride, " frag_stride: ", frag_stride, " nz_cnt: ", nz_cnt)

        if len(nz_shape) == 0:
            nz_shape = 1
            nz_stride = 0
        row_indicies_sview = fx.make_view(fx.get_iter(row_indicies_tview), fx.make_layout(nz_shape, nz_stride))
        row_indicies_frag = fx.make_fragment_like(row_indicies_sview)
        # print(" row_indicies_frag: ", row_indicies_frag)

        row_indicies_vec = row_indicies_sview.load()

        # store to rmem tensor usually do nothing after lowering?
        row_indicies_frag.store(row_indicies_vec)

        # reshape into partition_S thread-view form
        row_indicies_frag = fx.composition(row_indicies_frag, fx.make_layout(tview_shape, frag_stride))
        #row_indicies_frag = fx.make_view(fx.get_iter(row_indicies_frag), fx.make_layout(row_indicies_tview.shape, frag_stride))

        #print(" row_indicies_frag: ", row_indicies_frag)
        return row_indicies_frag

"""
subclass of fx.ThrCopy, more friendly to use, keep thread-views inside
provide load/store API work with mem-tensors which are more easier to understand
"""
class TCopy(fx.ThrCopy):
    def __init__(self, copy_atom, tv_layout = None, tile_mn = None, tiled_copy = None):
        if tiled_copy is None:
            assert tv_layout is not None and tile_mn is not None
            tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
        super().__init__(tiled_copy, fx.thread_idx.x)
        self.copy_atom = copy_atom
        self.thrview_S = {}
        self.thrview_D = {}

    def get_thrview(self, tensor: fx.Tensor, default=None):
        if tensor in self.thrview_S:
            return self.thrview_S[tensor]
        elif tensor in self.thrview_D:
            return self.thrview_D[tensor]
        else:
            if default is not None:
                return default
            assert False, f"tensor {tensor} is not prepared for copy"

    def make_fragment_like(self, t: fx.Tensor, coord: list = None, dtype = None):
        thrv = self.get_thrview(t)
        if coord is not None:
            coord.insert(0, None)
            return fx.make_fragment_like(thrv[coord], dtype=dtype)
        else:
            return fx.make_fragment_like(thrv, dtype=dtype)

    # prepare thread-view for source tensor, to be used in load()
    def prepare_S(self, *src: fx.Tensor):
        ret = []
        for s in src:
            assert s not in self.thrview_S
            thrv = self.partition_S(s)
            self.thrview_S[s] = thrv
            ret.append(thrv)
        return ret if len(ret) > 1 else ret[0]

    # prepare thread-view for destination tensor, to be used in store()
    def prepare_D(self, *dst: fx.Tensor):
        ret = []
        for d in dst:
            assert d not in self.thrview_D
            thrv = self.partition_D(d)
            self.thrview_D[d] = thrv
            ret.append(thrv)
        return ret if len(ret) > 1 else ret[0]

    def load(self, src: fx.Tensor, coord: list = None, frag = None):
        assert src in self.thrview_S, f"please call prepare_S(src) before load"
        thrv = self.thrview_S[src]
        if frag is None:
            frag = fx.make_fragment_like(thrv) # this is a purely compile-time OP
        else:
            frag = self.retile(frag) # this is a purely compile-time OP
        if coord is None:
            fx.copy(self.copy_atom, thrv, frag)
        else:
            coord.insert(0, None) # select FrgV
            fx.copy(self.copy_atom, thrv[coord], frag)
        return frag
    
    def store(self, frag, dst: fx.Tensor, coord: list = None):
        assert dst in self.thrview_D, f"please call prepare_D(dst) before store"
        thrv = self.thrview_D[dst]
        frag = self.retile(frag)
        if coord is None:
            fx.copy(self.copy_atom, frag, thrv)
        else:
            coord.insert(0, None) # select FrgV
            fx.copy(self.copy_atom, frag, thrv[coord])

    def custom(self, func, *args):
        new_args = []
        for i, v in enumerate(args):
            if isinstance(v, (tuple, list)):
                tensor, coord = v
                coord.insert(0, None) # select FrgV
                if tensor in self.thrview_S:
                    new_args.append(self.thrview_S[tensor][coord])
                elif tensor in self.thrview_D:
                    new_args.append(self.thrview_D[tensor][coord])
                else:
                    assert False, f"{i}'th input is not prepared for copy"
            elif isinstance(v, fx.Tensor):
                thrv = self.get_thrview(v, default=v)
                new_args.append(thrv)
            else:
                new_args.append(v)
        recurisve_apply(func, *new_args)

# load small amount of data into fragment with possible broadcast modes generating no redundant copy
# using memref_load_vec instead of copy-atom
def load_fragment(thr_view: fx.Tensor):
    # make_fragment_layout_like() reserves space for dimension with 0 stride, which is unexpected.
    tview_shape = thr_view.shape.to_py_value()
    tview_stride = thr_view.stride.to_py_value()
    nz_shape = []
    nz_stride = []
    nz_frag_stride = []
    fstride = 1
    def collect_nz_modes(shape, stride):
        nonlocal nz_shape, nz_stride, fstride
        frag_stride = []
        for s, d in zip(shape,stride):
            if isinstance(d, int):
                if d != 0:
                    nz_shape.append(s)
                    nz_stride.append(d)
                    nz_frag_stride.append(fstride)
                    frag_stride.append(fstride) # fragment stride is compact
                    fstride *= s
                else:
                    frag_stride.append(0) # fragment stride keeps all modes, even those with 0 stride
            else:
                frag_stride.append(collect_nz_modes(s, d))
        return frag_stride
    frag_stride = collect_nz_modes(tview_shape, tview_stride)
    nz_cnt = fstride
    # print(" thr_view shape: ", nz_shape, " stride: ", nz_stride, " frag_stride: ", frag_stride, " nz_cnt: ", nz_cnt)

    if len(nz_shape) == 0:
        nz_shape = 1
        nz_stride = 0
    thr_view_nz = fx.make_view(fx.get_iter(thr_view), fx.make_layout(nz_shape, nz_stride))
    frag = fx.make_rmem_tensor(fx.make_layout(nz_shape, nz_frag_stride), thr_view.dtype)

    vec = thr_view_nz.load()
    frag.store(vec) # store to rmem tensor usually do nothing after lowering

    # reshape back to thread-view domain
    #frag = fx.composition(frag, fx.make_layout(tview_shape, frag_stride))
    frag = fx.make_view(fx.get_iter(frag), fx.make_layout(tview_shape, frag_stride))

    return frag
"""

@fxu.fly
def kernel(a,b,c):
    # your flydsl kernel code

# directly invoke the kernel
kernel((1,1,1), (64,1,1), a, b, c)

"""
def fly(fun):
    import inspect
    from flydsl.compiler.ast_rewriter import ASTRewriter
    sig = inspect.signature(fun)
    params = sig.parameters
    nargs = len(params)
    fun = ASTRewriter.transform(fun)
    def call(grid, block, *args):
        # only at first invoke we got args
        # recover args to fx.Tensor with original static shape/stride
        def recover_static_shape_stride(fx_args):
            new_args = []
            for fx_a, orig_a in zip(fx_args, args):
                if isinstance(orig_a, torch.Tensor):
                    shape = list(orig_a.shape)
                    stride  = list(orig_a.stride())
                    if len(shape) == 1:
                        shape = shape[0]
                        stride = stride[0]
                    #print(fx_a.shape, fx_a.stride)
                    #print(">>>", shape, stride)
                    fx_a = fx.Tensor(fx.make_view(fx.get_iter(fx_a), fx.make_layout(shape, stride)))
                new_args.append(fx_a)
            return new_args

        @flyc.kernel
        def fly_kernel(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15):
            args = [a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15][0:nargs]
            args = recover_static_shape_stride(args)
            fun(*args)

        value_attrs = {"rocdl.waves_per_eu": 1,
                        "passthrough": [["amdgpu-agpr-alloc", "256,256"],]
                        }
        value_attrs = None

        @flyc.jit
        def launcher(
            grid0, grid1, grid2,
            block0, block1, block2,
            a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,
            stream = None):
            fly_kernel(a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15).launch(grid=(grid0,grid1,grid2),block=(block0,block1,block2),stream=stream)

        a = list(args)
        while(len(a) < 16):
            a.append(0)
        grid = list(grid)
        while(len(grid) < 3):
            grid.append(1)
        block = list(block)
        while(len(block) < 3):
            block.append(1)
        launcher(*grid, *block,
                a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15],
                stream=torch.cuda.current_stream())
        
        def pre_compiled_launch(grid, block, *args):
            a = list(args)
            while(len(a) < 16):a.append(0)
            grid = list(grid)
            while(len(grid) < 3):grid.append(1)
            block = list(block)
            while(len(block) < 3): block.append(1)
            launcher(*grid, *block,
                a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15],
                stream=torch.cuda.current_stream())
            return pre_compiled_launch        

        return pre_compiled_launch

    return call

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

