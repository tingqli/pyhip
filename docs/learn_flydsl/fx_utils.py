import os
import types
import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.utils.env import DebugEnvManager
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, scf, arith
from onnx_ir import val
import torch

def get_tv_layout(inverse_tv_layout, num_threads):
    tv_cnt = fx.cosize(inverse_tv_layout).get_static_leaf_int
    assert tv_cnt % num_threads == 0, f"Total tile size {tv_cnt} must be divisible by number of threads {num_threads}"
    num_values = tv_cnt // num_threads
    M = fx.size(inverse_tv_layout[0]).get_static_leaf_int
    N = fx.size(inverse_tv_layout[1]).get_static_leaf_int
    tv_layout = fx.composition(fx.right_inverse(inverse_tv_layout),
                               fx.make_layout((num_threads, num_values), (1, num_threads)))
    def show(self):
        print(self)
        print(f"inverse_tv_layout :  (M{M}, N{N}) -> (T{num_threads},V{num_values})")
        print(inverse_tv_layout)
        for m in range(M):
            print(f"M{m}: ", end="")
            t_last = -1
            for n in range(N):
                off = inverse_tv_layout(m, n).get_static_leaf_int
                t = off % num_threads
                v = off // num_threads
                if t == t_last:
                    item = f"..{v}"
                else:
                    item = f"{t}.{v}"
                print(f"{item:>6s}", end=" ")
                t_last = t
            print()

        print(f"tv_layout :   (T{num_threads},V{num_values}) -> (M{M}, N{N})")
        print(tv_layout)
        for t in range(num_threads):
            print(f"T{t}: ", end="")
            for v in range(num_values):
                off = tv_layout(t, v).get_static_leaf_int
                m = off % M
                n = off // M
                item = f"{m}.{n}"
                print(f"{item:>6s}", end=" ")
            print()
        return self
    tv_layout.show = types.MethodType(show, tv_layout)
    tv_layout.tvM = M
    tv_layout.tvN = N
    return tv_layout

def div_e(a, b):
    assert a % b == 0, f"div_e expect {a} evenly divisible by {b}"
    q = a // b
    assert q > 0, f"div_e expect {a} // {b} > 0"
    return q

def get_coalescing_tv_layout(tileM, tileN, vect_width, num_threads):
    num_threads_N = div_e(tileN, vect_width)
    num_threads_M = div_e(num_threads, num_threads_N)
    tv_idx = lambda t, v: v * num_threads + t * 1
    inverse_tv_layout = fx.make_layout((num_threads_M,            (vect_width, num_threads_N)),
                                        (tv_idx(num_threads_N, 0), (tv_idx(0, 1), tv_idx(1, 0))))
    tv_layout = get_tv_layout(inverse_tv_layout, num_threads)
    assert tileM % tv_layout.tvM == 0, f"tileM {tileM} must be divisible by tvM {tv_layout.tvM}"
    return tv_layout

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

def test_copy():
    num_threads = 1
    M = 4096
    N = 4096
    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, copy_atom: fx.CopyAtom):
        recurisve_apply(lambda a, b: fx.copy_atom_call(copy_atom, a, b), A, B)

    @flyc.jit
    def test(A: fx.Tensor, B: fx.Tensor, copy_bits: fx.Constexpr[int], stream):
        copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), A.dtype)
        copy_thr = copy_atom.layout_ref_tv.shape[0].get_static_leaf_int
        copy_val = copy_atom.layout_ref_tv.shape[1].get_static_leaf_int
        A = fx.Tensor(fx.make_view(fx.get_iter(A), fx.make_layout((M, N), (N, 1))))
        B = fx.Tensor(fx.make_view(fx.get_iter(B), fx.make_layout((M, N), (N, 1))))
        A = fx.zipped_divide(A, fx.make_tile(copy_val, copy_thr))
        B = fx.zipped_divide(B, fx.make_tile(copy_val, copy_thr))
        print("A:", A)
        print("B:", B)
        fx.printf("A.layout: {}", A.layout)
        kernel(A, B, copy_atom).launch(
            grid=(1, 1, 1),
            block=(num_threads, 1, 1), stream=stream
        )

    A = torch.randn(M, N, dtype=torch.float32, device="cuda")
    B = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    test(A, B, 32, torch.cuda.Stream())
    assert torch.allclose(A, B)

def enable_dump_ir(enable_debug_info = True):
    DebugEnvManager.enable_debug_info = enable_debug_info
    DebugEnvManager.dump_asm = True
    DebugEnvManager.dump_ir = True
    DebugEnvManager.dump_dir = "kkk"
    ir._globals.register_traceback_file_inclusion(__file__)
    ir._globals.register_traceback_file_exclusion(os.path.dirname(flydsl.__file__))
    ir._globals.set_loc_tracebacks_frame_limit(40)
    ir._globals.set_loc_tracebacks_enabled(True)
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
