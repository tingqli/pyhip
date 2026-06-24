import flydsl.compiler as flyc
import flydsl.expr as fx
from . import utils as fxu
from .tiled_copy import TiledCopy
import re
import magichash
hash_printer = magichash.register(mark="###", callback=magichash.do_print)
hash_printer = lambda func: func
def make_tiled_mma(mma_atom, thr_layout_mnk, permutation_mnk):
    return TiledMMA(mma_atom, thr_layout_mnk, permutation_mnk)

def make_tiled_copy_A(copy_atom, tiled_mma): return _make_tiled_copy(copy_atom, tiled_mma, "A")
def make_tiled_copy_B(copy_atom, tiled_mma): return _make_tiled_copy(copy_atom, tiled_mma, "B")
def make_tiled_copy_C(copy_atom, tiled_mma): return _make_tiled_copy(copy_atom, tiled_mma, "C")

def _make_tiled_copy(copy_atom, tiled_mma, ABC):
    """Create a TiledCopy matched to operand A of *tiled_mma*.
    可见 通过 tiled_mma 创建的 tiled_copy， 一次读取的数据布局是 tiled_mma.tv_layout_A_tiled
    """
    axises = {"A": (0, 2), "B": (1, 2), "C": (0, 1)}
    layout_tv = tiled_mma.get_layout_TV(ABC)
    tile_size = tiled_mma.tile_size_mnk
    tile_mn = (tile_size[axises[ABC][0]], tile_size[axises[ABC][1]])
    return TiledCopy(copy_atom, layout_tv, tile_mn)

# cutlass/include/cute/atom/mma_atom.hpp
"""
TiledMMA的核心对外接口是：
 - tv_layout_A/B/C_tiled 用于 make_tiled_copy_A/B/C 创建 TiledCopy
 - partition_A/B/C 用于将一个tensor切分成每个线程的片段，间接被 make_fragment_A/B/C 使用

而这一切的基础是 thrfrg() 也即是把一个 layout 切分为 thread+fragment 的布局。
该函数假定输入layout是矩阵乘法的 A/B/C 之一的原始逻辑布局, 将其按照 tiled-mma 
的配置转换为 thread+fragment 布局：


 - (M, K, ...) -> ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))
 - (N, K, ...) -> ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK,...)))
 - (M, N, ...) -> ((ThrV,(ThrM,ThrN)),(FrgV,(RestM,RestN,...)))

 - tv_layout: 用于 make_tiled_copy_A/B/C, 跟 thrfrg() 的区别是 tv_layout 
   必须凑满全部线程，thrfrg_A 中缺失的ThrN也要补上，因为这个方向上的 MMA-Atom
   跟水平方向的相同 ThrN 的那些 MMA-Atom 不能复用数据寄存器，大家都要搬入（从同一个位置）
   线程部分摊平展开
 - ( thr_idx,                   (FrgV,(RestM,RestK)) )

 - partition_? & make_fragment_?(tensor) : 根据需要放在寄存器中的tensor总数据量，为线程申请寄存器,
   分配的寄存器数目一般都是大于 MMA-Atom 的数目，代表 software-pipeline 的一个 stage 的需要的数目
   另外全部维度被摊平。
   (                            FrgV, RestM, RestK, ...)
   这也正是 fx.gemm 所假设的输入数据的布局：
      - fragA: (FrgV, RestM, RestK)
      - fragB: (FrgV, RestN, RestK)
      - fragC: (FrgV, RestM, RestN)

    def gemm(mma_atom, D, A, B, C):
        C/D : (c-tile, loop_m, loop_n)    always rank-3
         A  : (a-tile, loop_m, [loop_k])  [.] means optional, if not present, means 1
         B  : (b-tile, loop_n, [loop_k])  [.] means optional, if not present, means 1   

其中:
 - ThrV 是一个MMA-Atom协作线程数，cdna是64
 - ThrM 是并行 MMA-Atom 在M方向的个数，可以理解为 wave/warp 在m方向的索引 warp_m
 - ThrN 是并行 MMA-Atom 在N方向的个数，可以理解为 wave/warp 在n方向的索引 warp_n
 - ThrK 是并行 MMA-Atom 在K方向的个数，可以理解为 wave/warp 在k方向的索引 warp_k
 - FrgV 是一个MMA-Atom的值的索引
 - 上面这些维度切片经过 thrfrg 布局映射得到的投影就是所有线程同时执行一条 MMA-Atom 指令时所覆盖的计算量
 - 其后的维度 (RestM,RestK,...) 则代表了为了覆盖整个tensor, 需要如何重复前面这些 MMA-Atom 指令

          thrfrg_A: (((16, 4), (2, 1)), (4, (1, 2))):
                    (((1m,8k),(16m,0)),(1k,(0,4k)))<m32 k32>

tv_layout_?_tiledA: ((16, 4, 2, 2), (4, (1, 2))):
                    ((1m,8k,16m,0),(1k,(0,4k)))<m32 k32>

MmaAtom<!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>>,
thr_layout_mnk = Layout<(2,2,1):(1,2,0)>
permutation_mnk = None,None,Layout<(4,4,2):(1,8,4)>

                       thrfrg()
                    (( ThrV,  (ThrM,ThrK)),( FrgV,(RestM,RestK,...)))
AB: (32,32):(1,32)  (((16,  4),(   2,   1)),(4*f16,(    1,    2))):
                    ((( 1,256),(  16,   0)),(   32,(    0,  128)))
m,k=1,32            (((1m, 8k),( 16m,   0)),(   1k,(    0,   4k)))


permutation_m=Layout<(16,2):(1,16)>
        (((16, 4), (2, 1)), (4, (2, 2))):
        (((1m,8k),(16m,0)),(1k,(32m,4k)))<m64 k32>

permutation_m=Layout<(16,2):(1,32)>
        (((16, 4), (2, 1)), (4, (2, 2))):
        (((1m,8k),(32m,0)),(1k,(16m,4k)))<m64 k32>


increase M by a factor of 6, RestM will be increased by factor of 6 with step of 32m
AB: (192,32):(1,64)     (((16,  4), ( 2,    1)), (   4, (6,   2))):
                        (((1m, 8k), (16m,   0)), (  1k, (32m, 4k)))<m64 k32>


increase K by a factor of 5, RestK will be increased by factor of 5 with step of 32k
AB:  (192,160):(1,192)  (((16,  4), (2,    1)), (4,  (6,   (2, 5)))):
                        (((1m, 8k), (16m,  0)), (1k, (32m, (4k,32k))))<m192 k160>

permutation_m 从None变为 Layout<(16,2):(1,16)>



可以观察到 permutation_k 对 ThrV 中相邻16个线程映射到的列维度的影响

                    (( ThrV,  (ThrM,ThrN)),(    FrgV, (RestM,RestN,...)))
 C: (32,32):(1,32)  (((16, 4),(   2,   2)),((4*f32,1),(     1,   1))):
                    (((32, 4),(  16, 512)),((    1,0),(     0,   0)))
m,n=1,32            (((n, 4m),(  16m,16n)),((    m,0),(     0,   0)))



"""
class TiledMMA:

    def __repr__(self):
        perm = ",".join([str(p) for p in self.permutation_mnk_tuple])
        return f"TiledMMA(mma_atom={self.mma_atom}, thr_layout_mnk={self.thr_layout_mnk}, permutation_mnk={perm})"

    @hash_printer
    def __init__(self, mma_atom, thr_layout_mnk : fx.Layout, permutation_mnk):
        self.mma_atom = mma_atom ### mma_atom.layout_A_tv,mma_atom.layout_B_tv,mma_atom.layout_C_tv,mma_atom.shape_mnk,mma_atom.thr_layout,mma_atom.thr_id
        self.thr_layout_mnk = thr_layout_mnk ###
        self.permutation_mnk_tuple = fxu.to_py_value(permutation_mnk) ###
        
        str2dtype = {"f16":fx.Float16, "bf16":fx.BFloat16, "f32":fx.Float32}
        pattern = r'\(([^)]*)\)\s*->\s*([^>\s]+)'
        match = re.search(pattern, str(mma_atom))
        dtype_a, dtype_b = match.group(1).split(",")
        self.dtype_abc = {"A":str2dtype[dtype_a.strip()], "B":str2dtype[dtype_b.strip()],
                          "C":str2dtype[match.group(2).strip()]}
        #
        # mma_atom.thr_id(v)->thr_id 64:1
        # thr_layout_mnk(m,n,k)->thr_id (2,2,1):(1,2,0)  
        #       tiled_product(64:1, (2,2,1):(1,2,0)) => (64,2,2,1):(1,64,128,0)
        # thr_layout_vmnk_(v,m,n,k)->thr_id  v:lane m/n/k:mma_atom 
        #
        # ThrV: thread index inside an MMA atom, 0..63
        # ThrM: number of concurrent MMA-atom-threads tiled in M direction, 0..1
        # ThrN: number of concurrent MMA-atom-threads tiled in N direction, 0..1
        # ThrK: number of concurrent MMA-atom-threads tiled in K direction, 0..1
        #
        #                       A: ThrV   B (ThrM, ThrN, ThrK)
        # logical_product(A,B)   (ThrV, (ThrM, ThrN, ThrK)) -> thread_idx
        # tiled_product(A,B)     ((ThrV), ThrM, ThrN, ThrK) -> thread_idx
        #
        self.thr_layout_vmnk_ = fx.tiled_product(mma_atom.thr_id, thr_layout_mnk) ### mma_atom.thr_id,thr_layout_mnk,self.thr_layout_vmnk_

        self.thr_size_vmnk = [fx.get_scalar(fx.size(self.thr_layout_vmnk_[i])) for i in range(4)] ###
        self.atom_shape_mnk = [fx.get_scalar(fx.size(self.mma_atom.shape_mnk[i])) for i in range(3)] ###
        # tile-size
        self.permutation_mnk_ = []
        self.tile_size_mnk_ = []
        for i, perm in enumerate(self.permutation_mnk_tuple):
            if perm is None:
                atom_size = self.atom_shape_mnk[i]
                thr_size = self.thr_size_vmnk[i+1]
                self.permutation_mnk_.append(fx.make_layout(atom_size * thr_size, 1))
                self.tile_size_mnk_.append(atom_size * thr_size)
            else:
                self.permutation_mnk_.append(perm)
                self.tile_size_mnk_.append(fx.get_scalar(fx.size(perm)))
        # perm_K(mma_k) -> mem_k
        self.permutation_mnk_ ###

    # make_tiled_copy will use this property to get tv_layout of A/B/C
    @property
    def tv_layout_A_tiled(self): return self.get_layout_TV("A")
    @property
    def tv_layout_B_tiled(self): return self.get_layout_TV("B")
    @property
    def tv_layout_C_tiled(self): return self.get_layout_TV("C")
    @property
    def atom_layout(self): return self.thr_layout_mnk
    @property
    def tile_size_mnk(self): return self.tile_size_mnk_
    @property
    def permutation_mnk(self): return self.permutation_mnk_
    @property
    def thr_layout_vmnk(self): return self.thr_layout_vmnk_

    def abc2axis(self, ABC):
        if ABC == "A":
            return 0, 2
        elif ABC == "B":
            return 1, 2
        elif ABC == "C":
            return 0, 1
        else:
            raise ValueError(f"ABC must be 'A', 'B', or 'C', got {ABC}")

    def get_ref_layout(self, ABC, multiplier_mnk=(1,1,1)):
        axis0, axis1 = self.abc2axis(ABC)
        tile_size = self.tile_size_mnk
        s0 = tile_size[axis0] * multiplier_mnk[axis0]
        s1 = tile_size[axis1] * multiplier_mnk[axis1]
        ref_layout = fx.make_layout((s0, s1),(1, s0))
        return ref_layout

    @hash_printer
    def get_layout_TV(self, ABC):
        thr_vmnk = self.thr_layout_vmnk_.shape.to_py_value() ###
        thr_total = thr_vmnk[0] * thr_vmnk[1] * thr_vmnk[2] * thr_vmnk[3] # total number of threads

        if ABC == "A":
            tmp = fx.make_layout((thr_vmnk[1], thr_vmnk[2]),(1, 0)) # broadcast M
            atile = ((None, (tmp, None)), None)
        elif ABC == "B":
            tmp = fx.make_layout((thr_vmnk[1], thr_vmnk[2]),(0, 1)) # broadcast N
            atile = ((None, (tmp, None)), None)            
        elif ABC == "C":
            #tmp = fx.make_layout((thr_M, thr_N),(1, thr_M)) # identity
            #atile = ((None, (tmp, None)), None)
            atile = None
        else:
            raise ValueError(f"ABC must be 'A', 'B', or 'C', got {ABC}")

        refA = self.get_ref_layout(ABC) ###

        # (ThrV,(ThrM,ThrK)) ---- .compose(atile) -----> (ThrV,(ThrM,ThrN,ThrK))
        tmp1 = fx.make_layout((thr_total, 1), (1, 0))

        thr_comp = fx.complement(self.thr_layout_vmnk_) ### self.thr_layout_vmnk_,thr_comp
        thr_layout_vmnkc = fxu.make_layout(self.thr_layout_vmnk_, thr_comp) ###
        # (v,m,n,k) => thread-id
        thridx2vmnkc = fx.right_inverse(thr_layout_vmnkc) ###

        # thr_idx -> (ThrV,ThrM,ThrN,ThrK)
        thridx_2_thrid = fx.composition(tmp1, thridx2vmnkc) ###

        tfA = self.thrfrg(refA, ABC) ### tfA

        # tfA    ((ThrV,(ThrM,        ThrK)),(FrgV,(RestM,RestK)) )
        # atile  ((None,(tmp,         None)),      None           )
        # tmp1   ((ThrV,((ThrM, ThrN),ThrK)),(FrgV,(RestM,RestK)) )
        # tmp1   ((ThrV,(ThrM,        ThrN)),(FrgV,(RestM,RestK)) )
        if fx.const_expr(atile is not None):
            tmp1 = fxu.composition(tfA, atile) ###
        else:
            tmp1 = tfA

        # ret  (thr_idx,                   (FrgV,(RestM,RestK)) )
        ret = fx.composition(tmp1, fx.make_tile(thridx_2_thrid, None)) ###
        return ret

    def thr_slice(self, tid):
        self.tid = tid
        return self

    @hash_printer
    def thrfrg(self, atensor, ABC):
        """
        // Tile a tensor or a layout from shape
        //   (M,K,...)
        // to shape
        //   ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...))) -> (element in a_tensor)
        //
        // where
        //   ThrV: The threads local to an MMA-atom. layout<0>(ThrLayoutVMNK): ThrV -> thread_idx
        //   ThrM: The threads tiled in M.           layout<1>(ThrLayoutVMNK): ThrM -> thread_idx
        //   ThrK: The threads tiled in K.           layout<2>(ThrLayoutVMNK): ThrK -> thread_idx
        //   FrgV:  The values local to an MMA-atom.
        //   RestM: The values tiled in M.
        //   RestK: The values tiled in K.
        """
        # perm size is the size of whole TiledMMA
        #    size(perm_M) == size(atensor[0])
        #    size(perm_K) == size(atensor[1])
        # 
        perm_mnk = self.permutation_mnk ###
        AtomShape_mnk = self.mma_atom.shape_mnk.to_py_value()
        axis0, axis1 = self.abc2axis(ABC)
        if ABC == "A":
            atom_tv_layout = self.mma_atom.layout_A_tv
        elif ABC == "B":
            atom_tv_layout = self.mma_atom.layout_B_tv
        elif ABC == "C":
            atom_tv_layout = self.mma_atom.layout_C_tv
        else:
            raise ValueError(f"ABC must be 'A', 'B', or 'C', got {ABC}")

        # Reorder the tensor for the TiledAtom
        #       M ,             K,         ...
        #  (TileM, restM), (TileK, restK), ...
        t_tensor = fx.logical_divide(atensor, fx.make_tile(perm_mnk[axis0], perm_mnk[axis1])) ### atensor,t_tensor

        # Tile the tensor for the Atom
        
        #       M ,             K,         ....
        #  (TileM, restM), (TileK, restK), ... 
        # ((AtomM, AtomK), (RestM,RestK, ...)), 
        a_tensor = fx.zipped_divide(t_tensor, 
                                    fx.make_tile(
                                    fx.make_layout(AtomShape_mnk[axis0], 1),
                                    fx.make_layout(AtomShape_mnk[axis1], 1))) ### t_tensor,a_tensor

        # Transform the Atom mode from (M,K) to (Thr,Val)
        # ((AtomM,   AtomK), (RestM,RestK, ...)) o ((ThrV, Frg), None)
        # (( ThrV,     Frg), (RestM,RestK, ...))
        # (((16,4),      1) ,(2,    4, ))
        tv_tensor = fx.composition(a_tensor, fx.make_tile(atom_tv_layout, None)) ### a_tensor, tv_tensor, atom_tv_layout

        self.thr_size_vmnk ###

        # Tile the tensor for the Thread
        thr_tile = (None, (self.thr_size_vmnk[1+axis0], self.thr_size_vmnk[1+axis1])) ###
        
        #                A:    ((ThrV,    Frg), (    RestM,       RestK, ...))
        #                B:    (     None,      (    ThrM,        ThrK ))
        # logical_divide(A,B): ((ThrV,    Frg), ((ThrM, restM), (ThrK, restK), ...))
        # zipped_divide(A,B):  ((ThrV,    Frg), ((ThrM, ThrK), (restM, restK), ...)) zip-recursively
        #                      ((ThrV,(ThrM, ThrK)), (Frg, (restM, restK), ...)) -> (element in a_tensor)
        #
        # 从这个步骤可以发现，使用ThrM切分RestM这个行为代表使用相邻的mma-atom线程(wave)读取相邻的RestM数据
        # 但是别忘记之前logic-divide时已经compose了perm_M/perm_K，也就是，如果不使用perm_M/perm_K
        # 就是相邻mma-atom线程读取相邻数据，但是permM/permK可以改变这个映射。
        #
        # 例如没有传入permM时，2个M方向的 mma-atom 就会取0,1两个块，但是传入permM=(2,4):(4,1)之后
        # 2个M方向的 mma-atom 就会取(0,4)同时算，然后loop下一轮取(1,5),(2,6),(3,7),这样循环下来就是
        # 第一个mma-atom线程算了0123,第二个mma-atom线程算了4567。
        #thr_tensor = fx.zipped_divide(tv_tensor, thr_tile) # <==== bug
        if isinstance(tv_tensor, fx.Tensor) and isinstance(tv_tensor.layout, fx.ComposedLayout):
            clayout = tv_tensor.layout
            new_outer = fxu.zipped_divide(clayout.outer, thr_tile)
            thr_tensor = fxu.view_as(tv_tensor,
                                     fx.make_composed_layout(clayout.inner, clayout.offset, new_outer))
        else:
            thr_tensor = fxu.zipped_divide(tv_tensor, thr_tile) ###

        return thr_tensor

    def partition_A(self, tensor): return self.partition(tensor, "A")
    def partition_B(self, tensor): return self.partition(tensor, "B")
    def partition_C(self, tensor): return self.partition(tensor, "C")

    def partition(self, tensor, ABC, tid=None):
        if tid is None:
            tid = getattr(self, "tid", tid)
        assert tid is not None
        tid_vmnk = fx.get_flat_coord(tid, self.thr_layout_vmnk_)
        thr_tensor = self.thrfrg(tensor, ABC)
        # ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))
        # (                    FrgV, RestM, RestM, ...)
        slice_rests = (None,) * thr_tensor.layout[1][1].rank # to expand the nested structure
        axis0, axis1 = self.abc2axis(ABC)
        return thr_tensor[(tid_vmnk[0], (tid_vmnk[1+axis0], tid_vmnk[1+axis1])), (None, slice_rests)]

    def make_fragment_A(self, tensor): return self.make_fragment(tensor, "A")
    def make_fragment_B(self, tensor): return self.make_fragment(tensor, "B")
    def make_fragment_C(self, tensor): return self.make_fragment(tensor, "C")

    def make_fragment(self, tensor, ABC):
        part = self.partition(tensor, ABC, 1) # use any fake tid, just to get the shape of fragment
        return fx.make_fragment_like(part, dtype=self.dtype_abc[ABC])

@flyc.jit
def _test_tiled_mma(t:fx.Tensor, fake_tid: fx.Int32):
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16))
    thr_layout_mnk = fx.make_layout((2, 2, 1), (1, 2, 0))
    k_perm = fx.make_layout((4, 4), (4, 1))
    permutation_mnk = (None, None, k_perm)

    tiled_mma = TiledMMA(mma_atom, thr_layout_mnk, permutation_mnk)
    tiled_mma_ref = fx.make_tiled_mma(mma_atom, thr_layout_mnk, permutation_mnk)

    refA = tiled_mma_ref.tv_layout_A_tiled
    retA = tiled_mma.tv_layout_A_tiled
    assert fxu.is_same(retA, refA), f"{retA} != refA:{refA}"

    refB = tiled_mma_ref.tv_layout_B_tiled
    retB = tiled_mma.tv_layout_B_tiled
    assert fxu.is_same(retB, refB), f"{retB} != refB:{refB}"

    refC = tiled_mma_ref.tv_layout_C_tiled
    retC = tiled_mma.tv_layout_C_tiled
    assert fxu.is_same(retC, refC), f"{retC} != refC:{refC}"

    # thr_slice + partition_A/B/C
    tid = 1
    
    thrmma_ref = tiled_mma_ref.thr_slice(tid)
    thrmma = tiled_mma.thr_slice(tid)

    A = fx.Tensor(fx.make_view(fx.make_int_tuple(0), tiled_mma.get_ref_layout("A")))
    ref = thrmma_ref.partition_A(A)
    ret = thrmma.partition_A(A)
    assert fxu.is_same(ret, ref), f"{ret} != refA:{ref}"

    B = fx.Tensor(fx.make_view(fx.make_int_tuple(0), tiled_mma.get_ref_layout("B")))
    ref = thrmma_ref.partition_B(B)
    ret = thrmma.partition_B(B)
    assert fxu.is_same(ret, ref), f"{ret} != refB:{ref}"

    C = fx.Tensor(fx.make_view(fx.make_int_tuple(0), tiled_mma.get_ref_layout("C")))
    ref = thrmma_ref.partition_C(C)
    ret = thrmma.partition_C(C)
    assert fxu.is_same(ret, ref), f"{ret} != refC:{ref}"

    A = fxu.view_as(t, tiled_mma.get_ref_layout("A"))
    B = fxu.view_as(t, tiled_mma.get_ref_layout("B"))
    C = fxu.view_as(t, tiled_mma.get_ref_layout("C"))
    
    assert fxu.is_same(thrmma.make_fragment_A(A), thrmma_ref.make_fragment_A(A))
    assert fxu.is_same(thrmma.make_fragment_B(B), thrmma_ref.make_fragment_B(B))
    assert fxu.is_same(thrmma.make_fragment_C(C), thrmma_ref.make_fragment_C(C))

def test_tiled_mma():
    import os
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    import torch
    _test_tiled_mma(torch.randn(1024, dtype=torch.float32), 1)


@flyc.jit
def _exp_tiled_mma(t:fx.Tensor, fake_tid: fx.Int32):
    AB_dtype = fx.Float16
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, AB_dtype))
    thr_layout_mnk = fx.make_layout((2, 2, 1), (1, 2, 0))
    k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
    # k_perm = fx.recast_layout(k_perm, 32, AB_dtype.width)

    m_perm = None
    # m_perm = fx.make_layout((16, 2), (1, 32))
    permutation_mnk = (m_perm, None, k_perm)

    multiplier_mnk = (2, 2, 1)
    tiled_mma = TiledMMA(mma_atom, thr_layout_mnk, permutation_mnk)
    print(tiled_mma)
    for x, dtype in [("A", fx.Float16), ("B", fx.Float16), ("C", fx.Float32)]:
        tensor = fxu.view_as(t, tiled_mma.get_ref_layout(x, multiplier_mnk), dtype=dtype)
        thrfrg = tiled_mma.thrfrg(tensor, x)
        copy_tv_layout = tiled_mma.tv_layout_A_tiled
        print(f"============= matrix {x} {dtype} =============")
        print(f"{x}:", tensor)
        images = " ".join([f"{n}{d}" for n,d in zip(["m","k"], tensor.shape.to_py_value())])
        print(f"  thrfrg_{x}:", fxu.layout2str(thrfrg, images))

        ref_layout = tiled_mma.get_ref_layout(x)
        
        images = " ".join([f"{n}{d}" for n,d in zip(["m","k"], ref_layout.shape.to_py_value())])

        print(f"tv_layout{x}:", fxu.layout2str(copy_tv_layout, images))

        fragment = tiled_mma.make_fragment(tensor, x)
        print(f"fragment_{x}:", fragment)


def test_exp_tiled_mma():
    import os
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    import torch    
    _exp_tiled_mma(torch.randn(1024, dtype=torch.float32), 1)

##########################################
# contain relative import, trigger pytest with 
#   pytest --pyargs pyhip.contrib.flydsl.tiled_mma
#   pytest -s --pyargs pyhip.contrib.flydsl.tiled_mma::test_tiled_mma

