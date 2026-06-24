import flydsl.compiler as flyc
import flydsl.expr as fx
from . import utils as fxu
import magichash
hash_printer = magichash.register(mark="###", callback=magichash.do_print)

def make_tiled_copy(copy_atom, layout_thr_val, tile_mn):
    return TiledCopy(copy_atom, layout_thr_val, tile_mn)

"""
TiledCopy核心对外接口 get_slice(tid).partition_S / partition_D / retile(tensor)
其基础是 tile2thrfrg()

输入 layout (M, N, ...) -> offset in tensor
输出 layout ( thr_idx, (trg_val, rest_val), (RestM, RestN,...) ) -> offset in tensor
 - thr_idx : 线程索引
 - (trg_val, rest_val)： tv-layout 中要求的 value 的索引
    - trg_val 正好是一次 copy-atom 的搬运量
    - rest_val 是 copy-atom 的重复次数
 - RestM, RestN,... :  输入 (M, N, ...) 按照 (tile_m, tile_n) divide剩余的维度


partition_S 就是使用当前线程 tid 切片上面的布局（并且把Rest维度展平）得到的当前线程的搬运量，partition_D 同理：
     ((trg_val, rest_val), RestM, RestN, ...)


retile:
 - 输入一般来自 tiled_mma 的 make_fragment_A/B/C, 是 tiled_mma 的每个线程的寄存器总数据量：
        (FrgV, RestM, RestK, ...)
   - FrgV: 一次 MMA-Atom 需要的数据量
   - RestM, RestN,...: software-pipeline 中一个 stage 需要的寄存器总数据量

 - 输出是符合 partition_S / partition_D 要求的 view 布局，这里有一个布局的差别
   当 tiled_copy 也是根据 tiled_mma 给出的 tv-layout 生成的时候， tiled_mma
   给出的 tv-layout 是一个所有线程执行一次 MMA-Atom 指令时，所需的 A/B/C 的寄存器个数
   只有1份，RestK中保存着其余的部分，但是copy-atom需要把他们连续放在 fragment 中
   这样才能被copy-atom切割

"""

# cutlass/include/cute/atom/copy_atom.hpp
class TiledCopy:
    def __init__(self,
                 copy_atom, 
                 layout_thr_val, # (tid,vid) -> coord   [Need not be 2D...]
                 tile_mn         # coord space
                 ):
        if isinstance(tile_mn, fx.Tile):
            tile_mn = fxu.to_py_value(tile_mn)
        assert isinstance(tile_mn, (tuple, list))
        self.tile_mn_ = tile_mn
        self.copy_atom = copy_atom
        self.layout_thr_val = layout_thr_val
        
        self.Tiler_MN = tile_mn
        self.AtomThrID = copy_atom.thr_id            # thrid -> thr_idx
        self.AtomLayoutRef = copy_atom.layout_ref_tv # (thr,val) -> offset
        self.AtomLayoutSrc = copy_atom.layout_src_tv # (thr,val) -> offset
        self.AtomLayoutDst = copy_atom.layout_dst_tv # (thr,val) -> offset
        self.AtomNumThr = fx.get_scalar(fx.size(self.AtomLayoutRef[0]))
        self.AtomNumVal = fx.get_scalar(fx.size(self.AtomLayoutRef[1]))
        self.TiledLayout_TV = layout_thr_val

    def tile2thrfrg(self, tensor, ref2trg):

        # tensor 在外部已经被 self.Tiler_MN zipped_divide 过，布局为：
        #    ((TileM, TileN), (restM, restN, ...))

        # // Take the thrs/vals that the atom is interested in
        # (tid, vid) -> (m, n)

        atom_layout_TV = fxu.zipped_divide(self.TiledLayout_TV, (self.AtomNumThr, self.AtomNumVal)) ###
        # ((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)

        # // Transform to the trg layout
        trg_layout_TV = fxu.composition(atom_layout_TV, (ref2trg, None)) ###
        # // ((trg_tid,trg_val),(rest_tid,rest_val)) -> (m,n)

        # // Transform the thrs mode from thrid to thr_idx
        # // NOTE: Assumes the AtomNumThr are contiguous and identity within TiledThrID
        thrval2mn = fx.coalesce(fxu.zip_(trg_layout_TV), (1, (1, 1))) ###
        # // zip_      ((trg_tid,rest_tid),(trg_val,rest_val)) -> (m,n)
        # // coalesce  (   tid            ,(trg_val,rest_val)) -> (m,n)

        tv_tensor = fx.composition(tensor, fx.make_tile(thrval2mn, None)) ###
        # // ((thrid,val),(RestM,RestN,...))

        # flatten
        if isinstance(tv_tensor, fx.Tensor):
            ret = tv_tensor[(None, None), None] ###
        else:
            ret = tv_tensor((None, None), None) ###
        # ( thr_idx, val_idx, (RestM,RestN,...) )
        return ret

    def get_layout_TV(self, atom_layout):
        #// (M,N) -> (M,N)
        ref_S = fxu.make_layout((self.Tiler_MN, 1)) ###
        ref2trg = fx.composition(fx.right_inverse(self.AtomLayoutRef), atom_layout) ###
        # // (thr_idx,val_idx) -> (M,N)
        return self.tile2thrfrg(ref_S, ref2trg)(None,None,0)

    def get_layoutS_TV(self): return self.get_layout_TV(self.AtomLayoutSrc)
    def get_layoutD_TV(self): return self.get_layout_TV(self.AtomLayoutDst)

    def tidfrg_S(self, stensor): return self.tidfrg(stensor, self.AtomLayoutSrc)
    def tidfrg_D(self, stensor): return self.tidfrg(stensor, self.AtomLayoutDst)

    def tidfrg(self, stensor, atom_layout):
        # assert stensor.rank >= len(self.tile_mn_), "Rank of tensor to be partitioned too small."
        stensor = fx.zipped_divide(stensor, self.Tiler_MN)
        ref2trg = fx.composition(fx.right_inverse(self.AtomLayoutRef), atom_layout)
        return self.tile2thrfrg(stensor, ref2trg)

    def partition_S(self, src: fx.Tensor):
        tf_src = self.tidfrg_S(src)
        rank = getattr(src, "rank", src.layout.rank)
        return tf_src[self.thr_idx, None, [None]*rank]

    def partition_D(self, dst: fx.Tensor):
        tf_dst = self.tidfrg_D(dst)
        rank = getattr(dst, "rank", dst.layout.rank)
        return tf_dst[self.thr_idx, None, [None]*rank]

    @property
    def tile_mn(self): return self.tile_mn_

    @property
    def layout_tv_tiled(self): return self.layout_thr_val
    @property
    def layout_src_tv_tiled(self): return self.get_layoutS_TV()
    @property
    def layout_dst_tv_tiled(self): return self.get_layoutD_TV()

    def get_slice(self, thr_idx):
        self.thr_idx = thr_idx
        return self

    def thr_slice(self, thr_idx):
        return self.get_slice(thr_idx)

    def retile(self, tensor):
        tv_layout = self.layout_src_tv_tiled
        # 把 tiled_mma 
        # tv_layout    Layout<((16,4,2,2),(1,2)):((1,64,16,0),(0,32))>
        # tiler_mn     (32,8)
        # atom_num_val 1
        # tensor       Tensor<f32, register, (1,2,2):(0,2,1)>
        #
        # output:      Tensor<f32, register, ((1,2),2,1):((0,1),2,0)>
        R = tensor.layout.rank
        nThreads = fx.get_scalar(fx.size(tv_layout.shape[0]))
        V = fx.get_scalar(fx.size(tensor.layout.shape[0]))
        M, N = self.Tiler_MN
        
        """
        // Assert that AtomLayoutSrc|Dst is identity so we can skip the Ref transformation

        // Assume the first size<0>(tensor) elements are the first val_ids in TiledLayout_TV.
        // Then, we only need the shape+layout of those size<0>(tensor) elements in TiledLayout_TV
        //   and that shape is what we gather from the other modes of tensor    
        """

        mn_layout = fx.composition(fx.right_inverse(tv_layout), fx.make_layout((M, N),(1, M))) ###
        # ((16,2),(2,4)):((1,64),(256,16))
        frg_layout_mn = fx.recast_layout(mn_layout, 1, nThreads * V) ###
        #  (m,n) -> v_idx -- The shape and order of the V inside of TiledLayout_TV
        # ((1,1),(2,1)):((1,1),(1,1))

        d0 = fx.logical_product(fx.make_layout(V,1), fx.right_inverse(frg_layout_mn)) ###
        # 'Layout<(1,2):(0,1)>'
        d1 = fx.make_layout(self.AtomNumVal,1) ###
        # 'Layout<1:1>'

        frg_layout_v = fx.zipped_divide(d0, d1)  ###
        #  (atom_vals,rest_vals) -> (v,m,n)
        # (1,2):(0,1)

        # Tile the tensor for TileFrg
        
        #x = fx.prepend(product_each(frg_layout_mn), V)
        x = [V] + [fx.get_scalar(fx.size(frg_layout_mn[i])) for i in range(frg_layout_mn.rank)] # 
        # [1, 1, 2]

        t_tensor = fx.zipped_divide(tensor, x)  ###
        # ((TileV,TileM,TileN,...),(1,RestM,RestN,...))
        # ((1,1,2),(1,2,1)):((0,0,1),(0,2,0))>'

        # Transform the tile mode
        v_tensor = fx.composition(t_tensor, fx.make_tile(frg_layout_v,None))   ###
        # ((atom_vals,rest_vals),(1,RM,RN,...))
        # ((1,2),(1,2,1)):((0,1),(0,2,0))

        # Unfold and return
        coord = [0] + [None]*(R-1) # [0, None, None]
        ret = v_tensor[None, coord]  ###

        # Tensor<f32, register, ((1,2),2,1):((0,1),2,0)>

        return ret

@flyc.jit
def _test_tiled_copy(t:fx.Tensor, fake_tid: fx.Int32):
    tileM = 128
    tileN = 128
    num_threads = 256
    copy_bits = 128
    print("Testing TiledCopy...")
    copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), t.dtype)
    VECT_WIDTH = fxu.div_e(copy_bits, t.dtype.width)
    print("VECT_WIDTH: ", VECT_WIDTH)
    tv_layout, tv_tilemn = fxu.make_mem_coalescing_2d_tv_layout(num_threads, VECT_WIDTH, tileN)
    
    A = fx.Tensor(fx.make_view(fx.make_int_tuple(0), fx.make_layout((tileM, tileN), (1, tileM))))

    tiled_copy_ref = fx.make_tiled_copy(copy_atom, tv_layout, tv_tilemn)
    tiled_copy = TiledCopy(copy_atom, tv_layout, tv_tilemn)
    tid = 1
    
    assert fxu.is_same(tiled_copy_ref.layout_tv_tiled, tiled_copy.layout_tv_tiled)
    assert fxu.is_same(tiled_copy_ref.layout_src_tv_tiled, tiled_copy.layout_src_tv_tiled)
    assert fxu.is_same(tiled_copy_ref.layout_dst_tv_tiled, tiled_copy.layout_dst_tv_tiled)

    part_ref = tiled_copy_ref.get_slice(tid).partition_S(A)
    part_ret = tiled_copy.get_slice(tid).partition_S(A)
    assert fxu.is_same(part_ref, part_ret)

    part_ref = tiled_copy_ref.get_slice(tid).partition_D(A)
    part_ret = tiled_copy.get_slice(tid).partition_D(A)
    assert fxu.is_same(part_ref, part_ret)


def test_tiled_copy():
    import os
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")    
    import torch
    print("Testing TiledCopy...")
    _test_tiled_copy(torch.randn(1024, dtype=torch.float32), 1)

# test_tiled_copy()
##########################################
# contain relative import, trigger pytest with 
#   pytest --pyargs pyhip.contrib.flydsl.tiled_copy
#   pytest -s --pyargs pyhip.contrib.flydsl.tiled_copy::test_tiled_copy

