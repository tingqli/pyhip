import pyhip
import torch
import pytest
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

from functools import cache

# reuse voffset to save VGPR
@cache
def get_voffset(J, lane_rows, lane_cols, lane_bytes, stride_bytes):
    lane_id = J.gpr(J.threadIdx.x[0] & 63)
    if lane_cols == 1:
        voffset = J.gpr((lane_id % lane_rows)*stride_bytes + (lane_id // lane_rows)*(lane_bytes))
    else:
        voffset = J.gpr((lane_id // lane_cols)*stride_bytes + (lane_id % lane_cols)*(lane_bytes))
    return voffset

class RegTile:
    def __init__(self, J, shape, lane_layout, num_items, dtype, stride_bytes):
        """
        Parameters:
            shape       : the 2D tensor shape, must be (rows, cols)
            lane_layout : layout of 64 lanes
                    [1,  4] means row-major with row-size of 4 (each row has 4 cols)
                    [16, 1] means col-major with col-size of 16 (each col has 16 rows)
            num_items   : number of items of each lane
            dtype       : data type of each item
            stride      : the stride of external memory
            reg_strides : the stride of registers 
        """

        assert len(shape) == 2
        rows, cols = shape
        
        lane_bytes = num_items * dtype.itemsize
        assert (lane_bytes % 4) == 0, f"lane size {lane_bytes}bytes is not multiple of 32bit"
        assert lane_bytes == 16, f"dwordx4 is optimal lane size"
        assert (cols % num_items) == 0
        self.lane_regs = lane_bytes // 4
        self.lane_bytes = lane_bytes

        total_lanes = (rows * cols // num_items)
        assert (total_lanes % 64) == 0
        self.repeats = total_lanes//64

        assert len(lane_layout) == 2
        assert lane_layout[0] == 1 or lane_layout[1] == 1
        self.lane_rows, self.lane_cols = lane_layout
        assert (64 % self.lane_cols) == 0
        assert (64 % self.lane_rows) == 0

        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.J = J

        if self.lane_cols == 1:
            assert (self.rows % self.lane_rows) == 0
            self.n_rows = self.rows//self.lane_rows
            self.n_cols = (cols // num_items) // (64//self.lane_rows)
        else:
            assert ((self.cols//num_items) % self.lane_cols) == 0
            self.n_cols = (self.cols//num_items) // self.lane_cols
            self.n_rows = self.rows//(64 // self.lane_cols)

        self.gpr = J.gpr(f"vu32x{self.lane_regs * self.n_cols * self.n_rows}")
        self.voffset = get_voffset(J, self.lane_rows, self.lane_cols, self.lane_bytes, stride_bytes)
        self.stride_bytes = stride_bytes

    def __getitem__(self, key):
        return self.gpr.__getitem__(key)
    def __setitem__(self, key, value):
        return self.gpr.__setitem__(key, value)

    def load(self, buff, soffset, offset12=0, do_store=False):
        if self.n_cols == 1 and self.n_rows == 1:
            if do_store:
                buff.store_dwordx4(self.gpr, self.voffset, soffset, offset12=offset12)
            else:
                buff.load_dwordx4(self.gpr, self.voffset, soffset, offset12=offset12)
            return
        if self.lane_cols == 1:
            rep_step_h = (64 // self.lane_rows)*self.lane_bytes
            rep_step_v = self.lane_rows
            #voffset = J.gpr((lane_id % lane_rows)*stride_bytes + (lane_id // lane_rows)*(lane_bytes))
        else:
            rep_step_h = self.lane_cols * self.lane_bytes
            rep_step_v = (64 // self.lane_cols)
            #voffset = J.gpr((lane_id // lane_cols)*stride_bytes + (lane_id % lane_cols)*(lane_bytes))        
        soff = self.J.gpr("su32")
        soff[0] = soffset
        ireg = 0
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if do_store:
                    buff.store_dwordx4(self.gpr[ireg:ireg+self.lane_regs-1], self.voffset, soff, offset12=(offset12 + c*rep_step_h))
                else:
                    buff.load_dwordx4(self.gpr[ireg:ireg+self.lane_regs-1], self.voffset, soff, offset12=(offset12 + c*rep_step_h))
                ireg += self.lane_regs
            if r < self.n_rows - 1:
                soff[0] = soff[0] + rep_step_v * self.stride_bytes

    def store(self, buff, soffset, offset12=0):
        self.load(buff, soffset, offset12, do_store=True)

    def __repr__(self):
        ret = f"{self.rows}x{self.cols} {repr(self.dtype)} {self.gpr} {self.lane_bytes} bytes x {self.lane_rows} lane_rows, reps=({self.n_rows},{self.n_cols})"
        return ret


if __name__ == "__main__":
    #@pyhip.jit()
    #def kernel(J):
    #    reg_tile = RegTile(J, (32*2, 16*2), (32, 1), 8, torch.bfloat16, 128)
    #    print(reg_tile)
    #assert 0
    pass


@pytest.mark.parametrize("BM", [32,32*2])
@pytest.mark.parametrize("BN", [32,32*2])
@pytest.mark.parametrize("K", [16,16*2])
def test_mfma_32x32x16(BM, BN, K):
    assert K % 16 == 0
    assert BM % 32 == 0
    assert BN % 32 == 0
    @pyhip.jit()
    def kernel(J, pA:"void*", pB:"void*", pC:"void*", strideAB:"int", strideC:"int", K:"int"):
        lane_id = J.auto_gpr(J.threadIdx.x[0] & 63)

        n_block_m = BM//32
        n_block_n = BN//32
        print(f"{n_block_m=} {n_block_n=}")
        sizeof_half = 2
        sizeof_float = 4
        strideAB_bytes = J.auto_gpr(strideAB[0] * sizeof_half)
        strideC_bytes = J.auto_gpr(strideC * sizeof_float)

        # dwordx4 = 8xhalf
        # but mfma_32x32x8 requires 4xhalf per-lane, so we extend it to 32x32x16
        buff_a = J.Buffer(pA, K*(BM*sizeof_half))
        buff_b = J.Buffer(pB, K*(BN*sizeof_half))
        buff_c = J.Buffer(pC, BM*BN*sizeof_float)

        matA = RegTile(J, (32*n_block_m, 16), (32,1), 8, torch.float16, strideAB_bytes)
        matB = RegTile(J, (32*n_block_n, 16), (32,1), 8, torch.float16, strideAB_bytes)
        matC0 = RegTile(J, (32*n_block_m, 32*n_block_n), (32,1), 4, torch.float, strideC_bytes)

        k = J.new_gpr('s', 1, dtype="i32", align=1)
        k[0] = 0
        for n in range(16*n_block_m*n_block_n):
            matC0[n] = 0
        with J.While(k < K) as loop:
            matA.load(buff_a, k*sizeof_half)
            matB.load(buff_b, k*sizeof_half)
            J.s_waitcnt(mod=f"vmcnt({0})")

            for bm in range(n_block_m):
                for bn in range(n_block_n):
                    bi = (bm*n_block_n + bn)*16
                    # v_mfma_f32_32x32x8_f16         vdst:f32x16, vsrc0:f16x4,  vsrc1:f16x4,  src2:f32x16  cbsz abid blgp
                    J.v_mfma_f32_32x32x8_f16(matC0[bi:bi+15], matB[bn*4+0:bn*4+1], matA[bm*4+0:bm*4+1], matC0[bi:bi+15])
                    J.v_mfma_f32_32x32x8_f16(matC0[bi:bi+15], matB[bn*4+2:bn*4+3], matA[bm*4+2:bm*4+3], matC0[bi:bi+15])

            k[0] = k[0] + 16

        matC0.store(buff_c, 0)
        J.s_waitcnt(mod=f"vmcnt({0})")

    print(f">>>> {BM=} {BN=} {K=}")
    A = torch.randn(BM, K, dtype=torch.float16)
    B = torch.randn(BN, K, dtype=torch.float16)
    C = torch.randn(BM, BN, dtype=torch.float)
    ref = torch.nn.functional.linear(A, B).to(dtype=torch.float)
    kernel([1],[64], A.data_ptr(), B.data_ptr(), C.data_ptr(), K, BN, K)
    torch.cuda.synchronize()
    if not torch.allclose(ref, C, atol=0.01, rtol=0.01):
        print(ref)
        print(C)
        assert 0

if __name__ == "__main__":
    test_mfma_32x32x16(32*2, 32*2, 16)