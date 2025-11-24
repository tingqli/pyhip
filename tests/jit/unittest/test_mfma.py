import pyhip
import torch
import pytest
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

@pytest.mark.parametrize("BM", [32,32*2])
@pytest.mark.parametrize("BN", [32,32*2])
@pytest.mark.parametrize("K", [16,16*2])
def test_mfma_32x32x16(BM, BN, K):
    assert K % 16 == 0
    assert BM % 32 == 0
    assert BN % 32 == 0
    @pyhip.jit("(void* A, void* B, void* C, int strideAB, int strideC, int K)")
    def kernel(J):
        pkargs = J.new_gpr('s',[0,1])
        thread_id_x = J.new_gpr('v',[0,0], dtype="i32")
        lane_id = J.auto_gpr(thread_id_x[0] & 63)

        pA = J.new_gpr('s', 2, align=2)
        pB = J.new_gpr('s', 2, align=2)
        pC = J.new_gpr('s', 2, align=2)
        K = J.new_gpr('s', 1, align=1, dtype="i32")
        strideAB = J.new_gpr('s', 1, dtype="i32", align=1)
        strideC = J.new_gpr('s', 1, dtype="i32", align=1)
        J.s_load_dwordx2(pA, pkargs, 0)
        J.s_load_dwordx2(pB, pkargs, 8)
        J.s_load_dwordx2(pC, pkargs, 16)
        J.s_load_dword(strideAB, pkargs, 24)
        J.s_load_dword(strideC, pkargs, 28)
        J.s_load_dword(K, pkargs, 32)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        n_block_m = BM//32
        n_block_n = BN//32
        sizeof_half = 2
        sizeof_float = 4
        strideAB_bytes = J.auto_gpr(strideAB[0] * sizeof_half)

        # dwordx4 = 8xhalf
        # but mfma_32x32x8 requires 4xhalf per-lane, so we extend it to 32x32x16
        buff_a = J.Buffer(pA, K*(BM*sizeof_half))
        buff_b = J.Buffer(pB, K*(BN*sizeof_half))
        buff_c = J.Buffer(pC, BM*BN*sizeof_float)

        voffset = J.auto_gpr((lane_id & 31)*strideAB_bytes + (lane_id >> 5)*(8*sizeof_half),
                             name = "voffsetAB")

        matA = J.new_gpr('v', 4*n_block_m, align=4)
        matB = J.new_gpr('v', 4*n_block_n, align=4)
        matC0 = J.new_gpr('v', 16*n_block_m*n_block_n, align=4)

        k = J.new_gpr('s', 1, dtype="i32", align=1)
        k[0] = 0
        for n in range(16*n_block_m*n_block_n):
            matC0[n] = 0
        with J.While(k < K) as loop:
            for bm in range(n_block_m):
                soff = J.auto_gpr(k*sizeof_half + (bm*32)*strideAB_bytes)
                buff_a.load_dwordx4(matA[bm*4:(bm*4 + 3)], voffset, soff)
            for bn in range(n_block_n):
                soff = J.auto_gpr(k*sizeof_half + (bn*32)*strideAB_bytes)
                buff_b.load_dwordx4(matB[bn*4:(bn*4 + 3)], voffset, soff)

            J.s_waitcnt(mod=f"vmcnt({0})")

            for bm in range(n_block_m):
                for bn in range(n_block_n):
                    bi = (bm*n_block_n + bn)*16
                    # v_mfma_f32_32x32x8_f16         vdst:f32x16, vsrc0:f16x4,  vsrc1:f16x4,  src2:f32x16  cbsz abid blgp
                    J.v_mfma_f32_32x32x8_f16(matC0[bi:bi+15], matA[bm*4+0:bm*4+1], matB[bn*4+0:bn*4+1], matC0[bi:bi+15])
                    J.v_mfma_f32_32x32x8_f16(matC0[bi:bi+15], matA[bm*4+2:bm*4+3], matB[bn*4+2:bn*4+3], matC0[bi:bi+15])

            k[0] = k[0] + 16

        strideC_bytes = J.auto_gpr(strideC * sizeof_float)
        voffset = J.auto_gpr((lane_id & 31)*sizeof_float + (lane_id >> 5)*(4*strideC_bytes),
                             name="voffsetC")
        #voffset = J.auto_gpr(4*strideC_bytes)
        soffset = J.new_gpr('s', 1, dtype="i32")
        for bm in range(n_block_m):
            for bn in range(n_block_n):
                bi = (bm*n_block_n + bn)*16
                m0 = bm * 32
                n0 = bn * 32
                bi = (bm*n_block_n + bn)*16
                soffset[0] = 0
                for row in range(0,16,4):
                    # vdata, voffset, soffset, offset12=0
                    soffset[0] = (m0 + 2*row + 0)*strideC_bytes + n0*sizeof_float; buff_c.store_dword(matC0[bi+row+0], voffset, soffset) 
                    soffset[0] = (m0 + 2*row + 1)*strideC_bytes + n0*sizeof_float; buff_c.store_dword(matC0[bi+row+1], voffset, soffset)
                    soffset[0] = (m0 + 2*row + 2)*strideC_bytes + n0*sizeof_float; buff_c.store_dword(matC0[bi+row+2], voffset, soffset)
                    soffset[0] = (m0 + 2*row + 3)*strideC_bytes + n0*sizeof_float; buff_c.store_dword(matC0[bi+row+3], voffset, soffset)

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
    test_mfma_32x32x16(32*2, 32*1, 16)