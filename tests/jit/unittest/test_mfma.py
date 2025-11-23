import pyhip
import torch
import pytest
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

@pytest.mark.parametrize("K", [16, 16*2, 16*31])
def test_mfma_32x32x16(K):
    @pyhip.jit("(void* A, void* B, void* C, int strideAB, int strideC, int K)")
    def kernel(J):
        pkargs = J.new_gpr('s',[0,1],name="pkargs")
        thread_id_x = J.new_gpr('v',[0,0], dtype="i32")
        lane_id = J.auto_gpr(thread_id_x[0] & 63, name="lane_id")

        pA = J.new_gpr('s', 2, align=2, name="pA")
        pB = J.new_gpr('s', 2, align=2, name="pB")
        pC = J.new_gpr('s', 2, align=2, name="pC")
        K = J.new_gpr('s', 1, align=1, dtype="i32", name="K")
        strideAB = J.new_gpr('s', 1, dtype="i32", align=1, name="strideAB")
        strideC = J.new_gpr('s', 1, dtype="i32", align=1, name="strideC")
        J.s_load_dwordx2(pA, pkargs, 0)
        J.s_load_dwordx2(pB, pkargs, 8)
        J.s_load_dwordx2(pC, pkargs, 16)
        J.s_load_dword(strideAB, pkargs, 24)
        J.s_load_dword(strideC, pkargs, 28)
        J.s_load_dword(K, pkargs, 32)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        sizeof_half = 2
        sizeof_float = 4
        strideAB_bytes = J.auto_gpr(strideAB[0] * sizeof_half, name="strideAB_bytes")

        # dwordx4 = 8xhalf
        # but mfma_32x32x8 requires 4xhalf per-lane, so we extend it to 32x32x16
        buff_a = J.Buffer(pA, K*(32*sizeof_half))
        buff_b = J.Buffer(pB, K*(32*sizeof_half))
        buff_c = J.Buffer(pC, 32*32*sizeof_float)

        voffset = J.auto_gpr((lane_id & 31)*strideAB_bytes + (lane_id >> 5)*(8*sizeof_half),
                             name = "voffsetAB")

        matA = J.new_gpr('v', 4, align=4,name="matA")
        matB = J.new_gpr('v', 4, align=4,name="matB")
        matC0 = J.new_gpr('v', 16, align=4,name="matC0")

        k = J.new_gpr('s', 1, dtype="i32", align=1, name="k")
        k[0] = 0
        for n in range(16):
            matC0[n] = 0
        with J.While(k < K) as loop:
            soff = J.auto_gpr(k*sizeof_half)
            buff_a.load_dwordx4(matA, voffset, soff)
            buff_b.load_dwordx4(matB, voffset, soff)
            J.s_waitcnt(mod=f"vmcnt({0})")

            # v_mfma_f32_32x32x8_f16         vdst:f32x16, vsrc0:f16x4,  vsrc1:f16x4,  src2:f32x16  cbsz abid blgp
            J.v_mfma_f32_32x32x8_f16(matC0[0:15], matA[0:1], matB[0:1], matC0[0:15])
            J.v_mfma_f32_32x32x8_f16(matC0[0:15], matA[2:3], matB[2:3], matC0[0:15])

            k[0] = k[0] + 16

        strideC_bytes = J.auto_gpr(strideC * sizeof_float, name="strideC_bytes")
        voffset = J.auto_gpr((lane_id & 31)*sizeof_float + (lane_id >> 5)*(4*strideC_bytes),
                             name="voffsetC")
        #voffset = J.auto_gpr(4*strideC_bytes)
        soffset = J.new_gpr('s', 1, dtype="i32", align=1, name="soffset")
        soffset[0] = 0
        for row in range(0,16,4):
            # vdata, voffset, soffset, offset12=0
            soffset[0] = (2*row + 0)*strideC_bytes; buff_c.store_dword(matC0[row+0], voffset, soffset) 
            soffset[0] = (2*row + 1)*strideC_bytes; buff_c.store_dword(matC0[row+1], voffset, soffset)
            soffset[0] = (2*row + 2)*strideC_bytes; buff_c.store_dword(matC0[row+2], voffset, soffset)
            soffset[0] = (2*row + 3)*strideC_bytes; buff_c.store_dword(matC0[row+3], voffset, soffset)

        J.s_waitcnt(mod=f"vmcnt({0})")

    M,N = 32, 32
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(N, K, dtype=torch.float16)
    C = torch.randn(M, N, dtype=torch.float)
    ref = torch.nn.functional.linear(A, B).to(dtype=torch.float)
    kernel([1],[64], A.data_ptr(), B.data_ptr(), C.data_ptr(), K, M, K)
    torch.cuda.synchronize()
    if not torch.allclose(ref, C, atol=0.01, rtol=0.01):
        print(ref)
        print(C)
        assert 0

test_mfma_32x32x16(16)