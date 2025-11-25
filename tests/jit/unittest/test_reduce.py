import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_reduce():
    @pyhip.jit("(int*, int)")
    def kernel(J):
        kargs = J.new_gpr('s',[0,1])
        threadIdx_x = J.new_gpr('v',[0,0], dtype="i32")
        count = J.new_gpr('s', 1, dtype="i32")
        pA = J.new_gpr('s', 2, align=2)
        J.s_load_dwordx2(pA, kargs, 0)
        J.s_load_dword(count, kargs, 8)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        v1 = J.new_gpr("v", 1)
        voff = J.auto_gpr(threadIdx_x << 2)
        J.global_load_dword(v1, voff, pA)
        J.s_waitcnt(mod=f"vmcnt({0})")

        vmax = J.reduce("v_max_f32", v1)
        vsum = J.reduce("v_add_f32", v1)

        J.global_store_dword(voff, vmax, pA)
        J.global_store_dword(voff, vsum, pA, mod=f"offset:{64*4}")
        J.s_waitcnt(mod=f"vmcnt({0})")

    TOTAL_CNT = 64
    A = torch.randint(0, 1000, (2,TOTAL_CNT), dtype=torch.float)
    maxv = torch.max(A[0,:])
    sumv = torch.sum(A[0,:]).to(dtype=torch.float)
    #print(A[0,:])
    #print(maxv, sumv)
    kernel([1],[64], A.data_ptr(), 64)
    torch.cuda.synchronize()
    #print(A)
    assert torch.allclose(A[0,:], maxv)
    assert torch.allclose(A[1,:], sumv)

if __name__ == "__main__":
    test_reduce()