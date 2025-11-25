import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
from contextlib import contextmanager



def test_simt():
    @pyhip.jit("(int*, int)")
    def kernel(J):
        kargs = J.new_gpr('s',[0,1])
        threadIdx_x = J.new_gpr('v',[0,0], dtype="i32")
        count = J.new_gpr('s', 1, dtype="i32")
        pA = J.new_gpr('s',2,align=2)
        J.s_load_dwordx2(pA, kargs, 0)
        J.s_load_dword(count, kargs, 8)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        # vdst,vaddr,saddr offset13s sc0 nt sc1
        vi = J.auto_gpr(threadIdx_x[0])
        vdata = J.new_gpr('v',1,dtype="i32", align=1)

        with J.While() as loop:
            with J.ExecMask(vi < count):
                voff = J.auto_gpr(vi << 2)
                J.global_load_dword(vdata, voff, pA)
                J.s_waitcnt(mod=f"vmcnt({0})")

                vdata[0] = vdata + 102

                # vaddr,vdata,saddr offset13s sc0 nt sc1
                J.global_store_dword(voff, vdata, pA)
            J.s_cbranch_scc0(mod=loop["end"])
            vi[0] = vi + 64

        J.s_waitcnt(mod=f"vmcnt({0})")

    TOTAL_CNT = 640
    A = torch.ones(TOTAL_CNT, dtype=torch.int)
    CNT = 318
    kernel([1],[64], A.data_ptr(), CNT)
    torch.cuda.synchronize()
    
    print(A)
    assert((A == 1).sum().item() == TOTAL_CNT - CNT)
    assert((A == 103).sum().item() == CNT)



def test_get_amax():
    @pyhip.jit("(int*, int)")
    def kernel(J):
        kargs = J.new_gpr('s',[0,1])
        threadIdx_x = J.new_gpr('v',[0,0], dtype="i32")
        count = J.new_gpr('s', 1, dtype="i32")
        pA = J.new_gpr('s',2,align=2)
        J.s_load_dwordx2(pA, kargs, 0)
        J.s_load_dword(count, kargs, 8)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        # vdst,vaddr,saddr offset13s sc0 nt sc1
        vi = J.auto_gpr(threadIdx_x[0])
        vdata = J.new_gpr('v',1,dtype="i32", align=1)
        vmax = J.new_gpr('v',1,dtype="f32", align=1)
        vmax[0] = torch.finfo(torch.float).min
        with J.While() as loop:
            with J.ExecMask(vi < count):
                voff = J.auto_gpr(vi << 2)
                J.global_load_dword(vdata, voff, pA)
                J.s_waitcnt(mod=f"vmcnt({0})")
                J.v_max_f32_e32(vmax, vmax, vdata)
            J.s_cbranch_scc0(mod=loop["end"])
            vi[0] = vi + 64
        vmax = J.reduce("v_max_f32", vmax)
        voff = J.auto_gpr(threadIdx_x << 2)
        J.global_store_dword(voff, vmax, pA)
        J.s_waitcnt(mod=f"vmcnt({0})")

    TOTAL_CNT = 200
    A = torch.randint(-100,100, (TOTAL_CNT,), dtype=torch.float)
    A[128] = 120
    A[129] = 125
    CNT = 129
    ref = torch.max(A[:CNT])
    kernel([1],[64], A.data_ptr(), CNT)
    torch.cuda.synchronize()
    assert torch.allclose(ref, A[:64])

if __name__ == "__main__":
    test_simt()
    test_get_amax()

