import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
'''
    execmask 可以用来屏蔽内存访问
'''
def test_execmask_skip_memload():
    @pyhip.jit(with_debug_log=True)
    def kernel(J, p_selectors:"int*"):
        vaddr=J.gpr("vu32")
        vdst=J.gpr("vu32")
        mask = J.gpr(2, "su32")

        # J.v_mul_u32_u24(vdst, vdst, 0x100)
        # 构造非法地址
        J.s_mov_b32(p_selectors[0], 0)
        J.s_mov_b32(p_selectors[1], 0)

        J.s_mov_b32(mask[0], 0)
        J.s_mov_b32(mask[1], 0)
        J.s_mov_b64("exec", mask)
        # 非法地址的内存访问完全可以被execmask屏蔽
        J.global_load_dword(vaddr, J.threadIdx.x[0]*4, p_selectors)
        J.s_mov_b64("exec", -1)

        J.s_waitcnt(mod=f"vmcnt(0)")
        vdst[0] = 0

        J.ds_bpermute_b32(vdst, vaddr, J.threadIdx.x[0], mod=f"offset:{0}")
        J.debug_setup(J.blockIdx.x[0] == 0)
        J.debug_log(vdst, torch.int32)

    A = torch.arange(0,64*4,4, dtype=torch.int)
    A[1] = 4
    A[2] = 8
    A[3] = 4
    print(A)
    artifacts = kernel([1],[64], A.data_ptr())
    torch.cuda.synchronize()

def test_execmask_from_vcc():
    @pyhip.jit(with_debug_log=True)
    def kernel(J, p_selectors:"int*"):
        vaddr=J.gpr("vu32")
        vdst=J.gpr("vu32")

        J.global_load_dword(vaddr, J.threadIdx.x[0]*4, p_selectors)

        J.s_waitcnt(mod=f"vmcnt(0)")
        vdst[0] = 0

        J.SetMask("exec", J.threadIdx.x[0] >= 32)
        J.ds_bpermute_b32(vdst, vaddr, J.threadIdx.x[0], mod=f"offset:{4*8}")
        J.s_mov_b64("exec", -1)

        J.debug_setup(J.blockIdx.x[0] == 0)
        J.debug_log(vdst, torch.int32)

    A = torch.arange(0,64*4,4, dtype=torch.int)
    A[1] = 4
    A[2] = 8
    A[3] = 4
    print(A)
    kernel([1],[64], A.data_ptr())
    torch.cuda.synchronize()


if __name__ == "__main__":
    test_execmask_skip_memload()
    test_execmask_from_vcc()