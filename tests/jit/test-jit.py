import pyhip

def kernel(J):
    p_kargs = J.new_gpr('s',[0,1])
    pA = J.new_gpr('s',2,align=4)
    K = J.new_gpr('s',1)

    J.s_load_dwordx2(pA, p_kargs, 0)

    acc = J.new_gpr("a", 4)
    for i in range(4):
        J.v_accvgpr_write_b32(acc[i], 0)

    J.s_load_dword(K, p_kargs, 8)
    J.s_waitcnt(mod=f"lgkmcnt({0})")
    #T.v_mov_b32(v[2], s[2])
    # v_mov_b32(v3, s3)
    #with J.BB():
    #    J.v_lshl_add_u32(v[2], v[0], 2, v[2])
    s_idx = J.new_gpr('s',1)
    s_temp = J.new_gpr('s',2)

    J.s_mov_b32(s_idx, 0)

    J.Label("bb0")

    #J.s_lshl_b32(s_temp[1],1, s_idx)

    J.s_lshl_b32(s_temp[0],s_idx,2)
    #s_temp[0] = s_idx[0] << 2

    J.s_add_u32(s_temp[0], pA[0], s_temp[0])
    J.s_addc_u32(s_temp[1], pA[1], 0)

    J.s_store_dword(K, s_temp, 0, mod="glc")

    J.s_addk_i32(s_idx, 1)
    J.s_cmp_lt_i32(s_idx, 32)
    J.s_cbranch_scc0(mod="bb1%=")
    J.s_branch(mod="bb0%=")

    J.Label("bb1")

    # flat_store_dword(v[2:3], v0)

jit = pyhip.JIT()
kernel(jit)
kernel = jit.build("test(int* p, int K)")

import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

A = torch.ones(64, dtype=torch.int)
print(A)
kernel([1],[64], A.data_ptr(), 64)
torch.cuda.synchronize()
print(A)