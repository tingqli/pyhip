import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)


def test_loop():
    @pyhip.jit("(int*, int)")
    def kernel(J):
        s_pkargs = J.new_gpr('s',[0,1],name="s_pkargs")
        threadIdx_x = J.new_gpr('v',[0,0],name="threadIdx_x")
        s_pout = J.new_gpr('s', 2, align=2, name="s_pout")
        s_i = J.new_gpr('s', 1, dtype="i32", name="s_i")
        s_cnt = J.new_gpr('s', 1, dtype="i32", name="s_cnt")

        J.s_load_dwordx2(s_pout, s_pkargs, 0)
        J.s_waitcnt(mod=f"lgkmcnt({0})")

        s_i[0] = 0
        s_cnt[0] = 0
        with J.While(s_i[0] < 32) as loop:
            s_i[0] = s_i[0] + 1
            J.Jump(loop["begin"], s_i[0]==10) # continue
            s_cnt[0] = s_cnt[0] + 1
            J.Jump(loop["end"], s_i[0]==22) # break

        J.s_store_dword(s_cnt, s_pout, 0, mod="glc")

        J.s_waitcnt(mod=f"lgkmcnt({0})")

    OUT = torch.ones(32, dtype=torch.int)
    a = 123
    b = 3
    kernel([1],[64], OUT.data_ptr(), a, b)
    torch.cuda.synchronize()
    assert OUT[0] == 21, f"{OUT=}"

test_loop()