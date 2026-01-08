import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_if():
    @pyhip.jit()
    def kernel(J, s_pout:"int*"):
        s_i = J.gpr('si32')
        s_cnt = J.gpr('si32')
        s_i[0] = 0
        s_cnt[0] = 0
        with J.While(s_i[0] < 32) as loop:
            with J.If((s_i[0] & 1)==0):
                s_cnt[0] = s_cnt + s_i
            s_i[0] = s_i[0] + 1

        J.s_store_dword(s_cnt, s_pout, 0, mod="glc")
        J.s_waitcnt(mod=f"lgkmcnt({0})")

    OUT = torch.arange(0,32, dtype=torch.int)
    kernel([1],[64], OUT.data_ptr())
    torch.cuda.synchronize()
    assert OUT[0] == sum([i if (i & 1) == 0 else 0 for i in range(32)]), f"{OUT=}"

def test_loop():
    @pyhip.jit()
    def kernel(J, s_pout:"int*"):
        s_i = J.new_gpr('s', 1, dtype="i32", name="s_i")
        s_cnt = J.new_gpr('s', 1, dtype="i32", name="s_cnt")

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
    kernel([1],[64], OUT.data_ptr())
    torch.cuda.synchronize()
    assert OUT[0] == 21, f"{OUT=}"

if __name__ == "__main__":
    test_if()
    test_loop()