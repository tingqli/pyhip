import pyhip
import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

def test_debug_log():
    @pyhip.jit()
    def kernel(J, input:"__bf16*", count:"int", log_ptr:"int*"):
        s_i = J.gpr("su32")

        J.debug_setup(log_ptr, J.blockIdx.x[0] == 0)

        vdst = J.gpr("vu32")
        J.global_load_dword(vdst, (J.threadIdx.x[0] & 63)<<2, input)
        J.s_waitcnt(mod="vmcnt(0)")

        s_i[0] = 0
        with J.While(s_i[0] < count) as loop:
            J.debug_log(s_i, torch.int32)
            J.debug_log(vdst, torch.bfloat16)
            s_i[0] = s_i[0] + 1

    INPUT = torch.randn((64*2), dtype=torch.bfloat16)

    kernel([1],[64], INPUT.data_ptr(), 3, kernel.log_ptr())
    torch.cuda.synchronize()
    # will print all log items, and also returns a list of logs
    logs = kernel.get_logs()
    assert len(logs) == 2, f"{logs=}"
    s_i_count = 0
    vdst_count = 0
    for v in logs["s_i"]:
        assert len(v) == 1 and v[0] == s_i_count
        s_i_count += 1
    for v in logs["vdst"]:
        assert v.numel() == 128 and torch.allclose(v.reshape(INPUT.shape), INPUT.cpu())
        vdst_count += 1
    assert s_i_count == 3
    assert vdst_count == 3

if __name__ == "__main__":
    test_debug_log()