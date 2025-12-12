import pyhip
import torch
import pytest

torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

@pytest.mark.parametrize("M, N, dtype",[
    (5, 3, torch.float),
    (2, 2, torch.bfloat16),
    (4, 4, torch.bfloat16),
    (8, 4, torch.bfloat16),
    (4, 8, torch.bfloat16),
    (4, 4, torch.float8_e5m2),
    (4, 8, torch.float8_e5m2),
    (8, 4, torch.float8_e5m2),
])
def test_trans(M, N, dtype):
    dtype_bytes = dtype.itemsize
    @pyhip.jit()
    def kernel(J, pA:"void*", pB:"void*"):
        lane_id = J.gpr(J.threadIdx.x[0] % 64)

        # each lane has [M, N, dtype_bytes]
        buf_in = J.Buffer(pA, 64*M*N*dtype_bytes)
        buf_out = J.Buffer(pB, 64*M*N*dtype_bytes)

        num_dwords_per_row = N*dtype_bytes//4
        vinput = J.gpr(M, num_dwords_per_row, "vu32")
        voutput = J.gpr(M, num_dwords_per_row, "vu32")

        """
        加载数据令人头疼：每个lane有自己的地址，lane内部dword多次加载也可以做到有来自不同地址的数据
        """
        
        for m in range(M):
            for n in range(num_dwords_per_row):
                voffset = J.gpr(lane_id * (M*N*dtype_bytes) + m*N*dtype_bytes + n*4)
                buf_in.load_dword(vinput[m, n], voffset, 0, offset12=0)

        J.s_waitcnt(mod="vmcnt(0)")
        J.transpose_per_lane(M, N, dtype_bytes, vinput[...], voutput[...])

        for m in range(M):
            for n in range(num_dwords_per_row):
                voffset = J.gpr(lane_id * (M*N*dtype_bytes) + m*N*dtype_bytes + n*4)
                buf_out.store_dword(voutput[m, n], voffset, 0, offset12=0)
        J.s_waitcnt(mod="vmcnt(0)")
    
    inp = torch.randn(64, M, N).to(dtype=dtype)
    outp = torch.randn(64, N, M).to(dtype=dtype)
    kernel([1], [64], inp.data_ptr(), outp.data_ptr())
    for i in range(64):
        if not torch.allclose(outp[i].to(dtype=torch.float), inp[i].t().to(dtype=torch.float)):
            print(f"inp[{i}]")
            print(inp[i])
            print(f"inp[{i}].t()")
            print(inp[i].t())
            print(f"outp[{i}]")
            print(outp[i])
            assert 0

if __name__ == "__main__":
    #test_trans(5, 3, torch.float)
    #test_trans(2, 2, torch.bfloat16)
    #test_trans(4, 4, torch.bfloat16)
    #test_trans(8, 4, torch.float16)
    #test_trans(2, 8, torch.float16)
    test_trans(4, 4, torch.float8_e5m2)
    test_trans(4, 8, torch.float8_e5m2)
    