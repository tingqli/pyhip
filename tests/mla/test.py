
import pyhip
@pyhip.module("mla.cpp")
def gemm_qk_lds(q, k, P, kv_len, sm): ...

import torch
torch.cuda.set_device(6)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")

def check_all_close(out, out_ref, rtol=0.01, atol=0.01, verbose=False):
    if not torch.allclose(out, out_ref, rtol=0.01, atol=0.01):
        if verbose:
            print(out)
            print(out_ref)
        torch.testing.assert_close(out, out_ref, rtol=0.01, atol=0.01)
        # torch.testing.assert_close(out, out_ref)
    else:
        print("PASS")

def test_gemm_qk():

    #kernel = gemm_qk
    kernel = gemm_qk_lds
    '''
    (batch, heads, dim) (batch, kv_len, dim) => (batch, heads, kv_len)
    '''
    batch = 128
    heads = 128
    dim = 512
    kv_len = 16*1024

    q = torch.randn(batch, heads, dim, dtype=torch.float16)
    k = torch.randn(batch, kv_len, dim, dtype=torch.float16)
    out = torch.randn(batch, heads, 16, dtype=torch.float)

    kernel([heads//64, batch], [4*64], q.data_ptr(), k.data_ptr(), out.data_ptr(), kv_len, 1)

    temp = torch.einsum("bhd,bkd->bhk",[q.to(torch.float), k.to(torch.float)])
    out_ref = torch.einsum("bhkl->bhl",[temp.reshape(batch, heads, kv_len//16, 16)])

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for i in range(3):
        torch.cuda._sleep(1_000_000_000)
        ev_start.record()
        kernel([heads//64, batch], [4*64], q.data_ptr(), k.data_ptr(), out.data_ptr(), kv_len, 1)
        ev_end.record()
        torch.cuda.synchronize()
        dt_ms = ev_start.elapsed_time(ev_end)/1
        flops = batch*heads*kv_len*dim*2
        rd_bytes = batch*kv_len*dim*2
        print(f"dt = {dt_ms*1e3:.3f} us {flops*1e-9/dt_ms:.1f} TFLOPS  {rd_bytes*1e-6/dt_ms:.1f} GB/s per-layer  {batch=} {heads=} {kv_len=} {dim=}")

    check_all_close(out, out_ref)

test_gemm_qk()


'''
def test_swizzle():
    cols = 512//4
    lane_map = torch.zeros(16, cols, 2, dtype=torch.int32, device="cpu")
    for row in range(16):
        for col in range(cols):
            phase = col % 16
            row_swizzled = row ^ phase
            print(row, col, phase, row_swizzled)
            lane_map[row_swizzled, col, 0] = row
            lane_map[row_swizzled, col, 1] = col
    
    for row in range(16):
        for col in range(cols):
            r = lane_map[row, col, 0].item()
            c = lane_map[row, col, 1].item()
            print(f"{r:2} ", end="")
        print()

#test_swizzle()
'''