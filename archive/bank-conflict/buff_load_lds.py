import pyhip

hip = pyhip.module("buff_load_lds.cpp")

import torch

torch.cuda.set_device(6)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")

'''
load LDS with BUFFER_LOAD_DWORD and load them for use to MFMA, avoid bank-conflict w/o relying swizzling 

to make things more clear, we make an imaginary matmul operation, in which A matrix is resident in register
and B matrix needs to be loaded 

"mkb,knb->mn"
'''
def test_buff_load_lds(kernel):

    waves_per_CU = 1
    thread_blocks = 1 * 1 * waves_per_CU
    
    OC_blocks = 1024*4
    A = torch.randn(        1, 16, 512, dtype=torch.float16)*0.01
    B = torch.randn(OC_blocks, 16, 512, dtype=torch.float16)*0.01
    ref = torch.einsum("bmk,bnk->mn",[A.to(torch.float), B.to(torch.float)])

    out = torch.zeros(16, 16, dtype=torch.float32)
    kernel([thread_blocks],[64], out.data_ptr(), A.data_ptr(), B.data_ptr(), OC_blocks, force_occupancy=waves_per_CU)

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    #print(out)
    #print(ref)

    total_flops = 16*512*16*OC_blocks * thread_blocks * 2
    print(f"=============== {kernel.__name__} Correctness:{torch.allclose(out, ref, rtol=0.01, atol=0.01)}")
    for i in range(3):
        torch.cuda._sleep(1_00_000_000)
        ev_start.record()    
        kernel([thread_blocks], [64], out.data_ptr(), A.data_ptr(), B.data_ptr(), OC_blocks, force_occupancy=waves_per_CU)
        ev_end.record()
        torch.cuda.synchronize()
        dt_ms = ev_start.elapsed_time(ev_end)
        print(f" {i:5}  {thread_blocks} thread-blocks   dt: {dt_ms*1e3:7.1f} us   {total_flops*1e-9/dt_ms: .1f} TFLOPS")

test_buff_load_lds(hip.fake_mm_base)
test_buff_load_lds(hip.fake_mm_lds)
test_buff_load_lds(hip.fake_mm_b128)
test_buff_load_lds(hip.fake_mm_bload_b128)
test_buff_load_lds(hip.fake_mm_bload_b128_group_padding)
