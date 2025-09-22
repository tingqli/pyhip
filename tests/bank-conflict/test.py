import torch
import pyhip

torch.cuda.set_device(6)
torch.set_default_device('cuda')

cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")


@pyhip.kernel("dsread_test.cpp")
def dsread_test(data, off, sm): ...

@pyhip.kernel("dsread_test.cpp")
def dsread_testx4(data, off, sm): ...

def test_lds_bank_size():
    q = torch.ones(64//4, 1024, dtype=torch.uint32)
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    idx_base = torch.arange(0, 64, dtype=torch.int32, device="cpu")
    for i in [0,1,2,3,4,5,8,9,16,17,32,33,64,65,128,129,256,512,1024]:
        idx = (idx_base * i).cuda()
        torch.cuda._sleep(1_00_000_000)
        ev_start.record()    
        dsread_test([1], [16*64], q.data_ptr(), idx.data_ptr(), 0)
        ev_end.record()
        torch.cuda.synchronize()
        dt_ms = ev_start.elapsed_time(ev_end)
        print(f"stride_dword:{i:5}   dt: {dt_ms*1e3:7.1f} us")

def test_lds_bank_conflict(lane_src):
    q = torch.ones(64//4, 1024, dtype=torch.uint32)
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    print(f"======= lane_src: {lane_src} ========")
    for lane_dst in range(64):
        idx_base = torch.arange(0, 64, dtype=torch.int32, device="cpu")
        if lane_dst != lane_src:
            idx_base[lane_dst] = idx_base[lane_src] + 128
        idx = idx_base.cuda()

        torch.cuda._sleep(1_00_000_000)
        ev_start.record()    
        dsread_test([1], [16*64], q.data_ptr(), idx.data_ptr(), 0)
        ev_end.record()
        torch.cuda.synchronize()
        dt_ms = ev_start.elapsed_time(ev_end)
        print(f"\tlane: {lane_dst:5}   dt: {dt_ms*1e3:7.1f} us")

test_lds_bank_size()
test_lds_bank_conflict(0)

