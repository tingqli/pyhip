import pyhip


hip_module = pyhip.kernel("main.cpp")

@hip_module
def sum_1d_x4(gridDims, blockDims, count, data_f32, numElements, save_sum): ...
@hip_module
def sum_1d_x2(gridDims, blockDims, count, data_f32, numElements, save_sum): ...
@hip_module
def sum_1d_16x64(gridDims, blockDims, count, data_f32, numElements, save_sum): ...
@hip_module
def sum_1d_8x128(gridDims, blockDims, count, data_f32, numElements, save_sum): ...
@hip_module
def sum_1d_4x256(gridDims, blockDims, count, data_f32, numElements, save_sum): ...
@hip_module
def sum_1d_2x512(gridDims, blockDims, count, data_f32, numElements, save_sum): ...

import torch
torch.cuda.set_device(7)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")

def test_count_negatives(kfunc):
    act_sum = torch.tensor(0, dtype=torch.float32)

    numElements = 640*1024*1024
    num_CUs = 80
    Occupancy = 8
    grid_size = num_CUs * Occupancy
    # test correctness
    input = torch.randn(16, numElements//16, dtype=torch.float)*0.01

    # input[:, (640*4*4*4+16):] = 0

    ref_sum = input.sum()

    kfunc([grid_size], [256], act_sum.data_ptr(), input.data_ptr(), numElements, 1)
    torch.cuda.synchronize()

    correctness = "PASS" if (act_sum - ref_sum).abs() < 0.01 else f"FAILED"

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    # test performance, no negative exists
    input = -input.abs()
    print(f"================ {correctness=} ({act_sum.item()} vs {ref_sum.item()})")
    for i in range(3):
        torch.cuda._sleep(1_00_000_000)
        ev_start.record()    
        kfunc([grid_size], [256], act_sum.data_ptr(), input.data_ptr(), numElements, 0)
        ev_end.record()
        torch.cuda.synchronize()
        dt_ms = ev_start.elapsed_time(ev_end)
        print(f" {kfunc.__name__}  dt: {dt_ms*1e3:7.1f} us  {numElements*4e-6/dt_ms:.3f} GB/s")


test_count_negatives(sum_1d_x4)
test_count_negatives(sum_1d_x2)
test_count_negatives(sum_1d_16x64)
test_count_negatives(sum_1d_8x128)
test_count_negatives(sum_1d_4x256)
test_count_negatives(sum_1d_2x512)
