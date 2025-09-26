import torch
import pyhip

torch.cuda.set_device(6)
torch.set_default_device('cuda')

cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")


@pyhip.module("main.cpp")
def threadblock_test(gridDims, blockDims, busy_time, info): ...

# since our kernel uses very little resources, we got Occupancy [waves/SIMD]: 8 
# thus we can observe thread-block are launched in batch of 640 (80 CUs * 8 blocks/CU)
# by looking at the 5'th column (walltime of start for each block)
# 
batch_size = 80*10

info = torch.zeros(batch_size, 6, dtype=torch.int64)

busy_time_us = 1000
# we split batch_size as two factor to confirm the block launching sequence
threadblock_test([batch_size//10, 10],[256], busy_time_us*100, info.data_ptr())

min_time = info[:, 4].min().item()
info[:, 4] -= min_time

for b in range(batch_size):
    print(b, info[b,:].tolist())
