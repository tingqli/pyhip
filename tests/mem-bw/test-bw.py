import torch
import pyhip
import sys

torch.cuda.set_device(6)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
num_CU = torch.cuda.get_device_properties().multi_processor_count
print(f"{torch.get_default_device()=} with {num_CU=}")

WIDTH = 256 * 16 * 32 * 1 * 8
HEIGHT = num_CU * 4 * 80
hip = pyhip.module("bw.cpp", f"-D {WIDTH=}")

SIZE = WIDTH * HEIGHT
if SIZE >= 1024 * 1024 * 1024:
    buf_size_str = f'{SIZE / 1024 / 1024 / 1024:,} GB'
elif SIZE >= 1024 * 1024:
    buf_size_str = f'{SIZE / 1024 / 1024:,} MB'
else:
    buf_size_str = f'{SIZE / 1024:,} KB'
print(f'buffer size = {buf_size_str} with {WIDTH=:,} {HEIGHT=:,}')
A = torch.ones([HEIGHT, WIDTH//4], dtype=torch.int32)
B = torch.zeros([256], dtype=torch.int32)

myread = hip.myread

myread([HEIGHT],[256], A.data_ptr(), B.data_ptr())
#print(B)

for i in range(8):
    with pyhip.cudaPerf(0, (WIDTH * HEIGHT), name="read"):
        myread([HEIGHT],[256], A.data_ptr(), B.data_ptr())
