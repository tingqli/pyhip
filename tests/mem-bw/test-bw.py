import torch
import pyhip
import sys

torch.cuda.set_device(5)
torch.set_default_device('cuda')
torch.manual_seed(0)
cur_gpu_device =torch.cuda.get_device_name()
num_CU = torch.cuda.get_device_properties().multi_processor_count
print(f"{torch.get_default_device()=} with {num_CU=}")

WIDTH = 256 * 16 * 32 * 1 * 8
HEIGHT = num_CU * 4 * 64 * 2
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

force_occupancy = 0
if 1:
    myread_4m_4kx16mx16k = hip.myread_4m_4kx16mx16k

    myread_4m_4kx16mx16k([HEIGHT//64],[256], A.data_ptr(), B.data_ptr())
    #print(B)

    for i in range(8):
        with pyhip.cudaPerf(0, (WIDTH * HEIGHT), name="read(wave:fma16x16(4kx16mx16k))"):
            myread_4m_4kx16mx16k([HEIGHT//64],[256], A.data_ptr(), B.data_ptr(), force_occupancy=force_occupancy)

if 1:
    myread_4m_1mx64kx16k = hip.myread_4m_1mx64kx16k

    myread_4m_1mx64kx16k([HEIGHT//4],[256], A.data_ptr(), B.data_ptr())
    #print(B)

    for i in range(8):
        with pyhip.cudaPerf(0, (WIDTH * HEIGHT), name="read(wave:seq(1mx64kx16k))"):
            myread_4m_1mx64kx16k([HEIGHT//4],[256], A.data_ptr(), B.data_ptr(), force_occupancy=force_occupancy)
            
if 1:
    myread_4m_4mx16kx16k = hip.myread_4m_4mx16kx16k

    myread_4m_4mx16kx16k([HEIGHT//16],[256], A.data_ptr(), B.data_ptr())
    #print(B)

    for i in range(8):
        with pyhip.cudaPerf(0, (WIDTH * HEIGHT), name="read(wave:len(K)==256,read once(4mx16kx16k))"):
            myread_4m_4mx16kx16k([HEIGHT//16],[256], A.data_ptr(), B.data_ptr(), force_occupancy=force_occupancy)

if 1:
    myread_4m_8mx8kx16k = hip.myread_4m_8mx8kx16k

    myread_4m_8mx8kx16k([HEIGHT//32],[256], A.data_ptr(), B.data_ptr())
    #print(B)

    for i in range(8):
        with pyhip.cudaPerf(0, (WIDTH * HEIGHT), name="read(wave:len(K)==256,read twice(8mx8kx16k))"):
            myread_4m_8mx8kx16k([HEIGHT//32],[256], A.data_ptr(), B.data_ptr(), force_occupancy=force_occupancy)

if 1:
    myread_4m_16mx4kx16k = hip.myread_4m_16mx4kx16k

    myread_4m_16mx4kx16k([HEIGHT//64],[256], A.data_ptr(), B.data_ptr())
    #print(B)

    for i in range(8):
        with pyhip.cudaPerf(0, (WIDTH * HEIGHT), name="read(wave:len(K)==256,read four times(16mx4kx16k))"):
            myread_4m_16mx4kx16k([HEIGHT//64],[256], A.data_ptr(), B.data_ptr(), force_occupancy=force_occupancy)            