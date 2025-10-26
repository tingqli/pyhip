import torch
import pyhip

torch.cuda.set_device(3)
torch.set_default_device('cuda')

cur_gpu_device =torch.cuda.get_device_name()
print(f"{torch.get_default_device()=}")

hip = pyhip.module("test.cpp")

# since our kernel uses very little resources, we got Occupancy [waves/SIMD]: 8 
# thus we can observe thread-block are launched in batch of 640 (80 CUs * 8 blocks/CU)
# by looking at the 5'th column (walltime of start for each block)
# 
batch_size = 80+40

info = torch.zeros(batch_size, 8, dtype=torch.int64)

busy_time_us = 10
# we split batch_size as two factor to confirm the block launching sequence
hip.threadblock_test([batch_size],[256], busy_time_us*100, info.data_ptr(), force_occupancy=1)

min_time = info[:, 5].min().item()
info[:, 5] -= min_time

NUM_XCC = info[:, 0].max().item() + 1
NUM_SE = info[:, 1].max().item() + 1
NUM_CU = info[:, 2].max().item() + 1

time_filter = 0
while True:
    t0 = (time_filter)
    t1 = (time_filter + 500)

    summary = torch.full((NUM_XCC, NUM_SE, NUM_CU), -1, dtype=torch.int32)
    count = 0
    for b in range(batch_size):
        il = info[b,:].tolist()
        XCC,SE,CU,SIMD,TG,start,dt,_ = info[b,:].tolist()
        if start >= t0 and start < t1:
            #print(XCC, SE, CU)
            count += 1
            assert summary[XCC,SE,CU] == -1, "No CU is allocated more than 1 wave"
            summary[XCC,SE,CU] = b

    if count == 0: break

    time_filter += 1000
    print(f"============ {t0} ~ {t1}  {count} ===========")
    for xcc in range(NUM_XCC):
        print(f"XCC{xcc}:")
        for se in range(NUM_SE):
            item_ids = summary[xcc,se,:]
            valid_cu_mask = "".join(["1" if i >= 0 else "0" for i in item_ids])
            item_ids = item_ids[item_ids >= 0]
            items = item_ids.tolist()
            items.sort()
            
            print(f"\tSE{se} CU:{valid_cu_mask} : {items}")

print("============== test block's cache associativity ==============")
print(" if two waves accessing two different 4MB buffers were executing on same XCC,")
print(" they will compete for same L2-cache, thus much longer latency ")
print(" we can detect blocks sharing L2-caches using this way")
num_threads = 64*4*8
num_elements = 1*1024*1024  # 4MB float fits L2
d_sum = torch.zeros(num_threads, 4, dtype=torch.int32)

d_in1 = torch.ones(num_elements, dtype=torch.float)
d_in2 = torch.ones(num_elements, dtype=torch.float)

num_CU = 80
all_blocks = list(range(num_CU))
while len(all_blocks) > 0:
    block0 = all_blocks.pop(0)
    latencies = []
    for block1 in all_blocks:
        d_sum[:] = 0
        p = pyhip.cudaPerf(verbose=0)
        for i in range(2):
            with p:
                hip.compete_cache([num_CU], [64*4],
                    d_sum.data_ptr(),
                    d_in1.data_ptr(),
                    d_in2.data_ptr(),
                    num_elements, block0, block1, 0, force_occupancy=1)
        l = p.dt(excludes=1)
        latencies.append(l)
        if 0:
            loc0 = '.'.join([str(v) for v in d_sum[block0,:3].tolist()])
            loc1 = '.'.join([str(v) for v in d_sum[block1,:3].tolist()])
            print(f"blocks(XCC.SE.CU): {block0}({loc0}) & {block1}({loc1}): {l*1e6:.1f} us")
    
    # the slower one shares same L2 cache with block0
    block_group = [block0]
    avg = sum(latencies)/len(latencies)
    # print(avg)
    next_all_blocks = []
    dt1 = []
    dt0 = []
    for block1, dt in zip(all_blocks, latencies):
        if dt > avg:
            block_group.append(block1)
            dt1.append(dt)
        else:
            next_all_blocks.append(block1)
            dt0.append(dt)

    mean_dt1 = sum(dt1)/len(dt1)
    mean_dt0 = sum(dt0)/len(dt0)
    if (mean_dt1 - mean_dt0)/mean_dt0 < 0.1:
        block_group += next_all_blocks
        next_all_blocks = []

    print("block group: ", block_group)
    all_blocks = next_all_blocks

    
