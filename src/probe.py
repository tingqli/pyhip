import os

os.environ["PYHIP_JIT_LOG"] = "0"
os.environ["PYHIP_DEBUG_LOG"] = ""

import pyhip
import torch

torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
torch.set_default_device('cuda')
torch.manual_seed(0)
num_CU = torch.cuda.get_device_properties().multi_processor_count

@pyhip.jit(no_pass = ["pass_dse","pass_dce"])
def probe(J, ds_instruction, lane_bytes, offsets:"int*", p_cycles:"void*"):
    lds = J.alloc_lds(32*1024)
    vaddr = J.gpr("vu32", 0)

    J.global_load_dword(vaddr, J.threadIdx.x * J.sizeof_DW, offsets)
    J.s_waitcnt(mod=f"vmcnt(0)")

    vaddr[0] += lds

    lane_dw = J.div(lane_bytes, J.sizeof_DW)
    vdata = J.gpr(64, lane_dw, "vu32")

    mtime_start = J.gpr(2, "su32")
    J.s_memtime(mtime_start)

    k = J.gpr("su32", 0)
    with J.While(k[0] < 100):
        for i in range(64):
            if ds_instruction.startswith("ds_read"):
                getattr(J, ds_instruction)(vdata[i], vaddr)
            else:
                assert ds_instruction.startswith("ds_write")
                getattr(J, ds_instruction)(vaddr, vdata[i])
        k[0] += 1
        J.s_waitcnt(mod=f"lgkmcnt({0})")

    mtime_stop = J.gpr(2, "su32")
    J.s_memtime(mtime_stop)
    J.s_waitcnt(mod=f"lgkmcnt({0})")

    J.s_sub_u32(mtime_stop[0], mtime_stop[0], mtime_start[0])
    J.s_subb_u32(mtime_stop[1], mtime_stop[1], mtime_start[1])

    J.s_store_dwordx2(mtime_stop, p_cycles, 0, mod="glc")

def probe_lds_banks():
    print(" ======== LDS bank count probe using ds_read_b128 ======== ")
    cycles = torch.zeros(1, dtype=torch.uint64)
    lds_banks = 8
    # we need to use ds_read_b128 to probe since each lane-group can cover widest range
    ds_instruction = "ds_read_b128"
    lane_bytes = 16
    for i in range(6):
        lds_banks *= 2
        conflict_stride = lds_banks * 4
        voffsets = torch.arange(0,256, dtype=torch.int32)
        voffsets *= conflict_stride
        probe([1],[64], ds_instruction, lane_bytes, voffsets.data_ptr(), cycles.data_ptr())
        print(f"{lds_banks=:4} : {cycles.item()} cycles")

def probe_lanegroups(ds_instruction = "ds_read_b128"):

    cycles = torch.zeros(1, dtype=torch.uint64)

    lds_banks = 32 if pyhip.JIT.cdna < 4 else 64
    if ds_instruction.endswith("_b32"): lane_bytes = 4
    elif ds_instruction.endswith("_b64"): lane_bytes = 8
    elif ds_instruction.endswith("_b128"): lane_bytes = 16
    else: assert 0, f"{ds_instruction} not supported"
    warp_size = 64
    print(" ======== lange-groups of ", ds_instruction, " ======== ")
    probe_lanes = list(range(warp_size))
    while len(probe_lanes):
        src = probe_lanes[0]
        all_cycles = []
        for dst in range(warp_size):
            voffsets = torch.zeros(256, dtype=torch.uint32)
            voffsets[src] = lane_bytes
            voffsets[dst] = lane_bytes + (lds_banks * 4)
            probe([1],[256], ds_instruction, lane_bytes, voffsets.data_ptr(), cycles.data_ptr())
            all_cycles.append(cycles.item())
            #print(src, dst, cycles.item())
        
        avg_cycles = sum(all_cycles)/len(all_cycles)
        all_cycles[src] = avg_cycles * 2

        lane_group = [i for i,v in enumerate(all_cycles) if v > avg_cycles]
        print("\t", lane_group)

        for lane in lane_group:
            if lane in probe_lanes:
                probe_lanes.remove(lane)

@pyhip.jit(no_pass = ["pass_dse","pass_dce"])
def dummy_load(J, bytes_per_WG, access_type, dtype, data:"int*", p_cycles:"void*"):
    # work-group/thread-block cooperately loads data (but not use them at all)
    ele_size = J.sizeof(dtype)
    num_ele = bytes_per_WG // ele_size
    dw_count = J.div(ele_size, 4)

    data[:] += J.blockIdx.x[0] * bytes_per_WG

    num_threads = 256

    num_data_vgprs = 64

    def issue_inst(vdata, vaddr):
        if access_type == "load":
            getattr(J, f"global_load_{dtype}")(vdata, vaddr, data)
        elif access_type == "store":
            getattr(J, f"global_store_{dtype}")(vaddr, vdata, data)
        else:
            getattr(J, f"global_{access_type}")(vaddr, vdata, data)

    vaddr = J.gpr(J.threadIdx.x[0] * ele_size)
    if dw_count > 1:
        vdata = J.gpr(num_data_vgprs, dw_count, "abf16x2",0)
    else:
        vdata = J.gpr(num_data_vgprs, dw_count, "vbf16x2",0)
    for i in range(num_data_vgprs):
        issue_inst(vdata[i], vaddr)
        vaddr[0] += num_threads * ele_size

    access_bytes = J.gpr("su32", num_data_vgprs*num_threads * ele_size)

    with J.While(access_bytes[0] < bytes_per_WG):
        J.s_waitcnt(mod=f"vmcnt({num_data_vgprs//2})")
        for i in range(num_data_vgprs//2):
            issue_inst(vdata[i], vaddr)
            vaddr[0] += num_threads * ele_size

        J.s_waitcnt(mod=f"vmcnt({num_data_vgprs//2})")
        for i in range(num_data_vgprs//2):
            issue_inst(vdata[(num_data_vgprs//2)+i], vaddr)
            vaddr[0] += num_threads * ele_size

        access_bytes[0] += num_data_vgprs*num_threads * ele_size

    J.s_waitcnt(mod=f"vmcnt(0)")


def probe_vmem_bandwidth():
    # num_CU
    print(" ======== load bandwidth ======== ")
    bytes_per_WG = 5*1024*1024
    num_rounds = 32
    cycles = torch.zeros(num_CU, dtype=torch.uint64)
    data = torch.zeros([num_rounds, num_CU, bytes_per_WG//4], dtype=torch.int32)
    dtype = "dword"

    for r in range(10):
        with pyhip.cudaPerf(rw_bytes=bytes_per_WG*num_CU, name=f"load-{dtype}-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "load", dtype, data[r].data_ptr(), cycles.data_ptr())

    for r in range(10):
        with pyhip.cudaPerf(rw_bytes=bytes_per_WG*num_CU, name=f"load-dwordx2-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "load", "dwordx2", data[r].data_ptr(), cycles.data_ptr())

    for r in range(10):
        with pyhip.cudaPerf(rw_bytes=bytes_per_WG*num_CU, name=f"load-dwordx4-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "load", "dwordx4", data[r].data_ptr(), cycles.data_ptr())

    for r in range(10):
        with pyhip.cudaPerf(rw_bytes=bytes_per_WG*num_CU, name=f"store-{dtype}-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "store", dtype, data[r].data_ptr(), cycles.data_ptr())

    for r in range(10):
        with pyhip.cudaPerf(rw_bytes=bytes_per_WG*num_CU, name=f"atomic_pk_add_bf16-{dtype}-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "atomic_pk_add_bf16", "dword", data[r].data_ptr(), cycles.data_ptr())

    for r in range(10):
        with pyhip.cudaPerf(rw_bytes=bytes_per_WG*num_CU, name=f"global_atomic_xor-{dtype}-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "atomic_xor", "dword", data[r].data_ptr(), cycles.data_ptr())

@pyhip.jit(no_pass = ["pass_dse","pass_dce"])
def load_time(J, num_threads, data:"int*", K:"int", p_cycles:"void*"):
    data[:] += J.blockIdx.x[0] * K * J.sizeof_u32

    cycles_offset = 0
    def measure_memtime(f, repeat_cnt, *args):
        nonlocal cycles_offset
        avg = J.gpr("su32", 0)
        k = J.gpr("su32", 0)
        with J.While(k[0] < repeat_cnt):
            mtime_start = J.gpr(2, "su32")
            J.s_memtime(mtime_start)
            f(*args)
            mtime_stop = J.gpr(2, "su32")
            J.s_memtime(mtime_stop)
            J.s_waitcnt(mod=f"lgkmcnt({0})")

            J.s_sub_u32(mtime_stop[0], mtime_stop[0], mtime_start[0])
            J.s_subb_u32(mtime_stop[1], mtime_stop[1], mtime_start[1])
            avg[0] += mtime_stop[0]
            k[0] += 1

        avg[0] = avg[0] // repeat_cnt
        J.s_store_dword(avg[0], p_cycles, cycles_offset, mod=f"glc")
        cycles_offset += J.sizeof_DW

    buffer = J.Buffer(data, K * J.sizeof_u32)
    voffset = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_DW4)
    vdata = J.gpr(128, 4, "vu32")
    def empty(): pass
    def buffer_load_dwordx4(repeat_cnt):
        for i in range(repeat_cnt):
            buffer.load_dwordx4(vdata[i%128], voffset, 0)
            voffset[0] += num_threads*J.sizeof_DW4
        J.s_waitcnt(mod="vmcnt(0)")

    measure_memtime(empty, 1)
    measure_memtime(buffer_load_dwordx4, 1, 1)

    # reset voffset so cache hit happens
    voffset = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_DW4)
    measure_memtime(buffer_load_dwordx4, 1, 1)

    measure_memtime(buffer_load_dwordx4, 256, 1)
    measure_memtime(buffer_load_dwordx4, 256, 16)

    lds = J.alloc_lds(16*1024)
    voffset = J.gpr("vu32", J.threadIdx.x[0] * J.sizeof_DW4 + lds)
    lds_off = 0
    def lds_read(repeat_cnt):
        nonlocal lds_off
        for i in range(repeat_cnt):
            J.ds_read_b128(vdata[i%128], voffset, mod=f"offset:{lds_off % (16*1024)}")
            lds_off += num_threads*J.sizeof_DW4

    measure_memtime(lds_read, 256, 1)
    measure_memtime(lds_read, 256, 16)

        
def probe_cycles(num_threads = 256):
    # passing 4GB buffer into kernel has problem, all reads from kernel got 0
    # but <4GB has no such problem
    per_CU_K = (1024//num_CU)*1024*1024

    K=num_CU * per_CU_K
    if K >= 1024*1024*1024: K-= 1

    # each CU has it's own HBM data to test with
    A = torch.randint(0, per_CU_K-4096, (per_CU_K,), dtype=torch.int32)
    A = A.repeat(num_CU, 1) # num_CU, per_CU_K
    cycles = torch.zeros(1024, dtype=torch.int32)
    load_time([num_CU], [num_threads], num_threads, A.data_ptr(), per_CU_K, cycles.data_ptr())
    print(f"{'s_memtime overhead cycles':>60s} : {cycles[0]}")
    print(f"{'first buffer_load_dwordx4 cycles':>60s} : {cycles[1]}")
    print(f"{'cache-hit buffer_load_dwordx4 cycles':>60s} : {cycles[2]}")
    print(f"{'256 buffer_load_dwordx4_x1 avg cycles':>60s} : {cycles[3]}")
    print(f"{'256 buffer_load_dwordx4_x16 avg cycles':>60s} : {cycles[4]}")
    tput_cycles = (cycles[4]-cycles[3])//15
    freq = 2e9
    est_bw = num_CU * num_threads * 16 / (tput_cycles / freq)
    print(f"{'         derived buffer_load_dwordx4 tput cycles':>60s} : {tput_cycles}")
    print(f"{'         derived vmem bandwidth':>60s} : {est_bw*1e-9:.2f} GB/s    @freq{freq*1e-9:.1f}GHz")
    print(f"{'256 ds_read_b128_x1 avg cycles':>60s} : {cycles[5]}")
    print(f"{'256 ds_read_b128_x16 avg cycles':>60s} : {cycles[6]}")
    print(f"{'         derived ds_read_b128 tput cycles':>60s} : {(cycles[6]-cycles[5])//15}")

@pyhip.jit(no_pass = ["pass_dse","pass_dce"])
def compete_cache(J, block0:"int", block1:"int", data0:"void*", data1:"void*", K:"int", info:"void*"):
    # allocat all LDS so occupancy is 1
    lds = J.alloc_lds(J.lds_size_limit)
    num_threads = 256

    # only allow 2 blocks to access the memory
    with J.If((J.blockIdx.x[0] != block0) & (J.blockIdx.x[0] != block1)):
        J.s_endpgm()
    
    # compete cache with 2 big buffer access
    buffer = J.Buffer(data0, K * J.sizeof_u32)
    with J.If(J.blockIdx.x[0] == block1):
        buffer.setup(data1, K * J.sizeof_u32)
    
    vdata = J.gpr(2, 4, "vu32")
    voffset = J.threadIdx.x[0] * J.sizeof_DW4
    k = J.gpr("su32", 0)
    with J.While(k[0] < 4):
        i = J.gpr("su32", 0)
        buffer.load_dwordx4(vdata[0], voffset, i[0])
        i[0] += num_threads*J.sizeof_DW4

        with J.While(i[0] < K-2*num_threads*J.sizeof_DW4):
            buffer.load_dwordx4(vdata[1], voffset, i[0])
            i[0] += num_threads*J.sizeof_DW4
            J.s_waitcnt(mod="vmcnt(1)")

            buffer.load_dwordx4(vdata[0], voffset, i[0])
            i[0] += num_threads*J.sizeof_DW4
            J.s_waitcnt(mod="vmcnt(1)")

        J.s_waitcnt(mod="vmcnt(0)")
        k[0] += 1

    with J.If(J.warp_id[0] == 0):
        xcc_id = J.gpr("su32")
        cu_id = J.gpr("su32")
        se_id = J.gpr("su32")
        simd_id = J.gpr("su32")
        J.s_getreg_b32(simd_id, mod="hwreg(HW_REG_HW_ID, 4, 2)")
        J.s_getreg_b32(cu_id, mod="hwreg(HW_REG_HW_ID, 8, 4)")
        J.s_getreg_b32(se_id, mod="hwreg(HW_REG_HW_ID, 13, 3)")
        J.s_getreg_b32(xcc_id, mod="hwreg(HW_REG_XCC_ID, 0, 4)")

        info[:] += J.blockIdx.x[0] * J.sizeof_DW
        J.s_store_dword(simd_id, info, J.sizeof_DW*0, mod=f"glc")
        J.s_store_dword(cu_id, info, J.sizeof_DW*1, mod=f"glc")
        J.s_store_dword(se_id, info, J.sizeof_DW*2, mod=f"glc")
        J.s_store_dword(xcc_id, info, J.sizeof_DW*3, mod=f"glc")

def probe_cache_associativity():
    print("============== test block's cache associativity ==============")
    print(" if two waves accessing two different 4MB buffers were executing on same XCC,")
    print(" they will compete for same L2-cache, thus much longer latency ")
    print(" we can detect blocks sharing L2-caches using this way")
    num_elements = 4*1024*1024  # big enough to fits L2
    d_in1 = torch.ones(num_elements, dtype=torch.float)
    d_in2 = torch.ones(num_elements, dtype=torch.float)
    block_info = torch.zeros([num_CU, 4], dtype=torch.int)

    all_blocks = list(range(num_CU))
    while len(all_blocks) > 0:
        block0 = all_blocks.pop(0)
        latencies = []
        for block1 in all_blocks:
            p = pyhip.cudaPerf(verbose=0)
            for i in range(2):
                with p:
                    compete_cache([num_CU], [64*4],
                                  block0, block1,
                                  d_in1.data_ptr(),
                                  d_in2.data_ptr(),
                                  num_elements,
                                  block_info.data_ptr())
            l = p.dt(excludes=1)
            latencies.append(l*1e6)
        #print(block0)
        #print(latencies)
        #assert 0
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


if __name__ == "__main__":
    print(f"{pyhip.JIT.arch=}")
    print(f"{pyhip.JIT.gfx=}")
    print(f"{pyhip.JIT.cdna=}")
    print(f"{num_CU=}")

    probe_cycles(num_threads=64)
    probe_cycles(num_threads=256)

    probe_vmem_bandwidth()

    probe_lds_banks()

    probe_lanegroups("ds_read_b128")
    '''
            [0, 1, 2, 3, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27]
            [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 28, 29, 30, 31]
            [32, 33, 34, 35, 44, 45, 46, 47, 52, 53, 54, 55, 56, 57, 58, 59]
            [36, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50, 51, 60, 61, 62, 63]
    '''
    probe_lanegroups("ds_read_b64")
    '''
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    '''
    probe_lanegroups("ds_read_b32")
    '''
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
            [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    '''

    print("""
    we cannot test ds_write using the same method because it has no broad-cast
    but we can assume it follows the same lane-group as its peer ds_read
    """)

    probe_cache_associativity()
