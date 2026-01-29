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

        for lane in lane_group: probe_lanes.remove(lane)

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
        with pyhip.cuPerf(rw_bytes=bytes_per_WG*num_CU, name=f"load-{dtype}-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "load", dtype, data[r].data_ptr(), cycles.data_ptr())

    for r in range(10):
        with pyhip.cuPerf(rw_bytes=bytes_per_WG*num_CU, name=f"store-{dtype}-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "store", dtype, data[r].data_ptr(), cycles.data_ptr())

    for r in range(10):
        with pyhip.cuPerf(rw_bytes=bytes_per_WG*num_CU, name=f"atomic_pk_add_bf16-{dtype}-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "atomic_pk_add_bf16", "dword", data[r].data_ptr(), cycles.data_ptr())

    for r in range(10):
        with pyhip.cuPerf(rw_bytes=bytes_per_WG*num_CU, name=f"global_atomic_xor-{dtype}-{r}"):
            dummy_load([num_CU], [256], bytes_per_WG, "atomic_xor", "dword", data[r].data_ptr(), cycles.data_ptr())

if __name__ == "__main__":
    print(f"{pyhip.JIT.arch=}")
    print(f"{pyhip.JIT.gfx=}")
    print(f"{pyhip.JIT.cdna=}")

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
