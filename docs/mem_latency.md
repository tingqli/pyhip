# Memory loads latency

When pipelining a for loop with data-load stages & computation stages, 
we need rough memory access latency data to estimate how many cycles 
in advance the data loads should be.

Following code measures HBM & LDS reads latency using `s_memtime`. this markdown can be directly executed by:

```bash
python -m pyhip mem_latency.md
```

We got following measurements on MI308X @ 1-wave/CU *(LDS/HBM is exclusively used by one wave per CU)*:

| mem type                             | latency(cycles) |  throughput/CU (cycles) | throughput/CU (bytes/cycles) |
|--------------------------------------|----------------:|------------------------:|-----------------------------:|
| HBM load  dwordx4                    | 500~800         |   ~32                   |      32                      |
| LDS read (no bank-conflict) b128/b32 | 64/52           |  16/4      [^1]         |      64/64 [^1]              |
| LDS read b32 (4-bank-conflict)       | 120             |    16                   |      16                      |    
| LDS read b32 (full-bank-conflict)    | 119             |  64/64                  |       4                      |

 - $latency = clock\_of[load_1, wait]$
 - $throughput = (clock\_of[load_1,load_2, ...., load_{11}, wait] - latency)/10 $

[^1]: The doc says LDS's throughput is evenly divided between SIMD1&2 and SIMD3&4, since our test only uses 1 wave, the measured throughput of 16-cycles (or 64bytes/cycles) is actually just half of the real LDS capability.

The measurements are consistent with [LDS desciption](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/pipeline-descriptions.html#desc-lds):
 - The doc says a single wavefront can get 64B/cycle from LDS, which is consistent with ds_read_b32 throughput 4 (4B*64/4cycles = 64B/cycle).
 - at full bank-conflict(voff_stride=32), ds_read_b32 needs 64 cycles to get a new result in throughput mode, thus actual throughput of LDS is `4B*64/64cycles = 4B/cycles`, which means only 1 bank is working per cycle since all lanes fall into a single bank.
 - HBM load throughput `16B*64/32*80*GPU_freq = 2.56 TB/s/GHz` is roughly consistent with max HBM bandwidth.

Normally to fully use the VALU/MatrixCore's power, at least 4 waves per CU will be launched, and throughput/wave will be 1/4 of the number above.

# Example Usage in gemm

Suppose we want to implement following GEMM kernel:
 - 4 waves per CU
 - using `v_mfma_f32_32x32x8_f16`, latency/throughput is 32 cycles
 - each wave allocate 16(4x4) 32x32 C tiles, which is `128x128` floats accumulator registers, and just fit into `256x64` AccGPRs limitations.
 - 4 waves make up a work-group (thread block), calculate a `2x2x(128x128)=256x256` C tile
 - A/B tile are both in shape of `256x32`, and they are first loaded into LDS by 4 waves cooperatively, in memory-coalescing manner. then 4 waves load them into VGPR registers to feed to MFMA instruction.

To design a pipeline, the latency/throughput data was considered as following:
 - for each wave, the inner loop will do such gemm sub-problem : `128x32 @ 32x128 =+=> 128x128`, since each MFMA instruction do a small gemm of `32x8 @ 8x32 =+=> 32x32`, there will be `4*4*4=64` MFMA instructions, estimated total cycles are : `64*32=2048`
 - meanwhile/in-parallel, 4 waves need to prefetch `256x32x2xsizeof(half)=32768 Bytes`, the HBM throughput `32B/cycle` tells us that HBM is capable of loading upto `2048*32=64KB` data within 2048 computation cycles. Thus the problem is indeed compute-bounded.
 - considering HBM load latency of 500~800 cycles, we need to issue buffer load at least 800 cycles before we wait vmcnt and save the loaded data into LDS. and if we do HBM load using dwordx4 buffer load, and 4 waves together need to issue `32768/(64*16)=32` such loads, which is 8 dwordx4-load instructions per wave.
 - These instructions are issued together by 4 waves, and the HW execution pipeline serving them are shared by all waves in CU, and it needs time to send these mem-access-request down the complex memory-subsystem, the rate at which these request are sent will not exceed the 32-cycles throughput we measured, which means if we issue multiple requests faster than `1-request/32-cycles`, the issue pipeline is likely to be blocked, and consider all requests from 4 waves are actually sharing this `1-request/32-cycles` limitation, for each wave we better issue at rate of `1/4-request per 32-cycles = 1-request per 4*32 cycles` to avoid issue-blocking. as result, the issuing of totally 8 such instructions should be evenly distributed into `8*4=32` MFMA instructions
 - LDS write: .... ....(TO BE DONE)
 - within each inner loop, each wave needs to reads two `128x32` tiles (A & B) from LDS into register to feed to MFMA, which is `128*32*2*sizeof(half)=16384 bytes` corresponds to 16 `ds_read_b128` instructions:
   - these instructions have ~64 cycles of latency, thus need to be issued at leat 2 MFMA instructions ahead before the MFMA really using the load result.
   - these instructions have ~8 cycles of throughput(when being issued from at least 2 SIMDs), so they can be interleaved with MFMA in 1:1 ratio. 4 waves would issue 4 ds_read_b128 within 32 cycles in total and just saturate the throughput rate of `1-ds_read_b128 per 8-cycles`


<details>

<summary>source codes</summary>

### HIP device kernels
 - [AMDGPU inline asm](https://llvm.org/docs/LangRef.html#supported-constraint-code-list) with volatile prevents compiler from changing the code
 - `__builtin_amdgcn_sched_barrier` prevents compiler schedule unrelavant codes into the target assembly snippets.
 - `clock64()` is a HIP builtin for `s_memtime` instruction, which has very low-overhead (4-cycles) and perfect for self-profiling.

```c++

#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float16x32 = __attribute__((__vector_size__(32 * sizeof(__fp16)))) __fp16;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x16_t = __attribute__((__vector_size__(16 * sizeof(int)))) int;

using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;
__device__ __inline__
static int32x4_t llvm_buffer_load_dwordx4(int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

union BufferResource {
    __device__ __inline__ constexpr BufferResource()
        : config(0x00020000U) {}

    __device__ __inline__ constexpr BufferResource(void* buffer_address, uint32_t buffer_size)
        : address(buffer_address),
          range(buffer_size),
          config(0x00020000U) {}

    int32x4_t descriptor;
    struct{
        void* address;      // 8B, out of which first 48b is address, and 16b is stride (unused)
        uint32_t range;     // Byte range for the buffer resource
        uint32_t config;    // Constant, DFMT=32b
    };
};

// https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html#OutputOperands

template<typename T, int imm_offset=0>
__device__ T ds_read_b128(T* base, int index) {
    T v;
    static_assert(sizeof(T) == sizeof(int32x4_t));
    as3_uint32_ptr vaddr = (as3_uint32_ptr)(base + index);
    asm volatile("ds_read_b128 %[vdst], %[vaddr] offset:%[offset]"
                : [vdst]"=v"((int32x4_t&)(v))
                : [vaddr]"v"(vaddr),[offset]"i"(imm_offset));
    return v;
}

template<typename T, int imm_offset=0>
__device__ T ds_read_b32(T* base, int index) {
    T v;
    static_assert(sizeof(T) == sizeof(int32_t));
    as3_uint32_ptr vaddr = (as3_uint32_ptr)(base + index);
    asm volatile("ds_read_b32 %[vdst], %[vaddr] offset:%[offset]"
                : [vdst]"=v"((int32_t&)(v))
                : [vaddr]"v"(vaddr),[offset]"i"(imm_offset));
    return v;
}

template<uint16_t cnt>
__device__ void s_waitcnt_lgkmcnt() {
    asm volatile ("s_waitcnt lgkmcnt(%0)\n"::"i"(cnt));
}

template<typename T>
__device__ T buffer_load_dwordx4(BufferResource& buffer, int soffset, int voffset) {
    T v;
    asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
        :[vdst]"=v"(v)
        :[vaddr]"v"(voffset), [srsrc]"s"(buffer.descriptor), [soffset]"s"(soffset));
    return v;
}
template<uint16_t cnt>
__device__ void s_waitcnt_vmcnt() {
    asm volatile ("s_waitcnt vmcnt(%0)\n"::"i"(cnt));
}

template<typename F>
__device__ uint32_t get_cycles(F func, int* p_cycles = nullptr) {
    uint64_t start, end;
    __builtin_amdgcn_sched_barrier(0);
    //asm volatile ("s_memtime %0\n":"=s"(start):);
    start = clock64();
    __builtin_amdgcn_sched_barrier(0);

    func();

    __builtin_amdgcn_sched_barrier(0);
    //asm volatile ("s_memtime %0\n":"=s"(end):);
    end = clock64();
    __builtin_amdgcn_sched_barrier(0);
    

    uint32_t cycles = __builtin_amdgcn_readfirstlane(end) - __builtin_amdgcn_readfirstlane(start);

    return cycles;
}

__global__ __launch_bounds__(256, 1) void test(int* A, int K, int i0, int* indices) {
    __shared__ int32x4_t lds[1024*4];

    A += blockIdx.x * K;

    bool verbose = blockIdx.x == 0 && threadIdx.x == 0;
    if (verbose)
        printf("=============================\n");
    uint64_t start, end;
    uint64_t dc = 0;
    for(int i = 0; i < 10; i++) {
        dc += get_cycles([]{});
    }
    if (verbose)
        printf(" s_memtime overhead cycles: %lu\n", dc/10);

    BufferResource buff(A, K*sizeof(int));

    // buffer_load_dwordx4:latency
    dc = 0;
    int soffset = i0*sizeof(int32x4_t);
    int voffset = threadIdx.x * sizeof(int32x4_t);
    uint64_t count = 0;
    int32x4_t data;
    for(int i = 0; i < 64*1024; i += 64) {
        dc += get_cycles([&](){
            data = buffer_load_dwordx4<int32x4_t>(buff, soffset, voffset);
            s_waitcnt_vmcnt<0>();
        });
        count ++;
        soffset =  (__builtin_amdgcn_readfirstlane(data[0]) * sizeof(int)/sizeof(int32x4_t))*sizeof(int32x4_t);
        lds[i + threadIdx.x] = data;
    }
    auto buffer_load_dwordx4_latency =  dc/count;
    if (verbose)
        printf(" buffer_load_dwordx4 latency cycles: %lu\n", buffer_load_dwordx4_latency);

    // buffer_load_dwordx4:throughput
    dc = count = 0;
    int32x4_t datas[11];
    #pragma nounroll
    for(int i = 0; i < 64*1024; i += 64) {
        dc += get_cycles([&](){
            #pragma unroll
            for(int k = 0; k < 11; k++)
                datas[k] = buffer_load_dwordx4<int32x4_t>(buff, soffset + k*64*sizeof(int32x4_t), voffset);
            s_waitcnt_vmcnt<0>();
        });
        // prevent compiler allocate same vgprs for every datas[k]
        for(int k = 0; k < 11; k++) {
            lds[k + threadIdx.x] = datas[k];
        }
        soffset = (__builtin_amdgcn_readfirstlane(datas[0][0]) * sizeof(int)/sizeof(int32x4_t))*sizeof(int32x4_t);
        count ++;
    }
    auto buffer_load_dwordx4_latency_tput = dc/count;
    auto buffer_load_dwordx4_tput = (float)(buffer_load_dwordx4_latency_tput - buffer_load_dwordx4_latency)/10.0;
    if (verbose)
        printf(" buffer_load_dwordx4 throughput cycles: %.2f\n", buffer_load_dwordx4_tput);

    dc = 0;
    int index = indices[threadIdx.x];
    for(int i = 0; i < 10; i++) {
        dc += get_cycles([&](){
            ds_read_b128(lds, index);
            s_waitcnt_lgkmcnt<0>();
        });
    }
    auto lds_latency = dc/10;
    if (verbose)
        printf(" ds_read_b128 latency cycles: %lu\n", lds_latency);

    dc = 0;
    for(int i = 0; i < 10; i++) {
        dc += get_cycles([&](){
            datas[0] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*0>(lds, index);
            datas[1] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*1>(lds, index);
            datas[2] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*2>(lds, index);
            datas[3] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*3>(lds, index);
            datas[4] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*4>(lds, index);
            datas[5] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*5>(lds, index);
            datas[6] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*6>(lds, index);
            datas[7] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*7>(lds, index);
            datas[8] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*8>(lds, index);
            datas[9] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*9>(lds, index);
            datas[10] = ds_read_b128<int32x4_t, sizeof(int32x4_t)*64*10>(lds, index);
            s_waitcnt_lgkmcnt<0>();
        });
        for(int k = 0; k < 11; k++) {
            lds[i + k + threadIdx.x] = datas[k];
        }
    }
    auto lds_latency_tput = dc/10;
    auto lds_tput = (lds_latency_tput - lds_latency)/10;
    if (verbose)
        printf(" ds_read_b128 tput cycles: %lu\n", lds_tput);

    dc = 0;
    for(int i = 0; i < 10; i++) {
        dc += get_cycles([&](){
            ds_read_b32((int32_t*)lds, index);
            s_waitcnt_lgkmcnt<0>();
        });
    }
    lds_latency = dc/10;
    if (verbose)
        printf(" ds_read_b32 latency cycles: %lu\n", lds_latency);

    int32_t i32data[11];
    dc = 0;
    #pragma nounroll
    for(int i = 0; i < 10; i++) {
        auto dt = get_cycles([&](){
            i32data[0] = ds_read_b32<int32_t, sizeof(int32_t)*64*0>((int32_t*)lds, index);
            i32data[1] = ds_read_b32<int32_t, sizeof(int32_t)*64*1>((int32_t*)lds, index);
            i32data[2] = ds_read_b32<int32_t, sizeof(int32_t)*64*2>((int32_t*)lds, index);
            i32data[3] = ds_read_b32<int32_t, sizeof(int32_t)*64*3>((int32_t*)lds, index);
            i32data[4] = ds_read_b32<int32_t, sizeof(int32_t)*64*4>((int32_t*)lds, index);
            i32data[5] = ds_read_b32<int32_t, sizeof(int32_t)*64*5>((int32_t*)lds, index);
            i32data[6] = ds_read_b32<int32_t, sizeof(int32_t)*64*6>((int32_t*)lds, index);
            i32data[7] = ds_read_b32<int32_t, sizeof(int32_t)*64*7>((int32_t*)lds, index);
            i32data[8] = ds_read_b32<int32_t, sizeof(int32_t)*64*8>((int32_t*)lds, index);
            i32data[9] = ds_read_b32<int32_t, sizeof(int32_t)*64*9>((int32_t*)lds, index);
            i32data[10] = ds_read_b32<int32_t, sizeof(int32_t)*64*10>((int32_t*)lds, index);
            s_waitcnt_lgkmcnt<0>();
            asm volatile(";===============\n");
        });
        dc += dt;
        for(int k = 0; k < 11; k++) {
            ((int32_t*)A)[i + k + threadIdx.x] = i32data[k];
        }
    }
    lds_latency_tput = dc/10;
    lds_tput = (lds_latency_tput - lds_latency)/10;
    if (verbose)
        printf(" ds_read_b32 tput cycles: %lu\n", lds_tput);
}
```
### Python HOST side
```python

import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
num_CU = torch.cuda.get_device_properties().multi_processor_count

# passing 4GB buffer into kernel has problem, all reads from kernel got 0
# but <4GB has no such problem
per_CU_K = (1024//num_CU)*1024*1024

K=num_CU * per_CU_K
if K >= 1024*1024*1024: K-= 1

# each CU has it's own HBM data to test with
A = torch.randint(0, per_CU_K-4096, (per_CU_K,), dtype=torch.int32)
A = A.repeat(num_CU, 1) # num_CU, per_CU_K

voff_stride = 32 # full-bank conflict for ds_read_b32 & ds_read_b128
voff_stride = 8 # 4 bank conflicts for ds_read_b32
voff_stride = 1 # no bank conflict for ds_read_b32 & ds_read_b128
voff = [i*voff_stride for i in range(64)]
voff = torch.tensor(voff, dtype=torch.uint32)

threads = 64
hip.test([num_CU], [threads], A.data_ptr(), per_CU_K, 0, voff.data_ptr())
hip.test([num_CU], [threads], A.data_ptr(), per_CU_K, 1, voff.data_ptr())
hip.test([num_CU], [threads], A.data_ptr(), per_CU_K, 2, voff.data_ptr())
hip.test([num_CU], [threads], A.data_ptr(), per_CU_K, 3, voff.data_ptr())
```
</details>

