#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

/*

How thread-block is scheduled in real environment?

wall_clock64() => S_MEMREALTIME

CDNA3 ISA
    8.2.5. S_MEMREALTIME
This instruction reads a 64-bit "real time-counter" and returns the value into a pair of SGPRS: SDST and SDST+1.
The time value is from a constant 100MHz clock (not affected by power modes or core clock frequency
changes).


https://github.com/ROCm/ROCm/issues/2059
https://llvm.org/docs/AMDGPU/gfx940_hwreg.html#hwreg

CDNA3 ISA
3.12. Hardware ID Registers
WAVE_ID  3:0    Wave buffer slot number
SIMD_ID  5:4    SIMD which the wave is assigned to within the CU
PIPE_ID  7:6    Pipeline from which the wave was dispatched
CU_ID    11:8   Compute Unit the wave is assigned to
SH_ID    12     Shader Array (within an SE) the wave is assigned to. Is set to zero.
SE_ID    15:13  Shader Engine the wave is assigned to
TG_ID    19:16  Thread-group ID
VM_ID    23:20  Virtual Memory ID
QUEUE_ID 26:24  Queue from which this wave was dispatched
STATE_ID 29:27  State ID (UNUSED)
ME_ID    31:30  Micro-engine ID

Table 7. XCC ID (XCC_ID)
XCC_ID 3:0 ID of this XCC
*/

__global__ void threadblock_test(int busy_time, int64_t* info) {
    uint64_t t0 = wall_clock64();
    // the thread-block is scheduled along x-dim first
    auto block_id = blockIdx.x;

    uint64_t t1;
    do {
        t1 = wall_clock64(); 
    } while(t1 - t0 < busy_time);

    info += block_id * 8;

    uint simd_id;
    uint cu_id;
    uint se_id;
    uint tg_id;
    uint xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 4, 2)" : "=s"(simd_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=s"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=s"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 16, 3)" : "=s"(tg_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)" : "=s"(xcc_id));
    if (threadIdx.x == 0) {
        info[0] = xcc_id;
        info[1] = se_id;
        info[2] = cu_id;
        info[3] = simd_id; // when threads-per-block > 64, this id may have multiple value withing a block
        info[4] = tg_id;
        info[5] = t0;
        info[6] = t1 - t0;
    } 
}


__global__ void compete_cache(
        float* d_sum,
        float* d_in1, float* d_in2, int numElements,
        int block0, int block1, int save_sum) {
    auto numVectElements = (numElements >> 2);
    uint tid = threadIdx.x;

    // only allow 2 blocks to access the same memory buffer
    if (blockIdx.x != block0 && blockIdx.x != block1) {
        return;
    }
    float4 *vectorized_in = nullptr;
    if (blockIdx.x == block0)
        vectorized_in = reinterpret_cast<float4 *>(d_in1);
    else
        vectorized_in = reinterpret_cast<float4 *>(d_in2);

    float4 nc = make_float4(0.0f,0.0f,0.0f,0.0f);

    for(int i = 0; i < 10; i++) {
        float4 nc1 = make_float4(0.0f,0.0f,0.0f,0.0f);
        for (;tid < numVectElements; tid += blockDim.x) {
            auto va = vectorized_in[tid];
            nc1 += va;
        }
        for (;tid < numVectElements; tid += blockDim.x) {
            vectorized_in[tid] -= nc1;
        }
    }

    uint simd_id;
    uint cu_id;
    uint se_id;
    uint tg_id;
    uint xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 4, 2)" : "=s"(simd_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=s"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=s"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 16, 3)" : "=s"(tg_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 4)" : "=s"(xcc_id));
    if (save_sum) {
        d_sum[tid] = nc.x + nc.y + nc.z + nc.w;
    } else if (threadIdx.x == 0) {
        auto* info = reinterpret_cast<int32_t *>(d_sum);
        info[blockIdx.x*4 + 0] = xcc_id;
        info[blockIdx.x*4 + 1] = se_id;
        info[blockIdx.x*4 + 2] = cu_id;
    }
}
