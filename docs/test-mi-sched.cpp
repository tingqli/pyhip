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

#define SGB_VMEM_read_0x0020 0x0020
#define SGB_MFMA_0x0008      0x0008
#define SGB_DS_read_0x0100   0x0100
#define SGB_DS_write_0x0200  0x0200

__global__ void __launch_bounds__(256, 1) gemm(__fp16* A, float* C, int nstrideC, int K, int *offsets) {
    __shared__ __fp16 Abuff[16*1024];

    for(int i = threadIdx.x; i < 16*1024; i += blockDim.x) {
        Abuff[i] = A[i];
    }
    float16x4 Aregs0[4];
    float16x4 Bregs0[4];
    float16x4 Aregs1[4];
    float16x4 Bregs1[4];

    float32x4 c[16] = {0};

    auto compute0 = [&](){
        #pragma unroll
        for(int m=0;m<4;m++) {
            #pragma unroll
            for(int n=0;n<4;n++) {
                c[m*4 + n] = __builtin_amdgcn_mfma_f32_16x16x16f16(Aregs0[m], Bregs0[n], c[m*4 + n], 0, 0, 0);
            }
        }
    };
    auto compute1 = [&](){
        #pragma unroll
        for(int m=0;m<4;m++) {
            #pragma unroll
            for(int n=0;n<4;n++) {
                c[m*4 + n] = __builtin_amdgcn_mfma_f32_16x16x16f16(Aregs1[m], Bregs1[n], c[m*4 + n], 0, 0, 0);
            }
        }
    };
    auto load0 = [&](int voff){
        #pragma unroll
        for(int m=0;m<4;m++) {
            Aregs0[m] = *(float16x4*)&Abuff[voff + m];
            Bregs0[m] = *(float16x4*)&Abuff[voff + m + 8*1024];
        }
    };
    auto load1 = [&](int voff){
        #pragma unroll
        for(int m=0;m<4;m++) {
            Aregs1[m] = *(float16x4*)&Abuff[voff + m];
            Bregs1[m] = *(float16x4*)&Abuff[voff + m + 8*1024];
        }
    };

    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    int voff = offsets[threadIdx.x];

    load0(voff);

    for(int k = 0; k < K; k++, voff += 16) {
        // 从Abuff中加载数据到寄存器, 每个warp独立加载
        __syncthreads();

        __builtin_amdgcn_sched_barrier(0);
        asm volatile(";================= compute0(); load1();");
        __builtin_amdgcn_sched_barrier(0);

        compute0();
        load1(voff);

        voff = offsets[k + threadIdx.x + 1024];

#ifdef MI_SCHED_BARRIER
        for(int i = 0; i < 8; i++) {
            __builtin_amdgcn_sched_group_barrier(SGB_DS_read_0x0100, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 2, 0);
        }
#endif
        __builtin_amdgcn_sched_barrier(0);
        asm volatile(";================= compute1(); load0();");
        __builtin_amdgcn_sched_barrier(0);

        __syncthreads();
        compute1();
        load0(voff);

#ifdef MI_SCHED_BARRIER
        for(int i = 0; i < 8; i++) {
            __builtin_amdgcn_sched_group_barrier(SGB_DS_read_0x0100, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 2, 0);
        }
#endif
    }

    for(int i = 0; i < 16; i++) {
        *(float32x4*)&C[i*16] = c[i];
    }
}

