#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float16x8 = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__global__ void dsread_test(uint32_t* data, uint32_t* indices_u32, int sm) {
    __shared__ uint32_t lds_mem[64/4*1024]; // 64KB

    // this loads prevents compiler from optimizing all computation on uninitialized LDS data
    for(int i = threadIdx.x; i < sizeof(lds_mem)/sizeof(lds_mem[0]); i += blockDim.x) {
        lds_mem[i] = data[i];
    }
    uint32_t idx_u32 = indices_u32[threadIdx.x & 63];
    uint32_t sum = 0;
    for(int n = 0; n < 10000; n++) {
        // n&sm prevents compiler from optimizing the load
        sum += lds_mem[idx_u32 + (n&sm)];
    }
    if (threadIdx.x == 0) {
        data[0] = sum; // prevent opt
    }
}

__global__ void dsread_testx2(uint32_t* data, uint32_t* indices_u32, int sm) {
    __shared__ uint32_t lds_mem[64/4*1024]; // 64KB

    // this loads prevents compiler from optimizing all computation on uninitialized LDS data
    for(int i = threadIdx.x; i < sizeof(lds_mem)/sizeof(lds_mem[0]); i += blockDim.x) {
        lds_mem[i] = data[i];
    }
    uint32_t idx_u32 = indices_u32[threadIdx.x & 63];
    uint32_t sum = 0;
    for(int n = 0; n < 10000; n++) {
        // n&sm prevents compiler from optimizing the load
        uint2 v = ((uint2*)lds_mem)[idx_u32 + (n&sm)];
        sum += v.x + v.y;
    }
    if (threadIdx.x == 0) {
        data[0] = sum; // prevent opt
    }
}


__global__ void dsread_testx4(uint32_t* data, uint32_t* indices_u32, int sm) {
    __shared__ uint32_t lds_mem[64/4*1024]; // 64KB

    // this loads prevents compiler from optimizing all computation on uninitialized LDS data
    for(int i = threadIdx.x; i < sizeof(lds_mem)/sizeof(lds_mem[0]); i += blockDim.x) {
        lds_mem[i] = data[i];
    }
    uint32_t idx_u32 = indices_u32[threadIdx.x & 63];
    uint32_t sum = 0;
    for(int n = 0; n < 10000; n++) {
        // n&sm prevents compiler from optimizing the load
        uint4 v = ((uint4*)lds_mem)[idx_u32 + (n&sm)];
        sum += v.x + v.y + v.z + v.w;
    }
    if (threadIdx.x == 0) {
        data[0] = sum; // prevent opt
    }
}
