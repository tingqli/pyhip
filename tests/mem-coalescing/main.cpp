#include "hip/hip_runtime.h"
#include <stdio.h>


// https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/
template<bool to_lastlane>
__device__ float warp_reduce_sum(float sum) {
    asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 ": "=v"(sum): "0"(sum), "v"(sum), "v"(sum));
    asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 ": "=v"(sum): "0"(sum), "v"(sum), "v"(sum));
    asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 ": "=v"(sum): "0"(sum), "v"(sum), "v"(sum));
    asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 ": "=v"(sum): "0"(sum), "v"(sum), "v"(sum));
    //asm("\n\tv_add_f32 %0, %2, %3 wave_shr:1 bound_ctrl:0": "=v"(sum): "0"(sum), "v"(sum), "v"(sum));
    asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:15 bound_ctrl:0": "=v"(sum): "0"(sum), "v"(sum), "v"(sum));
    asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_bcast:31 bound_ctrl:0": "=v"(sum): "0"(sum), "v"(sum), "v"(sum));
    if constexpr(to_lastlane) return sum;
    return __shfl(sum, warpSize-1); // this last shfl will call ds_bpermute to broad-cast the last lane to all lanes in a warp
}

/*
    simplify the computations between read-instructions can further increase bandwidth
*/
__global__ void sum_1d_x4(float* d_sum, const float* d_in, uint numElements, int save_sum) {
    float4 const *vectorized_in = reinterpret_cast<float4 const *>(d_in);

    auto numVectElements = (numElements >> 2);

    uint tid = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // permutation of 64 lanes do not hurt coalescing performance
    tid = ((tid & 63)^(57)) | (tid & ~63);

    uint grid_tid_size = blockDim.x * gridDim.x;
    float4 nc = make_float4(0.0f,0.0f,0.0f,0.0f);
    for (;tid < numVectElements; tid += grid_tid_size) {
        auto v4 = vectorized_in[tid];
        nc.x += v4.x;
        nc.y += v4.y;
        nc.z += v4.z;
        nc.w += v4.w;
    }
    float sum = nc.x + nc.y + nc.z + nc.w;
    sum = warp_reduce_sum<false>(sum);
    //auto warp_id = threadIdx.x >> 6;
    auto lane_id = threadIdx.x & 63;
    if (save_sum && lane_id == (warpSize-1)) {
        // this atomic add slower down the kernel by a lot, so only save_sum when checking accuracy
        __atomic_add_fetch(d_sum, sum, __ATOMIC_RELAXED);
    }
}

__global__ void sum_1d_x2(float* d_sum, const float* d_in, uint numElements, int save_sum) {
    float2 const *vectorized_in = reinterpret_cast<float2 const *>(d_in);

    auto numVectElements = (numElements >> 1);

    uint tid = (blockIdx.x * blockDim.x + threadIdx.x);
    uint grid_tid_size = blockDim.x * gridDim.x;
    float2 nc = make_float2(0.0f,0.0f);
    for (;tid < numVectElements; tid += grid_tid_size) {
        auto va = vectorized_in[tid];
        nc.x += va.x;
        nc.y += va.y;
    }
    float sum = nc.x + nc.y;
    sum = warp_reduce_sum<false>(sum);
    auto lane_id = threadIdx.x & 63;
    if (save_sum && lane_id == (warpSize-1)) {
        __atomic_add_fetch(d_sum, sum, __ATOMIC_RELAXED);
    }
}


/*
    https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html

    1D data viewed as 16xN
    load 16 rows (each with 64bytes: 4 lanes x DWORD4)
*/

__global__ void sum_1d_16x64(float* d_sum, const float* d_in, uint numElements, int save_sum) {
    float4 const *vectorized_in = reinterpret_cast<float4 const *>(d_in);

    auto rowElements = (numElements >> 4);      // 16 rows, each row has such many elements
    auto rowVectElements = (rowElements >> 2);  // how many float4(DWORD4) in each row

    //size_t tid = (blockIdx.x * blockDim.x + threadIdx.x);

    auto warp_cnt = blockDim.x >> 6; // 4
    auto warp_id = threadIdx.x >> 6; // 0,1,2,3
    auto lane_id = threadIdx.x & 63; // 0...63
    auto row_idx = (lane_id >> 2);   // 0,1,...,15
    auto col_idx = (lane_id & 3);    // 0,1,2,3

    uint warp_col = (blockIdx.x*warp_cnt + warp_id)*4;
    uint idx = row_idx*rowVectElements + warp_col + col_idx;
    uint grid_idx_size = gridDim.x * warp_cnt * 4; // 640*4*4

    float4 nc = make_float4(0.0f,0.0f,0.0f,0.0f);
    for (;warp_col < rowVectElements; warp_col += grid_idx_size, idx += grid_idx_size)
    {
        auto v4 = vectorized_in[idx];
        nc.x += v4.x;
        nc.y += v4.y;
        nc.z += v4.z;
        nc.w += v4.w;
    }
    float sum = nc.x + nc.y + nc.z + nc.w;
    sum = warp_reduce_sum<false>(sum);

    if (save_sum && lane_id == (warpSize-1)) {
        // this atomic add slower down the kernel by a lot, so only save_sum when checking accuracy
        __atomic_add_fetch(d_sum, sum, __ATOMIC_RELAXED);
    }
}



__global__ void sum_1d_8x128(float* d_sum, const float* d_in, uint numElements, int save_sum) {
    float4 const *vectorized_in = reinterpret_cast<float4 const *>(d_in);

    auto rowElements = (numElements >> 3);      // 8 rows, each row has such many elements
    auto rowVectElements = (rowElements >> 2);  // how many float4(DWORD4) in each row

    //size_t tid = (blockIdx.x * blockDim.x + threadIdx.x);

    auto warp_cnt = blockDim.x >> 6; // 4
    auto warp_id = threadIdx.x >> 6; // 0,1,2,3
    auto lane_id = threadIdx.x & 63; // 0...63
    auto row_idx = (lane_id >> 3);   // 0,1,...,7
    auto col_idx = (lane_id & 7);    // 0,...7

    uint warp_col = (blockIdx.x*warp_cnt + warp_id) * 8;
    uint idx = row_idx*rowVectElements + warp_col + col_idx;
    uint grid_idx_size = gridDim.x * warp_cnt * 8; // 640*4*4

    float4 nc = make_float4(0.0f,0.0f,0.0f,0.0f);
    for (;warp_col < rowVectElements; warp_col += grid_idx_size, idx += grid_idx_size)
    {
        auto v4 = vectorized_in[idx];
        nc.x += v4.x;
        nc.y += v4.y;
        nc.z += v4.z;
        nc.w += v4.w;
    }
    float sum = nc.x + nc.y + nc.z + nc.w;
    sum = warp_reduce_sum<false>(sum);

    if (save_sum && lane_id == (warpSize-1)) {
        // this atomic add slower down the kernel by a lot, so only save_sum when checking accuracy
        __atomic_add_fetch(d_sum, sum, __ATOMIC_RELAXED);
    }
}




__global__ void sum_1d_4x256(float* d_sum, const float* d_in, uint numElements, int save_sum) {
    float4 const *vectorized_in = reinterpret_cast<float4 const *>(d_in);

    auto rowElements = (numElements >> 2);      // 4 rows, each row has such many elements
    auto rowVectElements = (rowElements >> 2);  // how many float4(DWORD4) in each row

    //size_t tid = (blockIdx.x * blockDim.x + threadIdx.x);

    auto warp_cnt = blockDim.x >> 6; // 4
    auto warp_id = threadIdx.x >> 6; // 0,1,2,3
    auto lane_id = threadIdx.x & 63; // 0...63
    auto row_idx = (lane_id >> 4);   // 0,...,3
    auto col_idx = (lane_id & 15);    // 0,...15

    uint warp_col = (blockIdx.x*warp_cnt + warp_id) * 16;
    uint idx = row_idx*rowVectElements + warp_col + col_idx;
    uint grid_idx_size = gridDim.x * warp_cnt * 16; // 640*4*4

    float4 nc = make_float4(0.0f,0.0f,0.0f,0.0f);
    for (;warp_col < rowVectElements; warp_col += grid_idx_size, idx += grid_idx_size)
    {
        auto v4 = vectorized_in[idx];
        nc.x += v4.x;
        nc.y += v4.y;
        nc.z += v4.z;
        nc.w += v4.w;
    }
    float sum = nc.x + nc.y + nc.z + nc.w;
    sum = warp_reduce_sum<false>(sum);

    if (save_sum && lane_id == (warpSize-1)) {
        // this atomic add slower down the kernel by a lot, so only save_sum when checking accuracy
        __atomic_add_fetch(d_sum, sum, __ATOMIC_RELAXED);
    }
}



__global__ void sum_1d_2x512(float* d_sum, const float* d_in, uint numElements, int save_sum) {
    float4 const *vectorized_in = reinterpret_cast<float4 const *>(d_in);

    auto rowElements = (numElements >> 1);      // 2 rows, each row has such many elements
    auto rowVectElements = (rowElements >> 2);  // how many float4(DWORD4) in each row

    //size_t tid = (blockIdx.x * blockDim.x + threadIdx.x);

    auto warp_cnt = blockDim.x >> 6; // 4
    auto warp_id = threadIdx.x >> 6; // 0,1,2,3
    auto lane_id = threadIdx.x & 63; // 0...63
    auto row_idx = (lane_id >> 5);   // 0,...,1
    auto col_idx = (lane_id & 31);    // 0,...15

    uint warp_col = (blockIdx.x*warp_cnt + warp_id) * 32;
    uint idx = row_idx*rowVectElements + warp_col + col_idx;
    uint grid_idx_size = gridDim.x * warp_cnt * 32; // 640*4*4

    float4 nc = make_float4(0.0f,0.0f,0.0f,0.0f);
    for (;warp_col < rowVectElements; warp_col += grid_idx_size, idx += grid_idx_size)
    {
        auto v4 = vectorized_in[idx];
        nc.x += v4.x;
        nc.y += v4.y;
        nc.z += v4.z;
        nc.w += v4.w;
    }
    float sum = nc.x + nc.y + nc.z + nc.w;
    sum = warp_reduce_sum<false>(sum);

    if (save_sum && lane_id == (warpSize-1)) {
        // this atomic add slower down the kernel by a lot, so only save_sum when checking accuracy
        __atomic_add_fetch(d_sum, sum, __ATOMIC_RELAXED);
    }
}
