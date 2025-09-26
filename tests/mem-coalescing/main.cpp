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

__device__ void reduce_save_sum(float* dst_ptr, float sum) {
    sum = warp_reduce_sum<false>(sum);
    //auto warp_id = threadIdx.x >> 6;
    auto lane_id = threadIdx.x & 63;
    if (lane_id == (warpSize-1)) {
        // this atomic add slower down the kernel by a lot, so only save_sum when checking accuracy
        __atomic_add_fetch(dst_ptr, sum, __ATOMIC_RELAXED);
    }
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
        nc += v4;
    }
    if (save_sum) reduce_save_sum(d_sum, nc.x + nc.y + nc.z + nc.w);
}

__global__ void sum_1d_x2(float* d_sum, const float* d_in, uint numElements, int save_sum) {
    float2 const *vectorized_in = reinterpret_cast<float2 const *>(d_in);

    auto numVectElements = (numElements >> 1);

    uint tid = (blockIdx.x * blockDim.x + threadIdx.x);
    uint grid_tid_size = blockDim.x * gridDim.x;
    float2 nc = make_float2(0.0f,0.0f);
    for (;tid < numVectElements; tid += grid_tid_size) {
        auto va = vectorized_in[tid];
        nc += va;
    }
    if (save_sum) reduce_save_sum(d_sum, nc.x + nc.y);
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
        nc += v4;
    }
    if (save_sum) reduce_save_sum(d_sum, nc.x + nc.y + nc.z + nc.w);
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
        nc += v4;
    }
    if (save_sum) reduce_save_sum(d_sum, nc.x + nc.y + nc.z + nc.w);
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
        nc += v4;
    }
    if (save_sum) reduce_save_sum(d_sum, nc.x + nc.y + nc.z + nc.w);
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
        nc += v4;
    }
    if (save_sum) reduce_save_sum(d_sum, nc.x + nc.y + nc.z + nc.w);
}


/*
evenly divide memory into pieces, each warp do their own part
*/
__global__ void sum_1d_x4e(float* d_sum, const float* d_in, uint numElements, int save_sum) {
    float4 const *vectorized_in = reinterpret_cast<float4 const *>(d_in);

    auto numVectElements = (numElements >> 2);

    auto lane_id = threadIdx.x & 63; // 0...63
    auto warp_id = threadIdx.x >> 6; // 0,1,2,3
    auto warp_cnt = blockDim.x >> 6; // 4
    auto total_warps = gridDim.x * warp_cnt;
    auto warpVectElements = numVectElements / total_warps;

    // permutation of 64 lanes do not hurt coalescing performance
    uint tid = (blockIdx.x*warp_cnt + warp_id) * warpVectElements + (lane_id^57);

    float4 nc = make_float4(0.0f,0.0f,0.0f,0.0f);
    for (int i = 0; i < warpVectElements; i+=64) {
        auto v4 = vectorized_in[tid + i];
        nc += v4;
    }
    if (save_sum) reduce_save_sum(d_sum, nc.x + nc.y + nc.z + nc.w);
}

__global__ void sum_1d_x4e2(float* d_sum, const float* d_in, uint numElements, int save_sum) {
    float4 const *vectorized_in = reinterpret_cast<float4 const *>(d_in);

    auto numVectElements = (numElements >> 2);

    auto lane_id = threadIdx.x & 63; // 0...63
    auto warp_id = threadIdx.x >> 6; // 0,1,2,3
    auto warp_cnt = blockDim.x >> 6; // 4
    auto total_warps = gridDim.x * warp_cnt;
    auto warpVectElements = numVectElements / total_warps;

    // permutation of 64 lanes do not hurt coalescing performance
    uint tid = (blockIdx.x*warp_cnt + warp_id) * warpVectElements + (lane_id^57);

    float4 nc = make_float4(0.0f,0.0f,0.0f,0.0f);
    for (int i = 0; i < warpVectElements; i+=512, tid += 512) {
        auto v40 = vectorized_in[tid];
        auto v41 = vectorized_in[tid + 64];
        auto v42 = vectorized_in[tid + 64*2];
        auto v43 = vectorized_in[tid + 64*3];
        auto v44 = vectorized_in[tid + 64*4];
        auto v45 = vectorized_in[tid + 64*5];
        auto v46 = vectorized_in[tid + 64*6];
        auto v47 = vectorized_in[tid + 64*7];
        nc += v40;
        nc += v41;
        nc += v42;
        nc += v43;
        nc += v44;
        nc += v45;
        nc += v46;
        nc += v47;
    } 
    if (save_sum) reduce_save_sum(d_sum, nc.x + nc.y + nc.z + nc.w);
}

// 
// https://github.com/mk1-project/quickreduce/blob/main/csrc/core/buffer.h
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using float32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) float;
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

__device__ __inline__
static float32x4_t buffer_load_dwordx4(int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

//  %raw_buffer0 = call <4 x float> @llvm.amdgcn.raw.buffer.load.v4f32(<4 x i32> %tmp0, i32 128, i32 0, i32 0) #0
//   call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc,
//       ptr addrspace(3) @lds.0,  // LDS base offset
//       i32 4,                    // Data byte size: 1/2/4 (/12/16 for gfx950)
//       i32 0,                    // voffset(VGPR, included in bounds checking and swizzling)
//       i32 0,                    // soffset(SGPR/imm, excluded from bounds checking and swizzling)
//       i32 0,                    // imm offset(imm, included in bounds checking and swizzling)
//       i32 0                     // auxiliary/cachepolicy(imm):
// )

__device__ __inline__
static void buffer_store_dwordx4(int32x4_t data,
                        int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__global__ void sum_1d_buffx4(float* d_sum, float* d_in, uint numElements, int save_sum) {
    BufferResource buffer(d_in, numElements * sizeof(float));

    float4 const *vectorized_in = reinterpret_cast<float4 const *>(d_in);

    auto numVectElements = (numElements >> 2);

    uint tid = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // permutation of 64 lanes do not hurt coalescing performance
    //tid = ((tid & 63)^(57)) | (tid & ~63);

    uint grid_tid_size = blockDim.x * gridDim.x;
    float32x4_t nc = {0};
    for (;tid < numVectElements; tid += grid_tid_size) {
        // auto v4 = vectorized_in[tid];
        auto vf4 = buffer_load_dwordx4(buffer.descriptor, tid*16, 0, 0);
        nc += vf4;
    }
    if (save_sum) reduce_save_sum(d_sum, nc[0] + nc[1] + nc[2] + nc[3]);
}
