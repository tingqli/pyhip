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
union float16x4_x2 {
    float16x4 f4[2];
};
extern __shared__ __align__(1024) __fp16 lds_mem[];
// https://github.com/ROCm/composable_kernel/blob/develop/include/ck_tile/core/arch/amd_buffer_addressing.hpp
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
static uint buffer_load_dword(int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i32");

using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;

__device__ __inline__
void llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,    //
                                as3_uint32_ptr lds_ptr, // LDS base offset
                                int32_t size,           // Data byte size: 1/2/4 (/12/16 for gfx950)
                                int32_t voffset,        // voffset(VGPR, included in bounds checking and swizzling)
                                int32_t soffset,        // soffset(SGPR/imm, excluded from bounds checking and swizzling)
                                int32_t offset,         // imm offset(imm, included in bounds checking and swizzling)
                                int32_t aux             // auxiliary/cachepolicy(imm):
                            ) __asm("llvm.amdgcn.raw.buffer.load.lds");


__global__ void fake_mm_base(float * C, __fp16 * A, __fp16 * B, int oc_blocks) {
    constexpr int K = 512;
    // load A from HBM into VGPRs
    float16x4 regA[K/16];
    int lane = threadIdx.x & 63;
    auto i_off = (lane & 15) * K;
    auto k_off = (lane >> 4) * 4;
    auto ik_off = i_off + k_off;
    for(int ik = 0; ik < K/16; ik ++)
        regA[ik] = *(float16x4*)(A + ik_off + ik*16);

    //__shared__ __fp16 lds_mem[2*16*K];

    float32x4 c = {0};
    for(int ioc = 0; ioc < oc_blocks; ioc ++, B += 16*K) {
        for(int ik = 0; ik < K/16; ik ++) {
            auto regB = *(float16x4*)(B + ik_off + ik*16);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regB, regA[ik], c, 0, 0, 0);
        }
    }

    *(float32x4*)(C + ((lane & 15))*16 + k_off) = c;
}


__global__ void fake_mm_lds(float * C, __fp16 * A, __fp16 * B, int oc_blocks) {
    constexpr int K = 512;
    // load A from HBM into VGPRs
    float16x4 regA[K/16];
    int lane = threadIdx.x & 63;
    auto i_off = (lane & 15) * K;
    auto k_off = (lane >> 4) * 4;
    auto ik_off = i_off + k_off;
    for(int ik = 0; ik < K/16; ik ++)
        regA[ik] = *(float16x4*)(A + ik_off + ik*16);

    //__shared__ __fp16 lds_mem[2*16*K];

    float32x4 c = {0};
    for(int ioc = 0; ioc < oc_blocks; ioc ++, B += 16*K) {
        // load into LDS in a way similar to BUFFER_LOAD_DWORD lds
        for(int m = 0; m < 16; m++) {
            for(int k = 0; k < K; k += 64*2) {
                // load 64 x DWORD
                auto regB = ((uint*)(B + m*K + k))[lane];
                ((uint*)(lds_mem + m*K + k))[lane] = regB;
            }
        }
        for(int ik = 0; ik < K/16; ik ++) {
            auto regB = *(float16x4*)(lds_mem + ik_off + ik*16);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regB, regA[ik], c, 0, 0, 0);
        }
    }

    *(float32x4*)(C + ((lane & 15))*16 + k_off) = c;
}

/*
b128 : interleave two 16x16 B matrix so ds_read_b128() can load two 16x16 B matrix
ds_read_b128() exists for a reason, it is much faster than ds_read2_b64()

a lot of VGPRs were allocated for being used as pipeline buffer in MEM=>LDS loading

VGPRs: 97

*/
__global__ void fake_mm_b128(float * C, __fp16 * A, __fp16 * B, int oc_blocks) {
    constexpr int K = 512;
    // load A from HBM into VGPRs
    float16x4 regA[K/16];
    int lane = threadIdx.x & 63;
    auto i_off = (lane & 15) * K;
    auto k_off = (lane >> 4) * 4;
    auto ik_off = i_off + k_off;
    for(int ik = 0; ik < K/16; ik ++)
        regA[ik] = *(float16x4*)(A + ik_off + ik*16);

    //__shared__ __fp16 lds_mem[2*16*K];

    float32x4 c = {0};
    const int table[] = {
        0,1,8,9,
        2,3,10,11,
        4,5,12,13,
        6,7,14,15
    };
    auto interleave_lane = (lane & 0xF0) | table[lane & 15];
    auto ik_off2 = i_off + (lane >> 4) * 8;

    for(int ioc = 0; ioc < oc_blocks; ioc ++, B += 16*K) {
        // load into LDS in a way similar to BUFFER_LOAD_DWORD lds
        for(int m = 0; m < 16; m++) {
            for(int k = 0; k < K; k += 64*2) {
                // load 64 x DWORD
                auto regB = ((uint*)(B + m*K + k))[interleave_lane];
                ((uint*)(lds_mem + m*K + k))[lane] = regB;
            }
        }
        for(int ik = 0; ik < K/16; ik +=2) {
            auto regBx2 = *(float16x4_x2*)(lds_mem + ik_off2 + ik*16);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regBx2.f4[0], regA[ik], c, 0, 0, 0);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regBx2.f4[1], regA[ik + 1], c, 0, 0, 0);
        }
    }

    *(float32x4*)(C + ((lane & 15))*16 + k_off) = c;
}

/*
use buffer load to LDS to reduce VGPR resource usage
*/

__global__ void fake_mm_bload_b128(float * C, __fp16 * A, __fp16 * B, int oc_blocks) {
    constexpr int K = 512;
    // load A from HBM into VGPRs
    float16x4 regA[K/16];
    int lane = threadIdx.x & 63;
    auto i_off = (lane & 15) * K;
    auto k_off = (lane >> 4) * 4;
    auto ik_off = i_off + k_off;
    for(int ik = 0; ik < K/16; ik ++)
        regA[ik] = *(float16x4*)(A + ik_off + ik*16);

    //__shared__ __fp16 lds_mem[16*K];

    float32x4 c = {0};
    const int table[] = {
        0,1,8,9,
        2,3,10,11,
        4,5,12,13,
        6,7,14,15
    };
    uint interleave_lane = (lane & 0xF0) | table[lane & 15];
    auto ik_off2 = i_off + (lane >> 4) * 8;
    uint mem_offset = 0;
    BufferResource buffB(B, oc_blocks*16*K*sizeof(__fp16));

    for(int ioc = 0; ioc < oc_blocks; ioc ++, mem_offset += 16*K*sizeof(__fp16)) {
        //__syncthreads();
        // load into LDS in a way similar to BUFFER_LOAD_DWORD lds
        __fp16 * __restrict__ lds_write = lds_mem;
        uint soffset = mem_offset;
        for(int m = 0; m < 16; m++, soffset += K*sizeof(__fp16), lds_write += 64*2*4) {
            static_assert(K / (64*2) == 4);
            // for(int k = 0; k < K; k += 64*2, soffset += 64*2*sizeof(__fp16), lds_write += 64*2)
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 0, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256*2, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256*3, 0);
        }

        for(int ik = 0; ik < K/16; ik +=2) {
            auto regBx2 = *(float16x4_x2*)(lds_mem + ik_off2 + ik*16);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regBx2.f4[0], regA[ik], c, 0, 0, 0);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regBx2.f4[1], regA[ik + 1], c, 0, 0, 0);
        }
    }

    *(float32x4*)(C + ((lane & 15))*16 + k_off) = c;
}


/*
groupped padding can avoid bank-conflict in a way simpler than swizzle and consumes
much less VGPRs for different swizzling patterns of each row.
*/
__global__ void fake_mm_bload_b128_group_padding(float * C, __fp16 * A, __fp16 * B, int oc_blocks) {
    constexpr int K = 512;
    // load A from HBM into VGPRs
    float16x4 regA[K/16];
    int lane = threadIdx.x & 63;
    auto i_off = (lane & 15) * K;
    auto k_off = (lane >> 4) * 4;
    auto ik_off = i_off + k_off;
    for(int ik = 0; ik < K/16; ik ++)
        regA[ik] = *(float16x4*)(A + ik_off + ik*16);

    constexpr int padding_fp16 = 3*4*16;
    //__shared__ __fp16 lds_mem[16*K + padding_fp16];

    float32x4 c = {0};
    const int table[] = {
        0,1,8,9,
        2,3,10,11,
        4,5,12,13,
        6,7,14,15
    };

    // for pack two 16x16 into 16x32, so bs_read_b128() can read them once for both
    uint interleave_lane = (lane & 0xF0) | table[lane & 15];
    auto ik_off2 = i_off + (lane >> 4) * 8;
    uint mem_offset = 0;
    BufferResource buffB(B, oc_blocks*16*K*sizeof(__fp16));

    uint lds_read_idx;
    {
        uint m = lane & 15;
        uint padding_fp16s = (m & 3)*4*4;
        uint lds_row = (m & 3)*4 + (m>>2);
        lds_read_idx = lds_row * K + padding_fp16s + (lane >> 4)*8;
    }

    for(int ioc = 0; ioc < oc_blocks; ioc ++, mem_offset += 16*K*sizeof(__fp16)) {
        //__syncthreads();
        // load into LDS in a way similar to BUFFER_LOAD_DWORD lds
        
        uint soffset = mem_offset;
        for(int m = 0; m < 16; m++, soffset += K*sizeof(__fp16)) {
            static_assert(K / (64*2) == 4);
            // groupped padding
            uint padding_fp16s = (m & 3)*4*4;
            uint lds_row = (m & 3)*4 + (m>>2);
            __fp16 * __restrict__ lds_write = lds_mem + lds_row * K + padding_fp16s;

            // for(int k = 0; k < K; k += 64*2, soffset += 64*2*sizeof(__fp16), lds_write += 64*2)
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 0, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256*2, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256*3, 0);
        }

        for(int ik = 0; ik < K/16; ik +=2) {
            auto regBx2 = *(float16x4_x2*)(lds_mem + lds_read_idx + ik*16);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regBx2.f4[0], regA[ik], c, 0, 0, 0);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regBx2.f4[1], regA[ik + 1], c, 0, 0, 0);
        }
    }

    *(float32x4*)(C + ((lane & 15))*16 + k_off) = c;
}


__global__ void fake_mm_bload_b128_group_padding_pipeline(float * C, __fp16 * A, __fp16 * B, int oc_blocks) {
    constexpr int K = 512;
    // load A from HBM into VGPRs
    float16x4 regA[K/16];
    int lane = threadIdx.x & 63;
    auto i_off = (lane & 15) * K;
    auto k_off = (lane >> 4) * 4;
    auto ik_off = i_off + k_off;
    for(int ik = 0; ik < K/16; ik ++)
        regA[ik] = *(float16x4*)(A + ik_off + ik*16);

    constexpr uint padding_fp16 = 3*4*16;
    constexpr uint LDS_buff_size = (16*K + padding_fp16);
    //__shared__ __fp16 lds_mem [LDS_buff_size * 2];

    float32x4 c = {0};
    const int table[] = {
        0,1,8,9,
        2,3,10,11,
        4,5,12,13,
        6,7,14,15
    };

    // for pack two 16x16 into 16x32, so bs_read_b128() can read them once for both
    uint interleave_lane = (lane & 0xF0) | table[lane & 15];
    auto ik_off2 = i_off + (lane >> 4) * 8;
    uint mem_offset = 0;
    BufferResource buffB(B, oc_blocks*16*K*sizeof(__fp16));

    uint lds_read_idx;
    {
        uint m = lane & 15;
        uint padding_fp16s = (m & 3)*4*4;
        uint lds_row = (m & 3)*4 + (m>>2);
        lds_read_idx = lds_row * K + padding_fp16s + (lane >> 4)*8;
    }

    {
        __fp16 * lds_w = lds_mem;
        uint soffset = mem_offset;
        for(int m = 0; m < 16; m++, soffset += K*sizeof(__fp16)) {
            static_assert(K / (64*2) == 4);
            // groupped padding
            uint padding_fp16s = (m & 3)*4*4;
            uint lds_row = (m & 3)*4 + (m>>2);
            __fp16 * __restrict__ lds_write = lds_w + lds_row * K + padding_fp16s;

            // for(int k = 0; k < K; k += 64*2, soffset += 64*2*sizeof(__fp16), lds_write += 64*2)
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 0, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256*2, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256*3, 0);
        }
    }

    for(int ioc = 0; ioc < oc_blocks; ioc ++, mem_offset += 16*K*sizeof(__fp16)) {
        __syncthreads();
        // load into LDS in a way similar to BUFFER_LOAD_DWORD lds
        __fp16 * __restrict__ lds_w = lds_mem + (1-(ioc&1)) * LDS_buff_size;
        __fp16 * __restrict__ lds_r = lds_mem + ((ioc&1)) * LDS_buff_size + lds_read_idx;
 
        uint soffset = mem_offset;
        for(int m = 0, ik=0; m < 16; m++, soffset += K*sizeof(__fp16), ik+= 2) {
            static_assert(K / (64*2) == 4);
            // groupped padding
            uint padding_fp16s = (m & 3)*4*4;
            uint lds_row = (m & 3)*4 + (m>>2);
            __fp16 * __restrict__ lds_write = lds_w + lds_row * K + padding_fp16s;

            // for(int k = 0; k < K; k += 64*2, soffset += 64*2*sizeof(__fp16), lds_write += 64*2)
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 0, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256*2, 0);
            llvm_amdgcn_raw_buffer_load_lds(buffB.descriptor, (as3_uint32_ptr)lds_write, 4, interleave_lane*4, soffset, 256*3, 0);

            auto regBx2 = *(float16x4_x2*)(lds_r + ik*16);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regBx2.f4[0], regA[ik], c, 0, 0, 0);
            c = __builtin_amdgcn_mfma_f32_16x16x16f16(regBx2.f4[1], regA[ik + 1], c, 0, 0, 0);
        }
    }

    *(float32x4*)(C + ((lane & 15))*16 + k_off) = c;
}
