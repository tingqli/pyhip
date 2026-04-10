#include <__clang_hip_runtime_wrapper.h>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using float16x4 = __fp16 __attribute__ ((ext_vector_type(2)));
using float16x8 = __fp16 __attribute__ ((ext_vector_type(8)));
using float16x32 = __fp16 __attribute__ ((ext_vector_type(32)));
using bfloat16x2 = __bf16  __attribute__ ((ext_vector_type(2)));
using bfloat16x4 = __bf16  __attribute__ ((ext_vector_type(4)));
using bfloat16x8 = __bf16  __attribute__ ((ext_vector_type(8)));
using bfloat16x32 = __bf16 __attribute__ ((ext_vector_type(32)));
using float32x16 = float  __attribute__ ((ext_vector_type(16)));
using float32x4 = float  __attribute__ ((ext_vector_type(4)));
using float32x2 = float  __attribute__ ((ext_vector_type(2)));
using float32x8 = float  __attribute__ ((ext_vector_type(8)));
using int32x4_t = int  __attribute__ ((ext_vector_type(4)));
using int32x16_t = int  __attribute__ ((ext_vector_type(16)));
using uint32x2_t = uint  __attribute__ ((ext_vector_type(2)));
using charx16 = char __attribute__ ((ext_vector_type(16)));

using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;
union BufferResource {
    __device__ __inline__ constexpr BufferResource()
        : config(0x00020000U) {}

    __device__ __inline__ constexpr BufferResource(void* buffer_address, uint32_t buffer_size)
        : address(buffer_address),
          range(buffer_size),
          config(0x00020000U) {}
    
    __device__ __inline__ constexpr void set_base(void* buffer_address) {
        address = buffer_address;
    }

    int32x4_t descriptor;
    struct{
        void* address;      // 8B, out of which first 48b is address, and 16b is stride (unused)
        uint32_t range;     // Byte range for the buffer resource
        uint32_t config;    // Constant, DFMT=32b
    };
};

using DWORDX4 = int32x4_t;


template <typename Object, std::enable_if_t<std::is_trivially_copyable_v<Object>, int> = 0>
__device__ inline auto amd_wave_read_first_lane(const Object& obj)
{
    constexpr size_t ObjectSize = sizeof(Object);
    constexpr size_t SGPR_size  = 4;
    constexpr size_t NumFull    = ObjectSize / SGPR_size;
    constexpr size_t Tail       = ObjectSize % SGPR_size;

    const unsigned char* src = reinterpret_cast<const unsigned char*>(&obj);
    alignas(Object) unsigned char dst[ObjectSize];

    #pragma unroll
    for (size_t Ic = 0; Ic < NumFull; Ic++) {
        size_t offset = Ic * SGPR_size;
        uint32_t read_src;
        __builtin_memcpy(&read_src, src + offset, SGPR_size);
        read_src = __builtin_amdgcn_readfirstlane(read_src);
        __builtin_memcpy(dst + offset, &read_src, SGPR_size);
    }

    if constexpr(Tail != 0)
    {
        constexpr size_t offset = NumFull * SGPR_size;
        uint32_t tail_loc       = 0;
        __builtin_memcpy(&tail_loc, src + offset, Tail);
        tail_loc = __builtin_amdgcn_readfirstlane(tail_loc);
        __builtin_memcpy(dst + offset, &tail_loc, Tail);
    }
    Object out;
    __builtin_memcpy(&out, dst, ObjectSize);
    return out;
}

__device__ DWORDX4 read_dwordx4(BufferResource& buffer, int soffset, int nstride) {
    DWORDX4 data;
    //volatile int dummy[40] = {0};
    int lane_offset = threadIdx.x * sizeof(DWORDX4);
    int soff = soffset;
    auto r = amd_wave_read_first_lane(buffer.descriptor);
    asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
        :[vdst]"=v"(data)
        :[vaddr]"v"(lane_offset), [srsrc]"s"(r), [soffset]"s"(soff));
    return data;
}

template<typename T>
__device__ T buffer_load_dwordx4(BufferResource& buffer, int soffset, int voffset, bool is_asm=false) {
    T v;
    if (is_asm) {
        auto r = amd_wave_read_first_lane(buffer.descriptor);
        asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
            :[vdst]"=v"(v)
            :[vaddr]"v"(voffset), [srsrc]"s"(r), [soffset]"s"(soffset));
    } else {
        v = *(T*)((char*)buffer.address + soffset + voffset);
    }
    return v;
}

__device__ float llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 int voffset,
                                 int soffset,
                                 int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ __inline__
void llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,    //
                                as3_uint32_ptr lds_ptr, // LDS base offset
                                int32_t size,           // Data byte size: 1/2/4 (/12/16 for gfx950)
                                int32_t voffset,        // voffset(VGPR, included in bounds checking and swizzling)
                                int32_t soffset,        // soffset(SGPR/imm, excluded from bounds checking and swizzling)
                                int32_t offset,         // imm offset(imm, included in bounds checking and swizzling)
                                int32_t aux             // auxiliary/cachepolicy(imm):
                            ) __asm("llvm.amdgcn.raw.buffer.load.lds");

template<typename T>
__device__ T buffer_load_dword(BufferResource& buffer, int soffset, int voffset, int coffset, bool is_asm=false) {
    T v;
    if (is_asm) {
        int32x4_t r = __builtin_bit_cast(int32x4_t, buffer);
        r.x         = __builtin_amdgcn_readfirstlane(r.x);
        r.y         = __builtin_amdgcn_readfirstlane(r.y);
        r.z         = __builtin_amdgcn_readfirstlane(r.z);
        r.w         = __builtin_amdgcn_readfirstlane(r.w);

        //auto r = amd_wave_read_first_lane(buffer.descriptor);
        asm volatile("buffer_load_dword %[vdst], %[vaddr], %[srsrc], 0 offen offset:%[coffset]\n"
            :[vdst]"=v"(v)
            :[vaddr]"v"(voffset), [srsrc]"s"(r), [coffset]"n"(coffset)
            : "memory");
    } else {
        int32x4_t r = __builtin_bit_cast(int32x4_t, buffer);
        auto d = llvm_amdgcn_raw_buffer_load_fp32(r, voffset + coffset, soffset, 0);
        v = __builtin_bit_cast(T, d);
        //v = *(T*)((char*)buffer.address + soffset + voffset + coffset);
    }
    return v;
}

template<typename T>
__device__ T global_load_dwordx4(void* addr, uint64_t voffset, bool is_asm=false) {
    T v;
    if (is_asm) {
        asm volatile("global_load_dwordx4 %[vdst], %[vaddr], off\n"
                :[vdst]"=v"(v)
                :[vaddr]"v"((char*)addr + voffset));
    } else {
        v = *(T*)((char*)addr + voffset);
    }
    return v;
}

__device__ void amdgcn_mfma_f32_16x16x16bf16(bfloat16x4 a, bfloat16x4 b, float32x4& c) {
    c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
    // asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %3\n"
    //             : "+v"(c)
    //             : "v"(a), "v"(b), "v"(c)
    //             :);
}

template<uint16_t cnt>
__device__ void s_waitcnt_vmcnt() {
    asm volatile ("s_waitcnt vmcnt(%0)\n"::"i"(cnt));
}
template<uint16_t cnt>
__device__ void s_waitcnt_lgkmcnt() {
    asm volatile ("s_waitcnt lgkmcnt(%0)\n"::"i"(cnt));
}


template<typename TD, typename TS>
__device__ TD& cast(TS& t) {
    return (TD&)t;
}


template<typename F, size_t... Is>
constexpr void compile_time_loop_impl(F&& f, std::index_sequence<Is...>) {
    (f(std::integral_constant<size_t, Is>{}), ...);
}

template<size_t N, typename F>
constexpr void compile_time_loop(F&& f) {
    compile_time_loop_impl(std::forward<F>(f), 
                          std::make_index_sequence<N>{});
}

#ifndef TOPK
#define BLOCK_SIZE_M 256
#define TOPK 20
#define ROW_PER_BLOCK 4
#define NUM_THREADS 64
#endif

//__launch_bounds__(NUM_THREADS, 2) 
__global__ void quant(
            __bf16* x,                  // [M, S]
            float* smooth_scale,        // [E, S]
            char* x_out,                // [M, TOPK, S]
            float* x_quant_scale,       // [sorted_expert_ids.shape[0], 256]
            uint* sorted_ids,           // [sorted_expert_ids.shape[0], 256]
            uint* sorted_expert_ids,    // [sorted_expert_ids.shape[0]]
            uint* num_valid_ids,        // [2]
            uint M, uint S, int is_gemm1) {
    // wg: [sorted_expert_ids.shape[0], 64]
    uint e_idx = blockIdx.x;
    uint r_idx = blockIdx.y;
    uint eid = sorted_expert_ids[e_idx];
    if (e_idx * BLOCK_SIZE_M >= num_valid_ids[0]) return;
    sorted_ids += e_idx * BLOCK_SIZE_M;
    smooth_scale += eid * S;
    __shared__ __bf16 lds[4096];

    for (int r = 0; r < ROW_PER_BLOCK; r++) {
        uint row_id = r_idx * ROW_PER_BLOCK + r;
        auto raw_id = sorted_ids[row_id];
        int64_t tok_id = raw_id & 0xffffff;
        auto topk_id = raw_id >> 24;
        // if (e_idx==36 &&threadIdx.x == 0) {
        //     printf("e_idx %d, row_id %d, raw_id %d, tok_id %ld, topk_id %d\n", e_idx, row_id, raw_id, tok_id, topk_id);
        // }
        if (tok_id >= M) return;

        auto p_x = is_gemm1 ? x + tok_id * S : x + tok_id * TOPK * S + topk_id * S;
        float cur_max = -FLT_MAX;
        for (int i = threadIdx.x; i < S / 8; i += NUM_THREADS) {
            auto scale = global_load_dwordx4<float32x8>(smooth_scale, i * 8 * sizeof(float));
            auto x_val = global_load_dwordx4<bfloat16x8>(p_x, i * 8 * sizeof(__bf16));
            for (int j = 0; j < 8; j++) {
                auto tmp = (scale[j] * (float)x_val[j]);
                lds[i * 8 + j] = tmp;
                cur_max = fmaxf(cur_max, abs(tmp));
            }
        }
        for(int mask = 64 / 2; mask >= 1; mask /= 2) {
            cur_max = fmaxf(cur_max, __shfl_xor(cur_max, mask));
        }
        __syncthreads();
        // if (e_idx==204 &&threadIdx.x == 0) {
        //     auto offset = tok_id * TOPK * S + topk_id * S;
        //     printf("e_idx %d, eid=%d row_id %d, tok_id %ld, topk_id %d, max %f,\n", e_idx, eid, row_id, tok_id, topk_id, cur_max);
        // }
        auto row_scale = cur_max / 128.0f;
        row_scale = fmaxf(row_scale, 1e-6f);
        auto inv_row_scale = 1.0f / row_scale;
        // if (e_idx==21 &&threadIdx.x == 0) {
        //     auto offset = tok_id * TOPK * S + topk_id * S;
        //     printf("e_idx %d, row_id %d, tok_id %ld, topk_id %d, max %f, scale %f\n", e_idx, row_id, tok_id, topk_id, cur_max, row_scale);
        // }
        charx16* p_x_out = reinterpret_cast<charx16*>(x_out + tok_id * TOPK * S + topk_id * S);
        for (int i = threadIdx.x; i < S / 16; i += NUM_THREADS) {
            charx16 qv;
            for (int j = 0; j < 16; j++) {
                auto val = lds[i * 16 + j] * inv_row_scale;
                qv[j] = max(-128, min(127, (int)roundf(val)));
            }
            p_x_out[i] = qv;
        }
        if (threadIdx.x == 0) {
            x_quant_scale[e_idx * BLOCK_SIZE_M + row_id] = row_scale;
        }
    }
}

#ifndef ROW_PER_BLOCK2
#define ROW_PER_BLOCK2 4
#endif
#define COL_PER_WAVE (64 / ROW_PER_BLOCK2)
#define NUM_THREADS2 256
__global__ void quant2(
            __bf16* x,                  // [M, S]
            float* smooth_scale,        // [E, S]
            char* x_out,                // [M, TOPK, S]
            float* x_quant_scale,       // [sorted_expert_ids.shape[0], 256]
            uint* sorted_ids,           // [sorted_expert_ids.shape[0], 256]
            uint* sorted_expert_ids,    // [sorted_expert_ids.shape[0]]
            uint* num_valid_ids,        // [2]
            uint M, uint S, int is_gemm1) {
    // wg: [sorted_expert_ids.shape[0], 32]
    uint e_idx = blockIdx.x;
    uint r_idx = blockIdx.y;
    uint eid = sorted_expert_ids[e_idx];
    if (e_idx * BLOCK_SIZE_M >= num_valid_ids[0]) return;
    sorted_ids += e_idx * BLOCK_SIZE_M;
    S = 1536;
    __shared__ __bf16 lds[ROW_PER_BLOCK2][1536];
    __shared__ float max_wave[ROW_PER_BLOCK2][4];

    uint lane_id = threadIdx.x % 64;
    uint wave_id = threadIdx.x / 64;
    uint lane_r = lane_id / COL_PER_WAVE;
    uint lane_c = lane_id % COL_PER_WAVE;
    uint row_id = r_idx * ROW_PER_BLOCK2 + lane_r;
    auto raw_id = sorted_ids[row_id];
    auto tok_id = raw_id & 0xffffff;
    auto topk_id = raw_id >> 24;
    auto org_tok_id = tok_id;
    tok_id = tok_id >= M ? 0 : tok_id; // oob
    smooth_scale += eid * S;
    auto p_x = x + (int64_t)tok_id * TOPK * S + topk_id * S;
    float cur_max = -FLT_MAX;
    int i = lane_c + wave_id * COL_PER_WAVE;
    auto scale = global_load_dwordx4<float32x8>(smooth_scale, i * 8 * sizeof(float));
    auto x_val = global_load_dwordx4<bfloat16x8>(p_x, i * 8 * sizeof(__bf16));
    for (; i < S / 8 - COL_PER_WAVE * 4; i += COL_PER_WAVE * 4) {
        auto next_scale = global_load_dwordx4<float32x8>(smooth_scale, (i + COL_PER_WAVE * 4) * 8 * sizeof(float));
        auto next_x_val = global_load_dwordx4<bfloat16x8>(p_x, (i + COL_PER_WAVE * 4) * 8 * sizeof(__bf16));
        for (int j = 0; j < 8; j++) {
            auto tmp = (scale[j] * (float)x_val[j]);
            lds[lane_r][i * 8 + j] = tmp;
            cur_max = fmaxf(cur_max, abs(tmp));
        }
        scale = next_scale;
        x_val = next_x_val;
    }
    for (int j = 0; j < 8; j++) {
        auto tmp = (scale[j] * (float)x_val[j]);
        lds[lane_r][i * 8 + j] = tmp;
        cur_max = fmaxf(cur_max, abs(tmp));
    }
    for(int mask = COL_PER_WAVE / 2; mask >= 1; mask /= 2) {
        cur_max = fmaxf(cur_max, __shfl_xor(cur_max, mask));
    }
    if (lane_c == 0) {
        max_wave[lane_r][wave_id] = cur_max;
    }
    __syncthreads();
    for (int i = 0; i < 4; i++) {
        cur_max = fmaxf(cur_max, max_wave[lane_r][i]);
    }
    if (org_tok_id >= M) return; // oob write
    auto row_scale = cur_max / 128.0f;
    row_scale = fmaxf(row_scale, 1e-6f);
    auto inv_row_scale = 1.0f / row_scale;
    // if (e_idx==1096 &&row_id == 166) {
    //     auto offset = tok_id * TOPK * S + topk_id * S;
    //     printf("x %d lanec %d wave_id %d e_idx %d, row_id %d, tok_id %ld, topk_id %d, max %f org_tok_id %d row_scale %f\n", threadIdx.x, lane_c, wave_id, e_idx, row_id, tok_id, topk_id, cur_max, org_tok_id, row_scale);
    // }

    charx16* p_x_out = reinterpret_cast<charx16*>(x_out + tok_id * TOPK * S + topk_id * S);
    #pragma unroll
    for (int i = lane_c + wave_id * COL_PER_WAVE; i < S / 16; i += COL_PER_WAVE * 4) {
        charx16 qv;
        for (int j = 0; j < 16; j++) {
            auto val = lds[lane_r][i * 16 + j] * inv_row_scale;
            qv[j] = max(-128, min(127, (int)roundf(val)));
        }
        p_x_out[i] = qv;
    }
    if (lane_c == 0 && wave_id == 0) {
        x_quant_scale[e_idx * BLOCK_SIZE_M + row_id] = row_scale;
    }
}

__global__ void quant1(
            __bf16* x,                  // [M, S]
            float* smooth_scale,        // [E, S]
            char* x_out,                // [M, TOPK, S]
            float* x_quant_scale,       // [M, TOPK]
            uint* expert_ids,           // [M, TOPK]
            uint M, uint S, int is_gemm1) {
    // wg: [M//4]
    uint m_block = blockIdx.x;
    uint lane_id = threadIdx.x % 64;
    uint wave_id = threadIdx.x / 64;
    uint lane_r = lane_id / COL_PER_WAVE;
    uint lane_c = lane_id % COL_PER_WAVE;
    S = 4096;
    __shared__ __bf16 lds[4096];
    __shared__ bfloat16x8 lds_tok[4096 / 8];

    auto loop_row = [&] (__bf16* p_x, uint token_id, uint topk_id) {
        auto p_smooth_scale = smooth_scale + expert_ids[token_id * TOPK + topk_id] * S;
        float cur_max = -FLT_MAX;
        # pragma unroll
        for (int i = threadIdx.x; i < S / 8; i += 64) {
            auto scale = global_load_dwordx4<float32x8>(p_smooth_scale, i * 8 * sizeof(float));
            bfloat16x8 x_val;
            if (topk_id == 0) {
                x_val = global_load_dwordx4<bfloat16x8>(p_x, i * 8 * sizeof(__bf16));
                lds_tok[i] = x_val;
            }
            else
                x_val = lds_tok[i];
            for (int j = 0; j < 8; j++) {
                auto tmp = (scale[j] * (float)x_val[j]);
                lds[i * 8 + j] = tmp;
                cur_max = fmaxf(cur_max, abs(tmp));
            }
        }
        for(int mask = 64 / 2; mask >= 1; mask /= 2) {
            cur_max = fmaxf(cur_max, __shfl_xor(cur_max, mask));
        }
        // if (e_idx==204 &&threadIdx.x == 0) {
        //     auto offset = tok_id * TOPK * S + topk_id * S;
        //     printf("e_idx %d, eid=%d row_id %d, tok_id %ld, topk_id %d, max %f,\n", e_idx, eid, row_id, tok_id, topk_id, cur_max);
        // }
        auto row_scale = cur_max / 128.0f;
        row_scale = fmaxf(row_scale, 1e-6f);
        auto inv_row_scale = 1.0f / row_scale;
        // if (e_idx==21 &&threadIdx.x == 0) {
        //     auto offset = tok_id * TOPK * S + topk_id * S;
        //     printf("e_idx %d, row_id %d, tok_id %ld, topk_id %d, max %f, scale %f\n", e_idx, row_id, tok_id, topk_id, cur_max, row_scale);
        // }
        charx16* p_x_out = reinterpret_cast<charx16*>(x_out + token_id * TOPK * S + topk_id * S);
        for (int i = threadIdx.x; i < S / 16; i += 64) {
            charx16 qv;
            for (int j = 0; j < 16; j++) {
                auto val = lds[i * 16 + j] * inv_row_scale;
                qv[j] = max(-128, min(127, (int)roundf(val)));
            }
            p_x_out[i] = qv;
        }
        // if (threadIdx.x == 0) {
        //     x_quant_scale[e_idx * BLOCK_SIZE_M + row_id] = row_scale;
        // }

    };
    #pragma unroll
    for (int t = 0; t < ROW_PER_BLOCK2; t++) {
        uint token_id = m_block * ROW_PER_BLOCK2 + t;
        if (token_id >= M) return;
        auto p_x = x + token_id * S;
        loop_row(p_x, token_id, 0);
        for (int topk_id = 1; topk_id < TOPK; topk_id++) {
            loop_row(p_x, token_id, topk_id);
        }
    }
}
