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
using charx8 = char __attribute__ ((ext_vector_type(8)));

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

__device__ float llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 int voffset,
                                 int soffset,
                                 int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ float32x4 llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc,
                                   int voffset,
                                   int soffset,
                                   int glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x2(float32x2 vdata,
                                    int32x4_t rsrc,
                                    int voffset,
                                    int soffset,
                                    int glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x4(float32x4 vdata,
                                    int32x4_t rsrc,
                                    int voffset,
                                    int soffset,
                                    int glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

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

enum struct amd_buffer_coherence_enum
{
    coherence_default = 0, // default value
    glc               = 1,
    slc               = 2,
    glc_slc           = 3,
    // gfx94: bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
    // SC[1:0] System Cache level: 0=wave, 1=group, 2=device, 3=system
    // NT Non-Temporal: 0=expect temporal reuse; 1=do not expect temporal reuse
    WAVE_NT0   = 0,
    WAVE_NT1   = 2,
    GROUP_NT0  = 1,
    GROUP_NT1  = 3,
    DEVICE_NT0 = 8,
    DEVICE_NT1 = 10,
    SYSTEM_NT0 = 9,
    SYSTEM_NT1 = 11,
};

template<typename T>
__device__ inline T buffer_load_dwordx4(BufferResource& buffer, uint soffset, uint voffset, uint coffset, int coherence=0, bool is_asm=false) {
    T v;
    if (is_asm) {
        int32x4_t r = __builtin_bit_cast(int32x4_t, buffer);
        r.x         = __builtin_amdgcn_readfirstlane(r.x);
        r.y         = __builtin_amdgcn_readfirstlane(r.y);
        r.z         = __builtin_amdgcn_readfirstlane(r.z);
        r.w         = __builtin_amdgcn_readfirstlane(r.w);

        //auto r = amd_wave_read_first_lane(buffer.descriptor);
        asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], 0 offen offset:%[coffset]\n"
            :[vdst]"=v"(v)
            :[vaddr]"v"(voffset), [srsrc]"s"(r), [coffset]"n"(coffset)
            : "memory");
    } else {
        int32x4_t r = __builtin_bit_cast(int32x4_t, buffer);
        auto d = llvm_amdgcn_raw_buffer_load_fp32x4(r, voffset + coffset, soffset, static_cast<int>(coherence));
        v = __builtin_bit_cast(T, d);
        // v = *(T*)((char*)buffer.address + soffset + voffset + coffset);
    }
    return v;
}

template<typename T>
__device__ void buffer_store_dwordx4(T& v, BufferResource& buffer, int soffset, int voffset, int coffset, amd_buffer_coherence_enum coherence=amd_buffer_coherence_enum::coherence_default) {
    auto d = __builtin_bit_cast(float32x4, v);
    int32x4_t r = __builtin_bit_cast(int32x4_t, buffer);
    llvm_amdgcn_raw_buffer_store_fp32x4(d, r, voffset + coffset, soffset, static_cast<int>(coherence));
}

template<typename T>
__device__ inline void buffer_store_dwordx2(T& v, BufferResource& buffer, int soffset, int voffset, int coffset, amd_buffer_coherence_enum coherence=amd_buffer_coherence_enum::coherence_default) {
    auto d = __builtin_bit_cast(float32x2, v);
    int32x4_t r = __builtin_bit_cast(int32x4_t, buffer);
    llvm_amdgcn_raw_buffer_store_fp32x2(d, r, voffset + coffset, soffset, static_cast<int>(coherence));
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

__device__ uint float4_to_uint(float f0, float f1, float f2, float f3) {
    uint out;
    asm ("v_cvt_i32_f32_sdwa %0, %1 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t"
                : "+v"(out)
                : "v"(f1)
                : );
    asm ("v_cvt_i32_f32_sdwa %0, %1 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t"
                : "+v"(out)
                : "v"(f0)
                : );
    asm ("v_cvt_i32_f32_sdwa %0, %1 dst_sel:BYTE_2 dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t"
                : "+v"(out)
                : "v"(f2)
                : );
    asm ("v_cvt_i32_f32_sdwa %0, %1 dst_sel:BYTE_3 dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t"
                : "+v"(out)
                : "v"(f3)
                : );
    return out;
}

__device__ float32x2 mul_add_half2(float f0_0, float f0_1, float f2) {
    float32x2 out;
    float32x2 f0 = {f0_0, f0_1};
    float32x2 f2_dup = {f2, f2};
    asm ("v_pk_fma_f32 %0, %1, %2, 0.5 \n\t"
                : "=v"(out)
                : "v"(f0), "v"(f2_dup)
                : );
    return out;
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
    __shared__ __bf16 lds[QUANT1_K];

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
        for (int i = threadIdx.x; i < S / 8; i += 64) {
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
        for (int i = threadIdx.x; i < S / 16; i += 64) {
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
__global__ void quant2(
            __bf16* x,                  // [M, S]
            float* smooth_scale,        // [E, S]
            char* x_out,                // [M, TOPK, S]
            float* x_quant_scale,       // [sorted_expert_ids.shape[0], 256]
            uint* sorted_ids,           // [sorted_expert_ids.shape[0], 256]
            uint* sorted_expert_ids,    // [sorted_expert_ids.shape[0]]
            uint* num_valid_ids,        // [2]
            uint M) {
    // wg: [sorted_expert_ids.shape[0], 32]
    uint e_idx = blockIdx.x;
    uint r_idx = blockIdx.y;
    uint eid = sorted_expert_ids[e_idx];
    if (e_idx * BLOCK_SIZE_M >= num_valid_ids[0]) return;
    sorted_ids += e_idx * BLOCK_SIZE_M;
    auto S = QUANT2_K;
    // __shared__ __bf16 lds[ROW_PER_BLOCK2][QUANT2_K];
    __shared__ float max_wave[ROW_PER_BLOCK2][4];

    uint lane_id = threadIdx.x % 64;
    uint wave_id = threadIdx.x / 64;
    uint lane_r = lane_id / COL_PER_WAVE;
    uint lane_c = lane_id % COL_PER_WAVE;
    smooth_scale += eid * S;
    //auto p_x = x + (int64_t)tok_id * TOPK * S + topk_id * S;
    static constexpr int ITEM_PER_LANE = QUANT2_K / 8 / (COL_PER_WAVE * 4);
    float32x8 scales[ITEM_PER_LANE];
    bfloat16x8 x_vals[ITEM_PER_LANE];
    float32x8 x_smooth[ITEM_PER_LANE];
    for (int i = 0; i < ITEM_PER_LANE; i++) {
        scales[i] = global_load_dwordx4<float32x8>(smooth_scale, (i * COL_PER_WAVE * 4 + lane_c + wave_id * COL_PER_WAVE) * 8 * sizeof(float));
    }
    // TODO: < 4G
    BufferResource x_res(x, M * S * TOPK * sizeof(__bf16));
    BufferResource x_out_res(x_out, 0xffffffff);
    auto load_token = [&](bfloat16x8& x_val, uint tok_id, uint topk_id, uint i) {
        // x_val = global_load_dwordx4<bfloat16x8>(p_x, i * 8 * sizeof(__bf16));
        x_val = buffer_load_dwordx4<bfloat16x8>(x_res, 
            0,
            (tok_id * TOPK * S + topk_id * S) * sizeof(__bf16), 
            i * 8 * sizeof(__bf16),
            (int)amd_buffer_coherence_enum::SYSTEM_NT1);//0x1f);
    };
    for (int block_i = 0; block_i < BLOCK_M2 / ROW_PER_BLOCK2; block_i++) {
        uint row_id = r_idx * BLOCK_M2 + block_i * ROW_PER_BLOCK2 + lane_r;
        auto raw_id = sorted_ids[row_id];
        auto tok_id = raw_id & 0xffffff;
        auto topk_id = raw_id >> 24;
        auto org_tok_id = tok_id;
        //tok_id = tok_id >= M ? 0 : tok_id; // oob
        float cur_max = -FLT_MAX;

        #pragma unroll
        for (int i = 0; i < ITEM_PER_LANE; i++) {
            load_token(x_vals[i], tok_id, topk_id, lane_c + wave_id * COL_PER_WAVE + i * COL_PER_WAVE * 4);
        }
        auto find_max = [&] (float32x8& scale, bfloat16x8& x_val, int i) {
            float32x8 tmp;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                auto tmp = (scale[j] * (float)x_val[j]);
                x_smooth[i][j] = tmp;
                cur_max = fmaxf(cur_max, abs(tmp));
            }
        };
        for (int i = 0; i < ITEM_PER_LANE; i++) {
            find_max(scales[i], x_vals[i], i);
        }

        if (COL_PER_WAVE / 2 == 8) {
            cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0xb1, 0xf, 0xf, true))); // 2 3 0 1
            cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0x1b, 0xf, 0xf, true))); // 0 1 2 3
            cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0x141, 0xf, 0xf, true))); //mirror 8 threads
            cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0x140, 0xf, 0xf, true))); //mirror 16 threads
        } else {
            for(int mask = COL_PER_WAVE / 2; mask >= 1; mask /= 2) {
                cur_max = fmaxf(cur_max, __shfl_xor(cur_max, mask));
            }
        }
        if (lane_c == 0) {
            max_wave[lane_r][wave_id] = cur_max;
        }
        __syncthreads();
        for (int i = 0; i < 4; i++) {
            cur_max = fmaxf(cur_max, max_wave[lane_r][i]);
        }
        __syncthreads();
        if (org_tok_id >= M) return; // oob write
        auto row_scale = cur_max / 128.0f;
        row_scale = fmaxf(row_scale, 1e-6f);
        auto inv_row_scale = __builtin_amdgcn_rcpf(row_scale); //1.0f / row_scale;
        // if (e_idx==1096 &&row_id == 166) {
        //     auto offset = tok_id * TOPK * S + topk_id * S;
        //     printf("x %d lanec %d wave_id %d e_idx %d, row_id %d, tok_id %ld, topk_id %d, max %f org_tok_id %d row_scale %f\n", threadIdx.x, lane_c, wave_id, e_idx, row_id, tok_id, topk_id, cur_max, org_tok_id, row_scale);
        // }

        // charx8* p_x_out = reinterpret_cast<charx8*>(x_out + tok_id * TOPK * S + topk_id * S);
        // #pragma unroll
        // for (int i = 0; i < ITEM_PER_LANE; i++) {
        //     charx8 qv;
        //     #pragma unroll
        //     for (int j = 0; j < 8; j++) {
        //         //auto val = lds[lane_r][i * 16 + j] * inv_row_scale;
        //         auto val = x_smooth[i][j] * inv_row_scale;
        //         qv[j] = val + 0.5f;
        //     }

        //     buffer_store_dwordx2(qv, 
        //         x_out_res, 
        //         0, 
        //         tok_id * TOPK * S + topk_id * S + (i * COL_PER_WAVE * 4 + lane_c + wave_id * COL_PER_WAVE) * 8, 
        //         0,
        //         amd_buffer_coherence_enum::SYSTEM_NT1);
        //     // p_x_out[i * COL_PER_WAVE * 4 + lane_c + wave_id * COL_PER_WAVE] = qv;
        // }
        #pragma unroll
        for (int i = 0; i < ITEM_PER_LANE; i++) {
            uint32x2_t qv;
            float32x2 result[4]; 
            //#pragma unroll
            // for (int j = 0; j < 8; j++) {
            //     //auto val = lds[i * 8 + j] * inv_row_scale;
            //     result[j] = x_smooth_buf[i][j] * inv_row_scale + 0.5f;
            //     //qv[j] = max(-128, min(127, (int)roundf(val)));
            // }
            result[0] = mul_add_half2(x_smooth[i][0], x_smooth[i][1], inv_row_scale);
            result[1] = mul_add_half2(x_smooth[i][2], x_smooth[i][3], inv_row_scale);
            result[2] = mul_add_half2(x_smooth[i][4], x_smooth[i][5], inv_row_scale);
            result[3] = mul_add_half2(x_smooth[i][6], x_smooth[i][7], inv_row_scale);
            qv[0] = float4_to_uint(result[0][0], result[0][1], result[1][0], result[1][1]);
            qv[1] = float4_to_uint(result[2][0], result[2][1], result[3][0], result[3][1]);
            //p_x_out[i * 64 + threadIdx.x] = qv;
            buffer_store_dwordx2(qv, 
                x_out_res, 
                0, 
                tok_id * TOPK * S + topk_id * S + (i * COL_PER_WAVE * 4 + lane_c + wave_id * COL_PER_WAVE) * 8, 
                0,
                amd_buffer_coherence_enum::SYSTEM_NT1);
        }
        if (lane_c == 0 && wave_id == 0) {
            x_quant_scale[e_idx * BLOCK_SIZE_M + row_id] = row_scale;
        }
    }
}

__global__ __launch_bounds__(64, 1) void quant2_seq(
            __bf16* x,                  // [M, S]
            float* smooth_scale,        // [E, S]
            char* x_out,                // [M, TOPK, S]
            float* x_quant_scale,       // [M, TOPK]
            uint* expert_ids,           // [M, TOPK]
            uint M) {
    // wg: [M//4]
    uint m_block = blockIdx.x;
    uint lane_id = threadIdx.x % 64;
    uint wave_id = threadIdx.x / 64;
    uint lane_r = lane_id / COL_PER_WAVE;
    uint lane_c = lane_id % COL_PER_WAVE;
    auto S = QUANT2_K;
    //__shared__ __bf16 lds[QUANT2_K];
    float32x8 x_org_buf[QUANT2_K / 64 / 8];
    float32x8 x_smooth_buf[QUANT2_K / 64 / 8];
    __shared__ uint expert_ids_lds[TOPK];
    if (threadIdx.x < TOPK) {
        expert_ids_lds[threadIdx.x] = expert_ids[m_block * ROW_PER_BLOCK2_SEQ * TOPK + threadIdx.x];
    }
    // __bf16 x_smooth_buf[QUANT2_K / 64 / 8][8];
    BufferResource x_res(x + (int64_t)m_block * ROW_PER_BLOCK2_SEQ * TOPK * S, 0xffffffff);
    BufferResource x_out_res(x_out + (int64_t)m_block * ROW_PER_BLOCK2_SEQ * TOPK * S, 0xffffffff);

    auto loop_row = [&] (uint token_id, uint token_id_in_block, uint topk_id) {
        auto p_smooth_scale = smooth_scale + expert_ids_lds[topk_id] * S;
        float cur_max = -FLT_MAX;
        float32x8 scale_buf[QUANT2_K / 64 / 8];
        for (int i = 0; i < QUANT2_K / 64 / 8; i++) {
            scale_buf[i] = global_load_dwordx4<float32x8>(p_smooth_scale, (threadIdx.x + i * 64) * 8 * sizeof(float));
        }
        # pragma unroll
        for (int i = 0; i < S / 8 / 64; i++) {
            auto& scale = scale_buf[i];
            auto x_val = x_org_buf[i];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                auto tmp = (scale[j] * x_val[j]);
                //lds[(i * 64 + threadIdx.x) * 8 + j] = tmp;
                x_smooth_buf[i][j] = tmp;
                cur_max = fmaxf(cur_max, abs(tmp));
            }
        }
        cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0xb1, 0xf, 0xf, true))); // 2 3 0 1
        cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0x1b, 0xf, 0xf, true))); // 0 1 2 3
        cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0x141, 0xf, 0xf, true))); //mirror 8 threads
        cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0x140, 0xf, 0xf, true))); //mirror 16 threads
        // auto tmp = __builtin_bit_cast(int, cur_max);
        // auto ret = __builtin_amdgcn_permlane16_swap(tmp, tmp, 0, 0);
        // cur_max = fmaxf(__builtin_bit_cast(float, ret[1]), __builtin_bit_cast(float, ret[0]));
        // tmp = __builtin_bit_cast(int, cur_max);
        // cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_permlane32_swap(tmp, tmp, 0, 0)[0]));
        for(int mask = 64 / 2; mask >= 16; mask /= 2) {
            cur_max = fmaxf(cur_max, __shfl_xor(cur_max, mask));
        }
        auto row_scale = cur_max / 128.0f;
        row_scale = fmaxf(row_scale, 1e-6f);
        auto inv_row_scale = __builtin_amdgcn_rcpf(row_scale); //1.0f / row_scale;
        //charx8* p_x_out = reinterpret_cast<charx8*>(x_out + token_id * TOPK * S + topk_id * S);
        #pragma unroll
        for (int i = 0; i < S / 8 / 64; i++) {
            uint32x2_t qv;
            float32x2 result[4];
            //#pragma unroll
            // for (int j = 0; j < 8; j++) {
            //     //auto val = lds[i * 8 + j] * inv_row_scale;
            //     result[j] = x_smooth_buf[i][j] * inv_row_scale + 0.5f;
            //     //qv[j] = max(-128, min(127, (int)roundf(val)));
            // }
            result[0] = mul_add_half2(x_smooth_buf[i][0], x_smooth_buf[i][1], inv_row_scale);
            result[1] = mul_add_half2(x_smooth_buf[i][2], x_smooth_buf[i][3], inv_row_scale);
            result[2] = mul_add_half2(x_smooth_buf[i][4], x_smooth_buf[i][5], inv_row_scale);
            result[3] = mul_add_half2(x_smooth_buf[i][6], x_smooth_buf[i][7], inv_row_scale);
            qv[0] = float4_to_uint(result[0][0], result[0][1], result[1][0], result[1][1]);
            qv[1] = float4_to_uint(result[2][0], result[2][1], result[3][0], result[3][1]);
            //p_x_out[i * 64 + threadIdx.x] = qv;
            buffer_store_dwordx2(qv, 
                x_out_res, 
                token_id_in_block * TOPK * S + topk_id * S, 
                (i * 64 + threadIdx.x) * 8, 
                0,
                amd_buffer_coherence_enum::SYSTEM_NT1);
        }
        x_quant_scale[token_id * TOPK + topk_id] = row_scale;
        // if (threadIdx.x == 0) {
        //     x_quant_scale[e_idx * BLOCK_SIZE_M + row_id] = row_scale;
        // }
    };

    auto load_token = [&] (bfloat16x8& x_val, uint token_id_in_block, uint topk_id, int i) {
        //x_val = global_load_dwordx4<bfloat16x8>(p_x, i * 8 * sizeof(__bf16));
        x_val = buffer_load_dwordx4<bfloat16x8>(x_res, 0, token_id_in_block * TOPK * S * sizeof(__bf16) + topk_id * S * sizeof(__bf16), i * 8 * sizeof(__bf16), (int)amd_buffer_coherence_enum::SYSTEM_NT1);
    };
    #pragma unroll
    for (int t = 0; t < ROW_PER_BLOCK2_SEQ; t++) {
        uint token_id = m_block * ROW_PER_BLOCK2_SEQ + t;
        if (token_id >= M) return;
        #pragma unroll(0)
        for (int topk_id = 0; topk_id < TOPK; topk_id++) {
            for (int i = 0; i < QUANT2_K / 64 / 8; i++) {
                bfloat16x8 tmp;
                load_token(tmp, t, topk_id, threadIdx.x + i * 64);
                for (int j = 0; j < 8; j++) {
                    x_org_buf[i][j] = tmp[j];
                }
            }
            loop_row(token_id, t, topk_id);
        }
    }
}


__global__ __launch_bounds__(64, 1) void quant1(
            __bf16* x,                  // [M, S]
            float* smooth_scale,        // [E, S]
            char* x_out,                // [M, TOPK, S]
            float* x_quant_scale,       // [M, TOPK]
            uint* expert_ids,           // [M, TOPK]
            uint M) {
    // wg: [M//4]
    uint m_block = blockIdx.x;
    uint lane_id = threadIdx.x % 64;
    uint wave_id = threadIdx.x / 64;
    uint lane_r = lane_id / COL_PER_WAVE;
    uint lane_c = lane_id % COL_PER_WAVE;
    auto S = QUANT1_K;
    //__shared__ __bf16 lds[QUANT1_K];
    float32x8 x_org_buf[QUANT1_K / 64 / 8];
    float32x8 x_smooth_buf[QUANT1_K / 64 / 8];
    __shared__ uint expert_ids_lds[TOPK];
    if (threadIdx.x < TOPK) {
        expert_ids_lds[threadIdx.x] = expert_ids[m_block * ROW_PER_BLOCK1 * TOPK + threadIdx.x];
    }
    // __bf16 x_smooth_buf[QUANT1_K / 64 / 8][8];
    BufferResource x_res(x + (int64_t)m_block * ROW_PER_BLOCK1 * S, 0xffffffff);
    BufferResource x_out_res(x_out + (int64_t)m_block * ROW_PER_BLOCK1 * TOPK * S, 0xffffffff);

    auto loop_row = [&] (__bf16* p_x, uint token_id, uint token_id_in_block, uint topk_id) {
        auto p_smooth_scale = smooth_scale + expert_ids_lds[topk_id] * S;
        float cur_max = -FLT_MAX;
        float32x8 scale_buf[QUANT1_K / 64 / 8];
        for (int i = 0; i < QUANT1_K / 64 / 8; i++) {
            scale_buf[i] = global_load_dwordx4<float32x8>(p_smooth_scale, (threadIdx.x + i * 64) * 8 * sizeof(float));
        }
        # pragma unroll
        for (int i = 0; i < S / 8 / 64; i++) {
            auto& scale = scale_buf[i];
            auto x_val = x_org_buf[i];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                auto tmp = (scale[j] * x_val[j]);
                //lds[(i * 64 + threadIdx.x) * 8 + j] = tmp;
                x_smooth_buf[i][j] = tmp;
                cur_max = fmaxf(cur_max, abs(tmp));
            }
        }
        cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0xb1, 0xf, 0xf, true))); // 2 3 0 1
        cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0x1b, 0xf, 0xf, true))); // 0 1 2 3
        cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0x141, 0xf, 0xf, true))); //mirror 8 threads
        cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, cur_max), 0x140, 0xf, 0xf, true))); //mirror 16 threads
        // auto tmp = __builtin_bit_cast(int, cur_max);
        // auto ret = __builtin_amdgcn_permlane16_swap(tmp, tmp, 0, 0);
        // cur_max = fmaxf(__builtin_bit_cast(float, ret[1]), __builtin_bit_cast(float, ret[0]));
        // tmp = __builtin_bit_cast(int, cur_max);
        // cur_max = fmaxf(cur_max, __builtin_bit_cast(float, __builtin_amdgcn_permlane32_swap(tmp, tmp, 0, 0)[0]));
        for(int mask = 64 / 2; mask >= 16; mask /= 2) {
            cur_max = fmaxf(cur_max, __shfl_xor(cur_max, mask));
        }
        auto row_scale = cur_max / 128.0f;
        row_scale = fmaxf(row_scale, 1e-6f);
        auto inv_row_scale = __builtin_amdgcn_rcpf(row_scale); //1.0f / row_scale;
        //charx8* p_x_out = reinterpret_cast<charx8*>(x_out + token_id * TOPK * S + topk_id * S);
        #pragma unroll
        for (int i = 0; i < S / 8 / 64; i++) {
            uint32x2_t qv;
            float32x2 result[4];
            //#pragma unroll
            // for (int j = 0; j < 8; j++) {
            //     //auto val = lds[i * 8 + j] * inv_row_scale;
            //     result[j] = x_smooth_buf[i][j] * inv_row_scale + 0.5f;
            //     //qv[j] = max(-128, min(127, (int)roundf(val)));
            // }
            result[0] = mul_add_half2(x_smooth_buf[i][0], x_smooth_buf[i][1], inv_row_scale);
            result[1] = mul_add_half2(x_smooth_buf[i][2], x_smooth_buf[i][3], inv_row_scale);
            result[2] = mul_add_half2(x_smooth_buf[i][4], x_smooth_buf[i][5], inv_row_scale);
            result[3] = mul_add_half2(x_smooth_buf[i][6], x_smooth_buf[i][7], inv_row_scale);
            qv[0] = float4_to_uint(result[0][0], result[0][1], result[1][0], result[1][1]);
            qv[1] = float4_to_uint(result[2][0], result[2][1], result[3][0], result[3][1]);
            //p_x_out[i * 64 + threadIdx.x] = qv;
            buffer_store_dwordx2(qv, 
                x_out_res, 
                token_id_in_block * TOPK * S + topk_id * S, 
                (i * 64 + threadIdx.x) * 8, 
                0,
                amd_buffer_coherence_enum::SYSTEM_NT1);
        }
        x_quant_scale[token_id * TOPK + topk_id] = row_scale;
        // if (threadIdx.x == 0) {
        //     x_quant_scale[e_idx * BLOCK_SIZE_M + row_id] = row_scale;
        // }
    };

    auto load_token = [&] (__bf16* p_x, bfloat16x8& x_val, uint token_id_in_block, int i) {
        //x_val = global_load_dwordx4<bfloat16x8>(p_x, i * 8 * sizeof(__bf16));
        x_val = buffer_load_dwordx4<bfloat16x8>(x_res, 0, token_id_in_block * S * sizeof(__bf16), i * 8 * sizeof(__bf16), (int)amd_buffer_coherence_enum::SYSTEM_NT1);
    };
    #pragma unroll
    for (int t = 0; t < ROW_PER_BLOCK1; t++) {
        uint token_id = m_block * ROW_PER_BLOCK1 + t;
        if (token_id >= M) return;
        auto p_x = x + token_id * S;
        for (int i = 0; i < QUANT1_K / 64 / 8; i++) {
            bfloat16x8 tmp;
            load_token(p_x, tmp, t, threadIdx.x + i * 64);
            for (int j = 0; j < 8; j++) {
                x_org_buf[i][j] = tmp[j];
            }
        }
        #pragma unroll(0)
        for (int topk_id = 0; topk_id < TOPK; topk_id++) {
            loop_row(p_x, token_id, t, topk_id);
        }
    }
}
