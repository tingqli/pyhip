#include <__clang_hip_runtime_wrapper.h>
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
using bfloat16x4 = __bf16  __attribute__ ((ext_vector_type(4)));
using bfloat16x8 = __bf16  __attribute__ ((ext_vector_type(8)));
using bfloat16x32 = __bf16 __attribute__ ((ext_vector_type(32)));
using float32x16 = float  __attribute__ ((ext_vector_type(16)));
using float32x4 = float  __attribute__ ((ext_vector_type(4)));
using int32x4_t = int  __attribute__ ((ext_vector_type(4)));
using int32x16_t = int  __attribute__ ((ext_vector_type(16)));

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

#define NUM_THREADS 256

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
__device__ T buffer_load_dwordx4(BufferResource& buffer, int soffset, int voffset) {
    T v;
    #if 1
    auto r = amd_wave_read_first_lane(buffer.descriptor);
    asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
        :[vdst]"=v"(v)
        :[vaddr]"v"(voffset), [srsrc]"s"(r), [soffset]"s"(soffset));
    #else
    v = *(T*)((char*)buffer.address + soffset + voffset);
    #endif
    return v;
}

__device__ void amdgcn_mfma_f32_16x16x16bf16(bfloat16x4 a, bfloat16x4 b, float32x4& c) {
    //c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
    asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %3\n"
                : "+a"(c)
                : "v"(a), "v"(b), "a"(c)
                :);
}

template<uint16_t cnt>
__device__ void s_waitcnt_vmcnt() {
    asm volatile ("s_waitcnt vmcnt(%0)\n"::"i"(cnt));
}
template<uint16_t cnt>
__device__ void s_waitcnt_lgkmcnt() {
    asm volatile ("s_waitcnt lgkmcnt(%0)\n"::"i"(cnt));
}

#ifndef HQ
#define HQ 32 
#define HK 4
#define S 128 
#define BLOCK_SIZE 1
#define SCALE 1.0 
#define KV_PART_SIZE 256
#endif

template<typename TS, typename TD>
TD& cast(TS& t) {
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

#define KV_PART_SIZE_WARP (KV_PART_SIZE / 4)
__global__ void __launch_bounds__(NUM_THREADS, 2) pa(
            __bf16* query,          // [B, HQ, S]
            __bf16* key_cache,      // [BLOCK, BLOCK_SIZE, HK, S]
            __bf16* value_cache,
            uint* kv_indptr,        // [B + 1]
            uint* kv_page_indices,  // [B * KV_LEN + 1]
            __bf16* out,            // [B, HQ, S]
            float* qk_ptr) {
    // wg: B, HK, kv_part
    uint b = blockIdx.x;
    uint hk = blockIdx.y;
    uint hq = hk * (HQ / HK);
    uint kv_part = blockIdx.z;
    uint lane_id = threadIdx.x % 64;
    uint warp_id = threadIdx.x / 64;
    const uint q_b_stride = HQ * S;
    const uint q_h_stride = S;
    query += b * q_b_stride + hq * q_h_stride;
    key_cache += hk * S;
    value_cache += hk * S;
    uint kv_len = kv_indptr[b + 1] - kv_indptr[b];
    if (kv_part * KV_PART_SIZE >= kv_len) return;
    uint kv_len_start = std::min(kv_part * KV_PART_SIZE + warp_id * KV_PART_SIZE_WARP, kv_len);
    uint kv_len_end = std::min(kv_len_start + KV_PART_SIZE_WARP, kv_len);
    kv_page_indices += kv_indptr[b] + kv_len_start / BLOCK_SIZE;
    out += b * HQ * S + hq * S;

    // stage1(q*k): [16, 32]x[64, 32]'
    BufferResource q_buf(query, (HQ / HK) * S * sizeof(__bf16));
    static_assert(HQ / HK <= 16, "use mfma16 requires M <= 16");
    bfloat16x8 q_cur[S / 32];
    // mfma16x16 layout: 16rows x 4cols
    uint col_id = lane_id / 16;      // 0 ~ 3
    uint row_id = lane_id % 16;      // 0 ~ 15

    uint k_offsets[KV_PART_SIZE_WARP / 16];
    if constexpr (BLOCK_SIZE == 1) {
        for (uint n = 0; n < KV_PART_SIZE_WARP / 16; n++) {
#if FAKE_K_IDX
            uint row = row_id + n * 16  + kv_len_start + 1;
            if (row > kv_len_end)
                row = 1;
            // auto ref = kv_page_indices[row_id + n * 16 < kv_len_end - kv_len_start ? row_id + n * 16 : 0];
            // if (ref != row) {
            //     printf("ref=%d row=%d x=%d\n", ref, row, threadIdx.x);
            // }
            k_offsets[n] = HK * S * sizeof(__bf16) * row + 8 * sizeof(__bf16) * col_id;
#else
            uint row = row_id + n * 16 < kv_len_end - kv_len_start ? row_id + n * 16 : 0;
            k_offsets[n] = HK * S * sizeof(__bf16) * kv_page_indices[row] + 8 * sizeof(__bf16) * col_id;
#endif
        }
    }
    // query -> reg
    if (row_id < HQ / HK) {
        #pragma unroll
        for (uint k = 0; k < S; k += 32) {
#if FAKE_Q
            q_cur[k / 32] = __bf16(b + 1.0f);
#else
            q_cur[k / 32] = buffer_load_dwordx4<bfloat16x8>(q_buf, k * sizeof(__bf16), (row_id * q_h_stride + col_id * 8) * sizeof(__bf16));
#endif
        }
    }
    //s_waitcnt_vmcnt<0>();
    // key -> reg
    float32x4 acc[KV_PART_SIZE_WARP / 16] = {0};
    BufferResource k_buf(key_cache, 0xffffffff);
    __builtin_amdgcn_s_waitcnt(3952);
    __builtin_amdgcn_sched_group_barrier(0,0,0);
#if 1 // prefetch 3 rows
    bfloat16x8 k_curs[S / 32 * 4];

    for (int k = 0; k < S; k += 32) {
        k_curs[k / 32] = buffer_load_dwordx4<bfloat16x8>(k_buf, k * sizeof(__bf16), k_offsets[0]);
    }
    for (int k = 0; k < S; k += 32) {
        k_curs[k / 32 + S / 32] = buffer_load_dwordx4<bfloat16x8>(k_buf, k * sizeof(__bf16), k_offsets[1]);
    }
    for (int k = 0; k < S; k += 32) {
        k_curs[k / 32 + S / 32 * 2] = buffer_load_dwordx4<bfloat16x8>(k_buf, k * sizeof(__bf16), k_offsets[2]);
    }

    uint idx_comupte = 0;
    uint idx_write = S / 32 * 3;
    uint n;
    #pragma unroll
    for (n = 3; n < KV_PART_SIZE_WARP / 16; n++) {
        compile_time_loop<S / 32>([&](auto k){
            k_curs[idx_write + k] = buffer_load_dwordx4<bfloat16x8>(k_buf, k * 32 * sizeof(__bf16), k_offsets[n]);
            if (n != KV_PART_SIZE_WARP / 16 - 1) {
                s_waitcnt_vmcnt<S / 32 * 3>();
            } else {
                s_waitcnt_vmcnt<S / 32 * 3 - k - 1>();
            }
            amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].lo, q_cur[k].lo, acc[n - 3]);
            amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].hi, q_cur[k].hi, acc[n - 3]);
        });

        idx_comupte = (idx_comupte + S / 32) % (S / 32 * 4);
        idx_write = (idx_write + S / 32) % (S / 32 * 4);
    }
    compile_time_loop<S / 32>([&](auto k){
        s_waitcnt_vmcnt<S / 32 * 2 - k - 1>();
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].lo, q_cur[k].lo, acc[n - 3]);
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].hi, q_cur[k].hi, acc[n - 3]);
    });
    idx_comupte = (idx_comupte + S / 32) % (S / 32 * 4);
    compile_time_loop<S / 32>([&](auto k){
        s_waitcnt_vmcnt<S / 32 - k - 1>();
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].lo, q_cur[k].lo, acc[n - 2]);
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].hi, q_cur[k].hi, acc[n - 2]);
    });
    idx_comupte = (idx_comupte + S / 32) % (S / 32 * 4);
    compile_time_loop<S / 32>([&](auto k){
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].lo, q_cur[k].lo, acc[n - 1]);
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].hi, q_cur[k].hi, acc[n - 1]);
    });

#elif 1 // prefetch 2 rows
    bfloat16x8 k_curs[S / 32 * 3];

    for (int k = 0; k < S; k += 32) {
        k_curs[k / 32] = buffer_load_dwordx4<bfloat16x8>(k_buf, k * sizeof(__bf16), k_offsets[0]);
    }
    for (int k = 0; k < S; k += 32) {
        k_curs[k / 32 + S / 32] = buffer_load_dwordx4<bfloat16x8>(k_buf, k * sizeof(__bf16), k_offsets[1]);
    }

    uint idx_comupte = 0;
    uint idx_write = S / 32 * 2;
    uint n;
    #pragma unroll
    for (n = 2; n < KV_PART_SIZE_WARP / 16; n++) {
        compile_time_loop<S / 32>([&](auto k){
            k_curs[idx_write + k] = buffer_load_dwordx4<bfloat16x8>(k_buf, k * 32 * sizeof(__bf16), k_offsets[n]);
            if (n != KV_PART_SIZE_WARP / 16 - 1) {
                s_waitcnt_vmcnt<S / 32 * 2>();
            } else {
                s_waitcnt_vmcnt<S / 32 * 2 - k - 1>();
            }
            amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].lo, q_cur[k].lo, acc[n - 2]);
            amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].hi, q_cur[k].hi, acc[n - 2]);
        });

        idx_comupte = (idx_comupte + S / 32) % (S / 32 * 3);
        idx_write = (idx_write + S / 32) % (S / 32 * 3);
    }
    compile_time_loop<S / 32>([&](auto k){
        s_waitcnt_vmcnt<S / 32 - k - 1>();
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].lo, q_cur[k].lo, acc[n - 2]);
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].hi, q_cur[k].hi, acc[n - 2]);
    });
    idx_comupte = (idx_comupte + S / 32) % (S / 32 * 3);
    compile_time_loop<S / 32>([&](auto k){
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].lo, q_cur[k].lo, acc[n - 1]);
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k].hi, q_cur[k].hi, acc[n - 1]);
    });

#elif 1   // prefetch 1 row
    if constexpr (BLOCK_SIZE == 1) {
        uint row = row_id + 0 < kv_len_end - kv_len_start ? row_id + 0 : 0;
        k_offset = HK * S * sizeof(__bf16) * kv_page_indices[row] + 8 * sizeof(__bf16) * col_id;
    }
    bfloat16x8 k_curs[S / 32 * 2];
    for (int k = 0; k < S; k += 32) {
        k_curs[k / 32] = buffer_load_dwordx4<bfloat16x8>(k_buf, k * sizeof(__bf16), k_offset);
    }
    uint idx_comupte = 0;
    uint idx_write = S / 32;
    s_waitcnt_vmcnt<0>();
    uint n;
    #pragma unroll
    for (n = 1; n < KV_PART_SIZE_WARP / 16; n++) {
        if constexpr (BLOCK_SIZE == 1) {
            uint row = row_id + n * 16 < kv_len_end - kv_len_start ? row_id + n * 16 : 0;
            k_offset = HK * S * sizeof(__bf16) * kv_page_indices[row] + 8 * sizeof(__bf16) * col_id;
        }
        #pragma unroll
        for (int k = 0; k < S; k += 32) {
            if (n != KV_PART_SIZE_WARP - 1)
                k_curs[idx_write + k / 32] = buffer_load_dwordx4<bfloat16x8>(k_buf, k * sizeof(__bf16), k_offset);
            amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k / 32].lo, q_cur[k / 32].lo, acc[n - 1]);
            amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k / 32].hi, q_cur[k / 32].hi, acc[n - 1]);
        }
        idx_comupte = S / 32 - idx_comupte;
        idx_write = S / 32 - idx_write;
        s_waitcnt_vmcnt<0>();
    }
    #pragma unroll
    for (int k = 0; k < S; k += 32) {
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k / 32].lo, q_cur[k / 32].lo, acc[n - 1]);
        amdgcn_mfma_f32_16x16x16bf16(k_curs[idx_comupte + k / 32].hi, q_cur[k / 32].hi, acc[n - 1]);
    }

#else
    for (uint n = 0; n < KV_PART_SIZE_WARP; n += 16) {
        if constexpr (BLOCK_SIZE == 1) {
            uint row = row_id + n < kv_len_end - kv_len_start ? row_id + n : 0;
            k_offset = HK * S * sizeof(__bf16) * kv_page_indices[row] + 8 * sizeof(__bf16) * col_id;
        }
        for (int k = 0; k < S; k += 32) {
            auto k_cur = buffer_load_dwordx4<bfloat16x8>(k_buf, k * sizeof(__bf16), k_offset);
            s_waitcnt_vmcnt<0>();
            amdgcn_mfma_f32_16x16x16bf16(k_cur.lo, q_cur[k / 32].lo, acc[n / 16]);
            amdgcn_mfma_f32_16x16x16bf16(k_cur.hi, q_cur[k / 32].hi, acc[n / 16]);
        }
    }
#endif
    __builtin_amdgcn_sched_group_barrier(0,0,0);
    if (qk_ptr) {
        // [B, HK, HQ / HK, stride]
        auto stride = (kv_len + KV_PART_SIZE - 1) / KV_PART_SIZE * KV_PART_SIZE;
        if (row_id < HQ / HK) {
            for (int n = 0; n < KV_PART_SIZE_WARP / 16; n++) {
                auto cur_tmp_ptr = qk_ptr + (hk * (HQ / HK) + row_id) * stride + kv_len_start + n * 16 + col_id * 4;
                for (int i = 0; i < 4; i++)
                    cur_tmp_ptr[i] = acc[n][i];
            }
        }
    }
            s_waitcnt_vmcnt<0>();
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 10 && lane_id == 0) {
             //printf("kv_page_indices=%p kv_len_start=%d threadIdx.x=%d kv_page_indices[row + n]=%d\n", kv_page_indices,  kv_len_start, threadIdx.x, k_offset);
            }
}
