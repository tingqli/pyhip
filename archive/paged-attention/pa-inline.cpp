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
using int32x4_t = int  __attribute__ ((ext_vector_type(4)));
using int32x16_t = int  __attribute__ ((ext_vector_type(16)));
using uint32x2_t = uint  __attribute__ ((ext_vector_type(2)));
using uint32x8_t = uint  __attribute__ ((ext_vector_type(8)));

using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;
union BufferResource {
    __device__ __inline__ constexpr BufferResource()
        : config(0x00020000U) {}

    __device__ __inline__ constexpr BufferResource(void* buffer_address, uint32_t buffer_size)
        : address(buffer_address),
          range(buffer_size),
          config(0x00020000U) {}
    __device__ __inline__ BufferResource(void* buffer_address, uint32_t buffer_size, uint64_t stride)
        : address((void*)((uint64_t)buffer_address | (stride << 48))),
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

template<typename T>
__device__ T buffer_load_dwordx4(BufferResource& buffer, int soffset, int voffset, bool is_asm=false) {
    T v;
    if (1||is_asm) {
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

template<typename T>
__device__ T buffer_load_dwordx4(BufferResource& buffer, int soffset, int voffset, int vindex, bool is_asm=false) {
    T v;
    if (1||is_asm) {
        auto r = amd_wave_read_first_lane(buffer.descriptor);
        uint32x2_t tmp;
        tmp[0] = vindex;
        tmp[1] = voffset;
        asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], 0 idxen, offen\n"
            :[vdst]"=v"(v)
            :[vaddr]"v"(tmp), [srsrc]"s"(r));
    } else {
        auto addr = (uint64_t)buffer.address & ((1ull << 48) - 1);
        auto stride = (uint64_t)buffer.address >> 48;
        v = *(T*)((char*)addr + soffset + voffset + stride * vindex);
    }
    return v;
}

template<typename T>
__device__ void lds_write_dwordx4(void* vaddr, T data, uint imm_offset=0, bool is_asm=false) {
    as3_uint32_ptr base = (as3_uint32_ptr)vaddr;
    if (1||is_asm) {
        if (imm_offset) {
            // ds_write_b128 v80, v[50:53] offset:1024          ;	LDS_MEM[v80 + 1024].b128 = v[50:53].b128
            asm volatile("ds_write_b128 %[vaddr], %[vdata] offset:%[offset]"
                        ::[vaddr]"v"(base), [vdata]"v"(data), [offset]"i"(imm_offset));
        } else {
            // ds_write_b128 v96, v[102:105]                    ;	LDS_MEM[v96 + 0].b128 = v[102:105].b128
            asm volatile("ds_write_b128 %[vaddr], %[vdata]"
                        ::[vaddr]"v"(base), [vdata]"v"(data));
        }

    } else {
        *((T*)vaddr) = data;
    }
}

template<typename T>
__device__ T lds_load_dwordx4(void* vaddr, uint imm_offset=0, bool is_asm=false) {
    T data;
    as3_uint32_ptr base = (as3_uint32_ptr)vaddr;
    if (1||is_asm) {
        if (imm_offset) {
            asm volatile("ds_read_b128 %[vdst], %[vaddr], offset:%[offset]"
                        : [vdst]"=v"((T&)(data))
                        : [vaddr]"v"(base), [offset]"i"(imm_offset));
        } else {
            // ds_read_b128 v[36:39], v132                      ;	v[36:39] = LDS_MEM[v132 + 0].b128; // read w/o any type convertion
            asm volatile("ds_read_b128 %[vdst], %[vaddr]"
                        : [vdst]"=v"((T&)(data))
                        : [vaddr]"v"(base));
        }
    } else {
        data = *((T*)vaddr);
    }
    return data;
}

template<typename T>
__device__ T lds_load_dwordx2(void* vaddr, uint imm_offset0=0, bool is_asm=false) {
    T data;
    as3_uint32_ptr base = (as3_uint32_ptr)vaddr;
    // ds_read_b64 v[0:1], v0 offset:1536               ;	v[0:1] = LDS_MEM[v0 + 1536].b64
    asm volatile("ds_read_b64 %[vdst], %[vaddr] offset:%[offset0]\n"
                : [vdst]"=v"((T&)(data))
                : [vaddr]"v"(base), [offset0]"i"(imm_offset0)
                : );
    return data;
}

template<typename T>
__device__ T lds_load_dwordx2x2(void* vaddr, uint imm_offset0=0, uint imm_offset1=0, bool is_asm=false) {
    T data;
    as3_uint32_ptr base = (as3_uint32_ptr)vaddr;
    // ds_read2_b64 v[74:77], v193 offset0:64 offset1:96
    asm volatile("ds_read2_b64 %[vdst], %[vaddr] offset0:%[offset0] offset1:%[offset1]\n"
                : [vdst]"=v"((T&)(data))
                : [vaddr]"v"(base), [offset0]"i"(imm_offset0), [offset1]"i"(imm_offset1)
                : );
    return data;
}

__device__ uint32x2_t transpose2x2(uint32_t s0, uint32_t s1) {
    auto p0 = __builtin_amdgcn_readfirstlane(0x01'00'05'04);
    auto p1 = __builtin_amdgcn_readfirstlane(0x03'02'07'06);
    uint32x2_t out;
    uint32_t a, b;
    // v_perm_b32 v76, v5, v3, s3
    asm volatile("v_perm_b32 %0, %1, %2, %3\n\t"
                : "=v"(a)
                : "v"(s0), "v"(s1),
                  "s"(p0)
                : );
    asm volatile("v_perm_b32 %0, %1, %2, %3\n\t"
                : "=v"(b)
                : "v"(s0), "v"(s1),
                  "s"(p1)
                : );
    out[0] = a;
    out[1] = b;
    return out;
}

__device__ void transpose4x4(bfloat16x4 src[4], bfloat16x4 dst[4]) {
    auto in = __builtin_bit_cast(uint32_t*, src);
    auto out = __builtin_bit_cast(uint32_t*, dst);
    uint32x2_t tmp;
    tmp = transpose2x2(in[0], in[2]);
    out[0] = tmp[0]; out[2] = tmp[1];
    tmp = transpose2x2(in[1], in[3]);
    out[4] = tmp[0]; out[6] = tmp[1];
    tmp = transpose2x2(in[4], in[6]);
    out[1] = tmp[0]; out[3] = tmp[1];
    tmp = transpose2x2(in[5], in[7]);
    out[5] = tmp[0]; out[7] = tmp[1];
}

template<typename T>
__device__ T buffer_load_dword(BufferResource& buffer, int soffset, int voffset, int coffset, bool is_asm=false) {
    T v;
    if (1||is_asm) {
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

__device__ void amdgcn_mfma_f32_16x16x16bf16(bfloat16x4 a, bfloat16x4 b, float32x4& c, bool is_asm=false, bool is_first=false) {
    if (1 && is_asm) {
        if (is_first) {
            asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, 0\n"
                        : "+v"(c)
                        : "v"(a), "v"(b)
                        :);
        } else {
            asm volatile("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %3\ns_nop 5\n"
                        : "+v"(c)
                        : "v"(a), "v"(b), "v"(c)
                        :);
        }
    } else {
        c = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
    }
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
#define KV_MIN_PART_SIZE 256

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

#define KV_PART_SIZE_WARP (KV_MIN_PART_SIZE / 4)

struct share_buf_t {
    __bf16* buf;
    __device__ share_buf_t(__bf16* buf) : buf(buf) {}
      // for output reduce: 4 warpx16xS
    __device__ __bf16* get_out_buf(uint warp_id) {
        return buf + 16 * S * warp_id;
    }
    // for key cache: 4 warpx16xS
    __device__ __bf16* get_key_buf(uint warp_id) {
        return buf + 16 * S * warp_id;
    }
    // for value cache: 4 warpx16xS
    __device__ __bf16* get_value_buf(uint warp_id) {
        return buf + 16 * S * warp_id;
    }
    // 4warp * (16m + 1)
    __device__ float* get_max_buf(uint m, uint warp_id) {
        return (float*)buf + m + warp_id * 16;
    }
    __device__ float* get_sum_buf(uint m, uint warp_id) {
        return (float*)buf + m + warp_id * 16 + 64;
    }
};

__global__ void __launch_bounds__(NUM_THREADS, 2) pa(
            __bf16* query,          // [B, HQ, S]
            __bf16* key_cache,      // [BLOCK, BLOCK_SIZE, HK, S]
            __bf16* value_cache,
            uint* kv_indptr,        // [B + 1]
            uint* kv_page_indices,  // [B * KV_LEN + 1]
            __bf16* out_seg,        // [B, HQ, PART, S]
            float* qk_ptr,          // [B, HK, HQ // HK, PART * KV_PART_SIZE]
            float* max_out,         // [B, HQ, PART, 1]
            float* sum_out) {       // [B, HQ, PART, 1]
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
    uint last_kv_idx = kv_indptr[b + 1];
    uint kv_len = last_kv_idx - kv_indptr[b];
    if (kv_part * KV_PART_SIZE >= kv_len) return;
    uint kv_len_start = std::min(kv_part * KV_PART_SIZE + warp_id * KV_PART_SIZE_WARP, kv_len);
    uint kv_len_end = std::min(kv_part * KV_PART_SIZE + KV_PART_SIZE, kv_len);
    kv_page_indices += kv_indptr[b];
    out_seg += b * HQ * gridDim.z * S + hq * gridDim.z * S + kv_part * S;
    max_out += b * HQ * gridDim.z * 1 + hq * gridDim.z * 1 + kv_part * 1;
    sum_out += b * HQ * gridDim.z * 1 + hq * gridDim.z * 1 + kv_part * 1;

    // stage1(q*k): [16, 32]x[64, 32]'
    BufferResource q_buf(query, (HQ / HK) * S * sizeof(__bf16));
    static_assert(HQ / HK <= 16, "use mfma16 requires M <= 16");
    bfloat16x8 q_cur[S / 32];
    // key load layout: 4rows x 16 cols
    uint key_load_col_id = lane_id % 16; // 0 ~ 15
    uint key_load_row_id = lane_id / 16; // 0 ~ 3
    // mfma16x16 layout: 16rows x 4cols
    uint fma_col_id = lane_id / 16;      // 0 ~ 3
    uint fma_row_id = lane_id % 16;      // 0 ~ 15
    float prev_max = - FLT_MAX;
    float prev_sum = 0;
    BufferResource kcache_buf(key_cache, 0xffffffff, HK * S * sizeof(__bf16));
    BufferResource vcache_buf(value_cache, 0xffffffff, HK * S * sizeof(__bf16));

    // query -> reg
    if (fma_row_id < HQ / HK) {
        #pragma unroll
        for (uint k = 0; k < S; k += 32) {
#if FAKE_Q
            q_cur[k / 32] = __bf16(1.0f);
#else
            q_cur[k / 32] = buffer_load_dwordx4<bfloat16x8>(q_buf, k * sizeof(__bf16), (fma_row_id * q_h_stride + fma_col_id * 8) * sizeof(__bf16), false);
#endif
        }
    }
    //s_waitcnt_vmcnt<0>();
    __shared__ __bf16 buff_lds[16 * S * 4];
    share_buf_t share_buf(buff_lds);
    float32x4 vout[S / 64 * 4] = {0};
    auto cur_max = prev_max;
    float cur_sum;

    BufferResource idx_buf(kv_page_indices, kv_len * sizeof(uint));
    // static_assert(KV_PART_SIZE / KV_MIN_PART_SIZE >= 2, "part size must be >= 2");
    uint k_idxs[KV_PART_SIZE_WARP / 4];
    auto cur_kv_len_start = kv_len_start + 0 * KV_MIN_PART_SIZE;
    bfloat16x8 k_reg_caches[KV_PART_SIZE_WARP / 4], v_reg_caches[KV_PART_SIZE_WARP / 4];

    #pragma unroll
    for (uint part_idx = 0; part_idx < KV_PART_SIZE / KV_MIN_PART_SIZE; part_idx++) {
        auto load_idx = [&] (uint n, bool should_load, uint cur_kv_len_start) {
            if (should_load)
                k_idxs[n] = buffer_load_dword<uint>(idx_buf, 0, (key_load_row_id + cur_kv_len_start) * sizeof(uint), n * 4 * sizeof(uint));
        };
        if (BLOCK_SIZE == 1) {
            for (uint n = 0; n < KV_PART_SIZE_WARP / 4; n++) {
    #if FAKE_K_IDX
                uint global_row_id = n * 4 + key_load_row_id + cur_kv_len_start;
                global_row_id = global_row_id < kv_len_end ? global_row_id : 0;
                k_idxs[n] = global_row_id + 1 + kv_indptr[b];
    #else
                load_idx(n, part_idx == 0, cur_kv_len_start);
    #endif
            }
        }
        // key -> reg
        float32x4 acc[KV_PART_SIZE_WARP / 16] = {0};

        auto load_kcache = [&] (auto n, bool should_load) {
            if (should_load) {
                s_waitcnt_vmcnt<KV_PART_SIZE_WARP / 4 - 1>();
                k_reg_caches[n] = buffer_load_dwordx4<bfloat16x8>(kcache_buf, 0, 8 * sizeof(__bf16) * key_load_col_id, k_idxs[n], false);
            }
        };
        auto load_vcache = [&] (auto n) {
            v_reg_caches[n] = buffer_load_dwordx4<bfloat16x8>(vcache_buf, 0, 8 * sizeof(__bf16) * key_load_col_id, k_idxs[n], false);
        };
        for (uint n = 0; n < KV_PART_SIZE_WARP / 4; n++) {
            load_kcache(n, part_idx == 0);
        }
        __bf16* cur_k_buff_lds = share_buf.get_key_buf(warp_id);
        auto write_kcache = [&] (uint n) {
            // swizzle: row ^ col
            lds_write_dwordx4(&cur_k_buff_lds[(key_load_row_id + 0 * 4) * S + (key_load_col_id ^ (key_load_row_id + 0)) * 8], k_reg_caches[4 * n + 0]);
            lds_write_dwordx4(&cur_k_buff_lds[(key_load_row_id + 1 * 4) * S + (key_load_col_id ^ (key_load_row_id + 4)) * 8], k_reg_caches[4 * n + 1]);
            lds_write_dwordx4(&cur_k_buff_lds[(key_load_row_id + 2 * 4) * S + (key_load_col_id ^ (key_load_row_id + 8)) * 8], k_reg_caches[4 * n + 2]);
            lds_write_dwordx4(&cur_k_buff_lds[(key_load_row_id + 3 * 4) * S + (key_load_col_id ^ (key_load_row_id + 12)) * 8], k_reg_caches[4 * n + 3]);
        };
        auto load_kcache_lds = [&] (uint n, uint k) {
            auto k_cur = lds_load_dwordx4<bfloat16x8>(&cur_k_buff_lds[fma_row_id * S + (fma_row_id ^ (fma_col_id + 4 * k)) * 8]);
            return k_cur;
        };
        auto qk_fma = [&] (auto& k, auto& q, auto& out, bool is_first=false) {
            amdgcn_mfma_f32_16x16x16bf16(k.lo, q.lo, out, true, is_first);
            amdgcn_mfma_f32_16x16x16bf16(k.hi, q.hi, out, true);
        };
        s_waitcnt_vmcnt<12>();
        write_kcache(0);
        auto k0 = load_kcache_lds(0, 0);
        load_vcache(0);
        auto k1 = load_kcache_lds(0, 1);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k0, q_cur[0], acc[0], true);
        load_vcache(1);
        k0 = load_kcache_lds(0, 2);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k1, q_cur[1], acc[0]);
        load_vcache(2);
        k1 = load_kcache_lds(0, 3);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k0, q_cur[2], acc[0]);
        load_vcache(3);
        s_waitcnt_lgkmcnt<0>();
        qk_fma(k1, q_cur[3], acc[0]);
        s_waitcnt_vmcnt<8 + 4>();
        write_kcache(1);
        k0 = load_kcache_lds(1, 0);
        load_vcache(4);
        k1 = load_kcache_lds(1, 1);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k0, q_cur[0], acc[1], true);
        load_vcache(5);
        k0 = load_kcache_lds(1, 2);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k1, q_cur[1], acc[1]);
        load_vcache(6);
        k1 = load_kcache_lds(1, 3);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k0, q_cur[2], acc[1]);
        load_vcache(7);
        s_waitcnt_lgkmcnt<0>();
        qk_fma(k1, q_cur[3], acc[1]);
        s_waitcnt_vmcnt<4 + 8>();
        write_kcache(2);
        k0 = load_kcache_lds(2, 0);
        load_vcache(8);
        k1 = load_kcache_lds(2, 1);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k0, q_cur[0], acc[2], true);
        load_vcache(9);
        k0 = load_kcache_lds(2, 2);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k1, q_cur[1], acc[2]);
        load_vcache(10);
        k1 = load_kcache_lds(2, 3);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k0, q_cur[2], acc[2]);
        load_vcache(11);
        s_waitcnt_lgkmcnt<0>();
        qk_fma(k1, q_cur[3], acc[2]);
        s_waitcnt_vmcnt<0 + 12>();
        write_kcache(3);
        k0 = load_kcache_lds(3, 0);
        load_vcache(12);
        k1 = load_kcache_lds(3, 1);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k0, q_cur[0], acc[3], true);
        load_vcache(13);
        k0 = load_kcache_lds(3, 2);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k1, q_cur[1], acc[3]);
        load_vcache(14);
        k1 = load_kcache_lds(3, 3);
        s_waitcnt_lgkmcnt<1>();
        qk_fma(k0, q_cur[2], acc[3]);
        load_vcache(15);
        s_waitcnt_lgkmcnt<0>();
        qk_fma(k1, q_cur[3], acc[3]);
        acc[0] *= SCALE;
        acc[1] *= SCALE;
        acc[2] *= SCALE;
        acc[3] *= SCALE;
    #if OUTPUT_QK
        // for debug
        if (qk_ptr) {
            // [B, HK, HQ / HK, stride]
            auto stride = (kv_len + KV_PART_SIZE - 1) / KV_PART_SIZE * KV_PART_SIZE;
            if (fma_row_id < HQ / HK) {
                for (int n = 0; n < KV_PART_SIZE_WARP / 16; n++) {
                    auto cur_tmp_ptr = qk_ptr + b * HK * HQ / HK * stride + (hk * (HQ / HK) + fma_row_id) * stride + cur_kv_len_start + n * 16 + fma_col_id * 4;
                    for (int i = 0; i < 4; i++)
                        cur_tmp_ptr[i] = acc[n][i];
                }
            }
        }
    #endif

        auto cur_kv_len_start_copy = cur_kv_len_start;
        // --------------------------
        cur_kv_len_start += KV_MIN_PART_SIZE;
    //     if (part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1)
    //     if (BLOCK_SIZE == 1) {
    //         for (uint n = 0; n < KV_PART_SIZE_WARP / 4; n++) {
    // #if FAKE_K_IDX
    //             uint global_row_id = n * 4 + key_load_row_id + cur_kv_len_start;
    //             global_row_id = global_row_id < kv_len_end ? global_row_id : 0;
    //             k_idxs[n] = global_row_id + 1;
    // #else
    //             k_idxs[n] = buffer_load_dword<uint>(idx_buf, 0, (key_load_row_id + cur_kv_len_start) * sizeof(uint), n * 4 * sizeof(uint));
    // #endif
    //         }
    //     }
        // -----------------------------
        // sum(exp(acc-max))
        if (0 || KV_PART_SIZE != kv_len_end - kv_part * KV_PART_SIZE) {
            for (uint n = 0; n < 4; n++) {
                load_idx(2 * n + 0, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1, cur_kv_len_start);
                load_idx(2 * n + 1, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1, cur_kv_len_start);
                for (uint i = 0; i < 4; i++) {
                    auto tmp = n * 16 + fma_col_id * 4 + i + cur_kv_len_start_copy < kv_len_end ? acc[n][i] : -FLT_MAX;
                    cur_max = fmaxf(cur_max, tmp);
                }
            }
            for(int mask = 64 / 2; mask >= 16; mask /= 2) {
                cur_max = fmaxf(cur_max, __shfl_xor(cur_max, mask));
            }
            cur_sum = 0;
            for (uint n = 0; n < 4; n++) {
                load_idx(2 * n + 0 + 8, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1, cur_kv_len_start);
                load_idx(2 * n + 1 + 8, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1, cur_kv_len_start);
                for (uint i = 0; i < 4; i++) {
                    acc[n][i] = n * 16 + fma_col_id * 4 + i + cur_kv_len_start_copy < kv_len_end ? __expf(acc[n][i] - cur_max) : 0;
                    cur_sum += acc[n][i];
                }
            }
        } else {
            for (uint n = 0; n < 4; n++) {
                load_idx(2 * n + 0, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1, cur_kv_len_start);
                load_idx(2 * n + 1, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1, cur_kv_len_start);
                for (uint i = 0; i < 4; i++) {
                    cur_max = fmaxf(cur_max, acc[n][i]);
                }
            }
            for(int mask = 64 / 2; mask >= 16; mask /= 2) {
                cur_max = fmaxf(cur_max, __shfl_xor(cur_max, mask));
            }
            cur_sum = 0;
            for (uint n = 0; n < 4; n++) {
                load_idx(2 * n + 0 + 8, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1, cur_kv_len_start);
                load_idx(2 * n + 1 + 8, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1, cur_kv_len_start);
                for (uint i = 0; i < 4; i++) {
                    acc[n][i] = __expf(acc[n][i] - cur_max);
                    cur_sum += acc[n][i];
                }
            }
        }
        for(int mask = 64 / 2; mask >= 16; mask /= 2) {
            cur_sum += __shfl_xor(cur_sum, mask);
        }
        // ----------------------------------
        auto prefetch_kcache = [&] (auto n, bool should_load) {
            if (should_load) {
                s_waitcnt_vmcnt<KV_PART_SIZE_WARP / 4 - 1>();
                k_reg_caches[n] = buffer_load_dwordx4<bfloat16x8>(kcache_buf, 0, 8 * sizeof(__bf16) * key_load_col_id, k_idxs[n], false);
            }
        };

        // if (part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1) {
        //     for (uint n = 0; n < KV_PART_SIZE_WARP / 4; n++) {
        //         s_waitcnt_vmcnt<KV_PART_SIZE_WARP / 4 - 1>();
        //         k_reg_caches[n] = buffer_load_dwordx4<bfloat16x8>(kcache_buf, 0, 8 * sizeof(__bf16) * key_load_col_id, k_idxs[n], false);
        //     }
        // }
        // ----------------------------------
        auto fixed_sum = prev_sum * __expf(prev_max - cur_max);
        cur_sum += fixed_sum;
        const float inv_sum_scale =
            __fdividef(1.f, cur_sum + 1e-6f);
        bfloat16x4 acc_low[KV_PART_SIZE_WARP / 16];
        for (uint n = 0; n < 4; n++) {
            for (uint i = 0; i < 4; i++) {
                auto tmp = acc[n][i] * inv_sum_scale;
                acc_low[n][i] = std::bit_cast<__bf16>((ushort)(std::bit_cast<uint>(tmp) >> 16));
            }
        }
        // compensation prev vout
        if (part_idx != 0) {
            for (int k = 0; k < S / 64 * 4; k++) {
                vout[k] = fixed_sum * inv_sum_scale * vout[k];
            }
        }
        //s_waitcnt_vmcnt<0>();
        __bf16* cur_v_buff_lds = share_buf.get_value_buf(warp_id);
        if (part_idx == KV_PART_SIZE / KV_MIN_PART_SIZE - 1) {
            s_waitcnt_vmcnt<0>();
        }
        auto write_vcache_lds = [&] (uint n) {
            lds_write_dwordx4(&cur_v_buff_lds[key_load_row_id * S + key_load_col_id * 8], v_reg_caches[4 * n + 0], 0 * 4 * S * sizeof(__bf16));
            lds_write_dwordx4(&cur_v_buff_lds[key_load_row_id * S + key_load_col_id * 8], v_reg_caches[4 * n + 1], 1 * 4 * S * sizeof(__bf16));
            lds_write_dwordx4(&cur_v_buff_lds[key_load_row_id * S + key_load_col_id * 8], v_reg_caches[4 * n + 2], 2 * 4 * S * sizeof(__bf16));
            lds_write_dwordx4(&cur_v_buff_lds[key_load_row_id * S + key_load_col_id * 8], v_reg_caches[4 * n + 3], 3 * 4 * S * sizeof(__bf16));
        };
        bfloat16x4 v_curs[4];
        bfloat16x4 v_curs_tr[4];
        // layout: [4tx4, 16tx4k]-> 16 tokens, 64 of head size used for one iter
        // K/N 0 1 2 3   4 5 6 7   8 9 10 11  12 13 14 15 ...  56 57 58 59  60 61 62 63 
        //   0 thread0   thread1   thread2    thread3          thread14     thread15
        //   1
        //   2
        //   3
        //   4 thread16  thread17  thread18   thread19         thread30     thread31
        //   5
        //   6
        //   7
        //   ...
        //  12 thread48  thread49  thread50   thread51         thread62     thread63
        //  13
        //  14
        //  15
        auto read_vcache_lds = [&] (uint n, uint k) {
            ((bfloat16x8*)v_curs)[0] = lds_load_dwordx2x2<bfloat16x8>(&cur_v_buff_lds[key_load_row_id * 4 * S + key_load_col_id * 4], (0 * S + k * 64) * sizeof(__bf16) / 8, (1 * S + k * 64) * sizeof(__bf16) / 8);
            ((bfloat16x8*)v_curs)[1] = lds_load_dwordx2x2<bfloat16x8>(&cur_v_buff_lds[key_load_row_id * 4 * S + key_load_col_id * 4], (2 * S + k * 64) * sizeof(__bf16) / 8, (3 * S + k * 64) * sizeof(__bf16) / 8);
        };
        auto kv_fma = [&] (uint n, uint k) {
            amdgcn_mfma_f32_16x16x16bf16(v_curs_tr[0], acc_low[n], vout[k * 4 + 0], true);
            amdgcn_mfma_f32_16x16x16bf16(v_curs_tr[1], acc_low[n], vout[k * 4 + 1], true);
            amdgcn_mfma_f32_16x16x16bf16(v_curs_tr[2], acc_low[n], vout[k * 4 + 2], true);
            amdgcn_mfma_f32_16x16x16bf16(v_curs_tr[3], acc_low[n], vout[k * 4 + 3], true);
        };
        prefetch_kcache(0, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        write_vcache_lds(0);
        prefetch_kcache(1, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        read_vcache_lds(0, 0);
        s_waitcnt_lgkmcnt<0>();
        transpose4x4(v_curs, v_curs_tr);
        prefetch_kcache(2, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        kv_fma(0, 0);
        read_vcache_lds(0, 1);
        prefetch_kcache(3, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        s_waitcnt_lgkmcnt<0>();
        transpose4x4(v_curs, v_curs_tr);
        kv_fma(0, 1);

        prefetch_kcache(4, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        write_vcache_lds(1);
        prefetch_kcache(5, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        read_vcache_lds(1, 0);
        s_waitcnt_lgkmcnt<0>();
        transpose4x4(v_curs, v_curs_tr);
        prefetch_kcache(6, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        kv_fma(1, 0);
        read_vcache_lds(1, 1);
        prefetch_kcache(7, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        s_waitcnt_lgkmcnt<0>();
        transpose4x4(v_curs, v_curs_tr);
        kv_fma(1, 1);

        prefetch_kcache(8, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        write_vcache_lds(2);
        prefetch_kcache(9, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        read_vcache_lds(2, 0);
        s_waitcnt_lgkmcnt<0>();
        transpose4x4(v_curs, v_curs_tr);
        prefetch_kcache(10, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        kv_fma(2, 0);
        read_vcache_lds(2, 1);
        prefetch_kcache(11, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        s_waitcnt_lgkmcnt<0>();
        transpose4x4(v_curs, v_curs_tr);
        kv_fma(2, 1);

        prefetch_kcache(12, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        write_vcache_lds(3);
        prefetch_kcache(13, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        read_vcache_lds(3, 0);
        s_waitcnt_lgkmcnt<0>();
        transpose4x4(v_curs, v_curs_tr);
        prefetch_kcache(14, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        kv_fma(3, 0);
        read_vcache_lds(3, 1);
        prefetch_kcache(15, part_idx != KV_PART_SIZE / KV_MIN_PART_SIZE - 1);
        s_waitcnt_lgkmcnt<0>();
        transpose4x4(v_curs, v_curs_tr);
        kv_fma(3, 1);

        prev_max = cur_max;
        prev_sum = cur_sum;
    }
    __syncthreads();
    // cross max/sum
    if (fma_col_id == 0) {
        *share_buf.get_max_buf(fma_row_id, warp_id) = cur_max;
        *share_buf.get_sum_buf(fma_row_id, warp_id) = cur_sum;
    }
    __syncthreads();
    // each wave will process HQ / HK / 4 tokens of output, each
    float maxs[HQ / HK], sums[HQ / HK];
    float real_max[HQ / HK / 4], real_sum[HQ / HK / 4];
    for (uint i = 0; i < HQ / HK / 4; i++) {
        uint m_token_id = HQ / HK / 4 * warp_id + i;
        maxs[4 * i + 0] = *share_buf.get_max_buf(m_token_id, 0);
        maxs[4 * i + 1] = *share_buf.get_max_buf(m_token_id, 1);
        maxs[4 * i + 2] = *share_buf.get_max_buf(m_token_id, 2);
        maxs[4 * i + 3] = *share_buf.get_max_buf(m_token_id, 3);
        real_max[i] = fmaxf(
            fmaxf(maxs[4 * i + 0], maxs[4 * i + 1]),
            fmaxf(maxs[4 * i + 2], maxs[4 * i + 3]));

        sums[4 * i + 0] = *share_buf.get_sum_buf(m_token_id, 0);
        sums[4 * i + 1] = *share_buf.get_sum_buf(m_token_id, 1);
        sums[4 * i + 2] = *share_buf.get_sum_buf(m_token_id, 2);
        sums[4 * i + 3] = *share_buf.get_sum_buf(m_token_id, 3);
        real_sum[i] = sums[4 * i + 0] * __expf(maxs[4 * i + 0] - real_max[i]) + 
                      sums[4 * i + 1] * __expf(maxs[4 * i + 1] - real_max[i]) + 
                      sums[4 * i + 2] * __expf(maxs[4 * i + 2] - real_max[i]) + 
                      sums[4 * i + 3] * __expf(maxs[4 * i + 3] - real_max[i]);
    }

    bfloat16x4 vout_low[S / 64 * 4];
    for (uint k = 0; k < S / 64 ; k++) {
        for (uint i = 0; i < 4; i++)
            for (uint j = 0; j < 4; j++)
                vout_low[k * 4 + j][i] = vout[k * 4 + i][j];
    }
    __syncthreads();

    auto shared_out = share_buf.get_out_buf(warp_id) + fma_row_id * S;
    for (int k = 0; k < S / 64; k++) {
        // swizzle: row ^ col
        ((bfloat16x4*)(shared_out + k * 64))[(fma_col_id * 4 + 0) ^ fma_row_id] = vout_low[k * 4 + 0];
        ((bfloat16x4*)(shared_out + k * 64))[(fma_col_id * 4 + 1) ^ fma_row_id] = vout_low[k * 4 + 1];
        ((bfloat16x4*)(shared_out + k * 64))[(fma_col_id * 4 + 2) ^ fma_row_id] = vout_low[k * 4 + 2];
        ((bfloat16x4*)(shared_out + k * 64))[(fma_col_id * 4 + 3) ^ fma_row_id] = vout_low[k * 4 + 3];
    }
    __syncthreads();
    for (uint i = 0; i < HQ / HK / 4; i++) {
        uint m_token_id = HQ / HK / 4 * warp_id + i;
        if (m_token_id < HQ / HK) {
            auto out_v = ((bfloat16x2*)(share_buf.get_out_buf(0) + m_token_id * S))[(m_token_id ^ (lane_id / 2)) * 2 + (lane_id & 1)];
            out_v = sums[4 * i + 0] * __expf(maxs[4 * i + 0] - real_max[i]) * out_v;

            for (int w = 1; w < 4; w++) {
                auto next_v = ((bfloat16x2*)(share_buf.get_out_buf(w) + m_token_id * S))[(m_token_id ^ (lane_id / 2)) * 2 + (lane_id & 1)];
                next_v = sums[4 * i + w] * __expf(maxs[4 * i + w] - real_max[i]) * next_v;
                out_v += next_v;
            }
            const float inv_sum_scale =
                __fdividef(1.f, real_sum[i] + 1e-6f);
            out_v = out_v * inv_sum_scale;
            ((bfloat16x2*)(out_seg + m_token_id * gridDim.z * S))[lane_id] = out_v;
            max_out[m_token_id * gridDim.z] = real_max[i];
            sum_out[m_token_id * gridDim.z] = real_sum[i];
        }
    }
}

__global__ void __launch_bounds__(NUM_THREADS, 2) pa_reduce(
            uint* kv_indptr,        // [B + 1]
            __bf16* out_seg,        // [B, HQ, PART, S]
            float* max_out,         // [B, HQ, PART, 1]
            float* sum_out,         // [B, HQ, PART, 1]
            __bf16* out,            // [B, HQ, S]
            uint max_part
            ) {
    // wg: B, HQ
    uint b = blockIdx.x;
    uint hq = blockIdx.y;
    uint lane_id = threadIdx.x % 64;
    uint warp_id = threadIdx.x / 64;
    out += b * HQ * S + hq * S;
    out_seg += b * HQ * max_part * S + hq * max_part * S;
    max_out += b * HQ * max_part * 1 + hq * max_part * 1;
    sum_out += b * HQ * max_part * 1 + hq * max_part * 1;
    uint kv_len = kv_indptr[b + 1] - kv_indptr[b];
    uint part_num = (kv_len + KV_PART_SIZE - 1) / KV_PART_SIZE;
    float real_max = -FLT_MAX;
    for (uint i = lane_id; i < part_num; i += 64) {
        real_max = fmaxf(real_max, max_out[i]);
    }
    for (uint mask = 64 / 2; mask >= 1; mask /= 2) {
        real_max = fmaxf(real_max, __shfl_xor(real_max, mask));
    }

    float32x2 cur_val = 0;
    float warp_sum = 0;
    for (uint i = warp_id; i < part_num; i += 4) {
        auto cur_max = max_out[i];
        auto cur_out_seg = out_seg + i * S;
        // TODO: support S is not 128
        auto cur_val_low = ((bfloat16x2*)cur_out_seg)[lane_id];
        float32x2 tmp_val;
        tmp_val[0] = cur_val_low[0];
        tmp_val[1] = cur_val_low[1];
        cur_val += tmp_val * sum_out[i] * __expf(cur_max - real_max);
        warp_sum += sum_out[i] * __expf(cur_max - real_max);
    }

    __shared__ float out_lds[4 * S];
    __shared__ float sum_lds[4];
    if (lane_id == 0) {
        sum_lds[warp_id] = warp_sum;
    }
    if (warp_id != 0)
        *(float32x2*)(&out_lds[warp_id * S + lane_id * 2]) = cur_val;
    __syncthreads();
    if (warp_id == 0) {
        warp_sum += sum_lds[1] + sum_lds[2] + sum_lds[3];
        for (int i = 1; i < 4; i++) {
            cur_val += *(float32x2*)(&out_lds[i * S + lane_id * 2]);
        }
        const float inv_sum_scale =
            __fdividef(1.f, warp_sum + 1e-6f);
        cur_val *= inv_sum_scale;
        bfloat16x2 tmp_val;
        tmp_val[0] = cur_val[0];
        tmp_val[1] = cur_val[1];
        ((bfloat16x2*)out)[lane_id] = tmp_val;
    }
}
