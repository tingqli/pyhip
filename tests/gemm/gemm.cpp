#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

#define INTERLEAVE_ACCESS 0
using float16x8 = __fp16 __attribute__ ((ext_vector_type(8)));
using float32x4 = float  __attribute__ ((ext_vector_type(4)));


// using float16x4 = __fp16 __attribute__ ((ext_vector_type(4)));
// using float16x8 = __fp16 __attribute__ ((ext_vector_type(8)));
// using float16x32 = __fp16 __attribute__ ((ext_vector_type(32)));
using float32x16 = float  __attribute__ ((ext_vector_type(16)));
using float32x4 = float  __attribute__ ((ext_vector_type(4)));
using float32x2 = float  __attribute__ ((ext_vector_type(2)));
using int32x4_t = int  __attribute__ ((ext_vector_type(4)));
using int32x16_t = int  __attribute__ ((ext_vector_type(16)));
using uint32x2_t = uint  __attribute__ ((ext_vector_type(2)));

__device__ __inline__
static uint buffer_load_dword(int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.i32");
__device__ __inline__
static int32x4_t buffer_load_dwordx4(int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

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
    __device__ __inline__ int32x4_t load_dwordx4(int32_t voffset, int32_t soffset) {
        return buffer_load_dwordx4(descriptor, voffset, soffset, 0);
    }
};



__global__ void __launch_bounds__(256, 1) run_mfma(__fp16* A_ptr, __fp16* B_ptr, float* C_ptr, int K,int N) {
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x) >> 6 ;
    int warp_id_m = warp_id / N_WAVES;
    int warp_id_n = warp_id % N_WAVES;

     __shared__ __fp16 Abuff[BM*BK*2];
    __shared__ __fp16 Bbuff[BN*BK*2];
    // printf("threadid:%d\n", threadIdx.x );
    int lane = threadIdx.x % 64;
    int fma_rowid = lane % 32;
    int fma_colid = lane / 32;
    int wgid_m = blockIdx.x;
    int wgid_n = blockIdx.y;
#define ELEMENTS  8// 128 bits load from HBM and store into LDS;
     //For one warp, each row would read BK from HBM and store BK into LDS. BK/ELEMENTS is how many lanes are needed for one row.
    int prefetch_rowid = lane / (BK/ELEMENTS);
    int prefetch_colid = lane % ((BK/ELEMENTS));
    constexpr uint waves = M_WAVES*N_WAVES;
    //For one warp, how many rows(each row BK) would be loaded.
    constexpr uint rows_per_load = 64 * ELEMENTS / BK;
    //For one lane, how many regs are needed to prefetch A and B.
    // assert(BM == BN);
    constexpr uint A_load_regs = BM*BK/(64*(waves)*ELEMENTS);
    constexpr uint B_load_regs = BN*BK/(64*(waves)*ELEMENTS);


    int prefetch_voff =  prefetch_rowid*K + prefetch_colid*(ELEMENTS);


    __fp16* A_ptr_prefetch = A_ptr + (wgid_m*BM + warp_id*(BM/(waves)))*K + prefetch_voff;
    __fp16* B_ptr_prefetch = B_ptr + (wgid_n*BN + warp_id*(BN/(waves)))*K +  prefetch_voff;


    A_ptr += wgid_m * BM * K;
    B_ptr += wgid_n * BN * K;

    BufferResource bufferA(A_ptr, BM * K * sizeof(__fp16));
    BufferResource bufferB(B_ptr, BN * K * sizeof(__fp16));

    //A,B LDS store ping-pong buffer.
    auto A_LDS0_store = Abuff + warp_id*(BM/(waves)) * BK + prefetch_rowid*BK;
    auto B_LDS0_store = Bbuff + warp_id*(BN/(waves)) * BK + prefetch_rowid*BK;
    auto A_LDS1_store = A_LDS0_store+BM*BK;
    auto B_LDS1_store = B_LDS0_store+BN*BK;

    //A,B LDS load ping-pong buffer.
    auto A_LDS0_load = Abuff + warp_id_m*WAVE_M * BK + fma_rowid*BK;
    auto B_LDS0_load = Bbuff + warp_id_n*WAVE_N * BK + fma_rowid*BK;
    auto A_LDS1_load = A_LDS0_load +BM*BK;
    auto B_LDS1_load = B_LDS0_load +BN*BK;

    //Store into 0 first.
    auto A_LDS_store = A_LDS0_store;
    auto B_LDS_store = B_LDS0_store;

    float* C_warp = C_ptr + (wgid_m*BM + warp_id_m*WAVE_M)*N + (wgid_n*BN + warp_id_n*WAVE_N);
    float16x8 prefecth_A[A_load_regs];
    float16x8 prefecth_B[B_load_regs];
    auto prefetch = [](float16x8 *vreg, __fp16* src) {
        vreg[0]= *(float16x8*)(src);
    };

    auto lds_write = [](float16x8 *prefetch_data, __shared__ __fp16* buffer) {
        *(float16x8*)(buffer) = prefetch_data[0];
    };

    auto lds_read = [](float16x8 *vreg, __shared__ __fp16* src, int fma_rowid, int fma_colid) {
        int  swizzle_colid;
        for (int idx = 0; idx < REG_K/2; idx++) {
            swizzle_colid = ((fma_rowid>>1)^(fma_colid + 2*idx)) % 4;
            vreg[idx] = *(float16x8*)(src+swizzle_colid*8);
        }
    };


    float32x16 c[REG_M*REG_N] = {0};
    float16x8 a[REG_M*REG_K/2];
    float16x8 b[REG_N*REG_K/2];

    auto tmp_pref_A = A_ptr_prefetch;
    #pragma unroll
    for(int m = 0; m < A_load_regs; m++) {
        prefetch(&prefecth_A[m],  tmp_pref_A);
        tmp_pref_A += rows_per_load*K;
    }

    #pragma unroll
    for(int m = 0; m < A_load_regs; m++) {
        auto swizzle_col_id = (prefetch_colid ^ (((prefetch_rowid + m*rows_per_load) % 32)/2)) % 4;
        auto tmp_store_A = A_LDS_store + (rows_per_load*BK)*m + swizzle_col_id *(ELEMENTS);
        lds_write(&prefecth_A[m], tmp_store_A);
    }

    auto tmp_pref_B = B_ptr_prefetch;
    #pragma unroll
    for(int n = 0; n < B_load_regs; n++) {
        prefetch(&prefecth_B[n],  tmp_pref_B);
        tmp_pref_B += rows_per_load*K;
    }


    #pragma unroll
    for(int n = 0; n < B_load_regs; n++) {
        auto swizzle_col_id = (prefetch_colid ^ (((prefetch_rowid + n*rows_per_load)%32)/2)) % 4;
        auto tmp_store_B = B_LDS_store + (rows_per_load*BK)*n + swizzle_col_id *(ELEMENTS);
        lds_write(&prefecth_B[n], tmp_store_B);
    }
    A_ptr_prefetch += BK;
    B_ptr_prefetch += BK;

    //prepare to load 0
    auto A_LDS_load = A_LDS0_load;
    auto B_LDS_load = B_LDS0_load;
    //prepare to store 1
    A_LDS_store = A_LDS1_store;
    B_LDS_store = B_LDS1_store;
    // __syncthreads();

    int cnt = 0;
    for(int k_idx = BK,cnt = 0; k_idx < K; k_idx += BK, cnt++)
    {
        __syncthreads();

        {
            auto tmp_load_A = A_LDS_load;
            #pragma unroll
            for(int m = 0; m < REG_M; m++) {
                lds_read(&a[m*REG_K/2],tmp_load_A, fma_rowid, fma_colid);
                tmp_load_A += BK*32;
            }
            auto tmp_load_B = B_LDS_load;
            #pragma unroll
            for(int n = 0; n < REG_N; n++) {
                lds_read(&b[n*REG_K/2],tmp_load_B, fma_rowid, fma_colid);
                tmp_load_B += BK*32;
            }
        }

        tmp_pref_A = A_ptr_prefetch;
        #pragma unroll
        for(int m = 0; m < A_load_regs; m++) {
            prefetch(&prefecth_A[m],  tmp_pref_A);
            tmp_pref_A += rows_per_load*K;
        }

        
        tmp_pref_B = B_ptr_prefetch;
        #pragma unroll
        for(int n = 0; n < B_load_regs; n++) {
            prefetch(&prefecth_B[n],  tmp_pref_B);
            tmp_pref_B += rows_per_load*K;
        }


        {
            #pragma unroll
            for(int m = 0; m < REG_M; m++) {
                #pragma unroll
                for(int n = 0; n < REG_N; n++) {
                    auto idx = m*REG_N + n;
                    #pragma unroll
                    for (int i = 0; i < REG_K/2; i++) {
                        c[idx] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[m*REG_K/2 + i].lo, b[n*REG_K/2 + i].lo, c[idx], 0, 0, 0);
                        c[idx] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[m*REG_K/2 + i].hi, b[n*REG_K/2 + i].hi, c[idx], 0, 0, 0);
                    }
                }
            }
        }
    
        // __syncthreads();

        {
            #pragma unroll
            for(int m = 0; m < A_load_regs; m++) {
                auto swizzle_col_id = (prefetch_colid ^ (((prefetch_rowid + m*rows_per_load) % 32)/2)) % 4;
                auto tmp_store_A = A_LDS_store + (rows_per_load*BK)*m + swizzle_col_id *(ELEMENTS);
                lds_write(&prefecth_A[m], tmp_store_A);
            }

            #pragma unroll
            for(int n = 0; n < B_load_regs; n++) {
                auto swizzle_col_id = (prefetch_colid ^ (((prefetch_rowid + n*rows_per_load)%32)/2)) % 4;
                auto tmp_store_B = B_LDS_store + (rows_per_load*BK)*n + swizzle_col_id *(ELEMENTS);
                lds_write(&prefecth_B[n], tmp_store_B);
            }
        }

        
        //update prefetch 
        A_ptr_prefetch += BK;
        B_ptr_prefetch += BK;
        //update ping-pong.
        A_LDS_store = cnt & 0x01 ? A_LDS1_store : A_LDS0_store;
        B_LDS_store = cnt & 0x01 ? B_LDS1_store : B_LDS0_store;
        A_LDS_load = cnt & 0x01 ? A_LDS0_load : A_LDS1_load;
        B_LDS_load = cnt & 0x01 ? B_LDS0_load : B_LDS1_load;
    }

    __syncthreads();
    //tails handling.
    {
        {
            auto tmp_load_A = A_LDS_load;
            #pragma unroll
            for(int m = 0; m < REG_M; m++) {

                lds_read(&a[m*REG_K/2],tmp_load_A, fma_rowid, fma_colid);
                tmp_load_A += BK*32;
            }
            auto tmp_load_B = B_LDS_load;
            #pragma unroll
            for(int n = 0; n < REG_N; n++) {
                lds_read(&b[n*REG_K/2],tmp_load_B, fma_rowid, fma_colid);
                tmp_load_B += BK*32;
            }
        }

        {
            #pragma unroll
            for(int m = 0; m < REG_M; m++) {
                #pragma unroll
                for(int n = 0; n < REG_N; n++) {
                    auto idx = m*REG_N + n;
                    #pragma unroll
                    for (int i = 0; i < REG_K/2; i++) {
                        c[idx] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[m*REG_K/2 + i].lo, b[n*REG_K/2 + i].lo, c[idx], 0, 0, 0);
                        c[idx] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[m*REG_K/2 + i].hi, b[n*REG_K/2 + i].hi, c[idx], 0, 0, 0);
                    }
                }
            }
        }
    }

    #pragma unroll
    for(int m = 0; m < REG_M; m++) {
        #pragma unroll
        for(int n = 0; n < REG_N; n++) {
            auto idx = m*REG_N + n;
            auto& v = c[idx];

            float* p0 = C_warp + (m*32+ fma_colid*4)*N + n*32+fma_rowid;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                *p0= v[i*4];
                p0 += N;
                *p0 = v[i*4+1];
                p0 += N;
                *p0 = v[i*4+2];
                p0 += N;
                *p0 = v[i*4+3];
                //Move to next 8M.
                p0 +=5*N;
            }
        }
    }
}
