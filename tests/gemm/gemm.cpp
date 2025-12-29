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
// using float32x4 = float  __attribute__ ((ext_vector_type(4)));
// using float32x2 = float  __attribute__ ((ext_vector_type(2)));
// using int32x4_t = int  __attribute__ ((ext_vector_type(4)));
// using int32x16_t = int  __attribute__ ((ext_vector_type(16)));
// using uint32x2_t = uint  __attribute__ ((ext_vector_type(2)));


__global__ void __launch_bounds__(256, 1) run_mfma(__fp16* A_ptr, __fp16* B_ptr, float* C_ptr, int K,int N) {
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x) >> 6 ;
    int warp_id_m = warp_id / N_WAVES;
    int warp_id_n = warp_id % N_WAVES;

     __shared__ __fp16 Abuff[BM*BK];
    __shared__ __fp16 Bbuff[BN*BK];

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
    constexpr uint A_load_regs = BM*BK/(64*(waves)*ELEMENTS);
    constexpr uint B_load_regs = BN*BK/(64*(waves)*ELEMENTS);

    int offset_load_hbm =  prefetch_rowid*K + prefetch_colid*(ELEMENTS);
    int offset_store_lds =  prefetch_rowid*BK + prefetch_colid*(ELEMENTS);

    __fp16* A_ptr_prefetch = A_ptr + (wgid_m*BM + warp_id*(BM/(waves)))*K + offset_load_hbm;
    __fp16* B_ptr_prefetch = B_ptr + (wgid_n*BN + warp_id*(BN/(waves)))*K +  offset_load_hbm;

    // auto lds_store_A = Abuff + warp_id*(BM/(waves)) * BK + offset_store_lds;
    // auto lds_store_B = Bbuff + warp_id*(BN/(waves)) * BK + offset_store_lds;
    // //Each lane would read REG_K*REG_KL number elements.
    // auto lds_load_A = Abuff + warp_id_m*WAVE_M * BK + fma_rowid*BK + fma_colid*(ELEMENTS*REG_K/2);
    // auto lds_load_B = Bbuff + warp_id_n*WAVE_N * BK + fma_rowid*BK + fma_colid*(ELEMENTS*REG_K/2);

    auto Abuff_store = Abuff + warp_id*(BM/(waves)) * BK + prefetch_rowid*BK;
    auto Bbuff_store = Bbuff + warp_id*(BN/(waves)) * BK + prefetch_rowid*BK;

    auto Abuff_load = Abuff + warp_id_m*WAVE_M * BK + fma_rowid*BK;
    auto Bbuff_load = Bbuff + warp_id_n*WAVE_N * BK + fma_rowid*BK;

    float* C_warp = C_ptr + (wgid_m*BM + warp_id_m*WAVE_M)*N + (wgid_n*BN + warp_id_n*WAVE_N);


    //REGK is register factor based on MFMA ISA. Here 2 REGK are combined to meet minim 128 bit HBM read for coalescing tranction.

    float32x16 c[REG_M*REG_N] = {0};
    constexpr int swizzle_cols=4;
    for(int k_idx = 0; k_idx < K; k_idx += BK)
    {
        __syncthreads();
        {
            float16x8 prefecth_A[A_load_regs];
            auto tmp_A = A_ptr_prefetch;
            #pragma unroll
            for(int m = 0; m < A_load_regs; m++) {
                auto swizzle_col_id = (prefetch_colid ^ (((prefetch_rowid + m*rows_per_load) % 32)/2)) % 4;
                auto tmp_store_A = Abuff_store + (rows_per_load*BK)*m + swizzle_col_id *(ELEMENTS);
                prefecth_A[m]= *(float16x8*)tmp_A;
                *(float16x8*)(tmp_store_A) = prefecth_A[m];
                tmp_A += rows_per_load*K;
            }
            // {
            //     auto& aa=a[0];
            //     printf("[%d, colid:%d]: %f, %f, %f, %f, %f, %f, %f,%f\n", rowid, colid,aa[0],aa[1],aa[2],aa[3],aa[4],aa[5],aa[6],aa[7]);
            // }
            // {
            //     auto& aa=a[1];
            //     printf("[%d, colid:%d]: %f, %f, %f, %f, %f, %f, %f,%f\n", rowid, colid,aa[0],aa[1],aa[2],aa[3],aa[4],aa[5],aa[6],aa[7]);
            // }

            auto tmp_B = B_ptr_prefetch;
            float16x8 prefecth_B[B_load_regs];
            #pragma unroll
            for(int n = 0; n < B_load_regs; n++) {
                auto swizzle_col_id = (prefetch_colid ^ (((prefetch_rowid + n*rows_per_load)%32)/2)) % 4;
                auto tmp_store_B = Bbuff_store + (rows_per_load*BK)*n + swizzle_col_id *(ELEMENTS);
                prefecth_B[n]= *(float16x8*)tmp_B;
                *(float16x8*)(tmp_store_B) = prefecth_B[n];
                tmp_B += rows_per_load*K;
            }
        }
        A_ptr_prefetch += BK;
        B_ptr_prefetch += BK;
        __syncthreads();
        float16x8 a[REG_M*REG_K/2];
        float16x8 b[REG_N*REG_K/2];
        {
            auto tmp_load_A = Abuff_load;

            #pragma unroll
            for(int m = 0; m < REG_M; m++) {

                int  swizzle_colid;
                for (int k = 0; k < REG_K/2; k++) {
                    swizzle_colid = ((fma_rowid/2)^(fma_colid + 2*k)) % 4;
                    a[m*REG_K/2 + k] = *(float16x8*)(tmp_load_A+swizzle_colid*8);
                }
                tmp_load_A += BK*32;
            }
            auto tmp_load_B = Bbuff_load;
            #pragma unroll
            for(int n = 0; n < REG_N; n++) {
                int  swizzle_colid;
                for (int k = 0; k < REG_K/2; k++) {
                    swizzle_colid = ((fma_rowid/2)^(fma_colid + 2*k)) % 4;
                    b[n*REG_K/2 + k] = *(float16x8*)(tmp_load_B+swizzle_colid*8);
                }
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
