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

    // printf("threadid:%d\n", threadIdx.x );
    int lane = threadIdx.x % 64;
    int rowid = lane % 32;
    int colid = lane / 32;
    int wgid_m = blockIdx.x;
    int wgid_n = blockIdx.y;
    __fp16* A_ptr_lane = A_ptr + (wgid_m*BM + warp_id_m*WAVE_M)*K + rowid*K + colid*(REG_K*MFMA_KL);
    __fp16* B_ptr_lane = B_ptr + (wgid_n*BN + warp_id_n*WAVE_N)*K + rowid*K + colid*(REG_K*MFMA_KL);
    float* C_warp = C_ptr + (wgid_m*BM + warp_id_m*WAVE_M)*N + (wgid_n*BN + warp_id_n*WAVE_N);

    //REGK is register factor based on MFMA ISA. Here 2 REGK are combined to meet minim 128 bit HBM read for coalescing tranction.
    float16x8 a[REG_M*REG_K/2];
    float16x8 b[REG_N*REG_K/2];
    float32x16 c[REG_M*REG_N] = {0};
    for(int k_idx = 0; k_idx < K; k_idx += BK)
    {
        auto tmp_A = A_ptr_lane;
        #pragma unroll
        for(int m = 0; m < REG_M; m++) {
            #pragma unroll
            //Contineous reading the REG_K*MFMA_KL
            for (int i = 0; i < REG_K/2; i++) {
                a[m*REG_K/2 + i] = *(float16x8*)(tmp_A+i*8);
            }
            tmp_A += 32*K;
        }
        // {
        //     auto& aa=a[0];
        //     printf("[%d, colid:%d]: %f, %f, %f, %f, %f, %f, %f,%f\n", rowid, colid,aa[0],aa[1],aa[2],aa[3],aa[4],aa[5],aa[6],aa[7]);
        // }
        // {
        //     auto& aa=a[1];
        //     printf("[%d, colid:%d]: %f, %f, %f, %f, %f, %f, %f,%f\n", rowid, colid,aa[0],aa[1],aa[2],aa[3],aa[4],aa[5],aa[6],aa[7]);
        // }

        auto tmp_B = B_ptr_lane;
        #pragma unroll
        for(int n = 0; n < REG_N; n++) {
            #pragma unroll
            for (int i = 0; i < REG_K/2; i++) {
                b[n*REG_K/2 + i] = *(float16x8*)(tmp_B+i*8);
            }
            tmp_B += 32*K;
        }
        A_ptr_lane += BK;
        B_ptr_lane += BK;
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

    #pragma unroll
    for(int m = 0; m < REG_M; m++) {
        #pragma unroll
        for(int n = 0; n < REG_N; n++) {
            auto idx = m*REG_N + n;
            auto& v = c[idx];

            float* p0 = C_warp + (m*32+ colid*4)*N + n*32+rowid;
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
