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
// using float32x16 = float  __attribute__ ((ext_vector_type(16)));
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
    int rowid = lane % 16;
    int colid = lane / 16;
    int wgid_m = blockIdx.x;
    int wgid_n = blockIdx.y;
    __fp16* A_warp = A_ptr + (wgid_m*BM + warp_id_m*WAVE_M)*K;
    __fp16* B_warp = B_ptr + (wgid_n*BN + warp_id_n*WAVE_N)*K;
    float* C_warp = C_ptr + (wgid_m*BM + warp_id_m*WAVE_M)*N + (wgid_n*BN + warp_id_n*WAVE_N);

    float16x8 a[REG_M];
    float16x8 b[REG_N];
    float32x4 c[REG_M*REG_N] = {0};
    #pragma unroll
    for(int m = 0; m < REG_M; m++) {
#if INTERLEAVE_ACCESS
        a[m] = *(float16x8*)(A_warp + (rowid*REG_M + m)*K + colid*8);
#else
        a[m] = *(float16x8*)(A_warp + (rowid + m*16)*K + colid*8);
#endif

    }
    #pragma unroll
    for(int n = 0; n < REG_N; n++) {
#if INTERLEAVE_ACCESS
     b[n] = *(float16x8*)(B_warp + (rowid*REG_N + n)*K + colid*8);
#else
    b[n] = *(float16x8*)(B_warp + (rowid + n*16)*K + colid*8);
#endif
    }

    // for(int k = 0; k < K; k += BK) {
        #pragma unroll
        for(int m = 0; m < REG_M; m++) {
            #pragma unroll
            for(int n = 0; n < REG_N; n++) {
                auto i = m*REG_N + n;
                // amdgcn_mfma_f32_16x16x16bf16(a[m].lo, b[n].lo, c[i]);
                // amdgcn_mfma_f32_16x16x16bf16(a[m].hi, b[n].hi, c[i]);
                c[i] = __builtin_amdgcn_mfma_f32_16x16x16f16(a[m].lo, b[n].lo, c[i], 0, 0, 0);
                c[i] = __builtin_amdgcn_mfma_f32_16x16x16f16(a[m].hi, b[n].hi, c[i], 0, 0, 0);

            }
        }
    // }
    #pragma unroll
    for(int m = 0; m < REG_M; m++) {
        #pragma unroll
        for(int n = 0; n < REG_N; n++) {
            auto idx = m*REG_N + n;
            auto& v = c[idx];
#if INTERLEAVE_ACCESS
            float* p0 = C_warp + (m+ colid*4*REG_M)*N + n + rowid*REG_N;
            *p0= v[0];
            p0 += REG_M*N;
            *p0 = v[1];
            p0 += REG_M*N;
            *p0 = v[2];
            p0 += REG_M*N;
            *p0 = v[3];
#else
            float* p0 = C_warp + (m*16+ colid*4)*N + n*16+rowid;
            *p0= v[0];
            p0 += N;
            *p0 = v[1];
            p0 += N;
            *p0 = v[2];
            p0 += N;
            *p0 = v[3];
#endif
        }
    }
}
