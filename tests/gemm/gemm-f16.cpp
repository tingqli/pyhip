#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

/*
GEMM Kernel trait

Workgroup size: 256
Wave number: 4
Wave size: 64
DataType: FP16
TileSize(MNK): 256 x 256 x 32
Layout: RCR, RRR
MMA: v_mfma_f32_32x32x8f16

for MI308 (80 Workgroup on 80 CUs)
    M=256*10
    N=256*8
    K=8192

clang/include/clang/Basic/BuiltinsAMDGPU.def
TARGET_BUILTIN(__builtin_amdgcn_mfma_f32_32x32x8f16, "V16fV4hV4hV16fIiIiIi", "nc", "mai-insts")
TARGET_BUILTIN(__builtin_amdgcn_mfma_f32_16x16x16f16, "V4fV4hV4hV4fIiIiIi", "nc", "mai-insts")

*/
using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

__global__ void __launch_bounds__(256, 1) gemm_tile_256x256x32(__fp16* A, __fp16* B, int nstrideAB, float* C, int nstrideC) {
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    int lane = threadIdx.x & 63;
    auto i_off = (lane & 31) * nstrideAB;
    auto k_off = (lane >> 5) * 4;
    
    float16x4 a[4];
    float16x4 b[4];
    float32x16 c[16] = {0};

    for(int m = 0; m < 4; m++) {
        a[m] = *(float16x4*)(A + i_off + k_off + m*32*nstrideAB);
    }

    for(int n = 0; n < 4; n ++) {
        b[n] = *(float16x4*)(B + i_off + k_off + n*32*nstrideAB);
    }

    for(int m = 0; m < 4; m++) {
        for(int n = 0; n < 4; n ++) {
            auto i = m*4 + n;
            c[i] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[m], b[n], c[i], 0, 0, 0);
        }
    }

    for(int m = 0; m < 4; m++) {
        for(int n = 0; n < 4; n ++) {
            auto i = m*4 + n;
            auto& v = c[i];
            auto* p0 = C + ((lane>>5)*4)*nstrideC + (lane & 31) + m*32*nstrideC + n*32;

            for (int i=0; i < 4; i++, p0 += 8*nstrideC) {
                auto* p = p0;
                p[0] = v[i*4+0]; p += nstrideC;
                p[0] = v[i*4+1]; p += nstrideC;
                p[0] = v[i*4+2]; p += nstrideC;
                p[0] = v[i*4+3]; p += nstrideC;
            }
        }
    }
}
