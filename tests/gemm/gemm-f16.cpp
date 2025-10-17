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
using float16x32 = __attribute__((__vector_size__(32 * sizeof(__fp16)))) __fp16;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x16_t = __attribute__((__vector_size__(16 * sizeof(int)))) int;

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
__device__ __inline__
static int32x4_t buffer_load_dwordx4(int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
/*
    thread block level tile copy

    instruction-level   : 32x32x8
    warp-level          : 128x128x32 : 4x4x4 of instruction-level
    block-level         : 256x256x32  : 2x2 of warp-level

    2x2 warps prefetch A_256x32 & B_256x32 data from HBM into registers
    for 256x32, each warp loads 32x64 fp16 data.
*/
struct block_copy_256x32 {
    union bigType {
        float16x32 h32;
        int32x4_t d4x4[4];
        int32x16_t d16;
    } data;
    __device__ void prefetch(__fp16* psrc, int soffset, int nstride) {
        data.h32 = *(float16x32*)(psrc + soffset + threadIdx.x*nstride);
    }
    __device__ void prefetch(BufferResource& buff, int soffset, int nstride) {
        data.d4x4[0] = buffer_load_dwordx4(buff.descriptor, threadIdx.x*nstride*sizeof(__fp16), (soffset + 0)*sizeof(__fp16), 0);
        data.d4x4[1] = buffer_load_dwordx4(buff.descriptor, threadIdx.x*nstride*sizeof(__fp16), (soffset + 8)*sizeof(__fp16), 0);
        data.d4x4[2] = buffer_load_dwordx4(buff.descriptor, threadIdx.x*nstride*sizeof(__fp16), (soffset + 16)*sizeof(__fp16), 0);
        data.d4x4[3] = buffer_load_dwordx4(buff.descriptor, threadIdx.x*nstride*sizeof(__fp16), (soffset + 24)*sizeof(__fp16), 0);
    }
    __device__ void store(__fp16* pdst, int nstride) {
        *(int32x16_t*)(pdst + threadIdx.x*nstride) = data.d16;
    }
};

__global__ void __launch_bounds__(256, 1) gemm_tile_256x256x32(__fp16* A, __fp16* B, int nstrideAB, float* C, int nstrideC, int OuterK) {
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    int lane = threadIdx.x & 63;
    auto i_off = (lane & 31) * 32;
    auto k_off = (lane >> 5) * 4;

    BufferResource bufferA(A, 256*nstrideAB * sizeof(__fp16));
    BufferResource bufferB(B, 256*nstrideAB * sizeof(__fp16));

    //auto *bufferA = A;
    //auto *bufferB = B;

    __shared__ __fp16 Abuff[256*32*2];
    __shared__ __fp16 Bbuff[256*32*2];

    block_copy_256x32 copyA;
    block_copy_256x32 copyB;

    float16x4 a[4];
    float16x4 b[4];
    float32x16 c[16] = {0};

    // prefetch(data0) to regs
    // store regs(data0) to LDS1
    // prefetch(data1) to regs
    //
    // for:
    //  __syncthreads();
    //     switch LDS0 & 1
    //     MFMA(data0) on LDS0
    //     store regs(data1) to LDS1
    //     prefetch(data2) to regs
    //

    copyA.prefetch(bufferA, 0*32, nstrideAB);
    copyB.prefetch(bufferB, 0*32, nstrideAB);
    copyA.store(Abuff, 32);
    copyB.store(Bbuff, 32);

    copyA.prefetch(bufferA, 1*32, nstrideAB);
    copyB.prefetch(bufferB, 1*32, nstrideAB);

    for(int ok = 0; ok < OuterK; ok ++) {
        __syncthreads();

        auto LDS0 = ((ok + 0)&1)*256*32;
        auto LDS1 = ((ok + 1)&1)*256*32;

        copyA.store(Abuff + LDS1, 32);
        copyB.store(Bbuff + LDS1, 32);

        copyA.prefetch(bufferA, (ok+2)*32, nstrideAB);
        copyB.prefetch(bufferB, (ok+2)*32, nstrideAB);

        for(int k = 0; k < 32; k += 8) {
            for(int m = 0; m < 4; m++) {
                a[m] = *(float16x4*)(Abuff + LDS0 + i_off + k_off + m*32*32 + k + (warp_id >> 1)*32*4*32);
            }
            for(int n = 0; n < 4; n++) {
                b[n] = *(float16x4*)(Bbuff + LDS0 + i_off + k_off + n*32*32 + k + (warp_id & 1)*32*4*32);
            }
            for(int m = 0; m < 4; m++) {
                for(int n = 0; n < 4; n ++) {
                    auto i = m*4 + n;
                    c[i] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[m], b[n], c[i], 0, 0, 0);
                }
            }
        }
    }

    for(int m = 0; m < 4; m++) {
        for(int n = 0; n < 4; n ++) {
            auto i = m*4 + n;
            auto& v = c[i];
            auto warp_off = (warp_id >> 1)*32*4*nstrideC + (warp_id & 1)*32*4;
            auto* p0 = C + ((lane>>5)*4)*nstrideC + (lane & 31) + m*32*nstrideC + n*32  + warp_off;

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
