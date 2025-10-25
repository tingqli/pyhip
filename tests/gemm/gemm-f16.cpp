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
    for mem-coalesing reason, each warp should load (256/4)x32 = 64x32 halfs
    each dwordx4 instruction should load 16x32 halfs, with 8-halfs per lane:

        lane0 lane1 lane2 lane3
        lane4 lane5 lane6 lane7 
        ... ...
*/

template<int shift, int mask>
__device__ __inline__ int swizzle_col(int logical_row, int logical_col) {
    return (logical_col^(logical_row >> shift))&(mask);
}

struct block_copy_256x32 {
    int32x4_t d4x4[4];
    __device__ void prefetch(BufferResource& buff, int soffset, int nstride) {
        auto lane_offset = ((threadIdx.x & 3)*8 + (threadIdx.x >> 2) * nstride) * sizeof(__fp16);
        //auto lane_offset = ((threadIdx.x & 7)*8 + (threadIdx.x >> 3) * nstride) * sizeof(__fp16);
        //auto lane_offset = (threadIdx.x * 8) * sizeof(__fp16);
        d4x4[0] = buffer_load_dwordx4(buff.descriptor, lane_offset, (soffset + 0)*sizeof(__fp16), 0);
        d4x4[1] = buffer_load_dwordx4(buff.descriptor, lane_offset, (soffset + 64*1*nstride)*sizeof(__fp16), 0);
        d4x4[2] = buffer_load_dwordx4(buff.descriptor, lane_offset, (soffset + 64*2*nstride)*sizeof(__fp16), 0);
        d4x4[3] = buffer_load_dwordx4(buff.descriptor, lane_offset, (soffset + 64*3*nstride)*sizeof(__fp16), 0);
    }
    __device__ void store(__fp16* pdst, int nstride) {
        auto lane_offset = ((threadIdx.x & 3)*8 + (threadIdx.x >> 2) * nstride);
        *(int32x4_t*)(pdst + lane_offset + 0) = d4x4[0];
        *(int32x4_t*)(pdst + lane_offset + 64*1*nstride) = d4x4[1];
        *(int32x4_t*)(pdst + lane_offset + 64*2*nstride) = d4x4[2];
        *(int32x4_t*)(pdst + lane_offset + 64*3*nstride) = d4x4[3];
    }
    template<int nstride>
    __device__ void store(__fp16* pdst) {
        auto row = (threadIdx.x >> 2);
        auto col = swizzle_col<1, 3>(row, threadIdx.x & 3);
        auto lane_offset = (col*8 + row * nstride);
        *(int32x4_t*)(pdst + lane_offset + 0) = d4x4[0];
        *(int32x4_t*)(pdst + lane_offset + 64*1*nstride) = d4x4[1];
        *(int32x4_t*)(pdst + lane_offset + 64*2*nstride) = d4x4[2];
        *(int32x4_t*)(pdst + lane_offset + 64*3*nstride) = d4x4[3];
    }
};

#define SGB_VMEM_read_0x0020 0x0020
#define SGB_MFMA_0x0008      0x0008
#define SGB_DS_read_0x0100   0x0100
#define SGB_DS_write_0x0200  0x0200
/*
__global__ void __launch_bounds__(256, 1) xxxtile(int* A, int N, int * B) {
    union bigtype {
        int32x4_t dwx4[2];
        float16x4 dhx4[4];
    } buff;

    BufferResource bufferA(A, N * 128 * sizeof(__fp16));

    auto lane_offset = ((threadIdx.x & 3)*8 + (threadIdx.x >> 2) * 256) * sizeof(__fp16);
    buff.dwx4[0] = buffer_load_dwordx4(bufferA.descriptor, lane_offset, (64*0)*sizeof(int), 0);
    buff.dwx4[1] = buffer_load_dwordx4(bufferA.descriptor, lane_offset, (64*1)*sizeof(int), 0);

    float32x16 c = {};
    for(int i=0; i < N; i++) {

        c = __builtin_amdgcn_mfma_f32_32x32x8f16(buff.dhx4[0], buff.dhx4[1], c, 0, 0, 0);
        c = __builtin_amdgcn_mfma_f32_32x32x8f16(buff.dhx4[2], buff.dhx4[3], c, 0, 0, 0);

        buff.dwx4[0] = buffer_load_dwordx4(bufferA.descriptor, lane_offset, (64*i)*sizeof(int), 0);
        buff.dwx4[1] = buffer_load_dwordx4(bufferA.descriptor, lane_offset, (64*i+32)*sizeof(int), 0);

        __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SGB_VMEM_read_0x0020, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SGB_VMEM_read_0x0020, 1, 0);
    }
    *(float32x16*)B = c;
}

*/
__global__ void __launch_bounds__(256, 1) gemm_tile_256x256x32(
        __fp16* A, __fp16* B, int nstrideAB, float* C, int nstrideC,
        int nblkK, int nblkM, int nblkN) {
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    int lane = threadIdx.x & 63;

    auto blk_index = blockIdx.x;
#if 0
    // naive block <-> CU mapping
    auto blkY = blk_index / nblkN;
    auto blkX = blk_index % nblkN;
#else
    // mapps blocks within same XCC/XCD closer
    // here we assume CU's of 4 XCC/XCD distributed into a 2x2 Tile
    auto xcc_id = blk_index & 3;
    blk_index = blk_index >> 2;
    auto blkY = (blk_index / (nblkN/2)) + (xcc_id >> 1)*(nblkM/2);
    auto blkX = (blk_index % (nblkN/2)) + (xcc_id & 1)*(nblkN/2);
#endif
    A += blkY * 256 * nstrideAB;
    B += blkX * 256 * nstrideAB;
    C += blkX * 256 + blkY * 256 * nstrideC;

    BufferResource bufferA(A, 256*nstrideAB * sizeof(__fp16));
    BufferResource bufferB(B, 256*nstrideAB * sizeof(__fp16));

    __shared__ __fp16 Abuff[256*32*2];
    __shared__ __fp16 Bbuff[256*32*2];

    block_copy_256x32 copyA;
    block_copy_256x32 copyB;

    constexpr int lds_nstride = 32;

    struct ABRegs {
        union ABType {
            // (M=32,K=8x4) halfs, each lane has 16 halfs
            float16x4 h[4]; // 32x8[x4]xhalfs
            int32x4_t d[2]; // 2 * 32x16xhalfs
        } regs[4];

        __device__ void load(__fp16 * buff) {
            int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
            int lane = threadIdx.x & 63;
            auto LDS0 = ((0 + 0)&1)*256*32;
            for(int ik = 0; ik < 2; ik ++) {
                auto row = (lane & 31);
                auto col = swizzle_col<1,3>(row, (lane >> 5) + ik*2);
                auto lane_off = row * lds_nstride + col * 8;
                for(int m = 0; m < 4; m++) {
                    regs[m].d[ik] = *(int32x4_t*)(buff + lane_off + m*32*lds_nstride);
                }
            }
        }
    };

    ABRegs a[2];
    ABRegs b[2];
    auto* Abuff_warp = Abuff + (warp_id >> 1)*32*4*lds_nstride;
    auto* Bbuff_warp = Bbuff + (warp_id & 1)*32*4*lds_nstride;
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
  
    /*
              copy.prefetch(data0)
              copy.store(data0, LDS0)

              copy.prefetch(data1)

              __syncthreads();
              copy.store(data1, LDS1)
              copy.prefetch(data2)
              REG1.lds_load(data0, LDS0)
        for:
              __syncthreads();

              copy.store(data2, LDS0)
              copy.prefetch(data3)

              REG1.MFMA(data0)
              REG1.lds_load(data1, LDS1)

              swap LDS0 with LDS1
    */

    copyA.prefetch(bufferA, 0*32, nstrideAB);
    copyB.prefetch(bufferB, 0*32, nstrideAB);
    copyA.store<32>(Abuff);
    copyB.store<32>(Bbuff);
    copyA.prefetch(bufferA, 1*32, nstrideAB);
    copyB.prefetch(bufferB, 1*32, nstrideAB);

    __syncthreads();
    a[0].load(Abuff_warp + 0);
    b[0].load(Bbuff_warp + 0);
    copyA.store<32>(Abuff + 256*32);
    copyB.store<32>(Bbuff + 256*32);
    copyA.prefetch(bufferA, 2*32, nstrideAB);
    copyB.prefetch(bufferB, 2*32, nstrideAB);
    
    for(int ok = 0; ok < nblkK; ok += 2) {
        __syncthreads();

        constexpr auto LDS0 = 0;
        constexpr auto LDS1 = 256*32;

        copyA.store<32>(Abuff + LDS0);
        copyB.store<32>(Bbuff + LDS0);
        copyA.prefetch(bufferA, (ok+3)*32, nstrideAB);
        copyB.prefetch(bufferB, (ok+3)*32, nstrideAB);

        for(int ik = 0; ik < 4; ik += 1) {
            for(int m = 0; m < 4; m++) {
                for(int n = 0; n < 4; n ++) {
                    auto i = m*4 + n;
                    c[i] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[0].regs[m].h[ik], b[0].regs[n].h[ik], c[i], 0, 0, 0);
                }
            }
            __builtin_amdgcn_sched_barrier(~SGB_MFMA_0x0008);
        }
        // as long as MFMA instruction issued, we can start READ a&b register for next iteration
        // and no need to worry about RAW ?
        a[1].load(Abuff_warp + LDS1);
        b[1].load(Bbuff_warp + LDS1);

        __syncthreads();

        copyA.store<32>(Abuff + LDS1);
        copyB.store<32>(Bbuff + LDS1);
        copyA.prefetch(bufferA, (ok+4)*32, nstrideAB);
        copyB.prefetch(bufferB, (ok+4)*32, nstrideAB);

        for(int ik = 0; ik < 4; ik += 1) {
            for(int m = 0; m < 4; m++) {
                for(int n = 0; n < 4; n ++) {
                    auto i = m*4 + n;
                    c[i] = __builtin_amdgcn_mfma_f32_32x32x8f16(a[1].regs[m].h[ik], b[1].regs[n].h[ik], c[i], 0, 0, 0);
                }
            }
            __builtin_amdgcn_sched_barrier(~SGB_MFMA_0x0008);
        }
        // as long as MFMA instruction issued, we can start READ a&b register for next iteration
        // and no need to worry about RAW ?
        a[0].load(Abuff_warp + LDS0);
        b[0].load(Bbuff_warp + LDS0);


        /*
        0x0200 DS write     ds_write_b128  x 8
        0x0100 DS read      ds_read2_b64 x 16
        0x0008 MFMA         v_mfma_f32_32x32x8_f16 x 64
        0x0020 VMEM read    buffer_load_dwordx4 x 8
        */ 
#if 1
        for(int i = 0; i < 8;i++) {
            __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_DS_write_0x0200, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_VMEM_read_0x0020, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 4, 0);
            //__builtin_amdgcn_sched_group_barrier(SGB_DS_read_0x0100, 1, 0);
        }
        for(int i = 0; i < 16; i++) {
            __builtin_amdgcn_sched_group_barrier(SGB_DS_read_0x0100, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 1, 0);
        }
        __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 64-5*8-16, 0);


        for(int i = 0; i < 8;i++) {
            __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_DS_write_0x0200, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_VMEM_read_0x0020, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 4, 0);
            //__builtin_amdgcn_sched_group_barrier(SGB_DS_read_0x0100, 1, 0);
        }
        for(int i = 0; i < 16; i++) {
            __builtin_amdgcn_sched_group_barrier(SGB_DS_read_0x0100, 1, 0);
            __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 1, 0);
        }
        __builtin_amdgcn_sched_group_barrier(SGB_MFMA_0x0008, 64-5*8-16, 0);

        //__builtin_amdgcn_sched_group_barrier(SGB_DS_read_0x0100, 16, 0);
#endif
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
