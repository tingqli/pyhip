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

how to load 2D-tile from HBM ?
use load_dwordx4, lane-offset from tile-shape

load from LDS into MFMA layout
store to LDS normal layout : col-major

*/

// SFINAE
// https://www.cppstories.com/2016/02/notes-on-c-sfinae/
// https://stackoverflow.com/questions/48045559/how-do-i-declare-sfinae-class

template<int shift, int mask>
__device__ __inline__ int swizzle_col(int logical_row, int logical_col) {
    return (logical_col^(logical_row >> shift))&(mask);
}

template<int shift, int mask>
struct SwizzleCol {
    __device__ __inline__ static int swizzle(int logical_row, int logical_col) {
        return (logical_col^(logical_row >> shift))&(mask);
    }
};


constexpr bool is_powerof2(int v) {
    return v && ((v & (v - 1)) == 0);
}

constexpr int clog2(int val) {
    if (val == 1) return 0;
    return 1 + clog2(val >> 1);
}

/*
    2D tile at Work-group-level

    threads: number of threads in work-group
*/
using DWORD4 = int32x4_t;
template<typename T, int rows, int cols, int nthreads>
struct WGTile {
    static_assert((sizeof(T) * cols) % sizeof(DWORD4) == 0, "cannot load a row using DWORD4");

    static constexpr int warp_Size = 64;

    static constexpr int lane_cols = sizeof(T) * cols / sizeof(DWORD4);

    static_assert(is_powerof2(lane_cols));
    static_assert(warp_Size % lane_cols == 0, "cannot load all rows using all lanes");

    static constexpr int wg_rows = nthreads / lane_cols;
    static constexpr int lane_rows = warp_Size / lane_cols;
    static constexpr int lane_col_shift = clog2(lane_cols);

    static_assert(rows % wg_rows == 0, "cannot load all rows using all threads");
    static constexpr int num_dwordx4 = rows / wg_rows;

    __device__ static int voff(int nstride) {
        return (threadIdx.x & (lane_cols-1))*sizeof(DWORD4) + (threadIdx.x >> lane_col_shift) * nstride * sizeof(T);
    }

    // register temp
    int32x4_t data[num_dwordx4];

    __device__ void prefetch(BufferResource& buffer, int soffset, int nstride) {
        auto lane_offset = voff(nstride);
        #pragma unroll
        for(int r = 0; r < num_dwordx4; r++) {
            // one such load at WG level produces lane_rows
            data[r] = buffer.load_dwordx4(lane_offset, (soffset + r * wg_rows * nstride)*sizeof(T));    
        }
    }

    template<int nstride, typename F>
    __device__ void store(T* pdst) {
        auto row = (threadIdx.x >> lane_col_shift);
        auto col = F::swizzle(row, threadIdx.x & (lane_cols-1));
        auto lane_offset = (col*sizeof(DWORD4)/sizeof(T) + row * nstride);
        #pragma unroll
        for(int r = 0; r < num_dwordx4; r++) {
            *(int32x4_t*)(pdst + lane_offset + r * wg_rows * nstride) = data[r];
        }
    }
};


/*
Warp level MFMA A/B register
*/
template<int BM, int BK, int IM=32, int IK=8>
struct Warp_MFMA_AB {
    static_assert((IM == 32 && IK == 8));
    static_assert((BM % IM) == 0);
    static_assert((BK % IK) == 0);

    static constexpr int M_shift = clog2(IM);
    static constexpr int nM = BM / IM;
    static constexpr int nK = BK / IK;

    union ABType {
        // (M=32,K=8x4) halfs, each lane has 16 halfs
        float16x4 h[nK]; // 32x8[x4]xhalfs
        int32x4_t d[nK/2]; // 2 * 32x16xhalfs
    } regs[nM];

    template<int nstride>
    __device__ void load(__fp16 * buff) {
        int lane = threadIdx.x & 63;
        #pragma unroll
        for(int ik = 0; ik < nK/2; ik ++) {
            auto row = (lane & (IM-1));
            auto col = swizzle_col<1,3>(row, (lane >> M_shift) + ik*2);
            auto lane_off = row * nstride + col * 8;
            #pragma unroll
            for(int m = 0; m < nM; m++) {
                regs[m].d[ik] = *(int32x4_t*)(buff + lane_off + m*32*nstride);
            }
        }
    }
};

using ABTile = WGTile<__fp16, 256, 32, 256>;
using ABRegs = Warp_MFMA_AB<128, 32>;


#define SGB_VMEM_read_0x0020 0x0020
#define SGB_MFMA_0x0008      0x0008
#define SGB_DS_read_0x0100   0x0100
#define SGB_DS_write_0x0200  0x0200

__global__ void __launch_bounds__(256, 1) gemm_tile_256x256x32(
        __fp16* A, __fp16* B, int nstrideAB, float* C, int nstrideC,
        int nblkK, int* blk_maps) {
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    int lane = threadIdx.x & 63;

    auto blk_index = blockIdx.x;
    auto blkY = blk_maps[blk_index*2 + 0];
    auto blkX = blk_maps[blk_index*2 + 1];

    A += blkY * 256 * nstrideAB;
    B += blkX * 256 * nstrideAB;
    C += blkX * 256 + blkY * 256 * nstrideC;

    BufferResource bufferA(A, 256*nstrideAB * sizeof(__fp16));
    BufferResource bufferB(B, 256*nstrideAB * sizeof(__fp16));

    __shared__ __fp16 Abuff[256*32*2];
    __shared__ __fp16 Bbuff[256*32*2];

    constexpr int lds_nstride = 32;

    ABRegs a[2];
    ABRegs b[2];
    auto* Abuff_warp = Abuff + (warp_id >> 1)*32*4*lds_nstride;
    auto* Bbuff_warp = Bbuff + (warp_id & 1)*32*4*lds_nstride;
    float32x16 c[16] = {0};

    auto mfma = [](ABRegs &a, ABRegs &b, float32x16 (&c)[16]) {
        for(int ik = 0; ik < 4; ik += 1) {
            for(int m = 0; m < 4; m++) {
                for(int n = 0; n < 4; n ++) {
                    auto i = m*4 + n;
                    c[i] = __builtin_amdgcn_mfma_f32_32x32x8f16(
                                a.regs[m].h[ik], b.regs[n].h[ik], c[i], 0, 0, 0);
                    //asm("v_mfma_f32_32x32x8_f16 %0, %1, %2, %3\n"
                    //            : "+a"(c[i])
                    //            : "v"(a.regs[m].h[ik]), "v"(b.regs[n].h[ik]), "a"(c[i])
                    //            :);
                }
            }
        }
    };

    ABTile tileA;
    ABTile tileB;

    // prelog
    using swizzle = SwizzleCol<1, 3>;
    tileA.prefetch(bufferA, 0*32, nstrideAB);
    tileB.prefetch(bufferB, 0*32, nstrideAB);
    tileA.store<32,swizzle>(Abuff);
    tileB.store<32,swizzle>(Bbuff);

    tileA.prefetch(bufferA, 1*32, nstrideAB);
    tileB.prefetch(bufferB, 1*32, nstrideAB);

    __syncthreads();
    a[0].load<lds_nstride>(Abuff_warp + 0);
    b[0].load<lds_nstride>(Bbuff_warp + 0);
    tileA.store<32,swizzle>(Abuff + 256*32);
    tileB.store<32,swizzle>(Bbuff + 256*32);
    tileA.prefetch(bufferA, 2*32, nstrideAB);
    tileB.prefetch(bufferB, 2*32, nstrideAB);

    // body
    for(int ok = 0; ok < nblkK; ok += 2) {
        __syncthreads();

        constexpr auto LDS0 = 0;
        constexpr auto LDS1 = 256*32;

        tileA.store<32,swizzle>(Abuff + LDS0);
        tileB.store<32,swizzle>(Bbuff + LDS0);
        tileA.prefetch(bufferA, (ok+3)*32, nstrideAB);
        tileB.prefetch(bufferB, (ok+3)*32, nstrideAB);

        mfma(a[0], b[0], c);
        // as long as MFMA instruction issued, we can start READ a&b register for next iteration
        // and no need to worry about RAW ?
        a[1].load<lds_nstride>(Abuff_warp + LDS1);
        b[1].load<lds_nstride>(Bbuff_warp + LDS1);

        __syncthreads();

        tileA.store<32,swizzle>(Abuff + LDS1);
        tileB.store<32,swizzle>(Bbuff + LDS1);
        tileA.prefetch(bufferA, (ok+4)*32, nstrideAB);
        tileB.prefetch(bufferB, (ok+4)*32, nstrideAB);

        mfma(a[1], b[1], c);
        a[0].load<lds_nstride>(Abuff_warp + LDS0);
        b[0].load<lds_nstride>(Bbuff_warp + LDS0);

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
