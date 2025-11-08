#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

/************************************************
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

    Is `s_waitcnt_vmcnt` in C++ dangerous because you can't stop compiler from insertting more
    VMEM-instructions between your-VMEMs & s_waitcnt_vmcnt?

    No, if compiler inserts more VMEMs, it only means your `s_waitcnt_vmcnt<N>()` do not guarentee exact N VMEMs
    is running, but it still guarentees your-VMEMs before N your-VMEMs are all finished.
    same argument can be made to s_waitcnt_lgkmcnt<N>(). so program is still valid, but the extra waits may be incured.
*************************************************/

using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float16x8 = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
using float16x32 = __attribute__((__vector_size__(32 * sizeof(__fp16)))) __fp16;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x16_t = __attribute__((__vector_size__(16 * sizeof(int)))) int;

using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;
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

template<int shift, int mask>
struct SwizzleNo {
    __device__ __inline__ static int swizzle(int logical_row, int logical_col) {
        return logical_col;
    }
};

constexpr bool is_powerof2(int v) {
    return v && ((v & (v - 1)) == 0);
}

constexpr int clog2(int val) {
    if (val == 1) return 0;
    return 1 + clog2(val >> 1);
}

template<uint16_t cnt>
__device__ void s_waitcnt_vmcnt() {
    asm volatile ("s_waitcnt vmcnt(%0)\n"::"i"(cnt));
}
template<uint16_t cnt>
__device__ void s_waitcnt_lgkmcnt() {
    asm volatile ("s_waitcnt lgkmcnt(%0)\n"::"i"(cnt));
}
/*
    2D tile at Work-group-level

    threads: number of threads in work-group
*/
using DWORD4 = int32x4_t;
template<typename T, int rows, int cols, int nthreads, typename FSwizzle>
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

    template<int r>
    __device__ void prefetch(BufferResource& buffer, int soffset, int nstride) {
        int lane_offset = voff(nstride);
        static_assert(r >=0 && r < num_dwordx4);
        // one such load at WG level produces lane_rows
        int soff = (soffset + r * wg_rows * nstride)*sizeof(T);
        //data[r] = buffer.load_dwordx4(lane_offset, soff);
        asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
            :[vdst]"=v"(data[r])
            :[vaddr]"v"(lane_offset), [srsrc]"s"(buffer.descriptor), [soffset]"s"(soff));
        //s_waitcnt_vmcnt<0>();
    }
    __device__ void prefetch(BufferResource& buffer, int soffset, int nstride) {
        prefetch<0>(buffer, soffset, nstride);
        prefetch<1>(buffer, soffset, nstride);
        prefetch<2>(buffer, soffset, nstride);
        prefetch<3>(buffer, soffset, nstride);
        s_waitcnt_vmcnt<0>();
    }

    template<int r>
    __device__ void store(T* pdst) {
        constexpr int nstride = cols;
        static_assert(r >=0 && r < num_dwordx4);
        auto row = (threadIdx.x >> lane_col_shift);
        auto col = FSwizzle::swizzle(row, threadIdx.x & (lane_cols-1));
        auto lane_offset = (col*sizeof(DWORD4)/sizeof(T) + row * nstride);
        constexpr int offset = r * wg_rows * nstride * sizeof(T);
        as3_uint32_ptr vaddr = (as3_uint32_ptr)(pdst + lane_offset);
        //*(int32x4_t*)(vaddr) = data[r];
        asm volatile("ds_write_b128 %[vaddr], %[vdata] offset:%[offset]"
                    ::[vaddr]"v"(vaddr), [vdata]"v"(data[r]), [offset]"i"(offset));
    }

    __device__ void store(T* pdst) {
        store<0>(pdst);
        store<1>(pdst);
        store<2>(pdst);
        store<3>(pdst);
        s_waitcnt_lgkmcnt<0>();
    }
};
 
#define NUM_THREADS 256

#if 1
    #define INST_M 32
    #define INST_N 32
    #define INST_K 8
#else
    #define INST_M 16
    #define INST_N 16
    #define INST_K 16
#endif
// number of MFMA C blocks within a warp
#define WARP_M 4
#define WARP_N 4
#define WARP_K 4
#define BLK_M (INST_M * WARP_M * 2)
#define BLK_N (INST_N * WARP_N * 2)
#define BLK_K (INST_K * WARP_K)



/*
Warp level LDS Buffer view, each warp has it's own sub-tile
*/
template<typename T, int nCols, typename FSwizzle>
struct LDS_buff {
    T * base;

    __device__ LDS_buff(T * base) : base(base) {}

    // (m,k) is (row,col) in unit of 32x16 tile
    template<int m, int k>
    __device__ float16x8 MFMA_ld_32x16_fp16(int offset=0) {
        constexpr int MFMA_M = 32;
        constexpr int M_shift = clog2(MFMA_M);
        float16x8 v;
        int lane = threadIdx.x & 63;
        auto row = (lane & (MFMA_M-1));
        auto col = FSwizzle::swizzle(row, (lane >> M_shift) + k * (MFMA_M == 32 ? 1:2));
        auto lane_off = row * nCols + col * 8;
        //(int32x4_t&)(regs[m][ik]) = *(int32x4_t*)(buff + lane_off + m*INST_M*nCols);
        as3_uint32_ptr vaddr = (as3_uint32_ptr)(base + offset + lane_off + m*MFMA_M*nCols);
        asm volatile("ds_read_b128 %[vdst], %[vaddr] offset:%[offset]"
                    : [vdst]"=v"((int32x4_t&)(v))
                    : [vaddr]"v"(vaddr),[offset]"i"(0));
        return v;
    }
};


__device__ void amdgcn_mfma_f32_32x32x8f16(float16x4 a, float16x4 b, float32x16& c) {
    //c = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0);
    asm volatile("v_mfma_f32_32x32x8_f16 %0, %1, %2, %3\n"
                : "+a"(c)
                : "v"(a), "v"(b), "a"(c)
                :);
};

template<int m, int n, int k, int nM, int nN, int nK>
__device__ void my_mfma_mnk(float16x4 (&a)[nM][nK], float16x4 (&b)[nN][nK], float32x16 (&c)[16]) {
    auto i = m*4 + n;
    //c[i] = __builtin_amdgcn_mfma_f32_32x32x8f16(a.regs[m][ik], b.regs[n][ik], c[i], 0, 0, 0);
    amdgcn_mfma_f32_32x32x8f16(a[m][k], b[n][k], c[i]);
}

#define SGB_VMEM_read_0x0020 0x0020
#define SGB_MFMA_0x0008      0x0008
#define SGB_DS_read_0x0100   0x0100
#define SGB_DS_write_0x0200  0x0200

__global__ void __launch_bounds__(256, 1) gemm(__fp16* A, __fp16* B, int nstrideAB, float* C, int nstrideC, int K) {
    auto nblkK = K / BLK_K;
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    int lane = threadIdx.x & 63;

    //auto blk_index = blockIdx.x;
    //auto blkY = blk_maps[blk_index*2 + 0];
    //auto blkX = blk_maps[blk_index*2 + 1];

    int blkX, blkY;
#if 1
    blkX = blockIdx.x;
    blkY = blockIdx.y;
    if (0)
    {
        // 1 XCD
        auto blk_index = blockIdx.x + blockIdx.y * gridDim.x;
        auto xcd_h = gridDim.y;
        auto xcd_w = gridDim.x;        
        auto xcd_off = blk_index;
        auto xcd_x0 = 0;
        auto xcd_y0 = 0;
        auto M01 = 9;
        auto xcd_panel_y_idx = (xcd_off / (M01*xcd_w));
        auto xcd_panel_y0 = xcd_panel_y_idx * M01;
        auto xcd_panel_height = M01;
        if (xcd_panel_y0 + xcd_panel_height > xcd_h)
            xcd_panel_height = xcd_h - xcd_panel_y0;

        auto xcd_panel_off = (xcd_off % (M01*xcd_w));
        auto x = xcd_panel_off / xcd_panel_height;
        auto y = xcd_panel_off % xcd_panel_height;

        blkY = xcd_y0 + (xcd_panel_y_idx * M01) + y;
        blkX = xcd_x0 + ((xcd_panel_y_idx & 1) ? (xcd_w - 1 - x) : (x));
    }
#else
    {
        auto blk_index = blockIdx.x + blockIdx.y * gridDim.x;
        auto xcd_h = gridDim.y / 2; // XCD's tile size(assume dividable)
        auto xcd_w = gridDim.x / 2;
        auto xcd_id = (blk_index & 3);   // on which XCD are we running?
        auto xcd_off = (blk_index >> 2); // block index within XCD area
        auto xcd_x0 = (xcd_id >> 1) * xcd_w;
        auto xcd_y0 = (xcd_id & 1) * xcd_h;
        //auto x = xcd_off % xcd_w;
        //auto y = xcd_off / xcd_w;
        auto M01 = 9;
        auto xcd_panel_y_idx = (xcd_off / (M01*xcd_w));
        auto xcd_panel_y0 = xcd_panel_y_idx * M01;
        auto xcd_panel_height = M01;
        if (xcd_panel_y0 + xcd_panel_height > xcd_h)
            xcd_panel_height = xcd_h - xcd_panel_y0;

        auto xcd_panel_off = (xcd_off % (M01*xcd_w));
        auto x = xcd_panel_off / xcd_panel_height;
        auto y = xcd_panel_off % xcd_panel_height;

        blkY = xcd_y0 + (xcd_panel_y_idx * M01) + y;
        blkX = xcd_x0 + ((xcd_panel_y_idx & 1) ? (xcd_w - 1 - x) : (x));
    }
#endif
    A += blkY * BLK_M * nstrideAB;
    B += blkX * BLK_N * nstrideAB;
    C += blkX * BLK_N + blkY * BLK_M * nstrideC;

    BufferResource bufferA(A, BLK_M*nstrideAB * sizeof(__fp16));
    BufferResource bufferB(B, BLK_N*nstrideAB * sizeof(__fp16));

    __shared__ __fp16 Abuff[BLK_M*BLK_K];
    __shared__ __fp16 Bbuff[BLK_N*BLK_K];
    constexpr int lds_nstride = BLK_K;

#if INST_M == 32
    using swizzle = SwizzleCol<1, 3>;
#else
    using swizzle = SwizzleCol<0, 7>;
#endif

    auto* Abuff_warp = Abuff + (warp_id >> 1)*(INST_M * WARP_M)*BLK_K;
    auto* Bbuff_warp = Bbuff + (warp_id & 1)*(INST_N * WARP_N)*BLK_K;

    LDS_buff<__fp16, lds_nstride, swizzle> ldsA0(Abuff_warp);
    LDS_buff<__fp16, lds_nstride, swizzle> ldsA1(Abuff_warp + BLK_M*BLK_K);

    LDS_buff<__fp16, lds_nstride, swizzle> ldsB0(Bbuff_warp);
    LDS_buff<__fp16, lds_nstride, swizzle> ldsB1(Bbuff_warp + BLK_M*BLK_K);

    float16x4 Aregs[WARP_M][WARP_K];
    float16x4 Bregs[WARP_N][WARP_K];
#if INST_M == 32
    float32x16 c[16] = {0};
#else
    float32x4 c[16] = {0};
#endif
    using ABTile = WGTile<__fp16, BLK_M, BLK_K, NUM_THREADS, swizzle>;
    ABTile tileA;
    ABTile tileB;

    #define MFMA(m,n,k) my_mfma_mnk<m, n, k>(Aregs, Bregs, c);
    #define LDA(m,k)    (float16x8&)(Aregs[m][k]) = ldsA0.MFMA_ld_32x16_fp16<m, k>();
    #define LDB(n,k)    (float16x8&)(Bregs[n][k]) = ldsB0.MFMA_ld_32x16_fp16<n, k>();

    // prelog: before entering main loop, we need
    //   - LDS contains A/B data at ok=0
            tileA.prefetch(bufferA, 0*BLK_K, nstrideAB);
            tileB.prefetch(bufferB, 0*BLK_K, nstrideAB);
            s_waitcnt_vmcnt<0>();
            tileA.store(Abuff);
            tileB.store(Bbuff);
            __syncthreads();
    //   - prefetching A/B data at ok=1
            tileA.prefetch(bufferA, 1*BLK_K, nstrideAB);
            tileB.prefetch(bufferB, 1*BLK_K, nstrideAB);
    //   - partially A/B registers has been loadded from LDS
            LDA(0,0);    LDB(0,0);
            LDA(1,0);    LDB(1,0);
            LDA(2,0);    LDB(2,0);
            LDA(3,0);    LDB(3,0);
    // loop body
    for(int ok = 0; ok < nblkK; ok ++) {
        // Stage1 :
        // ARegs(m=0,1,2,3, k=0/1) & BRegs(m=0,1,2,3, k=0/1) have been fetched into Registers, we can start MFMA directly
        // w/o waitting initial LDS read arrival, and interleaving with the rest LDS reads ABRegs(k=2/3)
        /*
            MFMA has 4x4x4 = 64 instances
        */

        s_waitcnt_lgkmcnt<4>();
        MFMA(0,0,0); LDA(0,2);
        MFMA(0,1,0); LDB(0,2);
        MFMA(1,0,0); LDA(1,2);
        MFMA(1,1,0); LDB(1,2);

        s_waitcnt_lgkmcnt<4>();
        MFMA(0,2,0); LDA(2,2);
        MFMA(0,3,0); LDB(2,2);
        MFMA(1,2,0); LDA(3,2);
        MFMA(1,3,0); LDB(3,2);
        // extra time to ensure LDS reads are all done before enter stage2
        MFMA(2,0,0);
        MFMA(2,1,0);
        MFMA(2,2,0);
        MFMA(2,3,0);

        MFMA(3,0,0);
        MFMA(3,1,0);
        MFMA(3,2,0);
        MFMA(3,3,0);
        s_waitcnt_lgkmcnt<0>();

        // Stage2 : LDS has been read by all waves,
        //          write data from next-iter & prefetch for next-next-iter
        __syncthreads();
        #define LDW_PFA(row, m,k) \
            s_waitcnt_vmcnt<7>(); \
            tileA.store<row>(Abuff); \
            MFMA(m,0,k); \
            tileA.prefetch<row>(bufferA, (ok+2)*BLK_K, nstrideAB); \
            MFMA(m,1,k); MFMA(m,2,k); MFMA(m,3,k);

        #define LDW_PFB(row, m,k) \
            s_waitcnt_vmcnt<7>(); \
            tileB.store<row>(Bbuff); \
            MFMA(m,0,k); \
            tileB.prefetch<row>(bufferB, (ok+2)*BLK_K, nstrideAB); \
            MFMA(m,1,k); MFMA(m,2,k); MFMA(m,3,k);

        LDW_PFA(0, 0, 1);
        LDW_PFA(1, 1, 1);
        LDW_PFA(2, 2, 1);
        LDW_PFA(3, 3, 1);

        LDW_PFB(0, 0, 2);
        LDW_PFB(1, 1, 2);
        LDW_PFB(2, 2, 2);
        LDW_PFB(3, 3, 2);

        // Stage3: wait for LDS writes from all waves to finish, so we can read ABregs for next-iter in advance
        // this also means we must finish using these ABregs before stage3
        // MFMA(0..3, 0..3, 3);
        s_waitcnt_lgkmcnt<0>();
        __syncthreads();
        LDA(0,0);MFMA(0,0,3);    LDB(0,0);MFMA(0,1,3);
        LDA(1,0);MFMA(0,2,3);    LDB(1,0);MFMA(0,3,3);
        LDA(2,0);MFMA(1,0,3);    LDB(2,0);MFMA(1,1,3);
        LDA(3,0);MFMA(1,2,3);    LDB(3,0);MFMA(1,3,3);

        MFMA(2,0,3);
        MFMA(2,1,3);
        MFMA(2,2,3);
        MFMA(2,3,3);

        MFMA(3,0,3);
        MFMA(3,1,3);
        MFMA(3,2,3);
        MFMA(3,3,3);
    }
    // this wait can fix GPU Memory access fault
    // maybe due to early termination before async loads return data to regs
    s_waitcnt_vmcnt<0>();

#if INST_M == 32
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
#else
    for(int m = 0; m < 4; m++) {
        for(int n = 0; n < 4; n ++) {
            auto i = m*4 + n;
            auto& v = c[i];
            auto warp_off = (warp_id >> 1)*16*4*nstrideC + (warp_id & 1)*16*4;
            auto* p0 = C + (lane>>4)*4*nstrideC + (lane & 15) + m*16*nstrideC + n*16  + warp_off;

            auto* p = p0;
            p[0] = v[0]; p += nstrideC;
            p[0] = v[1]; p += nstrideC;
            p[0] = v[2]; p += nstrideC;
            p[0] = v[3]; p += nstrideC;
        }
    }
#endif
}
