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

template<typename F, size_t... Is>
constexpr void compile_time_loop_impl(F&& f, std::index_sequence<Is...>) {
    (f(std::integral_constant<size_t, Is>{}), ...);
}

template<size_t N, typename F>
constexpr void compile_time_loop(F&& f) {
    compile_time_loop_impl(std::forward<F>(f), 
                          std::make_index_sequence<N>{});
}

using DWORDX4 = int32x4_t;

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

// MFMA_LDS: LDS buffer used for MFMA only, with swizzle / coorperative load
// to best use LDS load bandwidth, we load with ds_read_b128, each lane need to contain 8xhalf
template<typename T, int MFMA_M, int MFMA_K, int nRows, int nCols, int nthreads, typename FSwizzle>
struct MFMA_LDS_ABbuff {
    static_assert(MFMA_K == 16); // in MFMA_32x8 case, we interleave 2 such instruction togeter
    static_assert(nRows % MFMA_M == 0);
    static_assert(nCols % MFMA_K == 0);

    static constexpr int M_shift = clog2(MFMA_M);
    __device__ MFMA_LDS_ABbuff() {}

    static constexpr int lane_cols = sizeof(T) * nCols / sizeof(DWORDX4);
    static constexpr int lane_col_shift = clog2(lane_cols);
    static constexpr int wg_rows = nthreads / lane_cols;
    static_assert(nRows % wg_rows == 0, "cannot load all nRows using all threads");
    static constexpr int num_dwordx4 = nRows / wg_rows;

    // nstride : 
    template<int r>
    __device__ DWORDX4 prefetch_dwordx4(BufferResource& buffer, int soffset, int nstride) {
        DWORDX4 data;
        int lane_offset = (threadIdx.x & (lane_cols-1))*sizeof(DWORDX4) + (threadIdx.x >> lane_col_shift) * nstride * sizeof(T);
        static_assert(r >=0 && r < num_dwordx4);
        // one such load at WG level produces lane_rows
        int soff = (soffset + r * wg_rows * nstride)*sizeof(T);
        //data[r] = buffer.load_dwordx4(lane_offset, soff);
        asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
            :[vdst]"=v"(data)
            :[vaddr]"v"(lane_offset), [srsrc]"s"(buffer.descriptor), [soffset]"s"(soff));
        return data;
    }

    // coorperativly loaded DWORDx4 data from global memory
    // all threads within a thread-block(not just 1 wave)
    template<int r>
    __device__ void store_dwordx4(T* base, DWORDX4 data) {
        constexpr int nstride = nCols;
        static_assert(r >=0 && r < num_dwordx4);
        auto row = (threadIdx.x >> lane_col_shift);
        auto col = FSwizzle::swizzle(row, threadIdx.x & (lane_cols-1));
        auto lane_offset = (col*sizeof(DWORDX4)/sizeof(T) + row * nstride);
        constexpr int imm_offset = r * wg_rows * nstride * sizeof(T);
        as3_uint32_ptr vaddr = (as3_uint32_ptr)(base + lane_offset);
        //*(int32x4_t*)(vaddr) = data[r];
        asm volatile("ds_write_b128 %[vaddr], %[vdata] offset:%[offset]"
                    ::[vaddr]"v"(vaddr), [vdata]"v"(data), [offset]"i"(imm_offset));
    }

    __device__ void prefetch_dwordx4(BufferResource& buffer, int soffset, int nstride, DWORDX4 (&tempA)[num_dwordx4]) {
        compile_time_loop<num_dwordx4>([&](auto i){
            constexpr int index = i;
            tempA[index] = prefetch_dwordx4<index>(buffer, soffset, nstride);
        });
    }
    __device__ void store_dwordx4(T* base, DWORDX4 (&tempA)[num_dwordx4]) {
        compile_time_loop<num_dwordx4>([&](auto i){
            constexpr int index = i;
            store_dwordx4<index>(base, tempA[index]);
        });
    }

    // load into MFMA A/B register, each warp has it's own additional imm_offset
    template<int m, int k>
    __device__ float16x8 load(T* base) {
        float16x8 v;
        int lane = threadIdx.x & 63;
        auto row = (lane & (MFMA_M-1));
        auto col = FSwizzle::swizzle(row, (lane >> M_shift) + k * (MFMA_M == 32 ? 1:2));
        auto lane_off = row * nCols + col * 8;
        constexpr int imm_offset = m*MFMA_M*nCols*sizeof(T);
        as3_uint32_ptr vaddr = (as3_uint32_ptr)(base + lane_off);
        asm volatile("ds_read_b128 %[vdst], %[vaddr] offset:%[offset]"
                    : [vdst]"=v"((int32x4_t&)(v))
                    : [vaddr]"v"(vaddr),[offset]"i"(imm_offset));
        return v;
    }

    // prefetch directly into LDS
    // buffer_load_dword lds cannot swizzle in LDS store, so we have to
    // do it on the load side:
    //
    //                 load-raster                 store-swizzle
    //     global_row_col(lane_id) ====swizzle===> lds_row, lds_col
    //
    //                 load-swizzle                store-raster
    //     
    // so we want to load swizzled (row,col) of each lane:
    //
    static constexpr int lane_cols_dword = sizeof(T) * nCols / sizeof(int32_t);
    static_assert(lane_cols_dword <= 64);
    static_assert((64 % lane_cols_dword) == 0);
    static constexpr int wg_rows_dword = nthreads / lane_cols_dword;     // 16 wg_rows
    static constexpr int num_dwords = nRows / wg_rows_dword;

    template<int lds_base_idx, int r>
    __device__ void setM0(T* base) {
        int row = __builtin_amdgcn_readfirstlane(threadIdx.x >> clog2(lane_cols_dword));
        int m0_offset = __builtin_amdgcn_readfirstlane((r * wg_rows_dword + row) * nCols + lds_base_idx);
        as3_uint32_ptr lds_offset = (as3_uint32_ptr)(base + m0_offset);
        asm volatile("s_mov_b32 m0, %0\n\ts_nop 1\n"::"s"(lds_offset));
    }
    template<int dr=1>
    __device__ void incM0() {
        constexpr int d_m0 = (dr * wg_rows_dword) * nCols  * sizeof(T);
        asm volatile("s_addk_i32 m0, %0"::"i"(d_m0));
    }

    template<int r, int imm_off = 0>
    __device__ void prefetch_async_lds(BufferResource& buffer, int soffset, int nstride) {
        static_assert(r >= 0 && r < num_dwords);

        soffset = (soffset + r * wg_rows_dword * nstride) * sizeof(T);

        // swizzle was done in unit of dwordx4
        uint32_t row = (threadIdx.x >> clog2(lane_cols_dword));
        uint32_t dword_idx = threadIdx.x & 3;
        uint32_t col = (FSwizzle::swizzle(row, (threadIdx.x & (lane_cols_dword-1)) >> 2) << 2) + dword_idx;
        uint32_t lane_offset = col*sizeof(int32_t) + (row * nstride * sizeof(T));

        asm volatile("buffer_load_dword %[vaddr], %[srsrc], %[soffset] offen offset:%[ioffset] lds\n"
            ::[vaddr]"v"(lane_offset), [srsrc]"s"(buffer.descriptor), [soffset]"s"(soffset), [ioffset]"i"(imm_off));
    }

    template<int lds_base_idx = 0>
    __device__ void prefetch_async_lds_all(T* base, BufferResource& buffer, int soffset, int nstride) {
        // initialize M0 with per-wave offset
        setM0<lds_base_idx, 0>(base);
        compile_time_loop<num_dwords>([&](auto i){
            constexpr int index = i;
            //setM0<index>();
            prefetch_async_lds<index>(buffer, soffset, nstride);
            incM0(); // increase by constant step after each load
        });
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

template <size_t N, typename Func>
void unroll_loop(Func&& f) {
    [&f] <size_t... Is> (std::index_sequence<Is...>) {
        (f(std::integral_constant<size_t, Is>{}), ...);
    }(std::make_index_sequence<N>{});
}

__global__ void __launch_bounds__(NUM_THREADS, 1) gemm(__fp16* A, __fp16* B, int nstrideAB, float* C, int nstrideC, int K) {
    auto nblkK = K / BLK_K;
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    int lane = threadIdx.x & 63;

    //auto blk_index = blockIdx.x;
    //auto blkY = blk_maps[blk_index*2 + 0];
    //auto blkX = blk_maps[blk_index*2 + 1];

    int blkX, blkY;
#if 0
    blkX = blockIdx.x;
    blkY = blockIdx.y;
    if (1)
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
        // why M01=8~9 is best?
        // 
        auto M01 = 8;
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

    auto Abuff_warp_off = (warp_id >> 1)*(INST_M * WARP_M)*BLK_K;
    auto Bbuff_warp_off = (warp_id & 1)*(INST_N * WARP_N)*BLK_K;

#if INST_M == 32
    using swizzle = SwizzleCol<1, 3>;
    MFMA_LDS_ABbuff<__fp16, 32, 8*2, 32*8, 8*4, NUM_THREADS, swizzle> ldsA0;
    MFMA_LDS_ABbuff<__fp16, 32, 8*2, 32*8, 8*4, NUM_THREADS, swizzle> ldsB0;
#else
    using swizzle = SwizzleCol<0, 7>;
#endif

    float16x4 Aregs[WARP_M][WARP_K];
    float16x4 Bregs[WARP_N][WARP_K];
#if INST_M == 32
    float32x16 c[16] = {0};
#else
    float32x4 c[16] = {0};
#endif

    DWORDX4 tempA[ldsA0.num_dwordx4];
    DWORDX4 tempB[ldsB0.num_dwordx4];

    #define MFMA(m,n,k) my_mfma_mnk<m, n, k>(Aregs, Bregs, c);
    #define LDA(m,k)    (float16x8&)(Aregs[m][k]) = ldsA0.load<m, k>(Abuff + Abuff_warp_off);
    #define LDB(n,k)    (float16x8&)(Bregs[n][k]) = ldsB0.load<n, k>(Bbuff + Bbuff_warp_off);

    // prelog: before entering main loop, we need
    //   - LDS contains A/B data at ok=0
            ldsA0.prefetch_async_lds_all(Abuff, bufferA, 0*BLK_K, nstrideAB);
            ldsB0.prefetch_async_lds_all(Bbuff, bufferB, 0*BLK_K, nstrideAB);
            s_waitcnt_vmcnt<0>();
            /*
            ldsA0.prefetch_dwordx4(bufferA, 0*BLK_K, nstrideAB, tempA);
            ldsB0.prefetch_dwordx4(bufferB, 0*BLK_K, nstrideAB, tempB);
            s_waitcnt_vmcnt<0>();
            ldsA0.store_dwordx4(tempA);
            ldsB0.store_dwordx4(tempB);
            */
            __syncthreads(); 
    //   - prefetching A/B data at ok=1
            ldsA0.prefetch_dwordx4(bufferA, 1*BLK_K, nstrideAB, tempA);
            ldsB0.prefetch_dwordx4(bufferB, 1*BLK_K, nstrideAB, tempB);
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
            ldsA0.store_dwordx4<row>(Abuff, tempA[row]); \
            MFMA(m,0,k); \
            tempA[row] = ldsA0.prefetch_dwordx4<row>(bufferA, (ok+2)*BLK_K, nstrideAB); \
            MFMA(m,1,k); MFMA(m,2,k); MFMA(m,3,k);

        #define LDW_PFB(row, m,k) \
            s_waitcnt_vmcnt<7>(); \
            ldsB0.store_dwordx4<row>(Bbuff, tempB[row]); \
            MFMA(m,0,k); \
            tempB[row] = ldsB0.prefetch_dwordx4<row>(bufferB, (ok+2)*BLK_K, nstrideAB); \
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
    #pragma unroll
    for(int m = 0; m < 4; m++) {
        #pragma unroll
        for(int n = 0; n < 4; n ++) {
            auto& v = c[m*4 + n];
            //auto& v = c[i*16];
            auto warp_off = (warp_id >> 1)*32*4*nstrideC + (warp_id & 1)*32*4;
            auto* p0 = C + ((lane>>5)*4)*nstrideC + (lane & 31) + m*32*nstrideC + n*32  + warp_off;
            #pragma unroll
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
    #pragma unroll
    for(int m = 0; m < 4; m++) {
        #pragma unroll
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


__global__ void __launch_bounds__(NUM_THREADS, 1) gemm_async_lds(__fp16* A, __fp16* B, int nstrideAB, float* C, int nstrideC, int K) {
    auto nblkK = K / BLK_K;
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    int lane = threadIdx.x & 63;

    //auto blk_index = blockIdx.x;
    //auto blkY = blk_maps[blk_index*2 + 0];
    //auto blkX = blk_maps[blk_index*2 + 1];

    int blkX, blkY;
#if 0
    blkX = blockIdx.x;
    blkY = blockIdx.y;
    if (1)
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
        // why M01=8~9 is best?
        // 
        auto M01 = 8;
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

    __shared__ __fp16 Abuff[BLK_M*BLK_K*2];
    __shared__ __fp16 Bbuff[BLK_N*BLK_K*2];
    constexpr int lds_nstride = BLK_K;

    auto Abuff_warp_off = (warp_id >> 1)*(INST_M * WARP_M)*BLK_K;
    auto Bbuff_warp_off = (warp_id & 1)*(INST_N * WARP_N)*BLK_K;

#if INST_M == 32
    using swizzle = SwizzleCol<1, 3>;
    using MBuff = MFMA_LDS_ABbuff<__fp16, 32, 8*2, 32*8, 8*4, NUM_THREADS, swizzle>;
    MBuff lds;

    constexpr uint32_t LDS_STEP = BLK_N*BLK_K;
#else
    using swizzle = SwizzleCol<0, 7>;
#endif

    float16x4 Aregs[WARP_M][WARP_K];
    float16x4 Bregs[WARP_N][WARP_K];
#if INST_M == 32
    float32x16 c[16] = {0};
#else
    float32x4 c[16] = {0};
#endif

    int LDS0 = 0;
    int LDS1 = LDS_STEP;
    #define MFMA(m,n,k) my_mfma_mnk<m, n, k>(Aregs, Bregs, c);
    #define LDA0(m,k)    (float16x8&)(Aregs[m][k]) = lds.load<m, k>(&Abuff[LDS0 + Abuff_warp_off]);
    #define LDB0(n,k)    (float16x8&)(Bregs[n][k]) = lds.load<n, k>(&Bbuff[LDS0 + Bbuff_warp_off]);
    #define LDA1(m,k)    (float16x8&)(Aregs[m][k]) = lds.load<m, k>(&Abuff[LDS1 + Abuff_warp_off]);
    #define LDB1(n,k)    (float16x8&)(Bregs[n][k]) = lds.load<n, k>(&Bbuff[LDS1 + Bbuff_warp_off]);

    {
    // prelog: before entering main loop, we need
    // load LDS0 with A0&B0
        lds.prefetch_async_lds_all(&Abuff[LDS0], bufferA, 0*BLK_K, nstrideAB);
        lds.prefetch_async_lds_all(&Bbuff[LDS0], bufferB, 0*BLK_K, nstrideAB);
        s_waitcnt_vmcnt<0>();
        __syncthreads(); 
    // start async prefetch to LDS1 with A1&B1
        lds.prefetch_async_lds_all(&Abuff[LDS1], bufferA, 1*BLK_K, nstrideAB);
        lds.prefetch_async_lds_all(&Bbuff[LDS1], bufferB, 1*BLK_K, nstrideAB);
    //   - partially A/B registers has been loadded from LDS
            LDA0(0,0);    LDB0(0,0);
            LDA0(1,0);    LDB0(1,0);
            LDA0(2,0);    LDB0(2,0);
            LDA0(3,0);    LDB0(3,0);
    }
    // loop body
    for(int ok = 0; ok < nblkK; ok +=1) {
        // Stage1 :
        // ARegs(m=0,1,2,3, k=0/1) & BRegs(m=0,1,2,3, k=0/1) have been fetched into Registers, we can start MFMA directly
        // w/o waitting initial LDS read arrival, and interleaving with the rest LDS reads ABRegs(k=2/3)
        /*
            MFMA has 4x4x4 = 64 instances
        */
        LDS0 = (ok&1)?LDS_STEP:0;
        LDS1 = (ok&1)?0:LDS_STEP;
        #define PREFETCHA0(row) lds.prefetch_async_lds<row>(bufferA, (ok+2)*BLK_K, nstrideAB); lds.incM0();
        #define PREFETCHB0(row) lds.prefetch_async_lds<row>(bufferB, (ok+2)*BLK_K, nstrideAB); lds.incM0();

        s_waitcnt_lgkmcnt<4>();
        MFMA(0,0,0); LDA0(0,2);
        MFMA(0,1,0); LDB0(0,2);
        MFMA(1,0,0); LDA0(1,2);
        MFMA(1,1,0); LDB0(1,2);

        s_waitcnt_lgkmcnt<4>();
        MFMA(0,2,0); LDA0(2,2);
        MFMA(0,3,0); LDB0(2,2);
        MFMA(1,2,0); LDA0(3,2);
        MFMA(1,3,0); LDB0(3,2);
        
        MFMA(2,0,0);
        MFMA(2,1,0);
        MFMA(2,2,0);
        MFMA(2,3,0);

        MFMA(3,0,0);
        MFMA(3,1,0);
        MFMA(3,2,0);
        MFMA(3,3,0);

        s_waitcnt_lgkmcnt<0>();
        __syncthreads(); // all waves read LDS0 finished, we can load into it
        lds.setM0<0,0>(&Abuff[LDS0]);
        compile_time_loop<16>([&](auto i){
            constexpr int r = i;
            constexpr int m = i / 4;
            constexpr int n = i % 4;
            PREFETCHA0(r);
            MFMA(m, n, 1);
        });

        lds.setM0<0,0>(&Bbuff[LDS0]);
        compile_time_loop<8>([&](auto i){
            constexpr int r = i;
            PREFETCHB0(r);
            constexpr int m = r / 2;
            constexpr int n = r % 2;
            MFMA(m, 2*n, 2);
            MFMA(m, 2*n+1, 2);
        });
        compile_time_loop<8>([&](auto i){
            constexpr int r = i;
            PREFETCHB0(r + 8);
            constexpr int m = r / 2;
            constexpr int n = r % 2;
            MFMA(m, 2*n, 3);
            MFMA(m, 2*n+1, 3);
        });

        // Stage3: wait for LDS writes from all waves to finish, so we can read ABregs for next-iter in advance
        // this also means we must finish using these ABregs before stage3
        // MFMA(0..3, 0..3, 3);
        // wait loads of last iteration(which loads ok+1) to finish
        s_waitcnt_vmcnt<32>();
        //asm volatile("s_barrier\n");
        __syncthreads();
        /*
        MFMA(0,0,3);    MFMA(0,1,3);
        MFMA(0,2,3);    MFMA(0,3,3);
        MFMA(1,0,3);    MFMA(1,1,3);
        MFMA(1,2,3);    MFMA(1,3,3);
        LDA0(0,0);//MFMA(2,0,3);
        LDA0(1,0);//MFMA(2,1,3);
        LDA0(2,0);//MFMA(2,2,3);
        LDA0(3,0);//MFMA(2,3,3);

        LDB0(0,0);//MFMA(3,0,3);
        LDB0(1,0);//MFMA(3,1,3);
        LDB0(2,0);//MFMA(3,2,3);
        LDB0(3,0);//MFMA(3,3,3);

        */
        LDA1(0,0);    LDB1(0,0);
        LDA1(1,0);    LDB1(1,0);
        LDA1(2,0);    LDB1(2,0);
        LDA1(3,0);    LDB1(3,0);
    }

    // this wait can fix GPU Memory access fault
    // maybe due to early termination before async loads return data to regs
    s_waitcnt_vmcnt<0>();

#if INST_M == 32
    #pragma unroll
    for(int m = 0; m < 4; m++) {
        #pragma unroll
        for(int n = 0; n < 4; n ++) {
            auto& v = c[m*4 + n];
            //auto& v = c[i*16];
            auto warp_off = (warp_id >> 1)*32*4*nstrideC + (warp_id & 1)*32*4;
            auto* p0 = C + ((lane>>5)*4)*nstrideC + (lane & 31) + m*32*nstrideC + n*32  + warp_off;
            #pragma unroll
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
    #pragma unroll
    for(int m = 0; m < 4; m++) {
        #pragma unroll
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
