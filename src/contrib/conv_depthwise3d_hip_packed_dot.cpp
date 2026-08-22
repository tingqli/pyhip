/*
 * FP16/BF16 packed-dot depthwise Conv3D for PyHIP.
 *
 * gfx950 (CDNA4): FP16 via __builtin_amdgcn_fdot2 and BF16 via
 * __builtin_amdgcn_fdot2_f32_bf16 (dot12-insts).
 * gfx942 (CDNA3): FP16 via __builtin_amdgcn_fdot2 only; BF16 is routed to the
 * SGB kernel by conv_depthwise.py because dot12-insts is unavailable.
 *
 * IO_DTYPE must be __half or __hip_bfloat16. The kernel shares its packed
 * weights, LDS staging, 16-output tile, and FP32 accumulators across both data
 * types, while dot2_acc selects the matching dot-product builtin at compile
 * time.
 *
 * This is intentionally a separate implementation. The previous SGB and
 * original kernels remain available for comparison and fallback.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <type_traits>

//#define PaddingD 0
//#define PaddingH 2
//#define PaddingW 2
#define dilationD 1
#define dilationH 1
#define dilationW 1
#define strideT 1
#define strideH 1
#define strideW 1

template<typename T>
constexpr T div_up(T a, T b) {
    return (a + b - 1) / b;
}

constexpr int weight_size = KD*KH*KW;   // in unit of IO_DTYPE
/*
 * Allocate exactly the halo buffer we touch (KD * padded_H * padded_W) instead of a
 * fixed 32-KB slab: LDS is the occupancy limiter once the register pressure is gone,
 * so handing back the unused ~8 KB buys another workgroup per CU.
 *
 * The tail is slack for the innermost loop, which reads a whole dword-aligned run
 * per kernel row and so can reach up to KW-1 elements past the last output column
 * of the last row.  Those lanes are multiplied by the zero-padded high half of the
 * final weight dword, but they must still be backed by real, zeroed LDS: an
 * out-of-range LDS dword reads back as 0 and would take the valid low half with it.
 */
constexpr int halo_size = KD * (PaddingH*2 + BLOCK_H) * (PaddingW*2 + BLOCK_W);
constexpr int max_input_size = halo_size + 2 * KW;

static_assert(PaddingD == 0);
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;
using fp16x2_t = __attribute__((ext_vector_type(2))) _Float16;
using bf16x2_t = __attribute__((ext_vector_type(2))) __bf16;
using io_dtype_t = IO_DTYPE;

static_assert(std::is_same_v<io_dtype_t, __half> ||
                  std::is_same_v<io_dtype_t, __hip_bfloat16>,
              "depthwise_conv_packed_dot supports only FP16 and BF16");

template<uint16_t cnt>
__device__ __inline__ void s_waitcnt_vmcnt() {
    asm volatile ("s_waitcnt vmcnt(%0)\n"::"i"(cnt));
}

// Accumulate two packed 16-bit products into FP32 using the instruction that
// matches IO_DTYPE. The packed memory and register layout is shared by both
// formats; only interpretation of each 16-bit lane differs.
__device__ __inline__ void dot2_acc(float& acc, uint32_t a, uint32_t b) {
    if constexpr (std::is_same_v<io_dtype_t, __half>) {
        acc = __builtin_amdgcn_fdot2(__builtin_bit_cast(fp16x2_t, a),
                                     __builtin_bit_cast(fp16x2_t, b), acc, false);
    } else {
        acc = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2_t, a),
                                              __builtin_bit_cast(bf16x2_t, b), acc, false);
    }
}

__device__ __inline__ void set_m0(void* base) {
    as3_uint32_ptr lds_offset = (as3_uint32_ptr)(base);
    asm volatile("s_mov_b32 m0, %0\n\ts_nop 1\n"::"s"(lds_offset));
}

__device__ __inline__ void global_load_lds_dword(int vaddr, const void* saddr, int imm_offset = 0) {
    //void * saddr = __builtin_amdgcn_readfirstlane(_saddr);
    asm volatile("global_load_lds_dword %[vaddr], %[saddr] offset:%[offset]"
                :: [vaddr]"v"(vaddr), [saddr]"s"(saddr), [offset]"i"(imm_offset)
                : "memory"
            );
}

__global__ void __launch_bounds__(256, 1) conv_depthwise3d_hip(
    const IO_DTYPE* __restrict__ input,     // [B, iC, iD, H, W]
    IO_DTYPE* __restrict__ output,          // [B, oC, oD, H, W]
    const IO_DTYPE* __restrict__ kernel,    // [C, 1, KD, KH, KW]
    const IO_DTYPE* __restrict__ bias,      // [C]
    int iC,        // in_channels
    int iD,
    int iH,
    int iW,
    int oC,        // out_channels = channel_multiplier * in_channels
    int oD,
    int oH,
    int oW)
{
    const int channel_multiplier = oC / iC;
    const int batch = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int out_D = blockIdx.z;
    const int in_channel = out_channel / channel_multiplier;
    int blk_in = (batch * iC + in_channel) * iD + out_D;
    int blk_out = (batch * oC + out_channel) * oD + out_D;

    // 16B-aligned so the dword-wide LDS reads below are always naturally aligned
    __shared__ __attribute__((aligned(16))) IO_DTYPE s_input[max_input_size];

    kernel += out_channel * weight_size;
    input += blk_in * iH * iW;
    output += blk_out * oH * oW;

    // clear s_input with zero-padding
    constexpr int padded_H = PaddingH*2 + BLOCK_H;
    constexpr int padded_W = PaddingW*2 + BLOCK_W;
    constexpr int input_size_dw4 = (KD * padded_H * padded_W * sizeof(IO_DTYPE) + sizeof(int32x4_t) - 1)/sizeof(int32x4_t);
    constexpr int32x4_t vzero = {0};
    for (int i = threadIdx.x; i < max_input_size; i += blockDim.x) {
        s_input[i] = 0.0f; //vzero;
    }
    __syncthreads();

    constexpr int num_warps = 4;
    constexpr int warp_size = 64;
    constexpr int sizeof_dword = 4;
    const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int lane_id = threadIdx.x & (warp_size - 1);

    // load input
    for (int d = 0; d < KD; d++) {
        for (int h = warp_id; h < BLOCK_H; h += num_warps) {
            const int off_h0 = d * iH * iW + h * iW;
            const int off_h1 = d * padded_H * padded_W + (h + PaddingH) * padded_W + PaddingW;
            #pragma unroll
            for (int w = lane_id; w < BLOCK_W/2; w += warp_size) {
                // each thread loads 2-IO_DTYPE as 1-DWORD
                set_m0(s_input + off_h1 + 2*__builtin_amdgcn_readfirstlane(w));
                global_load_lds_dword(off_h0 * sizeof(IO_DTYPE) + w*sizeof_dword, input);
                // s_input[off_h1 + w] = input[off_h0 + w];
            }
        }
    }

    /*
     * Weights stay packed: two 16-bit values per register, three registers per
     * (d,h) row instead of
     * 5 f32 VGPRs.  Row layout: [w0|w1], [w2|w3], [w4|0].
     * The zero in the high half of the third dword makes the tail term of both
     * output columns fall out of a single v_dot2 with no masking.
     */
    constexpr int NROW = KD * KH;
    uint32_t wpk[NROW][3];
    {
        const uint16_t* __restrict__ kb = reinterpret_cast<const uint16_t*>(kernel);
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int b = r * KW;
            wpk[r][0] = (uint32_t)kb[b + 0] | ((uint32_t)kb[b + 1] << 16);
            wpk[r][1] = (uint32_t)kb[b + 2] | ((uint32_t)kb[b + 3] << 16);
            wpk[r][2] = (uint32_t)kb[b + 4];
            /*
             * The whole block shares one out_channel, so every weight is wave-uniform.
             * readfirstlane parks all 45 dwords in SGPRs, which v_dot2 can take as
             * src0 -- 45 VGPRs of weights become 0, and the register budget goes to
             * accumulators and a wider output tile instead.
             */
            #pragma unroll
            for (int j = 0; j < 3; j++)
                wpk[r][j] = (uint32_t)__builtin_amdgcn_readfirstlane((int)wpk[r][j]);
        }
    }
    float bias_value = 0.0f;
    if (bias != nullptr)
        bias_value = bias[out_channel];

    s_waitcnt_vmcnt<0>();
    __syncthreads();

    // conv
    constexpr int num_outputs = BLOCK_H * BLOCK_W;
    /*
     * Output columns computed per thread.  Each kernel row needs OW_TILE+KW-1
     * 16-bit inputs == OW_TILE/2 + 2 dwords, so a wider tile amortises the LDS
     * traffic: 3 dwords per 2 outputs at OW_TILE=2 vs 4 dwords per 4 outputs
     * here.  Halving the ds_read count is what the freed registers buy.
     */
    constexpr int OW_TILE = 16;
    constexpr int NDW = OW_TILE / 2 + 2;     // dwords of input per kernel row
    static_assert(OW_TILE % 2 == 0);
    static_assert(BLOCK_W % OW_TILE == 0);   // a tile never straddles a row
    static_assert(KW == 5 && KD == 3 && KH == 5);

    // LDS dword offset of kernel row r == (d,h), relative to the thread's base.
    #define ROW_WOFF(r) ((((r) / KH * (padded_H * padded_W) + (r) % KH * padded_W)) / 2)

    for (int oi = OW_TILE*threadIdx.x; oi < num_outputs; oi += OW_TILE*blockDim.x) {
        const int oh = oi / BLOCK_W;
        const int ow = oi % BLOCK_W;

        // element (oh + 0, ow + 0) of the padded halo buffer
        IO_DTYPE* base = s_input + oh * padded_W + ow;

        float sum[OW_TILE];
        #pragma unroll
        for (int t = 0; t < OW_TILE; t++) sum[t] = bias_value;

        /*
         * Stream one kernel row (NDW dwords) at a time instead of reading all
         * KD*KH*KW inputs up front, so only one row's worth of input is live at
         * any point rather than the whole 45-element window.  That, plus the
         * packed weights below, is what frees the VGPRs.
         *
         * These are plain LDS loads rather than ds_read_b32 inline asm: with asm
         * the scheduler is free to move the reads relative to a hand-written
         * s_waitcnt, so a manually counted lgkmcnt() does not reliably cover
         * them.  Letting the compiler own the loads lets it place the waits.
         */
        const uint32_t* rowp = reinterpret_cast<const uint32_t*>(base);

        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int rw = ROW_WOFF(r);
            uint32_t c[NDW];
            #pragma unroll
            for (int j = 0; j < NDW; j++)
                c[j] = rowp[rw + j];    // c[j] == (a[2j], a[2j+1])

            /*
             * Output column ow+t dots this row's 5 weights against a[t..t+4].
             * Odd t needs the window shifted by one 16-bit value, which v_alignbit_b32
             * re-packs in a single instruction.  The high half of wpk[r][2] is
             * zero, so the 6th lane it pulls in is multiplied away and needs no
             * masking -- including the (a[t+5], 0) tail at odd t.
             */
            #pragma unroll
            for (int t = 0; t < OW_TILE; t++) {
                const int j = t / 2;
                if ((t & 1) == 0) {
                    dot2_acc(sum[t], wpk[r][0], c[j + 0]);
                    dot2_acc(sum[t], wpk[r][1], c[j + 1]);
                    dot2_acc(sum[t], wpk[r][2], c[j + 2]);
                } else {
                    dot2_acc(sum[t], wpk[r][0], __builtin_amdgcn_alignbit(c[j + 1], c[j + 0], 16));
                    dot2_acc(sum[t], wpk[r][1], __builtin_amdgcn_alignbit(c[j + 2], c[j + 1], 16));
                    dot2_acc(sum[t], wpk[r][2], c[j + 2] >> 16);
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < OW_TILE; t++)
            output[oh * oW + ow + t] = (IO_DTYPE)(sum[t]);
    }
    #undef ROW_WOFF
}
