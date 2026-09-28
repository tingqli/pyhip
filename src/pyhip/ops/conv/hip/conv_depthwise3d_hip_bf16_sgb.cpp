/*
 * BF16 sched-group-barrier depthwise Conv3D fallback for gfx942.
 *
 * Ported from src/contrib/conv_depthwise3d_hip_sgb.cpp into the installed
 * pyhip.ops.conv package. The wrapper selects this path for the supported
 * 3x5x5 shape when gfx942 cannot execute BF16 packed-dot instructions.
 * Each workgroup computes one output depth plane for one channel and batch.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

template<typename T>
constexpr T div_up(T a, T b) {
    return (a + b - 1) / b;
}

// Keep the original SGB kernel's fixed 32 KiB LDS slab and its weight-size
// reservation. The Python dispatch checks that the padded input tile fits.
constexpr int LDS_SIZE = 32*1024;
constexpr int weight_size = KD*KH*KW;   // in unit of IO_DTYPE
constexpr int max_input_size = LDS_SIZE/sizeof(IO_DTYPE) - (weight_size + 31)/32 * 32;

static_assert(PaddingD == 0);
static_assert(KD == 3 && KH == 5 && KW == 5);
static_assert(KD * (PaddingH*2 + BLOCK_H) * (PaddingW*2 + BLOCK_W) <= max_input_size);
using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;

template<uint16_t cnt>
__device__ __inline__ void s_waitcnt_lgkmcnt() {
    asm volatile ("s_waitcnt lgkmcnt(%0)\n"::"i"(cnt));
}

template<uint16_t cnt>
__device__ __inline__ void s_waitcnt_vmcnt() {
    asm volatile ("s_waitcnt vmcnt(%0)\n"::"i"(cnt));
}

__device__ __inline__ uint32_t ds_read_b32(IO_DTYPE* psrc, int imm_offset=0) {
    uint32_t v;
    as3_uint32_ptr vaddr = (as3_uint32_ptr)(psrc);
    asm volatile("ds_read_b32 %[vdst], %[vaddr] offset:%[offset]"
                : [vdst]"=v"((uint32_t&)(v))
                : [vaddr]"v"(vaddr),[offset]"i"(imm_offset)
                : "memory"
            );
    return v;
}

__device__ __inline__ void v_fmac_f32(float& vdst, float src0, float vsrc1) {
    // The + modifier implies the operand appears in both input and output lists implicitly
    asm volatile("v_fmac_f32 %[vdst], %[src0], %[vsrc1]"
                : [vdst]"+v"(vdst)
                : [src0]"v"(src0),[vsrc1]"v"(vsrc1));
}

__device__ __inline__ void set_m0(void* base) {
    as3_uint32_ptr lds_offset = (as3_uint32_ptr)(base);
    asm volatile("s_mov_b32 m0, %0\n\ts_nop 1\n"::"s"(lds_offset));
}

__device__ __inline__ void global_load_lds_dword(int vaddr, const void* saddr, int imm_offset = 0) {
    asm volatile("global_load_lds_dword %[vaddr], %[saddr] offset:%[offset]"
                :: [vaddr]"v"(vaddr), [saddr]"s"(saddr), [offset]"i"(imm_offset)
                : "memory"
            );
}

__global__ void __launch_bounds__(256, 1) conv_depthwise3d_hip(
    const IO_DTYPE* __restrict__ input,     // [B, iC, iD, iH, iW]
    IO_DTYPE* __restrict__ output,          // [B, oC, oD, oH, oW]
    const IO_DTYPE* __restrict__ kernel,    // [oC, iC/groups=1, KD, KH, KW]
    const IO_DTYPE* __restrict__ bias,      // [oC]
    int iC,                                 // input channels
    int iD,                                 // input depth
    int iH,                                 // input height
    int iW,                                 // input width
    int oC,                                 // output channels
    int oD,                                 // output depth
    int oH,                                 // output height
    int oW)                                 // output width
{
    const int channel_multiplier = oC / iC;
    const int batch = blockIdx.x;
    const int out_channel = blockIdx.y;
    const int out_D = blockIdx.z;
    const int in_channel = out_channel / channel_multiplier;
    int blk_in = (batch * iC + in_channel) * iD + out_D;
    int blk_out = (batch * oC + out_channel) * oD + out_D;

    // ===== Register and LDS allocation =====
    // Weights: 75 float VGPRs (KD*KH*KW = 3*5*5), loaded from HBM to registers.
    // Input tile: padded spatial data in LDS, zero-filled for padding regions.
    float weight_reg[KD][KH][KW];
    __shared__ IO_DTYPE s_input[max_input_size];

    kernel += out_channel * weight_size;
    input += blk_in * iH * iW;
    output += blk_out * oH * oW;

    // ===== Phase 1: Clear LDS (zero-padding) =====
    // The entire padded tile [KD, padded_H, padded_W] is zeroed first.
    // Actual input data is then loaded into the interior (non-padding) region.
    constexpr int padded_H = PaddingH*2 + BLOCK_H;
    constexpr int padded_W = PaddingW*2 + BLOCK_W;
    for (int i = threadIdx.x; i < KD * padded_H * padded_W; i += blockDim.x) {
        s_input[i] = 0.0f;
    }
    __syncthreads();

    constexpr int num_warps = 4;
    constexpr int warp_size = 64;
    constexpr int sizeof_dword = 4;
    const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int lane_id = threadIdx.x & (warp_size - 1);

    // ===== Phase 2: Load input HBM -> LDS =====
    // Uses global_load_lds_dword (same on cdna3 and cdna4): data goes directly from HBM to LDS, bypassing VGPRs.
    // Each dword = 2 IO_TYPES (bf16/fp16) values. 4 warps cooperatively load all height rows.
    for (int d = 0; d < KD; d++) {
        for (int h = warp_id; h < BLOCK_H; h += num_warps) {
            const int off_h0 = d * iH * iW + h * iW;
            const int off_h1 = d * padded_H * padded_W + (h + PaddingH) * padded_W + PaddingW;
            #pragma unroll
            for (int w = lane_id; w < BLOCK_W/2; w += warp_size) {
                // each thread loads 2-IO_DTYPE as 1-DWORD
                set_m0(s_input + off_h1 + 2*__builtin_amdgcn_readfirstlane(w));
                global_load_lds_dword(off_h0 * sizeof(IO_DTYPE) + w*sizeof_dword, input);
            }
        }
    }

    // ===== Phase 3: Load weights HBM -> registers =====
    // All 75 filter taps loaded into float VGPRs. Each thread gets the same weights
    // (all threads in the block process the same channel).
    int ki = 0;
    #pragma unroll
    for(int d = 0; d < KD; d ++) {
        #pragma unroll
        for(int h = 0; h < KH; h++) {
            #pragma unroll
            for(int w = 0; w < KW; w++) {
                weight_reg[d][h][w] = kernel[ki++];
            }
        }
    }
    float bias_value = 0.0f;
    if (bias != nullptr)
        bias_value = bias[out_channel];

    // Wait for all global_load_lds_dword to complete before reading LDS.
    s_waitcnt_vmcnt<0>();
    __syncthreads();

    // ===== Phase 4: Compute — row-interleaved with sched_group_barrier =====
    //
    // Each thread computes 2 adjacent output pixels (p=0 and p=1) per iteration.
    // 256 threads cover up to 512 output pixels per iteration of the outer loop.
    //
    // Key difference from conv_depthwise3d_hip.cpp:
    //
    //   conv_depthwise3d_hip.cpp (batch reads, then batch compute):
    //     input_reg[KD][KH][KWR][2]      ← 45 VGPRs for ALL filter rows
    //     for d,h,w: ds_read_b32(...)    ← 45 reads issued at once
    //     s_waitcnt lgkmcnt(0)           ← wait for all 45
    //     for p: for d,h,w: v_fmac(...)  ← 150 fmacs
    //
    //   conv_depthwise3d_hip_sgb.cpp (per-row read + compute):
    //     input_reg[KWR][2]              ← 3 VGPRs reused per row
    //     for d,h:                       ← 15 rows (KD*KH = 3*5)
    //       ds_read_b32 × 3              ← read ONE row (3 pairs for KW=5)
    //       sched_group_barrier          ← hint: interleave reads with fmacs
    //       s_waitcnt lgkmcnt(0)         ← wait for just these 3 reads
    //       v_fmac × 10                  ← compute this row for both p=0 and p=1
    //
    // The sched_group_barrier tells LLVM to place the 3 ds_read_b32 instructions
    // BEFORE the v_fmac group in the final assembly. Without it, LLVM might
    // reorder instructions in ways that don't overlap read latency with compute.
    //
    //   sched_group_barrier(0x0100, 3, 0): schedule 3 DS_read instructions
    //   sched_group_barrier(0x0002, 10, 0): schedule 10 VALU instructions
    //
    //   Group type constants (from AMDGPU ISA & LLVM documentation):
    //     0x0002 = VALU
    //     0x0100 = DS_read

    constexpr int num_outputs = BLOCK_H * BLOCK_W;
    constexpr int KW_PACK = 2;                      // 2 output pixels per thread per iteration
    constexpr int KWR = div_up(KW, KW_PACK);        // ceil(KW/2) = 3 input pairs per filter row
    static_assert(BLOCK_W % 2 == 0);

    // Only 3 input pair registers, reused for each filter row.
    // V97-origin needs input_reg[KD][KH][KWR][KW_PACK] = 45 registers.
    IO_DTYPE input_reg[KWR][KW_PACK];

    for (int oi = KW_PACK*threadIdx.x; oi < num_outputs; oi += KW_PACK*blockDim.x) {
        const int oh = oi / BLOCK_W;                // output height position
        const int ow = oi % BLOCK_W;                // output width position (even, since KW_PACK=2)
        const int srci0 = oh * padded_W + ow;       // LDS base address for this output position

        // Accumulators for 2 adjacent output pixels.
        // sum0 = output[oh, ow], sum1 = output[oh, ow+1]
        float sum0 = bias_value;
        float sum1 = bias_value;

        // Loop over all 15 filter rows (KD=3 depths x KH=5 heights).
        // Each iteration: read 3 input pairs from LDS, compute 10 v_fmac.
        #pragma unroll
        for(int d = 0; d < KD; d ++) {
            #pragma unroll
            for(int h = 0; h < KH; h++) {

                // Read 3 input pairs for this (d,h) filter row.
                // Each ds_read_b32 loads 2 adjacent bf16 values as one dword.
                // For KW=5: reads pairs at kw=(0,1), (2,3), (4,_)
                #pragma unroll
                for(int w = 0; w < KW; w+=KW_PACK) {
                    int src_off = d*(padded_H * padded_W) + (PaddingH + h - KH/2)*padded_W + (PaddingW + w - KW/2);
                    reinterpret_cast<uint32_t&>(input_reg[w/KW_PACK]) = ds_read_b32(s_input + srci0, src_off*sizeof(IO_DTYPE));
                }

                // Scheduling hint: tell LLVM to place 3 ds_read_b32 then 10 VALU to overlap the latency of ds_read_b32
                // Otherwise, LLVM compiler will reorder the instructions: ds_read_b32, ds_read_b32, wait, fmac, fmac, ...
                __builtin_amdgcn_sched_group_barrier(0x0100, 3, 0);  // 3 DS_read (ds_read_b32)
                __builtin_amdgcn_sched_group_barrier(0x0002, 10, 0); // 10 VALU (v_fmac_f32)

                // Wait for this row's 3 reads to complete.
                s_waitcnt_lgkmcnt<0>();

                // Compute this filter row's contribution for both output pixels.
                //
                // For KW=5 with KW_PACK=2, the input pairs are:
                //   input_reg[0] = {in[kw=0], in[kw=1]}
                //   input_reg[1] = {in[kw=2], in[kw=3]}
                //   input_reg[2] = {in[kw=4], ___pad___}
                //
                // p=0 (output at ow):   uses in[kw] for kw=0..4
                // p=1 (output at ow+1): uses in[kw+1] for kw=0..4 (shifted by 1)
                //
                // The v_fmac pairs handle both even and odd filter taps:
                //   w=0: p=0 uses reg[0][0] (kw=0), p=1 uses reg[0][1] (kw=1)
                //   w=1: p=0 uses reg[0][1] (kw=1), p=1 uses reg[1][0] (kw=2)
                //   w=2: p=0 uses reg[1][0] (kw=2), p=1 uses reg[1][1] (kw=3)
                //   w=3: p=0 uses reg[1][1] (kw=3), p=1 uses reg[2][0] (kw=4)
                //   w=4: p=0 uses reg[2][0] (kw=4), p=1 would use reg[2][1] (padding, =0)
                #pragma unroll
                for(int w = 0; w < KW; w+=2) {
                    // p=0: accumulate weight[d][h][w] * input[w]
                    v_fmac_f32(sum0, weight_reg[d][h][w+0], (float)(input_reg[w/KW_PACK][0]));
                    if (w+1 < KW)
                        v_fmac_f32(sum0, weight_reg[d][h][w+1], (float)(input_reg[w/KW_PACK][1]));
                    // p=1: accumulate weight[d][h][w] * input[w+1] (shifted by 1)
                    v_fmac_f32(sum1, weight_reg[d][h][w+0], (float)(input_reg[w/KW_PACK][1]));
                    if (w+1 < KW)
                        v_fmac_f32(sum1, weight_reg[d][h][w+1], (float)(input_reg[(w+2)/KW_PACK][0]));
                }
            }
        }

        // Store 2 output pixels as IO_DTYPE (bf16/fp16).
        output[oh * oW + ow + 0] = (IO_DTYPE)(sum0);
        output[oh * oW + ow + 1] = (IO_DTYPE)(sum1);
    }
}
