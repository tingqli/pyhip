// Roofline probes for docs/record/moe_small_m_gfx942.md: sustained HBM read bandwidth and FP8 MFMA throughput.
// Kernels keep C++ linkage: pyhip.module reads the argument types from the mangled names.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

// Every workgroup streams one contiguous chunk with UNROLL 16 B loads per lane in flight per iteration.
template <int UNROLL>
__device__ void stream_read(const u32x4* __restrict__ src, unsigned long vecs_per_wg, unsigned int* __restrict__ sink) {
    const u32x4* base = src + blockIdx.x * vecs_per_wg;
    unsigned int acc = 0;
    for (unsigned long i = threadIdx.x; i < vecs_per_wg; i += blockDim.x * UNROLL) {
        u32x4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            unsigned long j = i + u * blockDim.x;
            v[u] = j < vecs_per_wg ? __builtin_nontemporal_load(base + j) : u32x4{0, 0, 0, 0};
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x9e3779b9u) sink[threadIdx.x] = acc;
}

__global__ void __launch_bounds__(256) read_u4(const u32x4* src, unsigned long vecs_per_wg, unsigned int* sink) {
    stream_read<4>(src, vecs_per_wg, sink);
}

__global__ void __launch_bounds__(256) read_u8(const u32x4* src, unsigned long vecs_per_wg, unsigned int* sink) {
    stream_read<8>(src, vecs_per_wg, sink);
}

__global__ void __launch_bounds__(512) read512_u8(const u32x4* src, unsigned long vecs_per_wg, unsigned int* sink) {
    stream_read<8>(src, vecs_per_wg, sink);
}

typedef float floatx4 __attribute__((ext_vector_type(4)));

// Back-to-back independent v_mfma_f32_16x16x32_fp8_fp8 (16 cycles each) from registers: the FP8 MFMA ceiling.
// Operands are runtime values so the power draw matches nonzero data.
__global__ void __launch_bounds__(256) mfma_fp8(int iters, unsigned long a_bits, unsigned long b_bits, float* sink) {
    floatx4 acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = floatx4{0, 0, 0, 0};
    long a = (long)(a_bits ^ threadIdx.x), b = (long)(b_bits ^ (threadIdx.x << 3));
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc[i], 0, 0, 0);
    }
    float s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) s += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
    if (s == 1234.5f) sink[threadIdx.x] = s;
}
