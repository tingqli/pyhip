// Strided-stream HBM read probe (docs/experiment/moe_small_m_gfx942): every workgroup reads a 1 MiB region as STREAMS
// contiguous sub-streams, CHUNK_VECS 16 B vectors of each stream per step, INFLIGHT steps of loads issued before use.
// The small-M kernels read 16 streams (16-row weight groups, 64 KiB apart) x 2 KiB per K block per workgroup.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

template <int STREAMS, int CHUNK_VECS, int INFLIGHT>
__device__ void streams_read(const u32x4* __restrict__ src, int steps, unsigned int* __restrict__ sink) {
    constexpr int LOADS = STREAMS * CHUNK_VECS / 512;
    static_assert(LOADS >= 1, "one 16 B load per thread per step at least");
    const u32x4* base = src + (size_t)blockIdx.x * STREAMS * steps * CHUNK_VECS;
    unsigned int acc = 0;
    for (int step = 0; step < steps; step += INFLIGHT) {
        u32x4 v[INFLIGHT][LOADS];
#pragma unroll
        for (int i = 0; i < INFLIGHT; i++) {
#pragma unroll
            for (int j = 0; j < LOADS; j++) {
                int u = j * 512 + threadIdx.x;
                int s = u / CHUNK_VECS, o = u % CHUNK_VECS;
                v[i][j] = __builtin_nontemporal_load(base + ((size_t)s * steps + step + i) * CHUNK_VECS + o);
            }
        }
#pragma unroll
        for (int i = 0; i < INFLIGHT; i++)
#pragma unroll
            for (int j = 0; j < LOADS; j++) acc ^= v[i][j].x ^ v[i][j].y ^ v[i][j].z ^ v[i][j].w;
    }
    if (acc == 0x9e3779b9u) sink[threadIdx.x] = acc;
}

// 128 KiB of loads in flight per workgroup in every variant (the small-M kernels keep 96 KiB of B in flight).
#define PROBE(S, C)                                                                                                    \
    __global__ void __launch_bounds__(512) streams_##S##_##C(const u32x4* src, int steps, unsigned int* sink) {      \
        streams_read<S, C, 8192 / ((S) * (C))>(src, steps, sink);                                                      \
    }
// (streams, 16 B vectors per stream per step): 16 x 2 KiB is the small-M pattern.
PROBE(16, 128)
PROBE(16, 256)
PROBE(8, 128)
PROBE(8, 256)
PROBE(8, 512)
PROBE(4, 512)
PROBE(1, 2048)
