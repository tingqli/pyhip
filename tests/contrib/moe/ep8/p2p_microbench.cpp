// P2P microbenchmarks for the gfx942 EP8 mega kernel design: xGMI put bandwidth and remote BF16 atomics.
#include <hip/hip_runtime.h>
#include <stdint.h>

// Every workgroup copies whole rows; row r of the grid goes to peer (me + r) % npes, so all xGMI links are busy.
__global__ void __launch_bounds__(256) p2p_put(const uint32_t* __restrict__ src_u32, const uint64_t* __restrict__ peer_base,
                                                          int rows, int row_u4, int npes, int me, int sc) {
    const uint4* src = reinterpret_cast<const uint4*>(src_u32);
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        int peer = (me + r) % npes;
        int per_peer = (rows + npes - 1) / npes;
        uint4* dst = reinterpret_cast<uint4*>(peer_base[peer]) + (int64_t)(me * per_peer + r / npes) * row_u4;
        const uint4* s = src + (int64_t)r * row_u4;
        for (int i = threadIdx.x; i < row_u4; i += blockDim.x) {
            uint4 v = s[i];
            if (sc) {
                __builtin_nontemporal_store(v.x, &dst[i].x);
                __builtin_nontemporal_store(v.y, &dst[i].y);
                __builtin_nontemporal_store(v.z, &dst[i].z);
                __builtin_nontemporal_store(v.w, &dst[i].w);
            } else {
                dst[i] = v;
            }
        }
    }
}

// Every lane adds packed BF16 pairs into the peer's [rows, row_u32] buffer with global_atomic_pk_add_bf16.
__global__ void __launch_bounds__(256) p2p_atomic_bf16(const uint32_t* __restrict__ src, const uint64_t* __restrict__ peer_base,
                                                                  int rows, int row_u32, int npes, int me, int dst_rows) {
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        int peer = (me + r) % npes;
        uint32_t* dst = reinterpret_cast<uint32_t*>(peer_base[peer]) + (int64_t)(r % dst_rows) * row_u32;
        const uint32_t* s = src + (int64_t)r * row_u32;
        for (int i = threadIdx.x; i < row_u32; i += blockDim.x) {
            uint32_t v = s[i];
            uint32_t* p = dst + i;
            asm volatile("global_atomic_pk_add_bf16 %0, %1, off" ::"v"(p), "v"(v) : "memory");
        }
    }
}
