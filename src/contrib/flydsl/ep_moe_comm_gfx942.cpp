// gfx942 EP8 MoE communication kernels (HIP): routing plan + count exchange, route-level P2P dispatch into
// expert-sorted rows, and the combine completion barrier. Peers are MORI symmetric-heap buffers (int64 tables).
//
// Workspace `ws` (symmetric, int32): counts [NPES][E_GLOBAL] (row s written by rank s), then flags
// [3][NPES]: count row / dispatch / combine of rank s finished, each holding the call's epoch, then
// arrived [s_cap / BLOCK_M]: routes already written into each sorted block here (added by the sources' producers).
// Schedule workspace `sched` (local, int32): [0, 8) task tickets of the per-XCD queues, [SCHED_EXPECTED + b] routes
// sorted block b receives, [SCHED_EXPECTED + s_cap / BLOCK_M + b] GateUp tiles finished of block b,
// [SCHED_EXPECTED + 2 s_cap / BLOCK_M + b] set once the mega kernel transposed block b's A scales.
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int NPES = 8;
constexpr int E_LOCAL = 32;
constexpr int E_GLOBAL = NPES * E_LOCAL;
constexpr int BLOCK_M = 256;
constexpr int CFLAG = NPES * E_GLOBAL;
constexpr int DFLAG = CFLAG + NPES;
constexpr int FFLAG = DFLAG + NPES;
constexpr int ARRIVED = FFLAG + NPES;
constexpr int SCHED_EXPECTED = 16;
// Send-order keys: a route's block position in its destination's queue (the last key: later or dropped).
constexpr int MAX_KEYS = 64;
// Routes per thread with their loads issued together (ep_prep); ep_plan slices of up to 256 * SLICE_PER_THREAD.
constexpr int ROUTE_BATCH = 8;
constexpr int SLICE_PER_THREAD = 4;
// ep_prep -> ep_plan tables (int32): totals [E_GLOBAL], fewest routes from one source [E_GLOBAL], expert bases
// [E_GLOBAL + NPES], first block of each destination's 8 mega-kernel queues and its block count [NPES][9], first send
// position of each key [MAX_KEYS], and the per-key cursors ep_plan's workgroups reserve their ranges with [MAX_KEYS].
constexpr int PLAN_TOTAL = 0;
constexpr int PLAN_MIN = E_GLOBAL;
constexpr int PLAN_BASE = 2 * E_GLOBAL;
constexpr int PLAN_QSTART = 3 * E_GLOBAL + NPES;
constexpr int PLAN_KEY_BASE = PLAN_QSTART + NPES * 9;
constexpr int PLAN_KEY_CURSOR = PLAN_KEY_BASE + MAX_KEYS;
constexpr int PLAN_INTS = PLAN_KEY_CURSOR + MAX_KEYS;
// send_info: destination row in bits [0, 17), global expert above.
constexpr int ROW_BITS = 17;

__device__ inline void store_release_sys(int* p, int v) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
__device__ inline int load_acquire_sys(int* p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); }
__device__ inline int load_relaxed_sys(int* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
__device__ inline void wait_eq_sys(int* p, int v) {
    while (load_acquire_sys(p) != v) __builtin_amdgcn_s_sleep(1);
}

// Mega-kernel row of this rank's idx-th route to global expert g: source s's i-th route sits at virtual position 8 i + s
// (compacted where sources run out). counts [NPES][E_GLOBAL]; fewest and base per global expert.
__device__ inline int mega_row(int g, int idx, int me, const int* counts, const int* fewest, const int* base) {
    int pos = idx * NPES + me;
    if (idx >= fewest[g]) {
        pos = 0;
        for (int s = 0; s < NPES; ++s) {
            int c = counts[s * E_GLOBAL + g], k = idx + (s < me ? 1 : 0);
            pos += c < k ? c : k;
        }
    }
    return base[g] + pos;
}

// Send-order key of a route to `row` of global expert g: the block's position in its destination's queue (qstart
// [NPES][9]); MAX_KEYS - 1 for later positions and dropped routes.
__device__ inline int route_key(int g, int row, int s_cap, const int* qstart) {
    if (row >= s_cap) return MAX_KEYS - 1;
    const int* qs = qstart + g / E_LOCAL * 9;
    int block = row / BLOCK_M, q = block >= qs[4] ? 4 : 0;
    q += block >= qs[q + 2] ? 2 : 0;
    q += block >= qs[q + 1] ? 1 : 0;
    int pos = block - qs[q];
    return pos < MAX_KEYS - 1 ? pos : MAX_KEYS - 1;
}

// One workgroup of 1024 threads per rank: count exchange and plan tables (ep_plan then plans the routes).
//   route_idx [T*TOPK]: rank of each route among this rank's routes to its global expert.
//   send_base [E_GLOBAL]: first destination row of this rank's routes to each global expert (ep_dispatch).
//   num_valid [1]: rows this rank receives (capped at s_cap).
//   status [8]: 0 dispatch counter (reset), 1 rows needed by this rank (for the capacity check), 2 / 3 / 4 low 32 bits
//   of s_memrealtime when the count exchange finished / at entry / at exit.
//   plan [PLAN_INTS]: tables for ep_plan.
__global__ void __launch_bounds__(1024) ep_prep(const int* __restrict__ topk_ids, int num_routes, int me, int epoch, int* route_idx,
                                                int* ws, const uint64_t* __restrict__ ws_peers, int* send_base,
                                                int* num_valid, int s_cap, int* status, int* sched, int* plan) {
    __shared__ int hist[E_GLOBAL];
    __shared__ int counts[NPES * E_GLOBAL];
    __shared__ int expert_base[E_GLOBAL + NPES];  // row base of each global expert at its destination; [E_GLOBAL + d] = rows of d
    __shared__ int sbase[E_GLOBAL];
    __shared__ int total[E_GLOBAL];
    __shared__ int min_count[E_GLOBAL];
    __shared__ int qstart[NPES * 9];
    __shared__ int key_hist[MAX_KEYS];
    const int tid = threadIdx.x;
    if (tid == 0) status[3] = static_cast<int>(__builtin_amdgcn_s_memrealtime());
    for (int i = tid; i < E_GLOBAL; i += blockDim.x) hist[i] = 0;
    // Before the count flag: no source adds to `arrived` until it has seen this rank's flag.
    for (int b = tid; b < s_cap / BLOCK_M; b += blockDim.x) ws[ARRIVED + b] = 0;
    __syncthreads();
    for (int r0 = 0; r0 < num_routes; r0 += ROUTE_BATCH * blockDim.x) {
        int g[ROUTE_BATCH];
#pragma unroll
        for (int j = 0; j < ROUTE_BATCH; ++j) {
            int r = r0 + j * blockDim.x + tid;
            g[j] = r < num_routes ? topk_ids[r] : 0;
        }
#pragma unroll
        for (int j = 0; j < ROUTE_BATCH; ++j) {
            int r = r0 + j * blockDim.x + tid;
            if (r < num_routes) route_idx[r] = atomicAdd(&hist[g[j]], 1);
        }
    }
    __syncthreads();
    for (int i = tid; i < NPES * E_GLOBAL; i += blockDim.x) {
        int peer = i / E_GLOBAL, e = i % E_GLOBAL;
        __hip_atomic_store(reinterpret_cast<int*>(ws_peers[peer]) + me * E_GLOBAL + e, hist[e], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __threadfence_system();
    __syncthreads();
    if (tid < NPES) store_release_sys(reinterpret_cast<int*>(ws_peers[tid]) + CFLAG + me, epoch);
    if (tid < NPES) wait_eq_sys(ws + CFLAG + tid, epoch);
    __syncthreads();
    // Every rank leaves the count exchange at about the same time: a common instant for traces (100 MHz clock).
    if (tid == 0) status[2] = static_cast<int>(__builtin_amdgcn_s_memrealtime());
    for (int i = tid; i < NPES * E_GLOBAL; i += blockDim.x) counts[i] = load_relaxed_sys(ws + i);
    __syncthreads();
    // Per global expert: routes from all sources (total), the fewest from one source, and this rank's rows before
    // its own.
    if (tid < E_GLOBAL) {
        int n = 0, before = 0, fewest = 1 << 30;
        for (int s = 0; s < NPES; ++s) {
            int c = counts[s * E_GLOBAL + tid];
            n += c;
            before += s < me ? c : 0;
            fewest = c < fewest ? c : fewest;
        }
        total[tid] = n;
        sbase[tid] = before;
        min_count[tid] = fewest;
    }
    __syncthreads();
    // Destination d's experts start on 256-row boundaries in expert order.
    if (tid < E_GLOBAL) {
        int d = tid / E_LOCAL, e = tid % E_LOCAL, base = 0;
        for (int k = d * E_LOCAL; k < tid; ++k) base += (total[k] + BLOCK_M - 1) / BLOCK_M * BLOCK_M;
        expert_base[tid] = base;
        if (e == E_LOCAL - 1) expert_base[E_GLOBAL + d] = base + (total[tid] + BLOCK_M - 1) / BLOCK_M * BLOCK_M;
        send_base[tid] = base + sbase[tid];
    }
    for (int b = tid; b < s_cap / BLOCK_M; b += blockDim.x) {
        sched[SCHED_EXPECTED + s_cap / BLOCK_M + b] = 0;
        sched[SCHED_EXPECTED + 2 * (s_cap / BLOCK_M) + b] = 0;
    }
    if (tid < SCHED_EXPECTED) sched[tid] = 0;
    __syncthreads();
    // Plan tables for ep_plan: totals, fewest routes from one source, expert bases, and the first block of each of
    // destination d's mega-kernel queues (its valid blocks split in eight), then the block count.
    for (int i = tid; i < E_GLOBAL; i += blockDim.x) {
        plan[PLAN_TOTAL + i] = total[i];
        plan[PLAN_MIN + i] = min_count[i];
    }
    for (int i = tid; i < E_GLOBAL + NPES; i += blockDim.x) plan[PLAN_BASE + i] = expert_base[i];
    if (tid < NPES * 9) {
        int d = tid / 9, q = tid % 9, rows = expert_base[E_GLOBAL + d];
        qstart[tid] = (rows < s_cap ? rows : s_cap) / BLOCK_M * q / 8;
        plan[PLAN_QSTART + tid] = qstart[tid];
    }
    if (tid < MAX_KEYS) {
        key_hist[tid] = 0;
        plan[PLAN_KEY_CURSOR + tid] = 0;
    }
    __syncthreads();
    // Routes per send key: ep_plan places every route at its key's base plus its rank within the key. This rank's
    // routes to expert g are idx = 0 .. C[me][g] - 1 and their rows grow with idx, so each block takes a range of idx.
    if (tid < E_GLOBAL) {
        const int g = tid, n = counts[me * E_GLOBAL + g], fewest = min_count[g];
        for (int lo = 0; lo < n;) {
            int row = mega_row(g, lo, me, counts, min_count, expert_base);
            int end = (row / BLOCK_M + 1) * BLOCK_M, hi = n;
            if (lo < fewest) {
                // Rows base + 8 idx + me up to fewest: the first idx at or past the next block.
                hi = (end - expert_base[g] - me + NPES - 1) / NPES;
                hi = hi < fewest ? hi : fewest;
            } else {
                for (int a = lo + 1, b = n; a < b;) {
                    int mid = (a + b) / 2;
                    if (mega_row(g, mid, me, counts, min_count, expert_base) >= end) b = hi = mid;
                    else a = mid + 1;
                }
            }
            hi = hi < n ? hi : n;
            atomicAdd(&key_hist[route_key(g, row, s_cap, qstart)], hi - lo);
            lo = hi;
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int k = 0, acc = 0; k < MAX_KEYS; ++k) {
            plan[PLAN_KEY_BASE + k] = acc;
            acc += key_hist[k];
        }
    }
    if (tid == 0) {
        int rows = expert_base[E_GLOBAL + me];
        num_valid[0] = rows < s_cap ? rows : s_cap;
        status[0] = 0;
        status[1] = rows;
        status[4] = static_cast<int>(__builtin_amdgcn_s_memrealtime());
    }
}

// Routing plan, one workgroup of 256 threads per slice of this rank's routes (slices = producers * 8):
//   send_route / send_info [T*TOPK]: all routes sorted by when the destinations' mega-kernel queues need them (block
//   position in the queue, the send key): workgroup j places the routes of slice j at their key's base (ep_prep) plus
//   a range reserved with an atomic on the key's cursor, so the order within a key varies from call to call. The
//   producers send this array chunk by chunk. info = destination row | global expert << ROW_BITS, -1 past s_cap.
//   gu_ids [s_cap] / expert_ids [s_cap/256] / sched expected rows per block: this rank's sorted blocks.
//   down_ids [s_cap] (symmetric, local view): padding rows set to -1 here; producers fill the valid rows.
// Mega-kernel rows of an expert interleave the sources (mega_row), so every block holds about 32 routes of each source
// and the first blocks of all queues load all senders evenly. (ep_dispatch keeps contiguous per-source rows at
// send_base.)
__global__ void __launch_bounds__(256) ep_plan(const int* __restrict__ topk_ids, const int* __restrict__ route_idx,
                                               int num_routes, int me, int s_cap, int* plan,
                                               const int* __restrict__ ws, int* send_route, int* send_info, int* gu_ids,
                                               int* down_ids, int* expert_ids, int* sched) {
    __shared__ int tab[PLAN_INTS];
    __shared__ int key_count[MAX_KEYS];
    __shared__ int key_next[MAX_KEYS];
    const int tid = threadIdx.x;
    for (int i = tid; i < PLAN_INTS; i += blockDim.x) tab[i] = plan[i];
    if (tid < MAX_KEYS) key_count[tid] = 0;
    __syncthreads();
    const int *total = tab + PLAN_TOTAL, *fewest = tab + PLAN_MIN, *base = tab + PLAN_BASE, *qstart = tab + PLAN_QSTART;
    int per = (num_routes + gridDim.x - 1) / gridDim.x, begin = blockIdx.x * per;
    int end = begin + per < num_routes ? begin + per : num_routes;
    int route[SLICE_PER_THREAD], info[SLICE_PER_THREAD], key[SLICE_PER_THREAD], rank[SLICE_PER_THREAD];
#pragma unroll
    for (int j = 0; j < SLICE_PER_THREAD; ++j) {
        route[j] = begin + j * blockDim.x + tid;
        if (route[j] < end) {
            int g = topk_ids[route[j]], row = mega_row(g, route_idx[route[j]], me, ws, fewest, base);
            info[j] = row < s_cap ? (row | g << ROW_BITS) : -1;
            key[j] = route_key(g, row, s_cap, qstart);
            rank[j] = atomicAdd(&key_count[key[j]], 1);
        }
    }
    __syncthreads();
    if (tid < MAX_KEYS) {
        int n = key_count[tid];
        key_next[tid] = tab[PLAN_KEY_BASE + tid] + (n > 0 ? atomicAdd(&plan[PLAN_KEY_CURSOR + tid], n) : 0);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < SLICE_PER_THREAD; ++j) {
        if (route[j] < end) {
            int at = key_next[key[j]] + rank[j];
            send_route[at] = route[j];
            send_info[at] = info[j];
        }
    }
    // This rank's blocks, one per workgroup at a time (a thread per row): expert, routes and row ids.
    int rows = base[E_GLOBAL + me], blocks = (rows < s_cap ? rows : s_cap) / BLOCK_M;
    for (int b = blockIdx.x; b < blocks; b += gridDim.x) {
        int e = 0;
        while (e + 1 < E_LOCAL && base[me * E_LOCAL + e + 1] <= b * BLOCK_M) ++e;
        int g = me * E_LOCAL + e, left = total[g] - (b * BLOCK_M - base[g]), n = left < BLOCK_M ? left : BLOCK_M;
        if (tid == 0) {
            expert_ids[b] = e;
            sched[SCHED_EXPECTED + b] = n;
        }
        for (int t = tid; t < BLOCK_M; t += blockDim.x) {
            int pos = b * BLOCK_M + t;
            gu_ids[pos] = t < n ? pos : -1;
            if (t >= n) down_ids[pos] = -1;
        }
    }
}

// Route-level dispatch: one wave per route copies the token's FP8 row (hidden bytes) and its FP32 scales
// (hidden / 128) to the route's sorted row on the destination, plus the Down sorted id and the routing weight.
__global__ void __launch_bounds__(256) ep_dispatch(const uint32_t* __restrict__ x, const float* __restrict__ x_scale,
                                                   const int* __restrict__ topk_ids, const float* __restrict__ topk_w,
                                                   const int* __restrict__ route_idx, const int* __restrict__ send_base,
                                                   int num_routes, int topk, int hidden, int me, int epoch, int token_shift, int s_cap,
                                                   const uint64_t* __restrict__ a_peers, const uint64_t* __restrict__ s_peers,
                                                   const uint64_t* __restrict__ id_peers, const uint64_t* __restrict__ w_peers,
                                                   int* ws, const uint64_t* __restrict__ ws_peers, int* status) {
    const int lane = threadIdx.x % 64;
    const int row_u4 = hidden / 16, scale_u4 = hidden / 128 / 4;
    const uint4* xa = reinterpret_cast<const uint4*>(x);
    const uint4* xs = reinterpret_cast<const uint4*>(x_scale);
    for (int r = blockIdx.x * (blockDim.x / 64) + threadIdx.x / 64; r < num_routes; r += gridDim.x * (blockDim.x / 64)) {
        int t = r / topk, k = r - t * topk;
        int g = topk_ids[r];
        int d = g / E_LOCAL;
        int row = send_base[g] + route_idx[r];
        if (row >= s_cap) continue;
        uint4* dst = reinterpret_cast<uint4*>(a_peers[d]) + (int64_t)row * row_u4;
        const uint4* src = xa + (int64_t)t * row_u4;
        for (int i = lane; i < row_u4; i += 64) dst[i] = src[i];
        if (lane < scale_u4) reinterpret_cast<uint4*>(s_peers[d])[(int64_t)row * scale_u4 + lane] = xs[(int64_t)t * scale_u4 + lane];
        if (lane == 0) {
            reinterpret_cast<int*>(id_peers[d])[row] = (me << token_shift | t) | (k << 24);
            reinterpret_cast<float*>(w_peers[d])[row] = topk_w[r];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        int prev = atomicAdd(&status[0], 1);
        if (prev == gridDim.x - 1) {
            __threadfence_system();
            for (int p = 0; p < NPES; ++p) store_release_sys(reinterpret_cast<int*>(ws_peers[p]) + DFLAG + me, epoch);
            for (int p = 0; p < NPES; ++p) wait_eq_sys(ws + DFLAG + p, epoch);
        }
    }
}

// After Down (same stream): tell every rank this rank's remote adds are done, then wait for all ranks.
__global__ void ep_finish(int me, int epoch, int* ws, const uint64_t* __restrict__ ws_peers) {
    __threadfence_system();
    if (threadIdx.x < NPES) {
        store_release_sys(reinterpret_cast<int*>(ws_peers[threadIdx.x]) + FFLAG + me, epoch);
        wait_eq_sys(ws + FFLAG + threadIdx.x, epoch);
    }
}

__device__ inline uint32_t bf16_rne(float f) {
    uint32_t u = __float_as_uint(f);
    return (u + 0x7FFF + ((u >> 16) & 1)) >> 16;
}

// FP8 combine on the source rank (after ep_finish): out[t] = BF16(sum over slots of comb[t][slot] * its 1x128 scale),
// in slot order. comb [T, topk, H] FP8 (e4m3fnuz), comb_scale [T, topk, H / 128]; 16 columns per thread, topk <= 8.
__global__ void __launch_bounds__(256) ep_combine_fp8(const uint8_t* __restrict__ comb, const float* __restrict__ comb_scale,
                                                      uint16_t* __restrict__ out, int num_tokens, int topk, int hidden) {
    constexpr int MAX_TOPK = 8;
    const int chunks = hidden / 16;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tokens * chunks) return;
    const int token = i / chunks, chunk = i % chunks;
    // All slots' loads first: a loop bound only known at run time would wait for each slot's load in turn.
    uint4 q[MAX_TOPK];
    float scale[MAX_TOPK];
#pragma unroll
    for (int slot = 0; slot < MAX_TOPK; ++slot) {
        const size_t row = static_cast<size_t>(token) * topk + slot;
        q[slot] = slot < topk ? *reinterpret_cast<const uint4*>(comb + row * hidden + chunk * 16) : make_uint4(0, 0, 0, 0);
        scale[slot] = slot < topk ? comb_scale[row * (hidden / 128) + chunk / 8] : 0.0f;
    }
    float acc[16] = {};
#pragma unroll
    for (int slot = 0; slot < MAX_TOPK; ++slot) {
        const uint32_t words[4] = {q[slot].x, q[slot].y, q[slot].z, q[slot].w};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(words[w], false);
            const auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(words[w], true);
            acc[w * 4 + 0] += lo[0] * scale[slot];
            acc[w * 4 + 1] += lo[1] * scale[slot];
            acc[w * 4 + 2] += hi[0] * scale[slot];
            acc[w * 4 + 3] += hi[1] * scale[slot];
        }
    }
    uint32_t packed[8];
    for (int k = 0; k < 8; ++k) packed[k] = bf16_rne(acc[2 * k]) | bf16_rne(acc[2 * k + 1]) << 16;
    uint4* dst = reinterpret_cast<uint4*>(out + static_cast<size_t>(token) * hidden + chunk * 16);
    dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
}
