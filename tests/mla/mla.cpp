#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float16x8 = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

/********************************************************************************
 256个线程，4个warp，模拟Q*K部分的性能瓶颈, 输出保持比较小的尺寸: m64x16
********************************************************************************/
__global__ void __launch_bounds__(256, 1) gemm_qk_lds(__fp16* Q, __fp16* K, float* P, int kv_len, int sm) {
    constexpr int dim = 512;
    constexpr int num_heads=128;

    // 32 个 16x16 query 子块
    uint64_t q[dim/16];

    int batch = blockIdx.y;
    int head_off = blockIdx.x * 64;

    int warp_id = threadIdx.x >> 6;
    int lane = threadIdx.x & 63;
    auto i_off = (lane & 15) * dim;
    auto k_off = (lane >> 4) * 4;
    auto ik_off = i_off + k_off;
    auto m_off = (batch * num_heads + head_off) + warp_id * 16; // 每个warp负责16行
    m_off *= sm;
    Q += m_off * dim; // 每个warp负责16行
    Q += ik_off;
    // K : [batch-size, kv_len, dim]
    K += batch * kv_len * dim;
    //K += ik_off;
    for(int qi = 0; qi < 32; qi ++) {
        q[qi] = *(uint64_t*)(Q + qi*16);
    }

    __shared__ __fp16 kv_shared[4*16*512];

    union bigType {
        float16x4 f4[2];
    };
    auto load_kv_16cols = [&](__fp16* Ksrc, int lds_off){
        for(int row = 0; row < 4; row ++) {
            // 多个warp分担不同行的写入
            auto row0 = (warp_id * 4 + row);
            auto row_off = row0 * dim;
            for(int col2 = lane; col2 < dim/8; col2 += 64) {
                // layout 不变复制到 LDS
                //*(uint4*)(kv_shared + row_off + col*8) = *(uint4*)(K + row_off + col*8);
                // MFMA_16x16x16_FP16 A 矩阵形式写入到LDS
                // 加载 8 个 fp16, 包含两个 float16x4

                // DWORDx2跟DWORDx4读取存在巨大的性能差距，为什么？
                // 此处从外存读入一定要使用DWORDx4
                bigType data = *((bigType*)(Ksrc + row_off) + col2);

                // float16x4 data = *((float16x4*)(K + row_off) + col);
                //swizzle
                //auto phase = (col & 15);
                //*((float16x4*)kv_shared + col*16 + (row0 ^ phase)) = data;

                auto col = col2 * 2;
                auto phase = (col2 & 15);
                auto row_off0 = (row0 ^ phase);
                int col_off = col * 16;
                *((float16x4*)(kv_shared + lds_off) + (col_off + row_off0)*(1)) = data.f4[0];
                *((float16x4*)(kv_shared + lds_off) + (col_off + 16 + row_off0)*(1)) = data.f4[1];
            }
        }
    };

    int kv_lds_offset = 0;
    
    load_kv_16cols(K, kv_lds_offset*16*512);
    // 循环n维度
    float32x4 c = {0};
    for(int n = 0; n < kv_len-16; n += 16, K += dim*16*sm) {
        // memory coalescing : 外存读入 K 到 LDS
        // 经过寄存器倒手，每个lane读入 dwordx4 也就是 halfx8 个数据，64个lane正好一次读完 512个half
        //   读入到LDS时，需要swizzle以避免LDS的bank-conflict
        //   swizzle设计合理时，写入LDS和读出LDS都不会发生bank-conflict
        //   因为我们以16x16xfp16为单位读取，但是以1x512为单位写入，swizzle方案会有点复杂
        //   首先尝试仅仅避免LDS读取bank-conflict, 写入时不避免，
        //   通过汇编修改发现，计算偏移开销不大，但是写入因为发生bank-conflict开销很大
        // 如何swizzle才能避免写入和读出 bank-conflict ?
        //
        __syncthreads();

        // LDS 加载 K 到寄存器, 如果出现bank-conflict会慢得离谱
        for(int qi = 0; qi < 32; qi ++) {
            int row = lane & 15;
            int col = (lane >> 4) + qi*4;
            int phase = (col >> 1) & 15;
            float16x4 key = *((float16x4*)(kv_shared + kv_lds_offset*16*512) + col*16 + (row ^ phase));

            c = __builtin_amdgcn_mfma_f32_16x16x16f16(key, (float16x4&)q[qi], c, 0, 0, 0);
        }
        auto kv_lds_offset_next = (kv_lds_offset + 1) & 1;
        load_kv_16cols(K + dim*16, kv_lds_offset_next*16*512);
        kv_lds_offset = kv_lds_offset_next;
    }

    __syncthreads();
    // LDS 加载 K 到寄存器, 如果出现bank-conflict会慢得离谱
    for(int qi = 0; qi < 32; qi ++) {
        int row = lane & 15;
        int col = (lane >> 4) + qi*4;
        int phase = (col >> 1) & 15;
        float16x4 key = *((float16x4*)(kv_shared + kv_lds_offset*16*512) + col*16 + (row ^ phase));
        c = __builtin_amdgcn_mfma_f32_16x16x16f16(key, (float16x4&)q[qi], c, 0, 0, 0);
    }

    {
        auto i = (lane & 15);
        auto j = (k_off);
        *(float32x4*)(P + (m_off + i)*16 + j) = c;
    }
}
