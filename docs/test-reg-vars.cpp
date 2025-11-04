/*
invoke with:

python -m pyhip ./test-reg-vars.cpp

    this example shows that inline asm reg type hint can determine the final
    physical register type.
*/
#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__global__ void __launch_bounds__(256, 1) test_mfma(__fp16* A, __fp16* B, int nstrideAB, float* C, int nstrideC) {
    int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    int lane = threadIdx.x & 63;
    auto m_off = (lane & 15) * nstrideAB;
    auto k_off = (lane >> 4) * 4;

    float16x4 a;
    float16x4 b;
    float32x4 c = {0};

    a = *(float16x4*)(A + m_off + k_off);
    b = *(float16x4*)(B + m_off + k_off);
#if 0
    c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
#else
    // check out the assembly we know the register type hint in asm
    // can determine the final actual physical register
    asm("v_mfma_f32_16x16x16_f16 %0, %1, %2, %3\n"
                : "+v"(c)
                : "a"(a), "a"(b), "v"(c)
                :);
#endif
    auto* p = C + (lane>>4)*4*nstrideC + (lane & 15);
    p[0] = c[0]; p += nstrideC;
    p[0] = c[1]; p += nstrideC;
    p[0] = c[2]; p += nstrideC;
    p[0] = c[3]; p += nstrideC;
}

/*PYHIP

import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

M,N,K=16,16,16
A = torch.randn(M, K, dtype=torch.float16)
B = torch.randn(N, K, dtype=torch.float16)
C = torch.randn(M, N, dtype=torch.float)
ref = torch.nn.functional.linear(A, B).to(dtype=torch.float)

hip.test_mfma([1], [64], A.data_ptr(), B.data_ptr(), K, C.data_ptr(), N)

print(torch.allclose(ref, C, atol=0.1, rtol=0.1))

*/