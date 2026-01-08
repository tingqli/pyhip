#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

__global__ void __launch_bounds__(256, 1) ttt(uint M, uint* K) {
    K[0] = M / 38;
}
