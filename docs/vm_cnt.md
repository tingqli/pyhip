# How VM_CNT works?

We want to find out the max possible VM_CNT that can be tracked by AMDGPU, we do this by issue N buffer load instruction into different VGPR, then we wait vm_cnt to be Q before we check the (N-1-Q)'th VGPR's content, if N is big enough, VM_CNT may overflow and VGPR's content will be wrong:

But our test shows that even we issued 4096 buffer-loads, and wait for 63 loads on the fly, the data before last-63 loads was indeed loaded into the register, this feature seems to track much more loads than we thought. OR it's more likely implemented in another simple way:

**It has an internal queue with only 64 (the VM_CNT stored in 6-bits) entries, any VMEM requests more than 64 will stall the issue pipe, thus HW tracks no more than 64 VMEM-requests.**


This also suggests that:
 - if any wave try to issue another VMEM while 64 concurrent VMEMs already on-the-fly, the wave has to stall (or switch out).

 - Maybe we can guess the same for 4-bit `LGKMcnt`. only 16 LDS requests can be run on-the-fly, but considering the super-fast throughput of LDS, that causes no stall most-likely.

Further tests using [rocprofiler thread-trace](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html) (need to uncomment the `Random Access` line) shows that if we issue VMEM continously, the 24'th VMEM load will stall the issue pipe which is even earlier than expected.

<details>

<summary>source codes</summary>

```c++
#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;

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

template<typename F, size_t... Is>
constexpr void compile_time_loop_impl(F&& f, std::index_sequence<Is...>) {
    (f(std::integral_constant<size_t, Is>{}), ...);
}

template<size_t N, typename F>
constexpr void compile_time_loop(F&& f) {
    compile_time_loop_impl(std::forward<F>(f), 
                          std::make_index_sequence<N>{});
}

template<uint16_t cnt>
__device__ void s_waitcnt_vmcnt() {
    asm volatile ("s_waitcnt vmcnt(%0)\n"::"i"(cnt));
}

template<typename T>
__device__ T buffer_load_dword(BufferResource& buffer, int soffset, int voffset) {
    T v;
    asm volatile("buffer_load_dword %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
        :[vdst]"=v"(v)
        :[vaddr]"v"(voffset), [srsrc]"s"(buffer.descriptor), [soffset]"s"(soffset));
    return v;
}

// constexpr伪随机数生成函数
constexpr uint32_t nextRandom(uint32_t seed, uint32_t max_mask) {
    constexpr uint32_t a = 1664525;
    constexpr uint32_t c = 1013904223;
    //constexpr unsigned long m = 4294967295; // 2^32 - 1
    return (a * seed + c) & max_mask;
}

__global__ __launch_bounds__(256, 1) void test(int* A, int cnt, int *matched) {
    BufferResource buff(A, cnt*sizeof(int));
    int voffset = threadIdx.x * sizeof(int);
    constexpr int NUM_VALUES = 128;
    int value[NUM_VALUES];

    A += blockIdx.x * cnt;
    matched += blockIdx.x * 64;
    auto clear = [&](){
        for(int i = 0; i < NUM_VALUES; i++) value[i] = 0;
    };
    {
        constexpr int N = 128;
        constexpr int waitN = 63;
        clear();
        asm volatile("buffer_inv sc0 sc1\n");
        s_waitcnt_vmcnt<0>();
        __builtin_amdgcn_sched_barrier(0);
        compile_time_loop<N>([&](auto i){
            //constexpr int index = nextRandom(i, (8192*8)-1);     // Random Access
            constexpr int index = i;
            constexpr int ti = i & (NUM_VALUES - 1);
            value[ti] = buffer_load_dword<int>(buff, index*64*sizeof(int), voffset);
        });
        s_waitcnt_vmcnt<waitN>();
        __builtin_amdgcn_sched_barrier(0);
        matched[threadIdx.x] = value[(N-1-waitN) & (NUM_VALUES - 1)];
        __builtin_amdgcn_sched_barrier(0);

        int sum = 0;
        for(int i = 0; i < NUM_VALUES; i++)
            sum += value[i];
        if (sum == -123) {
            printf("prevent compiler opt\n");
        }
    }
}

```

```python

import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)
num_CU = torch.cuda.get_device_properties().multi_processor_count

magic_num = 1234
cnt = 64*8192*8

for i in range(10):
    A = torch.arange(0, num_CU*cnt, dtype=torch.int32)
    B = torch.zeros([num_CU*64], dtype=torch.int32)
    hip.test([num_CU], [64], A.data_ptr(), cnt, B.data_ptr())
    print(B[0].item()//64)

```
</details>