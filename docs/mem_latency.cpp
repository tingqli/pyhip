/* Memory loads latency

When pipelining a for loop with data-load stages & computation stages, 
we need rough memory access latency data to estimate how many cycles 
in advance the data loads should be.

Following code measures latency.
*/
#include <hip/hip_fp16.h> // for __fp16
#include <hip/hip_bf16.h> // for bfloat16
#include "hip/hip_runtime.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using float16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float16x32 = __attribute__((__vector_size__(32 * sizeof(__fp16)))) __fp16;
using float32x16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x16_t = __attribute__((__vector_size__(16 * sizeof(int)))) int;

using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;
__device__ __inline__
static int32x4_t llvm_buffer_load_dwordx4(int32x4_t srsrc,
                        int32_t voffset,
                        int32_t soffset,
                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

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

// https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html#OutputOperands

template<typename T>
__device__ T ds_read_b128(T* base, int index) {
    T v;
    static_assert(sizeof(T) == sizeof(int32x4_t));
    as3_uint32_ptr vaddr = (as3_uint32_ptr)(base + index);
    asm volatile("ds_read_b128 %[vdst], %[vaddr] offset:%[offset]"
                : [vdst]"=v"((int32x4_t&)(v))
                : [vaddr]"v"(vaddr),[offset]"i"(0));
    return v;
}
template<uint16_t cnt>
__device__ void s_waitcnt_lgkmcnt() {
    asm volatile ("s_waitcnt lgkmcnt(%0)\n"::"i"(cnt));
}

template<typename T>
__device__ T buffer_load_dwordx4(BufferResource& buffer, int soffset, int voffset) {
    T v;
    asm volatile("buffer_load_dwordx4 %[vdst], %[vaddr], %[srsrc], %[soffset] offen\n"
        :[vdst]"=v"(v)
        :[vaddr]"v"(voffset), [srsrc]"s"(buffer.descriptor), [soffset]"s"(soffset));
    return v;
}
template<uint16_t cnt>
__device__ void s_waitcnt_vmcnt() {
    asm volatile ("s_waitcnt vmcnt(%0)\n"::"i"(cnt));
}

template<typename F>
__device__ uint64_t get_cycles(F func) {
    uint64_t start, end;
    __builtin_amdgcn_sched_barrier(0);
    asm volatile ("s_memtime %0\n":"=s"(start):);
    __builtin_amdgcn_sched_barrier(0);

    func();

    __builtin_amdgcn_sched_barrier(0);
    asm volatile ("s_memtime %0\n":"=s"(end):);
    __builtin_amdgcn_sched_barrier(0);

    return end - start;
}

__global__ __launch_bounds__(256, 1) void test(int* A, int K, int i0, int* indices) {
    __shared__ int32x4_t lds[1024*4];

    uint64_t start, end;
    uint64_t dc = 0;
    for(int i = 0; i < 10; i++) {
        dc += get_cycles([]{});
    }
    if (threadIdx.x == 0)
        printf(" s_memtime overhead cycles: %lu\n", dc/10);

    dc = 0;
    BufferResource buff(A, K*sizeof(int));
    int soffset = i0*sizeof(int32x4_t);
    int voffset = threadIdx.x * sizeof(int32x4_t);
    uint64_t count = 0;
    int32x4_t data;
    for(int i = 0; i < 64*1024; i += 64) {
        dc += get_cycles([&](){
            data = buffer_load_dwordx4<int32x4_t>(buff, soffset, voffset);
            s_waitcnt_vmcnt<0>();
        });
        count ++;
        soffset =  __builtin_amdgcn_readfirstlane(data[0]) * sizeof(int);
        lds[i + threadIdx.x] = data;
    }
    if (threadIdx.x == 0)
        printf(" buffer_load_dwordx4 latency cycles: %lu\n", dc/count);


    dc = 0;
    int index = indices[threadIdx.x];
    for(int i = 0; i < 10; i++) {
        dc += get_cycles([&](){
            ds_read_b128(lds, index);
            s_waitcnt_lgkmcnt<0>();
        });
    }
    if (threadIdx.x == 0)
        printf(" ds_read_b128 latency cycles: %lu\n", dc/10);
}

/*PYHIP

import torch
torch.set_printoptions(linewidth=300)
torch.cuda.set_device(2)
torch.set_default_device('cuda')
torch.manual_seed(0)

# passing 4GB buffer into kernel has problem, all reads from kernel got 0
# but <4GB has no such problem

K=1024*1024*1024-1
#A = torch.randn(K, dtype=torch.int32)
A = torch.randint(0, K-4096, (K,), dtype=torch.int32)

voff = [i*1 for i in range(64)]
voff = torch.tensor(voff, dtype=torch.uint32)

hip.test([1], [64], A.data_ptr(), K, 0, voff.data_ptr())
hip.test([1], [64], A.data_ptr(), K, 1, voff.data_ptr())
hip.test([1], [64], A.data_ptr(), K, 2, voff.data_ptr())
hip.test([1], [64], A.data_ptr(), K, 3, voff.data_ptr())

*/