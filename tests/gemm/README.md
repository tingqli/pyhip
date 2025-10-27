# GEMM optimization on AMDGPU

Setup [thread tracing](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-thread-trace.html) profiling environment is very helpful.

## single wave-per-SIMD mode
The most fast cache is register, allocating all registers of a SIMD to a single wave can maximize register reuse at the cost of giving-up thread-level parallelsim and relying on instruction-level parallelsim.

## pipelineing
a typical GEMM kernel has data-dependencies that prevent independent execution unit to work in parallel:
 - load data0 from HBM
 - cooperatively store data0 to LDS
 - sync
 - load data0 from LDS
 - MFMA on data0

W/o the help from thread-level parallelsim, we need to pipeline them manually:
 - prelog
 - for loop (following items are fully independent and can run in parallel)
   - sync
   1. MFMA on `data0` in REG_A0
   2. load `data1` from LDS_A into REG_A1
   3. store `data2` in REG_M into LDS_B
   4. prefetch `data3` from HBM into REG_M
   5. swap REG_A0 & REG_A1, swap LDS_A & LDS_B

note:

 - LDS_A & LDS_B ping-pong buffer avoids extra syncs (but this is not absolutely required); 
 - REG_A0 & REG_A1 ping-pong buffer is used but can be saved if we can pipeline in smaller granularity.
 - anti-dependency of step-4 on step-3 causes no extra cycles (possibly the register's content was copied to LDS's request fifo very soon within issue cycle).

## manual instruction scheduling

 - to reach compute-bound, MFMA must be issued continously w/o any bubbles
 - long latency vmem instruction requires carefully scheduled w/o blocking other instruction's issue


## thread-block swizzle
[thread-block tests](../threadblock-scheduling/test.py) demonstrated the fact that thread blocks are dispatched by GPU HW according to some pre-determined pattern, in which some groups of thread-blocks will be on the same die with shared L2-cache, thread-block swizzle use this fact to map work-loads that accessing overlaping/shared data into same block-group, so they can benefit from same L2-cache and increase L2 hit-rate.