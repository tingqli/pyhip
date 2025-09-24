
use self-instrumentation to understand HW thread-block scheduling behaviour.

some key points:

 - thread blocks are launched along x-dimension (the first one in dim3) first, then y & z dimensions, and they are launched concurrently in a batch of size `num_CU * ((SIMD/CU)*(waves/SIMD)/(waves/block))`, if total number of thread-blocks are bigger than that, the batches are launched in sequence.

 - depending on LDS&register resource required by compiled kernel, concurrent waves/warps per-SIMD is ranging from 1 to 8 (for CDNA3), and this number is shown by compiler as `Occupancy [waves/SIMD]` when option `-Rpass-analysis=kernel-resource-usage` is specified in command-line;

 - kernel can do their own mapping between thread-block index and work-loads index to make best use of L2 cache, this technique is called `Threadblock Swizzling`


