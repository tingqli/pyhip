# coalescing memory read
- `DWORDx4` is much faster than `DWORDx2`

- coalescing happens as long as data-addresses from 64-lanes fall in same area of cache-lines, they can be permuted w/o damage performance.

- coalescing is best if data-addresses from 64-lanes fall in continous cache-lines (check `sum_1d_16x64`,`sum_1d_8x128`,`sum_1d_4x256`, `sum_1d_2x512`). But, when actuall Occupancy [waves/SIMD] is forced to be 1 or 2 (by setting sharedMemBytes), we didn't see too much of differences. the reason could be that in small occupancy case, memory accessing is not well-pipelined, latency dominates.

- memory addresses between warps do not requires coalescing, the performance between `sum_1d_x4e` & `sum_1d_x4` are similar.

- `buffer_load_dwordx4` is slightly faster than `gloabl_load_dwordx4`, but with some extra features.

# How to further increase bandwidth when Occupancy is limited to 1 (due to some un-controllable factor)?

Why we can reach much higher memory-bandwidth when accessing it with higher occupancy? Especially considering the fact that the execution units are the same in both `occupancy=8` & `occupancy=1` cases.

This can be answered by `sum_1d_x4e2` wich almost doubled the `occupancy=1` case bandwith comparing to `sum_1d_x4e`. 

```asm
LOOP:
    LOAD_DWORDX4(R1, mem_ptr)
    R2 += R1
    branch LOOP
```

in the pseudocode above, second iteration of `LOAD_DWORDX4` will not start before first iteration completes, simply to avoid [data hazard](https://en.wikipedia.org/wiki/Hazard_(computer_architecture)#Examples). And the accessing speed is dominated by memory-access-latency instead of throughput (second request is sent only after first request returns).

This is not a problem in out-of-order execution unit like x86 CPU since HW will automatically do register renaming whenever an assignment happens, so second iteration will execute on another physical register w/o damaging the data in first iteration.

But on GPU with much simpler execution pipeline, it's only equipped with HW-deisgn like `scoreboarding` to simply prevent such hazards by detecting them and inserting neccessary stalls(pipeline bubbles). if occupancy>1, such stalls can be avoided by swithing to second wave(set of threads). but when `occupancy=1`, we have to do our own pipeline like `sum_1d_x4e2`
