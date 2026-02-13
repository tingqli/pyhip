# commit
[ck_tile(10/27/2025, 6d709dac)](https://github.com/ROCm/composable_kernel/commit/6d709dac41409a339b82a83ea59e03fbb37c7005)

# tips
 - install clangd extension in vscode and its intelligence should be helpful for writing ck code.
 - use thread trace to get micro-arch level trace: ``
 - use rocprof-compute to get bank conflict metric:
 ```
 rocprof-compute profile --no-roof -b 12 --name mytest -- python gemm-ck.py 0
 rocprof-compute analyze -p workloads/mytest/MI308X/
 ```

# optimization gemm using ck
| version | changes | performance | comment |
| --- | --- | --- | --- |
| base| - | 5.6T | - |
|| add `__launch_bounds__(256, 1)`| 22.0T |avoid spill: previous: `903`, now: `0`| 
|| add alignment `8` to tensor view | 64.0T | generated code will use `buffer_load_dwordx4` instead of `buffer_load_ushort`|
| swizzle | add swizzle for lds read/write | 84.2 T | avoid bank conflict using swizzle |
| prefetch | prefetch next global tile | 116 T | global load will run parallel with compute |
|| global prefetch 3 tiles, ds prefetch 1 tile | 117 T | |
| schedule | use `__builtin_amdgcn_sched_group_barrier` to interleave vmem/ds/mfma | 170 T | global prefetch 2 tiles |
| wave layout | 4 waves load 16x32 than repeat 4 times instead of load 64x32; remove `transpose` in mfma | 200 T | work closer for 4 waves will be good for memory access; `transpose` will not be good for storing memory access.|
