# commit
[ck_tile(10/27/2025, 6d709dac)](https://github.com/ROCm/composable_kernel/commit/6d709dac41409a339b82a83ea59e03fbb37c7005)

# tips
 - install clangd extension in vscode and its intelligence should be helpful for writing ck code.

# optimization gemm using ck
| version | changes | performance | comment |
| --- | --- | --- | --- |
| base| - | 5.6T | - |
|| add `__launch_bounds__(256, 1)`| 22.0T |avoid spill: previous: `903`, now: `0`| 
|| add alignment `8` to tensor view | 64.0T | generated code will use `buffer_load_dwordx4` instead of `buffer_load_ushort`|
| swizzle | add swizzle for lds read/write | 84.2 T | avoid bank conflict using swizzle |
