# thread trace

https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-thread-trace.html

 - ROCm 7.x
 - manually install [ROCprof trace decoder](https://github.com/ROCm/rocprof-trace-decoder/releases)
 - manually install [ROCprof Compute Viewer](https://github.com/ROCm/rocprof-compute-viewer/releases) on Windows to view results
 - use following example configs

```yaml
jobs:
    -
        kernel_exclude_regex:
        kernel_iteration_range: "[1, [3-4]]"
        output_file: out
        output_directory: ck_test
        output_format: [json, csv, otf2, pftrace]
        truncate_kernels: true
        sys_trace: true # enable for pftrace and otf2
        advanced_thread_trace: true # enable for att and ui folder
        att_target_cu: 1
        att_shader_engine_mask: "0xf" # collect one CU from 4 SEs
        att_simd_select: "0xf" # collect 4 SIMDs on single CU
        att_buffer_size: "0x6000000"
    -
        pmc: [SQ_WAVES, FETCH_SIZE]
``` 

