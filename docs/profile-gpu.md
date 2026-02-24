# Low Overhead Profiler

Comparing to torch profiler, GPU vendor provides profiler with much lower overheads which gives much more accurate measurements.

> ref: https://docs.vllm.ai/en/stable/contributing/profiling.html

# NVIDIA GPU:

https://developer.nvidia.com/nsight-systems/get-started

Download & install both:
 - Linux for capture log
 - Windows for analyse log

```bash
/opt/nvidia/nsight-systems/2025.5.1/bin/nsys profile --trace=cuda python3 your-torch-script.py
```

To profile at kernel level, we need install [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute-history) and using commond line tips shown [here](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#launching-nvidia-nsight-compute-from-a-cuda-kernel), for example:

```bash

# use nvidia-smi to check driver version
Thu Oct 30 10:22:54 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8     |

# download correct version of "NVIDIA Nsight Compute" from https://developer.nvidia.com/nsight-compute-history
# Nsight System right-click prompt can tell us  --kernel-name --launch-skip  --launch-count to profile the exact kernel launch instance.
/usr/local/NVIDIA-Nsight-Compute/ncu --kernel-name gemm_autotune_kernel --launch-skip 4 --launch-count 1 -o profile python3 example_gemm_autotune.py

# copy the generated file profile.ncu-rep to Windows host and view it with Windows's version of Nsight-Compute

```


# AMDGPU: kernel level
https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html

```bash
rocprofv3 --kernel-trace --output-format pftrace -- python myapp.py
```
Download and open the generated pftrace file with https://ui.perfetto.dev/.

# AMDGPU: instruction-level

thread trace: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/amd-mainline/how-to/using-thread-trace.html

 - ROCm 7.x
```bash
$ docker run -it --rm --cap-add=SYS_ADMIN --network=host --device=/dev/kfd --device=/dev/dri  --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v ~/pyhip/:/pyhip --entrypoint /bin/bash rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0 
```
 - manually install [ROCprof trace decoder](https://github.com/ROCm/rocprof-trace-decoder/releases)
 
```bash
wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
bash ./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --skip-license --prefix=./

```

 - manually install [ROCprof Compute Viewer](https://github.com/ROCm/rocprof-compute-viewer/releases) on Windows to view results
 - use following example configs

```yaml
jobs:
    -
        kernel_include_regex: (.*name_patten1.*)|(.*name_patten2.*)
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

```bash
rocprofv3 -i trace.yaml -- python myapp.py
```