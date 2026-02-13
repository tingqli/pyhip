

M = 256
N = 256
K = 2048
freq = 1.5e9 # GHz
element_size = 0.5 # fp4

num_CUs = 256
SIMD_per_CU = 4

# 1 CU (4xSIMD) do a [MxK] x [KxN] gemm
bytes_per_CU = (M*K + N*K) * element_size
mfma_cycles_per_CU = ((M//16) * (N//16) * (K//128) * 32) // SIMD_per_CU
mfma_time_per_CU = mfma_cycles_per_CU/freq

band_width = num_CUs * bytes_per_CU/mfma_time_per_CU

print(f"{band_width*1e-9:.2f} GB/s")

actual_band_width = 5000e9 # 5TB/s
mem_bound_times = (num_CUs * bytes_per_CU)/actual_band_width
mem_bound_flops = num_CUs*M*N*K*2/mem_bound_times

print(f"mem_bound_flops = {mem_bound_flops*1e-12:.2f} TFLOPS")