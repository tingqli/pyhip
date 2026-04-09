""" Some useful helpers """

__all__ = [
    'cudaPerf', 'torchPerf', 'calc_diff', 'div_up', "pre_shuffle", "run_perftest", "set_device"
]


class cudaPerf(object):
    def __init__(self, flops = 0, rw_bytes = 0, name="", verbose=1):
        global torch
        import torch
        self.flops = flops
        self.name = name
        self.verbose = verbose
        self.rw_bytes = rw_bytes
        self.ev_start = torch.cuda.Event(enable_timing=True)
        self.ev_end = torch.cuda.Event(enable_timing=True)
        self.latencies = []

    def __enter__(self):
        torch.cuda._sleep(1_000_000)
        self.ev_start.record()
        return self

    def __exit__(self, type, value, traceback):
        self.ev_end.record()
        torch.cuda.synchronize()
        self.dt_ms = self.ev_start.elapsed_time(self.ev_end)
        self.latencies.append(self.dt_ms * 1e-3)
        if self.verbose:
            self.show(self.flops, self.rw_bytes)

    def dt(self, excludes=0):
        return sum(self.latencies[excludes:])/len(self.latencies[excludes:])

    def bw(self, excludes=0):
        avg_dt_ms = self.dt(excludes) * 1e3
        return self.rw_bytes*1e-6/avg_dt_ms

    def show(self, flops = None, rw_bytes = None):
        if flops is None: flops = 0
        msg = f"{self.name} : {flops*1e-6:.0f} MFLOP / {self.dt_ms*1e3:.3f} us "
        if flops and flops > 0:
            msg += f"  {flops*1e-9/self.dt_ms:.1f} TFLOPS "
        if rw_bytes and rw_bytes > 0:
            msg += f"  {rw_bytes*1e-6/self.dt_ms:.1f} GB/s "
        print(msg)

    def tflops(self, excludes=0):
        avg_dt_ms = self.dt(excludes) * 1e3
        return self.flops*1e-9/avg_dt_ms

class torchPerf(object):
    def __init__(self, dir = ""):
        global torch
        import torch
        import os        
        self.TP_DIR = os.getenv("TORCH_PERF_DIR", dir)
        if self.TP_DIR != "":
            self.profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        self.TP_DIR, use_gzip=True))
        else:
            self.profiler = None

    def __enter__(self):
        if self.profiler is not None:
            self.profiler.start()
        return self

    def __exit__(self, type, value, traceback):
        if self.profiler is not None:
            self.profiler.stop()
            print(self.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

# type-less preshuffle
def pre_shuffle(x, mfma_MN):
    M, K = x.shape
    K_bytes = K * x.itemsize
    sizeof_DW4 = 16
    mfma_K_lanes = 64 // mfma_MN
    mfma_K_L = sizeof_DW4//x.itemsize
    mfma_K = mfma_K_lanes * mfma_K_L 

    assert M % mfma_MN == 0
    mfma_K_bytes = mfma_K_lanes * sizeof_DW4
    assert K_bytes % mfma_K_bytes == 0

    x = x.reshape(M//mfma_MN, mfma_MN, K//mfma_K, mfma_K_lanes, mfma_K_L)
    x = x.permute(0,2,3,1,4)
    return x.contiguous()

# https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/numeric.py#L5
def calc_diff(x: "torch.Tensor", y: "torch.Tensor", diff_thr=None):
    def get_diff(x, y):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        if denominator == 0:    # Which means that all elements in x and y are 0
            return 0.0
        sim = 2 * (x * y).sum() / denominator
        diff = (1 - sim).item()
        return diff

    diff = get_diff(x, y)
    if diff != diff or (diff_thr is not None and diff > diff_thr):
        if diff_thr is None or diff_thr < 0: return diff
        print(x)
        print(y)
        print(x.shape)
        print(y.shape)
        if len(x.shape) == 2:
            print_count = 0
            M, N = x.shape
            for m in range(0,M,16):
                dm = get_diff(x[m:m+16,:], y[m:m+16,:])
                if dm != dm:
                    print(x[m:m+16,:])
                    print(y[m:m+16,:])
                    assert 0
                elif dm >= diff_thr:
                    print_count += 1
                    assert print_count < 16, f"Too many errors in calc_diff with {diff_thr=:.3f}"
                    print(f"[{m:6}]: ", end="")
                    for n in range(0,N,16):
                        d = get_diff(x[m:m+16,n:n+16], y[m:m+16,n:n+16])
                        if d < diff_thr:
                            print(f"_.__ ", end="")
                        else:
                            print(f"{d:.2f} ", end="")
                    print()
            print()
        assert 0, f"{diff=} > {diff_thr=} !!!"
    assert diff == diff, "diff is nan!"
    return diff

def div_up(x, y):
    return (x + y - 1) // y

def run_perftest(kernel, *args, **kwargs):
    global torch
    import torch
    import copy

    def extract_attr(obj, name, default):
        nonlocal kwargs
        attr = kwargs.get(name, None)
        if attr is not None:
            del kwargs[name]
        else:
            attr = default
        return attr

    num_verbose = extract_attr(kwargs, 'num_verbose', 0)
    kernel_name = getattr(kernel, "__name__", "kernel?")
    perf = cudaPerf(name = kernel_name, verbose=num_verbose)

    num_iters = extract_attr(kwargs, 'num_iters', 10)
    num_warmup = extract_attr(kwargs, 'num_warmup', 2)
    num_copies = extract_attr(kwargs, 'num_copies', 0)
    num_flops = extract_attr(kwargs, 'num_flops', 0)
    num_bytes = extract_attr(kwargs, 'num_bytes', 0)
    num_spec_tag = extract_attr(kwargs, 'num_spec_tag', '')

    if num_copies == 0:
        copy_size = 0
        for a in args:
            if isinstance(a, torch.Tensor):
                #print(a.shape, a.dtype)
                copy_size += a.numel() * a.element_size()
        for k in kwargs:
            if isinstance(kwargs[k], torch.Tensor):
                #print(kwargs[k].shape, kwargs[k].dtype)
                copy_size += kwargs[k].numel() * kwargs[k].element_size()
        # up-to 4GB
        num_copies = max(int(4e9 / copy_size), num_warmup + num_iters)

    args_copies = []
    kwarg_copies = []
    for _ in range(num_copies):
        args_copies.append(copy.deepcopy(args))
        kwarg_copies.append(copy.deepcopy(kwargs))

    for i in range(num_warmup + num_iters):
        with perf:
            out = kernel(*args_copies[i%num_copies], **kwarg_copies[i%num_copies])

    dt = perf.dt(excludes=num_warmup)

    if num_flops or num_bytes:
        msg = f"{kernel_name} {num_spec_tag} :  {dt*1e6:.0f} us"
        if num_flops:
            msg += f", {num_flops/dt*1e-12:.3f} TFLOPS"
        if num_bytes:
            msg += f", {num_bytes/dt*1e-12:.3f} TB/s"
        print(msg)
    return out, dt*1e6

def set_device(selected_device = -1):
    global torch
    import torch
    if not torch.cuda.is_available():
        assert 0, "No CUDA device found in torch"

    if selected_device < 0:
        max_free_mem = -1
        for device_id in range(torch.cuda.device_count()):
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            free_mem_gb = free_mem / (1024 ** 3)
            total_mem_gb = total_mem / (1024 ** 3)
            if free_mem > max_free_mem:
                max_free_mem = free_mem
                max_total_mem = total_mem
                selected_device = device_id

    torch.set_default_device("cuda")
    torch.set_printoptions(linewidth=3000, sci_mode=False, edgeitems=8, )
    torch.cuda.set_device(selected_device)
    torch.manual_seed(0)
    free_mem, total_mem = torch.cuda.mem_get_info(selected_device)
    print(f"Use cuda device {selected_device} with {free_mem*100/total_mem:.1f}% Free mem : {free_mem/(1024**3):.0f}GB / {total_mem/(1024**3):.0f} GB")
    return selected_device