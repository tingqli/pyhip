
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
        msg = f"{self.name} dt = {self.dt_ms*1e3:.3f} us"
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

def calc_diff(x: "torch.Tensor", y: "torch.Tensor"):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:    # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    diff = (1 - sim).item()
    assert diff == diff, "diff is nan!"
    return diff

def div_up(x, y):
    return (x + y - 1) // y
