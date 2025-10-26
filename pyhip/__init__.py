from .hiptools import module

class torchPerf(object):
    def __init__(self):
        global torch
        import torch

        self.profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_flops=True,
                        with_stack=True,
                        #on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_trace_dir", use_gzip=True)
                        )

    def __enter__(self):
        self.profiler.start()
        return self

    def __exit__(self, type, value, traceback):
        torch.cuda.synchronize()
        self.profiler.stop()
        print(self.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

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

    def show(self, flops = None, rw_bytes = None):
        msg = f"{self.name} dt = {self.dt_ms*1e3:.3f} us"
        if flops and flops > 0:
            msg += f"  {flops*1e-9/self.dt_ms:.1f} TFLOPS "
        if rw_bytes and rw_bytes > 0:
            msg += f"  {rw_bytes*1e-6/self.dt_ms:.1f} GB/s "
        print(msg)
