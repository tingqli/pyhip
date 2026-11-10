"""Detects other processes' kernels on a GPU while a timed block runs (docs/record/moe_small_m_gfx942.md).

KFD reports every process with the GPU open and its live CU occupancy, also across containers. Processes
present before this process initializes the GPU are foreign; a background thread samples their occupancy
during a block and the block counts as contaminated if any is nonzero.
"""

import threading
import time


class ExternalLoadMonitor:
    def __init__(self, gpu_index, interval_s=0.002):
        import amdsmi

        self.amdsmi = amdsmi
        amdsmi.amdsmi_init()
        self.handle = amdsmi.amdsmi_get_processor_handles()[gpu_index]
        self.interval_s = interval_s
        self.before = {p["pid"] for p in amdsmi.amdsmi_get_gpu_process_list(self.handle)}
        self.mine = None

    def claim(self):
        """Call once this process has initialized the GPU: the new PIDs are this process."""
        now = {p["pid"] for p in self.amdsmi.amdsmi_get_gpu_process_list(self.handle)}
        self.mine = now - self.before

    def foreign_occupancy(self):
        procs = self.amdsmi.amdsmi_get_gpu_process_list(self.handle)
        return sum(p["cu_occupancy"] for p in procs if p["pid"] not in (self.mine or ()))

    def __enter__(self):
        self.peak, self.samples = 0, 0
        self._stop = threading.Event()

        def sample():
            while not self._stop.is_set():
                self.peak = max(self.peak, self.foreign_occupancy())
                self.samples += 1
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.foreign_occupancy())
        return False

    @property
    def clean(self):
        return self.peak == 0
