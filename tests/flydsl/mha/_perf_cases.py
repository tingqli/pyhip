"""Logical attention workload and effective FLOPs; presets live in test_mha_pa."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Workload:
    name: str
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    dq: int = 192
    dv: int = 128
    heads: int = 16
    kv_heads: int = 1
    page: int = 64
    causal: bool = False
    window: int = -1
    sink: bool = False
    scale_mode: str = "per-token"
    input_kind: str = "bf16-source"
    seed: int = 20260905

    @property
    def flops(self):
        pairs = 0
        for q, k in zip(self.q_lens, self.kv_lens):
            if not self.causal:
                pairs += q * k
            else:
                for row in range(q):
                    diagonal = k - q + row
                    left = max(0, diagonal - self.window) if self.window >= 0 else 0
                    pairs += max(0, min(k, diagonal + 1) - left)
        return 2 * self.heads * pairs * (self.dq + self.dv)

    def unsupported(self, backend):
        if self.window >= 0 and not self.causal:
            return "windowed attention requires bottom-right causal mode"
        if backend.name == "swa_bf16":
            if self.window < 0 or not self.causal:
                return "single-wave backend requires causal SWA"
        elif self.window >= 0 or self.sink:
            if backend.arch != "gfx950":
                return "this full-MHA backend does not support SWA/sink"
        if backend.name != "bf16_942" and self.page != 64:
            return "backend supports page64 only"
        if backend.name != "bf16_942" and self.dv != 128:
            return "backend supports V128 only"
        if not backend.empty_kv and any(q > 0 and k == 0 for q, k in zip(self.q_lens, self.kv_lens)):
            return "original BF16 pipeline requires nonempty active KV"
        if self.causal and not backend.causal_short_kv and any(k < q for q, k in zip(self.q_lens, self.kv_lens)):
            return "original BF16 causal pipeline requires KV>=Q per sequence"
        return None

    def to_dict(self):
        return {**asdict(self), "effective_flops": self.flops}