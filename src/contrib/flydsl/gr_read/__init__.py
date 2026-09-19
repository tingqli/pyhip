# SPDX-License-Identifier: MIT
"""gfx942 BF16 GRRead；公开入口按batch自动选择N2/N4/N8。"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import CombinedPaddedGRRead

__all__ = ["CombinedPaddedGRRead"]


def __getattr__(name):
    if name == "CombinedPaddedGRRead":
        from .runtime import CombinedPaddedGRRead
        globals()[name] = CombinedPaddedGRRead
        return CombinedPaddedGRRead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")